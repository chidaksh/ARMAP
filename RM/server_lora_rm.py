# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json

import os
from dataclasses import dataclass, field
from typing import Optional, List, Literal
import logging

import torch
import transformers
import argparse
from transformers import set_seed

from transformers import AutoTokenizer

from lora_utils import (
    SavePeftModelCallback,
    print_trainable_parameters,
    get_last_checkpoint,
    DEFAULT_PAD_TOKEN,
)
from data_utils.data_utils_rm import make_binary_reward_modeling_data_module
from models.reward_model import (
    RewardConfig,
    RewardModel,
    RewardModelTrainer as Trainer,
    compute_reward_modeling_metrics,
)

from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import tokenizer_image_token
from llava.constants import (
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)

from llava.train.train import smart_tokenizer_and_embedding_resize
from data_utils.common_utils import preprocess

torch.backends.cuda.matmul.allow_tf32 = True

logger = logging.getLogger(__name__)


class DisableLogger:
    def __enter__(self):
        logging.disable(logging.CRITICAL)

    def __exit__(self, exit_type, exit_value, exit_traceback):
        logging.disable(logging.NOTSET)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="EleutherAI/pythia-12b")
    trust_remote_code: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."
        },
    )
    # from LLaVA
    version: Optional[str] = field(default="v1")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(
        default=-1
    )  # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")


@dataclass
class DataArguments:
    dataset_path: str = field(default="tatsu-lab/alpaca_farm")
    dataset_name: str = field(default=None, metadata={"help": "Dataset name"})
    eval_dataset_path: str = field(default="tatsu-lab/alpaca_farm")
    eval_dataset_name: str = field(default="alpaca_human_preference")
    eval_size: int = field(
        default=500,
        metadata={
            "help": "Number of examples to split out from training to use for evaluation."
        },
    )
    # From LLaVA
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = "square"
    image_grid_pinpoints: Optional[str] = field(default=None)
    reward_prompt_file: Optional[str] = field(default=None)
    image_to_caption_file: Optional[str] = field(default=None)


@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    # From LLaVA
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    # From AlpacaFarm
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be left padded to this length always during training."
        },
    )
    query_len: int = field(default=None, metadata={"help": "Length of the query."})
    response_len: int = field(
        default=None, metadata={"help": "Length of the response."}
    )
    label_names: List[str] = field(
        default_factory=lambda: ["id", "index_0", "index_1", "choice"],
        metadata={
            "help": "Names of the labels in the dataset. "
            "This is needed to get transformers.Trainer to not throw those tensors away before `compute_loss`."
            "By default, the trainer throws away columns it doesn't recognize when creating the "
            "`train_dataloader` (see `_remove_unused_columns`). "
        },
    )
    padding: Literal["max_length", "longest"] = field(
        default="longest",
        metadata={
            "help": "Padding strategy. If 'max_length', pads to `model_max_length` always; this might lead to some "
            "redundant compute. If 'longest', pads to the longest sequence in the batch, capped by `model_max_length`."
        },
    )
    # From QLoRA
    full_finetune: bool = field(
        default=False, metadata={"help": "Finetune the entire model without adapters."}
    )
    adam8bit: bool = field(default=False, metadata={"help": "Use 8-bit adam."})
    double_quant: bool = field(
        default=True,
        metadata={
            "help": "Compress the quantization statistics through double quantization."
        },
    )
    quant_type: str = field(
        default="nf4",
        metadata={
            "help": "Quantization data type to use. Should be one of `fp4` or `nf4`."
        },
    )
    bits: int = field(default=4, metadata={"help": "How many bits to use."})
    lora_modules: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "Which modules to use LoRA on. If None, will use all linear layers."
        },
    )
    lora_r: int = field(default=64, metadata={"help": "Lora R dimension."})
    lora_alpha: float = field(default=16, metadata={"help": " Lora alpha."})
    lora_dropout: float = field(default=0.0, metadata={"help": "Lora dropout."})
    report_to: str = field(
        default="none",
        metadata={"help": "To use wandb or something else for reporting."},
    )
    resume_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the directory containing the checkpoint to resume."},
    )
    output_dir: str = field(
        default="./output", metadata={"help": "The output dir for logs and checkpoints"}
    )
    lora_dir: str = field(
        default="./lora", metadata={"help": "The lora dir for logs and checkpoints"}
    )
    save_path: str = field(
        default="./result/result.json", metadata={"help": "The path for saving result"}
    )
    optim: str = field(
        default="paged_adamw_32bit", metadata={"help": "The optimizer to be used"}
    )
    per_device_train_batch_size: int = field(
        default=1,
        metadata={
            "help": "The training batch size per GPU. Increase for better speed."
        },
    )
    gradient_accumulation_steps: int = field(
        default=16,
        metadata={
            "help": "How many gradients to accumulate before to perform an optimizer step"
        },
    )
    weight_decay: float = field(
        default=0.0, metadata={"help": "The L2 weight decay rate of AdamW"}
    )  # use lora dropout instead for regularization if needed
    learning_rate: float = field(default=0.0002, metadata={"help": "The learnign rate"})
    remove_unused_columns: bool = field(
        default=False,
        metadata={"help": "Removed unused columns. Needed to make this codebase work."},
    )
    max_grad_norm: float = field(
        default=0.3,
        metadata={
            "help": "Gradient clipping max norm. This is tuned and works well for all models tested."
        },
    )
    gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "Use gradient checkpointing. You want to use this."},
    )
    do_train: bool = field(
        default=False,
        metadata={"help": "To train or not to train, that is the question?"},
    )
    lr_scheduler_type: str = field(
        default="constant",
        metadata={
            "help": "Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis"
        },
    )
    warmup_ratio: float = field(
        default=0.03, metadata={"help": "Fraction of steps to do a warmup for"}
    )
    logging_steps: int = field(
        default=10,
        metadata={"help": "The frequency of update steps after which to log the loss"},
    )
    group_by_length: bool = field(
        default=True,
        metadata={
            "help": "Group sequences into batches with same length. Saves memory and speeds up training considerably."
        },
    )
    save_strategy: str = field(
        default="steps", metadata={"help": "When to save checkpoints"}
    )
    save_steps: int = field(default=250, metadata={"help": "How often to save a model"})
    save_total_limit: int = field(
        default=40,
        metadata={
            "help": "How many checkpoints to save before the oldest is overwritten"
        },
    )
    resume_from_training: bool = field(
        default=False, metadata={"help": "Resume from training"}
    )


def rank0_print(*args):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        print(*args)



save_path = './result/result.json'

result_dict = []
tokenizer = None
trainer = None
hfparser = transformers.HfArgumentParser(
    (ModelArguments, DataArguments, TrainingArguments)
)
(
    model_args,
    data_args,
    training_args,
    extra_args,
) = hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
args = argparse.Namespace(
    **vars(model_args), **vars(data_args), **vars(training_args)
)

def train():
    

    checkpoint_dir = args.lora_dir
    save_path = args.save_path

    # if args.resume_dir is not None:
    #     checkpoint_dir, completed_training = args.resume_dir, False
    # else:
    #     checkpoint_dir, completed_training = get_last_checkpoint(args.output_dir)

    # if completed_training:
    #     rank0_print("Detected that training was already completed!")

    # if checkpoint_dir is None:
    #     rank0_print("Training from scratch.")
    # else:
    #     rank0_print("Loading from checkpoint:", checkpoint_dir)
    #     if args.resume_from_training:
    #         rank0_print("Resuming from training not supported yet. Exiting.")
    #         exit(1)

    tokenizer_model_name = args.model_name_or_path
    TokenizerClass = AutoTokenizer

    global tokenizer
    # Tokenizer
    tokenizer = TokenizerClass.from_pretrained(
        tokenizer_model_name,
        cache_dir=args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        truncation_side="right",
        use_fast=False,
    )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[
                model_args.version
            ]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates[
                "vicuna_v1"
            ]



    config = RewardConfig(backbone_model_name_or_path=model_args.model_name_or_path)
    

    with DisableLogger():
        model = RewardModel(
            args=args,
            config=config,
            qlora=True,
            checkpoint_dir=checkpoint_dir,
            tokenizer=tokenizer,
        )

    model.backbone_model.config.use_cache = False
    if model_args.vision_tower is not None:
        from llava.model import LlavaLlamaForCausalLM

        with DisableLogger():
            model_tmp = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
            )

        vision_tower = model_tmp.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True
        model.config.mm_use_im_start_end = (
            data_args.mm_use_im_start_end
        ) = model_args.mm_use_im_start_end
        training_args.use_im_start_end = model_args.mm_use_im_start_end

    # import pdb;pdb.set_trace()
    print_trainable_parameters(args, model)
    print("loaded model")
    set_seed(args.seed)


    def save_result_and_compute_reward_modeling_metrics(eval_prediction):
        print("#### save result ####")
        # import pdb; pdb.set_trace()

        for score1, score2, id, choice in zip(eval_prediction.predictions[..., 0].squeeze(-1).tolist(),
        eval_prediction.predictions[..., 1].squeeze(-1).tolist(),
        eval_prediction.label_ids[0].squeeze(-1).tolist(),
        eval_prediction.label_ids[-1].squeeze(-1).tolist()):
            result_dict.append(dict(
                score1=score2,
                score2=score1,
                id=id,
                choice=choice,
            ))
        return compute_reward_modeling_metrics(eval_prediction)

    global trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=save_result_and_compute_reward_modeling_metrics,
    )

    # Callbacks
    if not args.full_finetune:
        trainer.add_callback(SavePeftModelCallback)

    # Verifying the datatypes.
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes:
            dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items():
        total += v
    for k, v in dtypes.items():
        print(k, v, v / total)



import os
import json
from datetime import datetime
from PIL import Image
import shutil
import random
import time


def get_rewards(preference_data, factual_data, images: dict[Image.Image], prompt):

    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d-%H-%M-%S")
    random.seed(time.time() + os.getpid())
    random_number = random.randint(1, 1000000)
    tmp_dir = f'server_tmp-{formatted_now}-{random_number}'

    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)

    data_args.dataset_path = os.path.join(tmp_dir, 'preference.json')
    data_args.eval_dataset_path = data_args.dataset_path
    data_args.image_folder = tmp_dir
    data_args.reward_prompt_file = os.path.join(tmp_dir, 'prompt.txt')
    data_args.image_to_caption_file = os.path.join(tmp_dir, 'factual.json')

    for k, v in images.items():
        v: Image.Image
        v.save(os.path.join(data_args.image_folder, k))

    with open(data_args.dataset_path , 'w') as f:
        f.write(preference_data)

    with open(data_args.image_to_caption_file , 'w') as f:
        f.write(factual_data)

    with open(data_args.reward_prompt_file , 'w') as f:
        f.write(prompt)

    # import pdb
    # pdb.set_trace()

    logger.info("*** Evaluate ***")
    data_module = make_binary_reward_modeling_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        training_args=training_args,
        do_train=args.do_train,
    )
    trainer.replace_data(data_module)
    global result_dict
    result_dict = []
    metrics = trainer.evaluate(metric_key_prefix="eval")
    shutil.rmtree(tmp_dir)
    return result_dict


from io import BytesIO
from flask import Flask, jsonify, request
def main_flask():
    app = Flask(__name__)
    @app.route('/', methods=["GET"])
    def health_check():
        """Confirms service is running"""
        return f"service '{training_args.lora_dir}' is up and running.\n"
    @app.route('/api/generate', methods=["POST"])
    def get_prediction():
        """Returns the prediction for the given image and text"""
        print("POST")
        preference_data = request.form['preference_data']
        print(preference_data)
        preference_data = json.loads(preference_data)

        for idx, sample in enumerate(preference_data):
            preference_data[idx]['id'] = idx
            preference_data[idx]['preference'] = 1
            preference_data[idx]['output_2'] = sample['output_1']


        sample_len = len(preference_data)
        preference_data = json.dumps(preference_data, indent=2)


        factual_data = request.form['factual_data']
        # image_folder = request.form['image_folder']

        images = {}

        for k, v in request.files.items():
            print(k)
            image_data = v.read()
            image_data = Image.open(BytesIO(image_data))
            images[k] = image_data

        prompt = request.form['prompt']

        rewards = get_rewards(preference_data, factual_data, images, prompt)
        score_dict = {}
        for sample in rewards:
            score_dict[sample['id']] = (sample['score1'] + sample['score2'] ) / 2.0

        scores = []
        for i in range(sample_len):
            scores.append(score_dict[i])

        return jsonify(scores)

    app.run(host="0.0.0.0", port=int(os.getenv('FLASK_PORT')))

if __name__ == "__main__":
    train()

    # test_get_rewards()
    main_flask()

    
