# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json

import os
from dataclasses import dataclass, field
from typing import Optional, List, Literal
import logging
from enum import Enum

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
from llava.mm_utils import (KeywordsStoppingCriteria, get_model_name_from_path,
                            process_images, tokenizer_image_token)
from llava.constants import (
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.model.builder import load_pretrained_model

from llava.train.train import smart_tokenizer_and_embedding_resize
from data_utils.common_utils import preprocess

torch.backends.cuda.matmul.allow_tf32 = True

logger = logging.getLogger(__name__)

# POMDP Core Components
class POMDPState(Enum):
    """
    Defines the latent (unobservable) user states.
    S = {NOVICE, EXPERT, CONFUSED}
    """
    NOVICE = 0
    EXPERT = 1
    CONFUSED = 2

class POMDPAction(Enum):
    """
    Defines the actions the agent can take.
    A = {GIVE_SIMPLE_EXAMPLE, GIVE_TECHNICAL_DEFINITION, ASK_FOR_CLARIFICATION}
    """
    GIVE_SIMPLE_EXAMPLE = 0
    GIVE_TECHNICAL_DEFINITION = 1
    ASK_FOR_CLARIFICATION = 2

def transition_model(state: POMDPState, action: POMDPAction) -> torch.Tensor:
    """
    Defines the transition function T(s' | s, a).
    Returns a probability distribution over the next states.
    """
    # Placeholder transition logic
    num_states = len(POMDPState)
    # For now, assume the state doesn't change
    next_state_dist = torch.zeros(num_states)
    next_state_dist[state.value] = 1.0
    return next_state_dist

def observation_model(observation: str, state: POMDPState) -> float:
    """
    Defines the observation function O(o | s').
    Returns the probability of an observation given a state.
    """
    # Placeholder observation logic
    # This should be a more sophisticated model in a real implementation
    if state == POMDPState.CONFUSED and "?" in observation:
        return 0.7
    return 0.3

def reward_function(state: POMDPState, action: POMDPAction) -> float:
    """
    Defines the reward function R(s, a).
    Returns the immediate reward for taking action a in state s.
    """
    reward_matrix = {
        POMDPState.NOVICE: {
            POMDPAction.GIVE_SIMPLE_EXAMPLE: 10.0,
            POMDPAction.GIVE_TECHNICAL_DEFINITION: -5.0,
            POMDPAction.ASK_FOR_CLARIFICATION: 1.0,
        },
        POMDPState.EXPERT: {
            POMDPAction.GIVE_SIMPLE_EXAMPLE: -5.0,
            POMDPAction.GIVE_TECHNICAL_DEFINITION: 10.0,
            POMDPAction.ASK_FOR_CLARIFICATION: 1.0,
        },
        POMDPState.CONFUSED: {
            POMDPAction.GIVE_SIMPLE_EXAMPLE: 5.0,
            POMDPAction.GIVE_TECHNICAL_DEFINITION: -5.0,
            POMDPAction.ASK_FOR_CLARIFICATION: 10.0,
        },
    }
    return reward_matrix[state][action]

class BeliefUpdater(torch.nn.Module):
    """
    A lightweight Transformer-based module to update the belief state.
    b_t = Transformer(b_{t-1}, a_{t-1}, o_t)
    """
    def __init__(self, num_states: int, hidden_size: int, embedding_dim: int = 128):
        super().__init__()
        self.num_states = num_states
        self.embedding_dim = embedding_dim

        # Projection layers for belief, actions, and observations
        self.belief_projection = torch.nn.Linear(num_states, embedding_dim)
        self.action_embedding = torch.nn.Embedding(len(POMDPAction), embedding_dim)
        self.observation_projection = torch.nn.Linear(hidden_size, embedding_dim)

        # Transformer Encoder Layer
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=4, batch_first=True)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Final layer to produce the new belief distribution
        self.output_layer = torch.nn.Linear(embedding_dim, num_states)

    def forward(self, belief: torch.Tensor, action: POMDPAction, observation_embedding: torch.Tensor):
        # Project all components to the same embedding dimension
        belief_emb = self.belief_projection(belief)
        action_emb = self.action_embedding(torch.tensor(action.value, device=belief.device))
        obs_emb = self.observation_projection(observation_embedding)

        # Create a sequence for the transformer: (batch, seq_len, embedding_dim)
        # Sequence: [belief, action, observation]
        transformer_input = torch.stack([belief_emb, action_emb, obs_emb], dim=1)

        # The belief is for a single step, so we add a batch dimension if it's not there
        if transformer_input.dim() == 2:
            transformer_input = transformer_input.unsqueeze(0)

        transformer_output = self.transformer_encoder(transformer_input)
        
        # We take the output corresponding to the last element of the sequence (observation)
        # as it has attended to the belief and action.
        new_belief_logits = self.output_layer(transformer_output[:, -1, :])
        
        return torch.nn.functional.softmax(new_belief_logits, dim=-1).squeeze(0)

class BeliefState:
    """
    Represents the agent's belief over the user's latent state.
    It's a probability distribution over the states in POMDPState.
    """
    def __init__(self, num_states: int, belief_updater: BeliefUpdater, device):
        # Initialize with a uniform belief distribution on the correct device
        self.belief = torch.ones(num_states, device=device) / num_states
        self.belief_updater = belief_updater

    def update(self, action, observation_embedding):
        """
        Update the belief state using the Transformer-based belief updater.
        """
        # Ensure belief is on the correct device before passing to the updater
        self.belief = self.belief_updater(self.belief.to(observation_embedding.device), action, observation_embedding)

    def get_belief(self):
        return self.belief

class StateAwareRewardModel(RewardModel):
    """
    A reward model that is aware of the user's latent state.
    The reward computation will depend on the current belief state.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_states = len(POMDPState)
        self.belief_updater = BeliefUpdater(num_states, self.config.hidden_size)
        self.belief_state = BeliefState(num_states, self.belief_updater, self.device)

    def forward(self, chosen_input_ids, rejected_input_ids, **kwargs):
        # Get the full outputs from the backbone model to access hidden states
        chosen_outputs = self.backbone_model(
            input_ids=chosen_input_ids, output_hidden_states=True, **kwargs
        )
        # We only need the rewards for the rejected side, not the hidden states
        rejected_rewards = super().forward(rejected_input_ids, **kwargs)["rewards"]

        chosen_rewards = self.value_head(chosen_outputs.hidden_states[-1]).squeeze(-1)

        # Use the embedding of the last token as the observation
        observation_embedding = chosen_outputs.hidden_states[-1][:, -1, :]

        # For now, let's assume a fixed action for demonstration
        action = POMDPAction.GIVE_SIMPLE_EXAMPLE

        # Update belief state
        self.belief_state.update(action, observation_embedding)

        # Calculate the expected reward based on the new belief state
        belief = self.belief_state.get_belief()
        expected_reward = 0
        for s_idx, s in enumerate(POMDPState):
            expected_reward += belief[s_idx] * reward_function(s, action)

        # Add the expected state-aware reward to the base reward
        chosen_rewards += expected_reward
        rejected_rewards += expected_reward # Or a different logic for rejected

        return {"chosen_rewards": chosen_rewards, "rejected_rewards": rejected_rewards}


class POMDPPavoTrainer(Trainer):
    """Custom trainer to handle POMDP logic if needed in the future."""
    def compute_loss(self, model, inputs, return_outputs=False):
        # The belief update is handled within the model's forward pass.
        # We just need to ensure the loss is computed correctly.
        rewards = model(**inputs)
        loss = -torch.nn.functional.logsigmoid(rewards["chosen_rewards"] - rewards["rejected_rewards"]).mean()
        return (loss, {"rewards": rewards}) if return_outputs else loss

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
    tokenizer: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(
        default=-1
    )  # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    # change patch to cls_patch
    mm_vision_select_feature: Optional[str] = field(default="cls_patch")


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
        default_factory=lambda: ["labels"],
        metadata={
            "help": "Names of the labels in the dataset. This is needed to get transformers.Trainer to find the labels."
        },
    )
    padding_side: Literal["left", "right"] = field(
        default="right",
        metadata={"help": "Side to pad, left or right."},  # ignored, will be set by tokenizer
    )
    full_finetune: bool = field(
        default=False, metadata={"help": "Finetune the entire model without adapters."}
    )
    lora_r: int = field(default=8, metadata={"help": "Lora R value."})
    lora_alpha: int = field(default=32, metadata={"help": "Lora alpha."})
    lora_dropout: float = field(default=0.05, metadata={"help": "Lora dropout."})
    lora_modules: List[str] = field(
        default_factory=lambda: None,
        metadata={
            "help": "Which modules to use LoRA on. If None, will default to the model's preferred modules."
        },
    )
    lora_bf16: bool = field(
        default=False,
        metadata={
            "help": "Whether to use bf16 for LoRA. If True, will use bf16. If False, will use fp16."
        },
    )
    deepspeed_plugin: str = field(
        default=None,
        metadata={
            "help": "[experimental] Deepspeed plugin to use. See https://github.com/tatsu-lab/stanford_alpaca/pull/135"
        },
    )
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    # From QLoRA
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
    # PagedAdamW
    paged_adam: bool = field(default=False, metadata={"help": "Use PagedAdamW."})
    paged_adam_bf16: bool = field(
        default=False, metadata={"help": "Use bfloat16 for PagedAdamW."}
    )
    # training
    optim: str = field(
        default="adamw_hf", metadata={"help": "Optimizer to use, e.g. adamw_hf"}
    )
    learning_rate: float = field(
        default=1e-5, metadata={"help": "Learning rate for the optimizer."}
    )
    weight_decay: float = field(
        default=0.0, metadata={"help": "Weight decay for the optimizer."}
    )
    gradient_accumulation_steps: int = field(
        default=1, metadata={"help": "Gradient accumulation steps."}
    )
    num_train_epochs: int = field(
        default=1, metadata={"help": "Number of training epochs."}
    )
    # evaluation
    evaluation_strategy: str = field(
        default="steps",
        metadata={"help": "When to evaluate the model."},  # steps or epoch
    )
    eval_steps: int = field(
        default=250, metadata={"help": "How often to evaluate the model."}
    )
    # saving
    output_dir: str = field(
        default="./output/reward_model",
        metadata={"help": "Output directory for the model."},
    )
    save_strategy: str = field(
        default="steps",
        metadata={"help": "When to save the model."},  # steps or epoch
    )
    save_steps: int = field(default=250, metadata={"help": "How often to save a model"})
    save_total_limit: int = field(
        default=40,
        metadata={
            "help": "How many checkpoints to save before the oldest is overwritten."
        },
    )


def rank0_print(*args):
    if os.environ.get("RANK", "0") == "0":
        print(*args)


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    args = argparse.Namespace(**vars(model_args), **vars(data_args), **vars(training_args))

    checkpoint_dir, completed_training = get_last_checkpoint(args.output_dir)
    if completed_training:
        print("Detected that training was already completed!")

    # model
    model_name_or_path = args.model_name_or_path

    model_kwargs = {}
    if args.lora_bf16:
        model_kwargs["torch_dtype"] = torch.bfloat16
    else:
        model_kwargs["torch_dtype"] = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        cache_dir=args.cache_dir,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=True,
    )
    tokenizer.pad_token = tokenizer.unk_token

    if model_args.vision_tower is not None:
        if 'mpt' in model_args.model_name_or_path.lower():
            raise NotImplementedError("MPT is not supported yet.")
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    config = RewardConfig(backbone_model_name_or_path=model_args.model_name_or_path)
    
    with DisableLogger():
        model = StateAwareRewardModel(
            args=args,
            config=config,
            qlora=True,
            checkpoint_dir=checkpoint_dir,
            tokenizer=tokenizer,
        )

    model.backbone_model.config.use_cache = False
    model.backbone_model.config.cache_shape = (4096, 4096)
    print_trainable_parameters(args, model)
    print("loaded model")

    if model_args.vision_tower is not None:
        from llava.model import LlavaLlamaForCausalLM

        with DisableLogger():
            from transformers import AutoConfig
            import os

            config = AutoConfig.from_pretrained(model_args.model_name_or_path)

            if hasattr(config, "mm_vision_tower") and not os.path.isabs(config.mm_vision_tower):
                main_model_dir = os.path.dirname(model_args.model_name_or_path.rstrip('/'))
                vision_tower_path = os.path.join(main_model_dir, 'vision_tower')
                print(f"Correcting mm_vision_tower path for temporary model to: {vision_tower_path}")
                config.mm_vision_tower = vision_tower_path

            model_tmp = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
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

    data_module = make_binary_reward_modeling_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        training_args=training_args,
    )

    if args.do_train:
        training_data = data_module["train_dataset"]
        rank0_print("Training data size:", len(training_data))
        rank0_print("Training data example:")

    set_seed(args.seed)

    trainer = POMDPPavoTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=compute_reward_modeling_metrics,
        **{k: v for k, v in data_module.items() if k != "predict_dataset"},
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

    all_metrics = {"run_name": args.run_name}

    # Training
    if args.do_train:
        logger.info("*** Train ***")
        # torch.autograd.set_detect_anomaly(True)
        # with torch.autograd.detect_anomaly():
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        all_metrics.update(metrics)

    # Evaluation
    if args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        all_metrics.update(metrics)

    if args.do_train or args.do_eval:
        with open(os.path.join(args.output_dir, "metrics.json"), "w") as fout:
            fout.write(json.dumps(all_metrics))


if __name__ == "__main__":
    train()
