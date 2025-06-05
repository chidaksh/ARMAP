import torch
from llava.model.multimodal_encoder.vision_encoder import VisionTower, VisionTowerS2

from transformers import AutoConfig, PretrainedConfig, AutoModel
from .siglip import (
    SiglipVisionConfig,
    SiglipVisionModel,
    SiglipImageProcessor,
)


class SiglipVisionTower(VisionTower):
    def __init__(self, model_name_or_path: str, config: PretrainedConfig, state_dict=None):
        super().__init__(model_name_or_path, config)
        current_device = torch.cuda.current_device()
        self.image_processor = SiglipImageProcessor.from_pretrained(model_name_or_path)
        #self.image_processor = SiglipImageProcessor.from_pretrained(model_name_or_path,device_map={"": current_device},)
        # import pdb
        # pdb.set_trace()
        self.vision_tower = SiglipVisionModel.from_pretrained(
            # TODO(ligeng): why pass config here leading to errors?
            model_name_or_path,state_dict=state_dict
            #model_name_or_path,device_map={"": current_device}, state_dict=state_dict
        )
        self.is_loaded = True


class SiglipVisionTowerS2(VisionTowerS2):
    def __init__(self, model_name_or_path: str, config: PretrainedConfig):
        super().__init__(model_name_or_path, config)
        self.image_processor = SiglipImageProcessor.from_pretrained(model_name_or_path)
        self.vision_tower = SiglipVisionModel.from_pretrained(
            model_name_or_path, torch_dtype=eval(config.model_dtype)
        )

        # Make sure it crops/resizes the image to the largest scale in self.scales to maintain high-res information
        self.image_processor.size['height'] = self.image_processor.size['width'] = self.scales[-1]

        self.is_loaded = True


AutoConfig.register("siglip_vision_model", SiglipVisionConfig)
AutoModel.register(SiglipVisionConfig, SiglipVisionModel)

