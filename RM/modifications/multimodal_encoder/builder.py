from .clip_encoder import CLIPVisionTower
from .siglip_encoder import SiglipVisionTower

def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    if vision_tower.startswith("openai") or vision_tower.startswith("laion"):
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif "vila" in vision_tower.lower():
        vision_tower = SiglipVisionTower(vision_tower, vision_tower_cfg)
        return vision_tower

    raise ValueError(f'Unknown vision tower: {vision_tower}')
