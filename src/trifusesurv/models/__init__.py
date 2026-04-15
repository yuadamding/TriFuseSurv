"""Model building blocks for TriFuseSurv."""

from trifusesurv.models.swinunetr_backbone_utils import (
    build_swinunetr_backbone,
    convert_swinvit_feats_to_channel_first,
    load_swinunetr_pretrained,
    swinvit_features,
)
from trifusesurv.models.swinunetr_shared_roi_token_backbone import (
    ContourAwareROITokenBackbone,
)
from trifusesurv.models.survival_model import (
    SwinUNETRTokenMoEDiscrete,
    gate_entropy_penalty_presence,
    gate_load_balance_penalty_presence,
)
from trifusesurv.models.lora import (
    LoRALinear,
    inject_lora_into_module,
    inject_lora_from_state_dict,
    freeze_all_params,
    mark_only_lora_trainable,
    count_trainable,
    is_lora_param_name,
)

__all__ = [
    "ContourAwareROITokenBackbone",
    "SwinUNETRTokenMoEDiscrete",
    "build_swinunetr_backbone",
    "convert_swinvit_feats_to_channel_first",
    "load_swinunetr_pretrained",
    "swinvit_features",
    "gate_entropy_penalty_presence",
    "gate_load_balance_penalty_presence",
    "LoRALinear",
    "inject_lora_into_module",
    "inject_lora_from_state_dict",
    "freeze_all_params",
    "mark_only_lora_trainable",
    "count_trainable",
    "is_lora_param_name",
]
