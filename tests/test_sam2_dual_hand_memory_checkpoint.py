import sys
from pathlib import Path

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


from training.utils.sam2_dual_hand_memory_checkpoint import (
    DuplicateMemoryWeights,
)


def main():
    attention_weight = torch.randn(2, 2)
    encoder_weight = torch.randn(2, 2)
    original_state_dict = {
        "image_encoder.weight": torch.randn(2, 2),
        "memory_attention.layers.0.self_attn.q_proj.weight": attention_weight,
        "memory_encoder.mask_downsampler.encoder.0.weight": encoder_weight,
        "maskmem_tpos_enc": torch.randn(7, 1, 1, 64),
    }

    converted_state_dict = DuplicateMemoryWeights()(original_state_dict)

    assert not any(
        key.startswith(("memory_attention.", "memory_encoder."))
        for key in converted_state_dict
    )

    for hand_name in ("left", "right"):
        assert converted_state_dict[
            f"{hand_name}_memory_attention.layers.0.self_attn.q_proj.weight"
        ] is attention_weight
        assert converted_state_dict[
            f"{hand_name}_memory_encoder.mask_downsampler.encoder.0.weight"
        ] is encoder_weight

    assert converted_state_dict["image_encoder.weight"] is original_state_dict[
        "image_encoder.weight"
    ]
    assert converted_state_dict["maskmem_tpos_enc"] is original_state_dict[
        "maskmem_tpos_enc"
    ]
    assert "memory_attention.layers.0.self_attn.q_proj.weight" in original_state_dict
    assert "memory_encoder.mask_downsampler.encoder.0.weight" in original_state_dict

    print("SAM2DualHandMemory checkpoint mapping: OK")


if __name__ == "__main__":
    main()
