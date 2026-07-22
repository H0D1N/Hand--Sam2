from training.utils.sam2_modified_checkpoint import (
    DuplicateMaskDecoderWeights,
)


def main():
    original_state_dict = {
        "image_encoder.weight": 1,
        "sam_mask_decoder.mask_tokens.weight": 2,
        "sam_mask_decoder.conv_s0.weight": 3,
    }

    kernel = DuplicateMaskDecoderWeights()

    converted_state_dict = kernel(
        state_dict=original_state_dict
    )

    assert (
        "sam_mask_decoder.mask_tokens.weight"
        not in converted_state_dict
    )

    assert (
        "left_mask_decoder.mask_tokens.weight"
        in converted_state_dict
    )

    assert (
        "right_mask_decoder.mask_tokens.weight"
        in converted_state_dict
    )

    assert (
        converted_state_dict[
            "left_mask_decoder.mask_tokens.weight"
        ]
        == 2
    )

    assert (
        converted_state_dict[
            "right_mask_decoder.mask_tokens.weight"
        ]
        == 2
    )

    assert (
        "image_encoder.weight"
        in converted_state_dict
    )

    print("SAM2Modified checkpoint mapping: OK")


if __name__ == "__main__":
    main()