class DuplicateMemoryWeights:
    """Copy one official Memory state into left and right branches."""
    def __call__(self, state_dict):

        # Input:
        # state_dict = {
        #     "image_encoder.weight": tensor(...),
        #     "memory_attention.mask_tokens.weight": tensor(...),
        # }
        # Output:
        # converted_state_dict = {
        #     "image_encoder.weight": tensor(...),
        #     "left_memory_attention.mask_tokens.weight": tensor(...),
        #     "right_memory_attention.mask_tokens.weight": tensor(...),
        # }
        converted_state_dict = state_dict.copy()

        prefix_mappings = (
            (
                "memory_attention.",
                "left_memory_attention.",
                "right_memory_attention.",
            ),
            (
                "memory_encoder.",
                "left_memory_encoder.",
                "right_memory_encoder.",
            ),
        )

        for source_prefix, left_prefix, right_prefix in prefix_mappings:

            keys_to_modify = [
                key for key in state_dict
                if key.startswith(source_prefix)
            ]

            for key in keys_to_modify:
                suffix = key[len(source_prefix):]
                value = converted_state_dict.pop(key)

                left_key = left_prefix + suffix
                right_key = right_prefix + suffix

                converted_state_dict[left_key] = value
                converted_state_dict[right_key] = value

        return converted_state_dict
