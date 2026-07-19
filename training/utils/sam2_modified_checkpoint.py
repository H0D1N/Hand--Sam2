class DuplicateMaskDecoderWeights:
    """Copy one official decoder state into left and right branches."""
    def __call__(self, state_dict):

        # Input:
        # state_dict = {
        #     "image_encoder.weight": tensor(...),
        #     "sam_mask_decoder.mask_tokens.weight": tensor(...),
        # }
        # Output:
        # converted_state_dict = {
        #     "image_encoder.weight": tensor(...),
        #     "left_mask_decoder.mask_tokens.weight": tensor(...),
        #     "right_mask_decoder.mask_tokens.weight": tensor(...),
        # }
        converted_state_dict = state_dict.copy()

        source_prefix = "sam_mask_decoder."
        left_prefix = "left_mask_decoder."
        right_prefix = "right_mask_decoder."

        keys_to_modify = []
        for key in state_dict:
            if key.startswith(source_prefix):
                keys_to_modify.append(key)
        
        for key in keys_to_modify:
            suffix = key[len(source_prefix):]

            value = converted_state_dict.pop(key)

            left_key = left_prefix + suffix
            right_key = right_prefix + suffix

            converted_state_dict[left_key] = value
            converted_state_dict[right_key] = value
        
        return converted_state_dict



