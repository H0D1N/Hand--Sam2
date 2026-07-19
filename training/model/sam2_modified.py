import copy

import torch
import torch.nn.functional as F

from sam2.modeling.sam2_base import NO_OBJ_SCORE

from .sam2 import SAM2Train

class SAM2Modified(SAM2Train):

    def _build_sam_heads(self):
        super()._build_sam_heads()
        
        # 删掉原有的self.sam_mask_decoder，选择的mask_decoder都改成直接传参
        self._build_two_sam_heads()
    
    def _build_two_sam_heads(self):

        self.left_mask_decoder = copy.deepcopy(self.sam_mask_decoder)
        self.right_mask_decoder = copy.deepcopy(self.sam_mask_decoder)

        del self.sam_mask_decoder

        assert (self.left_mask_decoder is not self.right_mask_decoder)

    def forward_image(self, img_batch):

        backbone_out = self.image_encoder(img_batch)

        if self.use_high_res_features_in_sam:
            backbone_fpn = backbone_out["backbone_fpn"]

            left_high_res_features = (
                self._project_high_res_features(
                    backbone_fpn=backbone_fpn,
                    mask_decoder=self.left_mask_decoder,
                )
            )

            right_high_res_features = (
                self._project_high_res_features(
                    backbone_fpn=backbone_fpn,
                    mask_decoder=self.right_mask_decoder,
                )
            )
        else:
            left_high_res_features = None
            right_high_res_features = None

        backbone_out["left_high_res_features"] = (
            left_high_res_features
        )

        backbone_out["right_high_res_features"] = (
            right_high_res_features
        )

        return backbone_out
    
    def _forward_one_sam_head(
        self,
        mask_decoder,
        prompt_encoder,
        backbone_features,
        point_inputs=None,
        mask_inputs=None,
        high_res_features=None,
        multimask_output=False,
    ):

        B = backbone_features.size(0)
        device = backbone_features.device
        assert backbone_features.size(1) == self.sam_prompt_embed_dim
        assert backbone_features.size(2) == self.sam_image_embedding_size
        assert backbone_features.size(3) == self.sam_image_embedding_size

        # a) Handle point prompts
        if point_inputs is not None:
            sam_point_coords = point_inputs["point_coords"]
            sam_point_labels = point_inputs["point_labels"]
            assert sam_point_coords.size(0) == B and sam_point_labels.size(0) == B
        else:
            # If no points are provide, pad with an empty point (with label -1)
            sam_point_coords = torch.zeros(B, 1, 2, device=device)
            sam_point_labels = -torch.ones(B, 1, dtype=torch.int32, device=device)

        # b) Handle mask prompts
        if mask_inputs is not None:
            # If mask_inputs is provided, downsize it into low-res mask input if needed
            # and feed it as a dense mask prompt into the SAM mask encoder
            assert len(mask_inputs.shape) == 4 and mask_inputs.shape[:2] == (B, 1)
            if mask_inputs.shape[-2:] != prompt_encoder.mask_input_size:
                sam_mask_prompt = F.interpolate(
                    mask_inputs.float(),
                    size=prompt_encoder.mask_input_size,
                    align_corners=False,
                    mode="bilinear",
                    antialias=True,  # use antialias for downsampling
                )
            else:
                sam_mask_prompt = mask_inputs
        else:
            # Otherwise, simply feed None (and SAM's prompt encoder will add
            # a learned `no_mask_embed` to indicate no mask input in this case).
            sam_mask_prompt = None

        sparse_embeddings, dense_embeddings = prompt_encoder(
            points=(sam_point_coords, sam_point_labels),
            boxes=None,
            masks=sam_mask_prompt,
        )
        (
            low_res_multimasks,
            ious,
            sam_output_tokens,
            object_score_logits,
        ) = mask_decoder(
            image_embeddings=backbone_features,
            image_pe=prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=False,  # the image is already batched
            high_res_features=high_res_features,
        )
        if self.pred_obj_scores:
            is_obj_appearing = object_score_logits > 0

            # Mask used for spatial memories is always a *hard* choice between obj and no obj,
            # consistent with the actual mask prediction
            low_res_multimasks = torch.where(
                is_obj_appearing[:, None, None],
                low_res_multimasks,
                NO_OBJ_SCORE,
            )

        # convert masks from possibly bfloat16 (or float16) to float32
        # (older PyTorch versions before 2.1 don't support `interpolate` on bf16)
        low_res_multimasks = low_res_multimasks.float()
        high_res_multimasks = F.interpolate(
            low_res_multimasks,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        sam_output_token = sam_output_tokens[:, 0]
        if multimask_output:
            # take the best mask prediction (with the highest IoU estimation)
            best_iou_inds = torch.argmax(ious, dim=-1)
            batch_inds = torch.arange(B, device=device)
            low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            if sam_output_tokens.size(1) > 1:
                sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
        else:
            low_res_masks, high_res_masks = low_res_multimasks, high_res_multimasks

        # Extract object pointer from the SAM output token (with occlusion handling)
        obj_ptr = self.obj_ptr_proj(sam_output_token)
        if self.pred_obj_scores:
            # Allow *soft* no obj ptr, unlike for masks
            if self.soft_no_obj_ptr:
                lambda_is_obj_appearing = object_score_logits.sigmoid()
            else:
                lambda_is_obj_appearing = is_obj_appearing.float()

            if self.fixed_no_obj_ptr:
                obj_ptr = lambda_is_obj_appearing * obj_ptr
            obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr

        return (
            low_res_multimasks,
            high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        )

    def forward_single_image(
        self,
        images,
        point_inputs=None,
        mask_inputs=None,
        multimask_output=False,
    ):
        assert images.dim() == 4

        # image_encoder
        backbone_out = self.forward_image(images)

        # backbone_out = {
        #     "vision_features": ...,
        #     "vision_pos_enc": ...,
        #     "backbone_fpn": ...,
        #     "left_high_res_features": ...,
        #     "right_high_res_features": ...,
        #     }

        # mask_decoder 主要输入
        pix_feat = backbone_out["vision_features"]

        left_outputs = self._forward_one_sam_head(
            mask_decoder=self.left_mask_decoder,
            prompt_encoder=self.sam_prompt_encoder,
            backbone_features=pix_feat,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            high_res_features=backbone_out["left_high_res_features"],
            multimask_output=multimask_output,
        )

        right_outputs = self._forward_one_sam_head(
            mask_decoder=self.right_mask_decoder,
            prompt_encoder=self.sam_prompt_encoder,
            backbone_features=pix_feat,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            high_res_features=backbone_out["right_high_res_features"],
            multimask_output=multimask_output,
        )
        # right_outputs:
        # return (
        # low_res_multimasks,
        # high_res_multimasks,
        # ious,
        # low_res_masks,
        # high_res_masks,
        # obj_ptr,
        # object_score_logits,
        # )
        
        return {
            "left": {
                "low_res_multimasks": left_outputs[0],
                "high_res_multimasks": left_outputs[1],
                "ious": left_outputs[2],
                "low_res_masks": left_outputs[3],
                "high_res_masks": left_outputs[4],
                "obj_ptr": left_outputs[5],
                "object_score_logits": left_outputs[6],
            },
            "right": {
                "low_res_multimasks": right_outputs[0],
                "high_res_multimasks": right_outputs[1],
                "ious": right_outputs[2],
                "low_res_masks": right_outputs[3],
                "high_res_masks": right_outputs[4],
                "obj_ptr": right_outputs[5],
                "object_score_logits": right_outputs[6],
            }
        }

    def _project_high_res_features(
        self,
        backbone_fpn,
        mask_decoder,
    ):
        feature_s0 = mask_decoder.conv_s0(backbone_fpn[0])
        feature_s1 = mask_decoder.conv_s1(backbone_fpn[1])

        return [feature_s0, feature_s1]