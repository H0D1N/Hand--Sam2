import copy
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
    

    def _project_high_res_features(
        self,
        backbone_fpn,
        mask_decoder,
    ):
        feature_s0 = mask_decoder.conv_s0(backbone_fpn[0])
        feature_s1 = mask_decoder.conv_s1(backbone_fpn[1])

        return [feature_s0, feature_s1]