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