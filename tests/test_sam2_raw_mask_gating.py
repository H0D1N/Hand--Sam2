from types import SimpleNamespace

import torch
from torch import nn

from sam2.modeling.sam2_base import NO_OBJ_SCORE
from training.model.sam2_modified import SAM2Modified


class PromptEncoder:
    mask_input_size = (2, 2)

    def __call__(self, points, boxes, masks):
        batch_size = points[0].size(0)
        empty = points[0].new_zeros(batch_size, 1, 1)
        return empty, empty

    def get_dense_pe(self):
        return torch.zeros(1, 1, 2, 2)


class MaskDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.raw_mask = nn.Parameter(torch.tensor([[[[2.0, -2.0], [1.0, -1.0]]]]))

    def forward(self, image_embeddings, **kwargs):
        batch_size = image_embeddings.size(0)
        return (
            self.raw_mask.expand(batch_size, -1, -1, -1),
            image_embeddings.new_full((batch_size, 1), 0.8),
            image_embeddings.new_zeros(batch_size, 1, 1),
            image_embeddings.new_full((batch_size, 1), -0.1),
        )


def main():
    owner = SimpleNamespace(
        sam_prompt_embed_dim=1,
        sam_image_embedding_size=2,
        image_size=8,
        pred_obj_scores=True,
        obj_ptr_proj=nn.Identity(),
        soft_no_obj_ptr=False,
        fixed_no_obj_ptr=False,
        no_obj_ptr=torch.zeros(1),
    )
    decoder = MaskDecoder()
    outputs = SAM2Modified._forward_one_sam_head(
        owner,
        mask_decoder=decoder,
        prompt_encoder=PromptEncoder(),
        backbone_features=torch.zeros(1, 1, 2, 2),
    )

    raw_low, raw_high, _, gated_low, gated_high, _, object_logits = outputs
    assert torch.equal(raw_low, decoder.raw_mask)
    assert raw_high.shape == (1, 1, 8, 8)
    assert torch.all(gated_low == NO_OBJ_SCORE)
    assert torch.all(gated_high == NO_OBJ_SCORE)
    assert object_logits.item() < 0

    (raw_low.sum() + raw_high.sum()).backward()
    assert torch.count_nonzero(decoder.raw_mask.grad).item() > 0
    print("SAM2 raw supervision and gated prediction split: OK")


if __name__ == "__main__":
    main()
