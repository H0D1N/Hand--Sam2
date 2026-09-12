import sys
from pathlib import Path

import torch
from torch import nn


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


from training.model.sam2_multiview_dual_hand_memory import (
    SAM2MultiViewDualHandMemory,
)


class FakeDecoder(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.conv_s0 = nn.Conv2d(1, 1, kernel_size=1, bias=False)
        self.conv_s1 = nn.Conv2d(1, 1, kernel_size=1, bias=False)
        nn.init.constant_(self.conv_s0.weight, scale)
        nn.init.constant_(self.conv_s1.weight, scale)


class FakeChunkedEncoder(SAM2MultiViewDualHandMemory):
    def __init__(self):
        nn.Module.__init__(self)
        self.image_encoder = nn.Conv2d(1, 1, kernel_size=1, bias=False)
        nn.init.ones_(self.image_encoder.weight)
        self.left_mask_decoder = FakeDecoder(scale=2.0)
        self.right_mask_decoder = FakeDecoder(scale=3.0)
        self.chunk_sizes = []

    def forward_image(self, images):
        self.chunk_sizes.append(images.size(0))
        main_feature = self.image_encoder(images)
        return {
            "backbone_fpn": [main_feature + 10, main_feature + 20, main_feature],
            "vision_pos_enc": [main_feature + 30, main_feature + 40, main_feature + 50],
            "left_high_res_features": [
                self.left_mask_decoder.conv_s0(main_feature),
                self.left_mask_decoder.conv_s1(main_feature),
            ],
            "right_high_res_features": [
                self.right_mask_decoder.conv_s0(main_feature),
                self.right_mask_decoder.conv_s1(main_feature),
            ],
        }


def check_chunked_encoding_preserves_batch_view_order():
    model = FakeChunkedEncoder().eval()
    frame_images = torch.arange(1.0, 7.0).reshape(2, 3, 1, 1, 1)

    chunked = model._encode_frame_views(frame_images, views_per_encode=2)
    assert model.chunk_sizes == [4, 2]

    model.chunk_sizes.clear()
    unchunked = model._encode_frame_views(frame_images, views_per_encode=3)
    assert model.chunk_sizes == [6]

    for chunked_tensor, unchunked_tensor in zip(chunked[:3], unchunked[:3]):
        if isinstance(chunked_tensor, torch.Tensor):
            assert torch.equal(chunked_tensor, unchunked_tensor)
        else:
            assert chunked_tensor == unchunked_tensor

    main_feature, main_pos_embed, feature_size, high_res = chunked
    assert feature_size == (1, 1)
    assert main_feature[:, :, 0].tolist() == [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]
    assert main_pos_embed[:, :, 0].tolist() == [[51.0, 52.0, 53.0, 54.0, 55.0, 56.0]]
    assert high_res["left"][0].flatten().tolist() == [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
    assert high_res["right"][0].flatten().tolist() == [3.0, 6.0, 9.0, 12.0, 15.0, 18.0]


def check_encoding_keeps_only_required_gradients():
    model = FakeChunkedEncoder().train()
    frame_images = torch.ones(1, 2, 1, 1, 1)

    model.requires_grad_(False)
    frozen_outputs = model._encode_frame_views(frame_images, views_per_encode=1)
    assert not frozen_outputs[0].requires_grad
    assert not frozen_outputs[3]["left"][0].requires_grad

    model.image_encoder.requires_grad_(True)
    trainable_outputs = model._encode_frame_views(frame_images, views_per_encode=1)
    assert trainable_outputs[0].requires_grad
    assert trainable_outputs[3]["left"][0].requires_grad

    model.requires_grad_(False)
    model.left_mask_decoder.conv_s0.requires_grad_(True)
    projection_outputs = model._encode_frame_views(frame_images, views_per_encode=1)
    assert not projection_outputs[0].requires_grad
    assert projection_outputs[3]["left"][0].requires_grad
    assert not projection_outputs[3]["right"][0].requires_grad


def check_invalid_views_per_encode():
    model = FakeChunkedEncoder()
    frame_images = torch.ones(1, 2, 1, 1, 1)

    try:
        model._encode_frame_views(frame_images, views_per_encode=0)
    except ValueError as error:
        assert "views_per_encode" in str(error)
    else:
        raise AssertionError("views_per_encode=0 应该报错")


def main():
    check_chunked_encoding_preserves_batch_view_order()
    check_encoding_keeps_only_required_gradients()
    check_invalid_views_per_encode()
    print("Dual-hand multiview chunked encoding: OK")


if __name__ == "__main__":
    main()
