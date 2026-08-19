import tempfile
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


from projects.framewise_sam2_modified import run_evaluate


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))


class FakeDataset:
    def __init__(self, size, num_streams):
        self.size = size
        self.streams = {
            f"stream-{index}": {}
            for index in range(num_streams)
        }

    def __len__(self):
        return self.size


class FakeSampler:
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size


def build_args(output_dir):
    return SimpleNamespace(
        checkpoint=Path("framewise-best.pt"),
        output_dir=Path(output_dir),
        dataset_root=Path("multiserver"),
        dataset_names=["private-a", "private-b", "private-c"],
        dex_ycb_root=Path("dexycb"),
        test_seq_count=3,
        frames_per_second=3.0,
        prompt_mode="point",
        batch_size=2,
        num_workers=0,
        prefetch_factor=2,
        device="cpu",
        seed=42,
        disable_tf32=False,
        channels_last=False,
        skip_visualizations=True,
        log_interval=50,
    )


def check_checkpoint_loading():
    checkpoint = {
        "model_state": {"weight": torch.tensor([3.0])},
        "args": {
            "image_size": 768,
            "use_image_adapter": True,
            "use_decoder_adapter": False,
            "adapter_dim": 32,
            "adapter_dropout": 0.2,
            "adapter_init_scale": 1e-2,
        },
    }

    with (
        patch.object(run_evaluate.torch, "load", return_value=checkpoint),
        patch.object(
            run_evaluate,
            "create_sam2_modified_tiny",
            return_value=TinyModel(),
        ) as create_model,
        patch.object(
            run_evaluate,
            "inject_sam2_modified_adapters",
        ) as inject_adapters,
    ):
        model, checkpoint_metadata, train_args = (
            run_evaluate.load_trained_model(
                Path("framewise-best.pt"),
                torch.device("cpu"),
            )
        )

    create_model.assert_called_once_with(image_size=768)
    inject_adapters.assert_called_once_with(
        model=model,
        use_image_adapter=True,
        use_decoder_adapter=False,
        adapter_dim=32,
        adapter_dropout=0.2,
        adapter_init_scale=1e-2,
    )
    assert torch.equal(model.weight, torch.tensor([3.0]))
    assert not model.training
    assert checkpoint_metadata == {
        "epoch": None,
        "best_val_iou": None,
    }
    assert train_args is checkpoint["args"]


def check_dataset_separation():
    args = build_args("outputs")

    with (
        patch.object(
            run_evaluate,
            "MultiServerDualHandDataset",
            return_value="multiserver-dataset",
        ) as build_multiserver,
        patch.object(
            run_evaluate,
            "DexYCBDataset",
            return_value="dexycb-dataset",
        ) as build_dexycb,
    ):
        datasets = run_evaluate.build_eval_datasets(args, image_size=768)

    assert datasets == {
        "MultiServer": "multiserver-dataset",
        "DexYCB": "dexycb-dataset",
    }
    build_multiserver.assert_called_once_with(
        dataset_root=args.dataset_root,
        split="val",
        test_seq_count=3,
        image_size=768,
        use_augmentation=False,
        dataset_names=["private-a", "private-b", "private-c"],
    )
    build_dexycb.assert_called_once_with(
        dataset_root=args.dex_ycb_root,
        split="val",
        setup="s0",
        image_size=768,
        use_augmentation=False,
    )


def check_separate_evaluation_summary():
    multiserver = FakeDataset(size=30, num_streams=6)
    dexycb = FakeDataset(size=40, num_streams=8)
    sampler_sizes = {id(multiserver): 9, id(dexycb): 12}
    validation_calls = []
    written = {}

    def build_loader(dataset, _args, _device):
        return dataset, FakeSampler(sampler_sizes[id(dataset)])

    def validate(model, loader, device, epoch, args, point_prompt_fn):
        validation_calls.append(
            (loader, args.output_dir, point_prompt_fn)
        )
        value = 0.7 if loader is multiserver else 0.8
        return {
            "loss": 1.0 - value,
            "iou": value,
            "dice": value,
            "object_accuracy": value,
            "object_precision": value,
            "object_recall": value,
            "object_f1": value,
        }

    def record_json(data, output_path):
        written["data"] = data
        written["output_path"] = output_path

    with tempfile.TemporaryDirectory() as output_dir:
        args = build_args(output_dir)
        checkpoint = {"epoch": 4, "best_val_iou": 0.75}
        train_args = {
            "image_size": 768,
            "use_point_prompt": True,
        }

        with (
            patch.object(run_evaluate, "parse_args", return_value=args),
            patch.object(run_evaluate, "set_seed"),
            patch.object(run_evaluate, "configure_runtime"),
            patch.object(
                run_evaluate,
                "load_trained_model",
                return_value=(TinyModel(), checkpoint, train_args),
            ),
            patch.object(
                run_evaluate,
                "build_eval_datasets",
                return_value={
                    "MultiServer": multiserver,
                    "DexYCB": dexycb,
                },
            ),
            patch.object(
                run_evaluate,
                "build_eval_loader",
                side_effect=build_loader,
            ),
            patch.object(
                run_evaluate,
                "run_validation_epoch",
                side_effect=validate,
            ),
            patch.object(run_evaluate, "dump_json", side_effect=record_json),
        ):
            run_evaluate.main()

        assert [call[0] for call in validation_calls] == [
            multiserver,
            dexycb,
        ]
        assert [call[1].name for call in validation_calls] == [
            "multiserver",
            "dexycb",
        ]
        assert all(
            call[2] is run_evaluate.build_center_point_prompt
            for call in validation_calls
        )

        summary = written["data"]
        assert summary["prompt_mode"] == "point"
        assert summary["prompt_source"] == "ground_truth_mask_center"
        assert summary["datasets"]["MultiServer"]["raw_samples"] == 30
        assert summary["datasets"]["MultiServer"]["sampled_samples"] == 9
        assert summary["datasets"]["DexYCB"]["raw_samples"] == 40
        assert summary["datasets"]["DexYCB"]["sampled_samples"] == 12
        assert written["output_path"] == Path(output_dir) / "metrics.json"


def main():
    check_checkpoint_loading()
    check_dataset_separation()
    check_separate_evaluation_summary()
    print("Framewise separate evaluation: OK")


if __name__ == "__main__":
    main()
