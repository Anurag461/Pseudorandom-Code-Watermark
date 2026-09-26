import argparse
import json
from pathlib import Path
import torch
from baselines import generation as baseline_generation
from prc_watermark.generation import generate_batch_and_collect_online
from .config import StudySetting
from .generation import generate_response_batch
from .repeat import RepeatSetting, generate_repeat_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--setting",
        choices=[
            "prc",
            "null",
            "textseal",
            "synthid",
            "synthid_depth2",
            "synthid_depth30",
            "gumbel",
            "textseal_repeat_on",
            "synthid_repeat_off",
            "gumbel_repeat_on",
        ],
        required=True,
    )
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    args = parser.parse_args()
    if args.setting == "prc" and args.artifact is None:
        parser.error("--artifact is required for PRC")
    names = {"prc": "online_prc", "synthid": "synthid_text", "gumbel": "gumbel_max"}
    method = names.get(args.setting.split("_")[0], args.setting.split("_")[0])
    options = {}
    if args.setting == "synthid_depth2":
        options["depth"] = 2
    if args.setting == "synthid_depth30":
        options["depth"] = 30
    repeat = "_repeat_" in args.setting
    setting = (
        RepeatSetting(method, repeat_fallback=args.setting.endswith("_repeat_on"))
        if repeat
        else StudySetting(method, **options)
    )
    baseline_generation.MODEL_ROOT = str(args.model_directory)
    model = baseline_generation.load_qwen3_8b()
    prompts = [
        json.loads(line)["prompt_tokens"]
        for line in args.prompts.read_text().splitlines()
    ][:50]
    artifact = (
        torch.load(args.artifact, map_location="cpu", weights_only=False)
        if args.artifact
        else None
    )
    args.output.mkdir(parents=True, exist_ok=True)
    execution = {
        "torch": torch.__version__,
        "device_name": torch.cuda.get_device_name(),
        "batch_size": args.batch_size,
    }
    for response_index, seed in enumerate([12345, 67890]):
        for offset in range(0, len(prompts), args.batch_size):
            indices = list(range(offset, min(offset + args.batch_size, len(prompts))))
            kwargs = dict(
                setting=setting,
                sampling_seed=seed,
                response_index=response_index,
                execution=execution,
            )
            if method == "online_prc":
                kwargs.update(
                    prc_artifact=artifact,
                    online_sampler=generate_batch_and_collect_online,
                )
            batch = (generate_repeat_batch if repeat else generate_response_batch)(
                model, [prompts[i] for i in indices], indices, **kwargs
            )
            (args.output / f"response{response_index}_{offset:04d}.json").write_text(
                json.dumps(batch) + "\n"
            )


if __name__ == "__main__":
    main()
