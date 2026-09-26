import argparse
import csv
import json
from pathlib import Path
import torch
from baselines.official import official_gumbel_scores, synthid_processor
from baselines.scoring import (
    deduplicated_positions,
    gumbel_gamma_test,
    synthid_normal_test,
)
from prc_watermark.detectors import detect_online_hoeffding
from prc_watermark.qwen import completion_only_partition_trace_batch, load_model
from .config import StudySetting


def detect(
    responses, method, model_directory=None, artifact=None, depth=10, cache_root=None
):
    result = []
    model = None
    textseal = None
    if method == "prc":
        model, _ = load_model(model_directory, "8B")
        artifact = torch.load(artifact, map_location="cpu", weights_only=False)
    elif method == "textseal":
        from baselines.textseal import (
            TextSealCompletionDetector,
            load_model as load_textseal_model,
            runtime_identity,
        )

        root = Path(__file__).resolve().parents[2]
        environments = json.loads((root / "experiments/environments.json").read_text())
        request = {
            "model": json.loads(Path(__file__).with_name("model.json").read_text()),
            "runtime": {"dependencies": environments["textseal"]["packages"]},
        }
        runtime_identity(request)
        textseal = TextSealCompletionDetector(load_textseal_model(request, cache_root))
    processor = (
        synthid_processor(
            "cpu", keys=StudySetting("synthid_text", depth=depth).synthid_keys
        )
        if method == "synthid"
        else None
    )
    for start in range(0, len(responses), 50):
        batch = responses[start : start + 50]
        tokens = torch.tensor(
            [row["token_ids"][:1024] for row in batch], dtype=torch.long
        )
        if tokens.shape[1] != 1024:
            raise ValueError("Expected 1,024-token completions")
        if method == "prc":
            probabilities = (
                completion_only_partition_trace_batch(
                    model,
                    tokens.cuda(),
                    artifact["partition"][1],
                    kv_cache_implementation="static",
                )
                .cpu()
                .numpy()
            )
        if method == "synthid":
            values = processor.compute_g_values(tokens).numpy()
            masks = (
                processor.compute_context_repetition_mask(tokens).numpy().astype(bool)
            )
        for index, row in enumerate(batch):
            ids = tokens[index].tolist()
            if method == "prc":
                decision, info = detect_online_hoeffding(
                    artifact["online_key"],
                    tokens[index],
                    probabilities[index],
                    artifact["partition"],
                    fpr=0.001,
                    weight="map",
                    return_info=True,
                )
                score = {
                    "decision": decision,
                    "statistic": info["statistic"],
                    "threshold": info["threshold"],
                }
            elif method == "textseal":
                info = textseal.detect(ids)
                score = {
                    "decision": info["comparison"]["decision"],
                    "p_value": info["comparison"]["p_value"],
                }
            elif method == "synthid":
                score = synthid_normal_test(values[index][masks[index]])
            else:
                positions = deduplicated_positions(ids)
                score = gumbel_gamma_test(official_gumbel_scores(ids, positions))
            result.append(
                {
                    "response_id": row["response_id"],
                    "prompt_index": row["prompt_index"],
                    "response_index": row["response_index"],
                    "detector": method,
                    "detected": score["decision"],
                    "p_value": score.get("p_value", ""),
                }
            )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--method", choices=["prc", "textseal", "synthid", "gumbel"], required=True
    )
    parser.add_argument("--model-directory", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--depth", type=int, choices=[2, 10, 30], default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.method == "prc" and (args.model_directory is None or args.artifact is None):
        parser.error("PRC requires --model-directory and --artifact")
    if args.method == "textseal" and args.cache_root is None:
        parser.error("TextSeal requires --cache-root")
    rows = []
    for path in sorted(args.input.glob("response*.json")):
        rows.extend(json.loads(path.read_text())["responses"])
    scores = detect(
        rows,
        args.method,
        args.model_directory,
        args.artifact,
        args.depth,
        args.cache_root,
    )
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scores[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(scores)


if __name__ == "__main__":
    main()
