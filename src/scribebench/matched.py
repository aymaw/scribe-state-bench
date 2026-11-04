"""Closed-template integration check. Individual utterance types may repeat train."""

import argparse
import json
from pathlib import Path

import torch

from .data import fingerprint, generate
from .evaluate import evaluate
from .model import JointParser, predict
from .train import group_predictions


def run(runs, out):
    dialogues = generate(600, seed=901)["train"][:120]
    for i, dialogue in enumerate(dialogues):
        dialogue["id"] = f"matched-901-{i}"
    train = generate(600)["train"]

    def signature(dialogue):
        return tuple(turn["text"] for turn in dialogue["turns"])

    assert not ({signature(d) for d in dialogues} & {signature(d) for d in train})
    flat = [turn for d in dialogues for turn in d["turns"]]
    report = {
        "scope": "same-template synthetic evaluation; individual utterances may repeat train",
        "data_sha256": fingerprint(dialogues),
        "runs": [],
    }
    torch.set_num_threads(1)
    for path in sorted(Path(runs).glob("joint-*.pt")):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model = JointParser(len(checkpoint["vocabulary"]))
        model.load_state_dict(checkpoint["state_dict"])
        parsed = group_predictions(dialogues, predict(model, flat, checkpoint["vocabulary"]))
        report["runs"].append({"checkpoint": path.name, **evaluate(dialogues, parsed)})
    if not report["runs"]:
        raise ValueError("No trained checkpoints found")
    Path(out).write_text(json.dumps(report, indent=2) + "\n")
    print([(r["checkpoint"], round(r["joint_state_accuracy"], 3)) for r in report["runs"]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs/reference")
    parser.add_argument("--out", default="runs/matched-template.json")
    args = parser.parse_args()
    run(args.runs, args.out)
