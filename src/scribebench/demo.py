"""Inspect a synthetic text dialogue using rules or a locally trained parser."""

import argparse
import json
from dataclasses import asdict

import torch

from .baseline import parse_rule
from .model import JointParser, predict
from .schema import State, transition


def run(texts, checkpoint=None):
    if checkpoint:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model = JointParser(len(payload["vocabulary"]))
        model.load_state_dict(payload["state_dict"])
        parsed = predict(model, texts, payload["vocabulary"])
    else:
        parsed = [parse_rule(text) for text in texts]
    state, results = State(), []
    for text, (frame, spans) in zip(texts, parsed, strict=True):
        state, action = transition(state, frame)
        results.append(
            {
                "text": text,
                "frame": asdict(frame),
                "spans": spans,
                "action": action,
                "state": state.snapshot(),
            }
        )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--text",
        nargs="+",
        default=[
            "i take medorin",
            "actually 20 mg",
            "i take calmexa",
            "i do not know the dose",
            "i stopped medorin",
        ],
    )
    args = parser.parse_args()
    print(json.dumps(run(args.text, args.checkpoint), indent=2))
