import argparse
import copy
import json
import platform
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from .baseline import parse_rule
from .data import fingerprint, generate
from .evaluate import evaluate
from .model import JointParser, batch, predict, vocabulary


def group_predictions(dialogues, flat):
    result, cursor = [], 0
    for d in dialogues:
        count = len(d["turns"])
        result.append(flat[cursor : cursor + count])
        cursor += count
    return result


def run(n=600, epochs=25, seeds=(7, 19, 31), out="runs/reference"):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    data = generate(n)
    (out / "dataset.json").write_text(json.dumps(data, indent=2))
    vocab = vocabulary(data["train"])
    flat = {k: [t for d in rows for t in d["turns"]] for k, rows in data.items()}
    report = {
        "synthetic_only": True,
        "n_dialogues": n,
        "epochs": epochs,
        "data_sha256": fingerprint(data),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": "cpu",
        "rule_baseline": evaluate(
            data["test"], [[parse_rule(t["text"]) for t in d["turns"]] for d in data["test"]]
        ),
        "runs": [],
    }
    for seed in seeds:
        start = time.perf_counter()
        torch.manual_seed(seed)
        model = JointParser(len(vocab))
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
        best, checkpoint, history = float("inf"), None, []
        for epoch in range(epochs):
            model.train()
            total = 0.0
            for idx in torch.randperm(len(flat["train"])).split(64):
                turns = [flat["train"][i] for i in idx]
                x, lengths, tags, intents = batch(turns, vocab)
                i_logits, s_logits = model(x, lengths)
                loss = F.cross_entropy(i_logits, intents) + F.cross_entropy(
                    s_logits.flatten(0, 1), tags.flatten()
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                total += loss.item() * len(turns)
            model.eval()
            with torch.no_grad():
                x, lengths, tags, intents = batch(flat["validation"], vocab)
                il, sl = model(x, lengths)
                val = (
                    F.cross_entropy(il, intents) + F.cross_entropy(sl.flatten(0, 1), tags.flatten())
                ).item()
            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": total / len(flat["train"]),
                    "validation_loss": val,
                }
            )
            if val < best:
                best, checkpoint = val, copy.deepcopy(model.state_dict())
        model.load_state_dict(checkpoint)
        predictions = group_predictions(data["test"], predict(model, flat["test"], vocab))
        result = {
            "seed": seed,
            "best_validation_loss": best,
            "history": history,
            "seconds": time.perf_counter() - start,
            **evaluate(data["test"], predictions),
        }
        report["runs"].append(result)
        torch.save({"state_dict": checkpoint, "vocabulary": vocab}, out / f"joint-{seed}.pt")
        print(
            f"seed={seed} span-F1={result['span_micro_f1']:.3f} joint-state={result['joint_state_accuracy']:.3f}",
            flush=True,
        )
        (out / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=600)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--seeds", nargs="+", type=int, default=[7, 19, 31])
    p.add_argument("--out", default="runs/reference")
    a = p.parse_args()
    run(a.n, a.epochs, a.seeds, a.out)
