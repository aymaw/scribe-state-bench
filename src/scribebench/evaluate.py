import numpy as np

from .schema import INTENTS, State, transition


def evaluate(dialogues, predictions):
    tp = fp = fn = intent_correct = parse_correct = joint_correct = turns = 0
    per_dialogue = []
    errors = []
    intent_counts = {k: [0, 0] for k in INTENTS}
    for dialogue, parsed in zip(dialogues, predictions, strict=True):
        state, joint = State(), []
        for turn, (prediction, spans) in zip(dialogue["turns"], parsed, strict=True):
            gold = {(s["slot"], s["start"], s["end"]) for s in turn["spans"]}
            pred = {(s["slot"], s["start"], s["end"]) for s in spans}
            tp += len(gold & pred)
            fp += len(pred - gold)
            fn += len(gold - pred)
            intent_correct += prediction.intent == turn["intent"]
            intent_counts[turn["intent"]][0] += prediction.intent == turn["intent"]
            intent_counts[turn["intent"]][1] += 1
            exact = prediction.intent == turn["intent"] and prediction.slots == turn["slots"]
            parse_correct += exact
            state, action = transition(state, prediction)
            is_joint = state.snapshot() == turn["state"]
            joint.append(is_joint)
            joint_correct += is_joint
            turns += 1
            if not exact and len(errors) < 30:
                errors.append(
                    {
                        "text": turn["text"],
                        "gold_intent": turn["intent"],
                        "predicted_intent": prediction.intent,
                        "gold_slots": turn["slots"],
                        "predicted_slots": prediction.slots,
                        "action": action,
                    }
                )
        per_dialogue.append(float(np.mean(joint)))
    rng = np.random.default_rng(9)
    samples = [float(np.mean(rng.choice(per_dialogue, len(per_dialogue)))) for _ in range(500)]
    return {
        "n_dialogues": len(dialogues),
        "n_turns": turns,
        "intent_accuracy": intent_correct / turns,
        "semantic_exact_match": parse_correct / turns,
        "span_precision": tp / (tp + fp) if tp + fp else 0,
        "span_recall": tp / (tp + fn) if tp + fn else 0,
        "span_micro_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0,
        "joint_state_accuracy": joint_correct / turns,
        "joint_state_dialogue_bootstrap_95": np.quantile(samples, [0.025, 0.975]).tolist(),
        "per_intent_accuracy": {k: a / b if b else None for k, (a, b) in intent_counts.items()},
        "errors": errors,
    }
