import pytest
import torch

from scribebench.baseline import parse_rule
from scribebench.data import generate, render
from scribebench.demo import run
from scribebench.model import JointParser, batch, predict, spans_from_tags, vocabulary
from scribebench.schema import Parse, State, transition


def test_name_alone_preserves_unknown_details():
    state, action = transition(State(), Parse("add_medication", {"drug": "medorin"}))
    assert action == "accepted"
    assert state.medications == {"medorin": {"dose": None, "frequency": None}}


def test_context_correction_unknown_and_removal():
    original = State()
    state, _ = transition(original, Parse("add_medication", {"drug": "medorin"}))
    assert original.medications == {}
    state, _ = transition(state, Parse("correct_dose", {"dose": "20"}))
    assert state.medications["medorin"]["dose"] == "20"
    state, _ = transition(state, Parse("unknown_dose", {}))
    assert state.medications["medorin"]["dose"] is None
    state, _ = transition(state, Parse("stop_medication", {"drug": "medorin"}))
    assert state.medications == {} and state.focus is None


def test_unbound_correction_and_low_confidence_do_not_mutate():
    state = State()
    assert transition(state, Parse("correct_dose", {"dose": "20"})) == (state, "clarify")
    assert transition(state, Parse("add_medication", {"drug": "medorin"}, 0.2)) == (
        state,
        "clarify",
    )


def test_negation_updates_only_named_symptom():
    state = State(symptoms={"headache": "present", "nausea": "present"})
    state, _ = transition(state, Parse("deny_symptom", {"symptom": "headache"}))
    assert state.symptoms == {"headache": "denied", "nausea": "present"}


def test_offsets_and_multitoken_bio_roundtrip():
    turn = render(
        "add_medication", 1, {"drug": "luma plus", "dose": "20", "frequency": "twice daily"}
    )
    assert spans_from_tags(turn["text"], turn["tags"]) == sorted(
        turn["spans"], key=lambda x: x["start"]
    )
    for span in turn["spans"]:
        assert turn["text"][span["start"] : span["end"]] == span["value"]


def test_template_and_dialogue_ids_are_disjoint():
    data = generate(60)
    groups = [set(t["family"] for d in rows for t in d["turns"]) for rows in data.values()]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    ids = [d["id"] for rows in data.values() for d in rows]
    assert len(ids) == len(set(ids))


def test_padding_does_not_change_neural_prediction():
    torch.manual_seed(7)
    data = generate(60)["train"]
    vocab = vocabulary(data)
    short, long = data[0]["turns"][5], data[0]["turns"][0]
    model = JointParser(len(vocab)).eval()
    x, lengths, _, _ = batch([short], vocab)
    single = model(x, lengths)[0][0]
    x, lengths, _, _ = batch([short, long], vocab)
    assert torch.allclose(single, model(x, lengths)[0][0], atol=1e-6)


def test_rule_negation():
    parsed, _ = parse_rule("um i have not noticed blurred vision")
    assert parsed.intent == "deny_symptom"
    assert parsed.slots == {"symptom": "blurred vision"}


@pytest.mark.parametrize("confidence", [float("nan"), float("inf"), -1.0, 1.1])
def test_invalid_confidence_never_mutates_state(confidence):
    state = State()
    assert transition(state, Parse("add_medication", {"drug": "medorin"}, confidence)) == (
        state,
        "clarify",
    )


@pytest.mark.parametrize(
    "slots",
    [
        {"drug": "medorin", "dose": "headache"},
        {"drug": "medorin", "dose": "0"},
        {"drug": "medorin", "unexpected": "x"},
    ],
)
def test_invalid_semantic_slots_require_clarification(slots):
    state = State()
    assert transition(state, Parse("add_medication", slots)) == (state, "clarify")


def test_plain_text_inference_and_empty_text():
    data = generate(60)["train"]
    vocab = vocabulary(data)
    parsed = predict(JointParser(len(vocab)), ["i take medorin"], vocab)
    assert len(parsed) == 1
    with pytest.raises(ValueError):
        predict(JointParser(len(vocab)), ["..."], vocab)


def test_slot_uncertainty_limits_frame_confidence():
    class UncertainSlots(torch.nn.Module):
        def forward(self, x, lengths):
            il = torch.full((len(x), 7), -10.0)
            il[:, 0] = 10.0
            sl = torch.zeros((*x.shape, 9))
            sl[:, :, 1] = 0.1  # B-drug, but very weak evidence
            return il, sl

    parsed, _ = predict(UncertainSlots(), ["medorin"], {"<pad>": 0, "<unk>": 1, "medorin": 2})[0]
    assert parsed.slots == {"drug": "medorin"}
    assert parsed.confidence < 0.55


@pytest.mark.parametrize(
    "text",
    [
        "i have not stopped medorin",
        "i do not take medorin",
        "i take medorin and calmexa",
        "actually 10 mg or 20 mg",
    ],
)
def test_ambiguous_or_negated_medication_is_rejected(text):
    frame, _ = parse_rule(text)
    assert transition(State(), frame)[1] == "clarify"


def test_multi_medication_demo_binds_focus_and_preserves_other_record():
    result = run(
        [
            "i take medorin",
            "actually 20 mg",
            "i take calmexa",
            "i do not know the dose",
            "i stopped medorin",
        ]
    )[-1]
    assert result["state"] == {
        "medications": {"calmexa": {"dose": None, "frequency": None}},
        "symptoms": {},
        "focus": "calmexa",
    }
