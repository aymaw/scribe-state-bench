import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .data import tokens
from .schema import INTENTS, TAGS, Parse


def vocabulary(dialogues):
    words = sorted({m.group() for d in dialogues for t in d["turns"] for m in tokens(t["text"])})
    return {"<pad>": 0, "<unk>": 1, **{w: i + 2 for i, w in enumerate(words)}}


def batch(turns, vocab):
    ids = [[vocab.get(m.group(), 1) for m in tokens(t["text"])] for t in turns]
    lengths = torch.tensor([len(x) for x in ids])
    x = torch.zeros((len(ids), int(lengths.max())), dtype=torch.long)
    tags = torch.full_like(x, -100)
    for i, (row, turn) in enumerate(zip(ids, turns)):
        x[i, : len(row)] = torch.tensor(row)
        tags[i, : len(row)] = torch.tensor([TAGS.index(tag) for tag in turn["tags"]])
    return x, lengths, tags, torch.tensor([INTENTS.index(t["intent"]) for t in turns])


class JointParser(nn.Module):
    def __init__(self, vocab_size, width=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, width, padding_idx=0)
        self.encoder = nn.LSTM(width, width, batch_first=True, bidirectional=True)
        self.slot_head = nn.Linear(width * 2, len(TAGS))
        self.intent_head = nn.Linear(width * 2, len(INTENTS))

    def forward(self, x, lengths):
        packed = pack_padded_sequence(
            self.embedding(x), lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        encoded, _ = self.encoder(packed)
        hidden, _ = pad_packed_sequence(encoded, batch_first=True, total_length=x.shape[1])
        mask = torch.arange(x.shape[1])[None, :] < lengths[:, None]
        pooled = (hidden * mask[..., None]).sum(1) / lengths[:, None]
        return self.intent_head(pooled), self.slot_head(hidden)


def spans_from_tags(text, tags):
    """Treat an orphan I tag as a new span; retain exact character offsets."""
    matches, spans, active = tokens(text), [], None
    for match, tag in zip(matches, tags):
        if tag == "O":
            active = None
            continue
        prefix, slot = tag.split("-", 1)
        if prefix == "B" or active is None or active["slot"] != slot:
            active = {"slot": slot, "start": match.start(), "end": match.end()}
            spans.append(active)
        else:
            active["end"] = match.end()
    for span in spans:
        span["value"] = text[span["start"] : span["end"]].lower()
    return spans


def predict(model, turns, vocab):
    model.eval()
    results = []
    with torch.no_grad():
        for start in range(0, len(turns), 64):
            rows = turns[start : start + 64]
            x, lengths, _, _ = batch(rows, vocab)
            intent_logits, slot_logits = model(x, lengths)
            confidence, intent = intent_logits.softmax(-1).max(-1)
            tags = slot_logits.argmax(-1)
            for i, row in enumerate(rows):
                spans = spans_from_tags(row["text"], [TAGS[t] for t in tags[i, : lengths[i]]])
                # Duplicate slot types are ambiguous; reject instead of silently overwriting.
                slots = {s["slot"]: s["value"] for s in spans}
                conf = confidence[i].item() if len(slots) == len(spans) else 0.0
                results.append((Parse(INTENTS[intent[i]], slots, conf), spans))
    return results
