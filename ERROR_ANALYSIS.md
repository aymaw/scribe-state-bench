# Error analysis

The benchmark intentionally holds out complete surface-template families.
All three neural runs select their first epoch by validation loss; later training
reduces training loss while increasing shifted validation loss. This is direct
evidence of overfitting to the small authored language inventory.

Unseen function words map to `<unk>`, while medication/symptom tokens were seen
in training. The model can still put entity boundaries in the wrong places,
confuse symptom polarity, or misclassify an utterance. Inspect each run's retained
`errors` in `results/metrics.json`; examples include exact predicted/gold frames.

Joint state accuracy is stricter than per-turn semantic accuracy: an incorrect
addition or dose update can persist until a later correcting turn. Re-evaluating
every turn from an empty state would conceal this accumulation and is not done.

The perfect rule score reflects a closed lexicon and explicit cues. It should
motivate broader independently authored challenge data, not a claim that the
clinical language problem is solved. A stronger follow-up experiment would hold
out entity names, vary dialogue ordering, add ambiguous references and multiple
medications, then compare calibrated abstention and context-aware encoders.

## Conservative decision policy

The reviewed parser bounds frame confidence by the minimum confidence of its
predicted entity tokens. The reducer also rejects non-finite scores, irrelevant
slots and malformed dose strings. The stronger guard reduces state mutations
under shifted phrasing; it does not improve the underlying neural parser.
The separate familiar-template slice demonstrates fitting and state integration,
while the primary held-out-template result remains a failed generalization test.
