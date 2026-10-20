# Frozen training membership

`replay_selection.json` is a metadata-only projection of the original
39,561-row ChronoReplay training pool. It records:

- serialized row order and unique IDs;
- source grouping identifiers when present;
- component and target-interface names;
- exact sample loss weights and supervised assistant-turn ordinals;
- image counts and SHA-256 hashes of native prompt/answer text;
- the original training Parquet file's recorded SHA-256.

It contains no questions, answers, image/video pixels, checkpoint weights,
predictions, or evaluation scores.

`python -m scripts.materialize_frozen` matches locally obtained converted
source rows to this manifest and restores their frozen order and weights.
It rejects missing/duplicate IDs, changed text, incompatible supervision
masks, and changed observation counts. Text hashes do not establish pixel
identity; source media must still match the original source records.

The manifest identifies the selected examples. It does not contain the
converted source records themselves or replace access to the licensed
source datasets.
