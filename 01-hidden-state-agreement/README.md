# Study 01 — Hidden-State Agreement

## Research question
Do independently trained language models organize the same texts into similar nearest-neighbour structures? How does agreement vary by layer, normalization, and comparison baseline?

## Methods
- **Dataset:** 2,000 AG News texts (English; four news topics).
- **Models:** Pythia-160M, BERT-base, MiniLM-L6; random-initialization controls.
- **Representations:** per-layer token mean pooling; per-dimension z-scoring except where noted.
- **Metrics:** mean top-10 nearest-neighbour overlap and CKA; TF-IDF lexical baseline.
- **Validation:** explore/held-out confirmation split for a selected layer pair.
- **Runtime:** Google Colab; notebook requires an internet connection to download pretrained models and data.

## Results

| Experiment | Observation |
|---|---|
| 1 — Cross-layer scan | Pythia × BERT: **0.375** mean overlap; random control **0.118**; trained higher in every evaluated layer pair |
| 2 — Depth gap | Pythia × BERT: **0.408** near matching depth vs **0.343** at large relative-depth difference |
| 3 — Lexical baseline | Mean TF-IDF overlap: Pythia **0.224**, BERT **0.234**, MiniLM **0.237** |
| 4 — Metric check | Raw CKA: trained above random in **57%** of pairs; standardized CKA: **100%** |
| 5 — Within-topic | Pythia × BERT agreement remains elevated within the same AG News topic at k=5 |
| 6 — Held-out check | Selected pair Pythia layer 11 × BERT layer 11: held-out overlap **0.497**, random control **0.092** |
| 7 — Matched depths | Trained agreement: Pythia × BERT **0.408**; Pythia × MiniLM **0.372**; BERT × MiniLM **0.426** |

Full machine-readable values: [results.json](results.json). Full console summary: [report.txt](report.txt).

## Running the analysis

1. Open [01_hidden_state_agreement.ipynb](01_hidden_state_agreement.ipynb) in Google Colab.
2. Select a GPU runtime, if available.
3. Run all cells in order; experiment outputs and figures are generated in the notebook.
4. Compare the new summary with `results.json` and `report.txt`.

The notebook in the repository is the **clean, unexecuted analysis version**. The numeric results were obtained in a separately executed Colab run; they have not been regenerated from this GitHub copy yet.

## Important limitations
- Neighbour agreement alone **does not establish shared semantics or identical internal features**.
- Results are from one English news dataset, mean pooling, one text sample, and two random seeds per architecture.
- Relative depth correlations are descriptive and are not evidence of causal relationships.
- Normalization affects results, notably for CKA.
- In the held-out analysis, bootstrap confidence intervals were formed by resampling per-query overlap scores from a shared neighbour corpus. Dependence among queries may yield intervals that are too narrow.
- The confirmation half uses a different neighbour candidate count from the full 2,000-sample scan, so raw overlap scores are not directly comparable.

The goal is **transparent exploratory evidence**, not a proof of semantic equivalence.
