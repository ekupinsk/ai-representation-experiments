# Study 01 — Cross-Model Hidden-State Agreement

## Question

Do independently trained language models organize the *same texts* into similar nearest-neighbour structures, and how do their layers, normalization choices and baselines affect the answer?

## Setup

- **Dataset:** 2,000 English text samples from AG News (four topics).
- **Main models:** Pythia-160M, BERT-base and MiniLM-L6.
- **Controls:** randomly initialized BERT variants and a TF-IDF lexical baseline.
- **Representation:** per-layer token mean pooling; standardization of each feature dimension unless stated otherwise.
- **Primary metric:** top-10 nearest-neighbour set overlap, averaged across samples. Chance baseline is about **0.5%** at 2,000 samples.
- **Additional metric:** linear CKA, with raw versus standardized comparisons.

## Results from the executed Colab run

| Experiment | Main observation |
|---|---|
| 1. Pythia × BERT cross-layer scan | Average overlap **0.375** trained vs **0.118** random control; trained higher in every layer pair |
| 2. Relative depth gap | Mean Pythia × BERT agreement falls from **0.408** near matching depth to **0.343** for far-apart depths |
| 3. TF-IDF lexical baseline | Mean overlap with TF-IDF: Pythia **0.224**, BERT **0.234**, MiniLM **0.237** |
| 4. Metric/preprocessing check | With raw CKA, trained exceeds random in only **57%** of cells; with standardized CKA, **100%** |
| 5. Within-topic analysis | Agreement remains elevated after restricting comparisons within topical classes (top-5 neighbours) |
| 6. Held-out confirmation | Pythia layer 11 × BERT layer 11 selected on exploration split, then **0.497** on confirmation half, vs **0.092** random control |
| 7. Matched relative depth | Mean trained overlaps: Pythia × BERT **0.408**, Pythia × MiniLM **0.372**, BERT × MiniLM **0.426** |

These figures describe **this particular dataset and run**. They do not establish that different architectures encode identical semantic features.

## Main figures

![Cross-layer overlap and random baseline](figures/exp1_heatmaps.png)

![Layer depth gap](figures/exp2_depth.png)

![TF-IDF baseline](figures/exp3_tfidf.png)

![Raw vs standardized metrics](figures/exp4_metric_check.png)

![Within-topic analysis](figures/exp5_within_topic.png)

![Held-out confirmation](figures/exp6_heldout.png)

![Three model pairs](figures/exp7_three_models.png)

## Methodological cautions

1. **Nearest-neighbour overlap is not proof of shared semantics.** Lexical and topical regularities can contribute; TF-IDF is a useful but incomplete lexical baseline.
2. **Uncertainty on the held-out chart is descriptive.** Bootstrap intervals resample per-query overlap scores after neighbour sets have been computed. Queries share the underlying corpus; this dependency may make the reported 95% intervals too narrow.
3. **The held-out estimate is not directly comparable to the full-sample scan.** Top-10 overlap uses a different candidate count when the dataset is split in half; the chance baseline consequently changes.
4. **Depth matters, but is not causal evidence.** Relative-depth patterns are descriptive averages over many correlated layer pairs.
5. **Normalization strongly affects CKA.** Raw and per-dimension-standardized results should not be combined or cherry-picked.
6. **Scope:** one English news dataset, mean pooling, two random initialization seeds per architecture, one sample of texts. Broader replication is needed.

## Files and reproducibility

- `report.txt` — text summary of the executed results.
- `results.json` — machine-readable results.
- `figures/` — graphical outputs.
- The executed Colab notebook is being transferred into this directory; until then, the results are documented but code reproduction is incomplete.

All reported result values are taken from the provided executed experiment, not reconstructed from the figures.
