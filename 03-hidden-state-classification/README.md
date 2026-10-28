# Study 03 — Hidden-State Classification and Sparse Autoencoders

## Research question
Can classifiers recover topic information more effectively from transformed language-model hidden states than from raw representations? This study compares PCA, ICA, sparse autoencoders (SAE), and supervised probe features.

## Methods
- **Models:** BERT and E5 hidden-state representations.
- **AG News:** four topic classes; 100, 500, 2,000, and 6,000 labeled examples in the reported experiment.
- **20 Newsgroups:** twenty classes; 200, 1,000, and 4,000 labeled examples.
- **Feature variants:** raw vectors, PCA-64, ICA-64, supervised probe, and SAE. The Newsgroups experiment evaluates SAE mean and SAE max pooling separately.
- **Classifiers:** logistic regression, k-nearest neighbours, decision tree, random forest, and boosting.
- **Outcome:** classification test accuracy. Exported numerical outputs are included in [results/](results/).

## Key results from the exported CSV files
- **AG News:** highest observed accuracy across all tested settings was **0.9030** (E5, raw, kNN). SAE is **not** universally the best feature representation.
- **20 Newsgroups:** highest observed accuracy was **0.7065** (E5, SAE mean, logistic regression, 4,000 labeled examples). At the same 4,000-example setting with E5/logistic regression, raw features achieved **0.65075**; SAE max **0.7010**; PCA-64 **0.6725**; ICA-64 **0.66775**.
- These are *observed* outcomes in the executed run, not evidence that SAE systematically dominates across datasets or models. Settings were compared on the same test results, so the maximum should not be interpreted as an unbiased estimate of the best approach.

## Figures and data
- [AG News results](results/ag_news_results.csv)
- [20 Newsgroups results](results/newsgroups_results.csv)
- [AG News accuracy comparison](figures/ag_news_accuracy.svg)
- [20 Newsgroups accuracy comparison](figures/newsgroups_accuracy.svg)

The summary plots are reconstructed from the saved CSVs and are not pixel-perfect reproductions of Colab's original PNG figures.

## Reproduce
Open [03_hidden_state_classification.ipynb](03_hidden_state_classification.ipynb) in Google Colab, enable a GPU, then run cells top-to-bottom. The GPU accelerates transformer feature extraction and SAE training, while CPU routines are used for several classical methods.

**Limitations:** Findings depend on the dataset, model, pooling and feature dimensions, train/test split, seeds, and classifier hyperparameters. A stronger claim about generalization requires repeated splits and validation-set model selection; check that feature transforms and supervised probes are fitted only on training partitions. This is an exploratory study.
