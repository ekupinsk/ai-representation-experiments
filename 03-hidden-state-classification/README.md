# Study 03 — Hidden-State Classification and Sparse Autoencoders

## 1. What did we want to discover? — Research hypothesis
Can classifiers recover topic information more effectively from transformed language-model hidden states than from raw representations? This study compares PCA, ICA, sparse autoencoders (SAE), and supervised probe features.

**Hypothesis:** Transforms of frozen language-model hidden states — particularly sparse-autoencoder features — may expose classification-relevant structure that a standard classifier cannot exploit as easily in raw vectors. The gain might vary with the number of labelled examples.

## 2. How did we test it? — Experiments, metrics and controls
- **Models:** BERT and E5 hidden-state representations.
- **AG News:** four topic classes; 100, 500, 2,000, and 6,000 labeled examples in the reported experiment.
- **20 Newsgroups:** twenty classes; 200, 1,000, and 4,000 labeled examples.
- **Feature variants:** raw vectors, PCA-64, ICA-64, supervised probe, and SAE. The Newsgroups experiment evaluates SAE mean and SAE max pooling separately.
- **Classifiers:** logistic regression, k-nearest neighbours, decision tree, random forest, and boosting.
- **Outcome:** classification test accuracy. Exported numerical outputs are included in [results/](results/).

**Comparisons / controls:** the raw feature vector is the principal baseline; PCA-64 and ICA-64 are alternative unsupervised transforms; supervised probe features offer another comparison. Both datasets were evaluated at multiple labelled-data sizes, across five classifier families. **Metric:** test accuracy.

## 3. What did we learn? — Results, interpretation and limitations

### Observed results
- **AG News:** highest observed accuracy across all tested settings was **0.9030** (E5, raw, kNN). SAE is **not** universally the best feature representation.
- **20 Newsgroups:** highest observed accuracy was **0.7065** (E5, SAE mean, logistic regression, 4,000 labeled examples). At the same 4,000-example setting with E5/logistic regression, raw features achieved **0.65075**; SAE max **0.7010**; PCA-64 **0.6725**; ICA-64 **0.66775**.
- These are *observed* outcomes in the executed run, not evidence that SAE systematically dominates across datasets or models. Settings were compared on the same test results, so the maximum should not be interpreted as an unbiased estimate of the best approach.

### Interpretation

SAE mean features outperform raw E5 features with logistic regression in the observed 4,000-label Newsgroups setting, but this does not generalize to all tasks or classifiers: the strongest AG News result uses raw features. The comparison does not establish that SAE features are more semantically interpretable.

### Limitations

- Repeated seeds, independent test splits and selection of hyperparameters on a validation set would be needed to establish robust generalization.
- Transform fitting and supervised probe training must be confined to training partitions to prevent leakage. This documentation alone does not constitute an independent leakage audit.
- The maximum observed test accuracy over many configurations is a descriptive maximum, not an unbiased performance estimate for the selected method.

## Figures and data
- [AG News results](results/ag_news_results.csv)
- [20 Newsgroups results](results/newsgroups_results.csv)
- [AG News accuracy comparison](figures/ag_news_accuracy.svg)
- [20 Newsgroups accuracy comparison](figures/newsgroups_accuracy.svg)

The summary plots are reconstructed from the saved CSVs and are not pixel-perfect reproductions of Colab's original PNG figures.

## Reproduce
Open [03_hidden_state_classification.ipynb](03_hidden_state_classification.ipynb) in Google Colab, enable a GPU, then run cells top-to-bottom. The GPU accelerates transformer feature extraction and SAE training, while CPU routines are used for several classical methods.

**Additional scope note:** Findings depend on the dataset, model, pooling and feature dimensions, train/test split, random seeds and classifier hyperparameters. This is an exploratory study.

## Executed Google Colab notebook

[View executed notebook with saved outputs](https://colab.research.google.com/drive/1OkiCF16IhP-Pggox1YR3NtmDO1UxfkQQ?usp=sharing) — the shared Colab copy showing the experiment run. For the version-controlled source, use the notebook linked above in this repository. The shared notebook is viewable without signing in (checked by the repository owner in a private browsing window).
