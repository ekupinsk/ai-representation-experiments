# AI Representation Experiments

A collection of reproducible exploratory studies on language-model hidden states, representation similarity, and machine learning.

## Studies

| Directory | Topic | Status |
|---|---|---|
| [01-hidden-state-agreement](01-hidden-state-agreement/) | Pythia, BERT, and MiniLM nearest-neighbour agreement, depth effects, lexical and random controls | Results recorded, notebook available |
| 02-cross-model-analysis | Expanded comparison across model families | Planned |
| 03-hidden-state-classification | Probes, PCA, ICA, and sparse autoencoders | Planned |
| 04-visual-counting | CNN shape counting and generalization | Planned |

The analysis treats neighbour overlap as **representational geometry**, not direct evidence of identical semantics. See each study's README for assumptions and caveats.

## Running

Open the `.ipynb` file in [Google Colab](https://colab.research.google.com/), select a GPU if available, and use **Runtime → Run all**.

Other independent demos are maintained in [Hebrew Image Search](https://github.com/ekupinsk/hebrew-image-search) and [Hebrew QA Retrieval](https://github.com/ekupinsk/hebrew-qa-retrieval).
