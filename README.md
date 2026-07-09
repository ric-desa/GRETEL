<!-- # XPlore

Official repository for **XPlore**.

To run any explainer, run the corresponding configuration file.
For example, to run XPlore on TreeCyclesRand, run the following:
``` bash
python main.py XPlore_config/GRAPH/TCR/TCR-5000-28-0.3_GCN_XPlore++.jsonc
```

The algorithmic implentation is found at [XPlore](src/explainer/XPlore.py).

Be sure to have installed all requirements and the correct version of the libraries. -->

# XPlore: Beyond Edge Deletion for Counterfactual GNN Explanations

<div align="center">

<!-- [![Paper](https://img.shields.io/badge/Paper-arXiv%3A2603.04209-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2603.04209)
[![Framework](https://img.shields.io/badge/Built%20on-GRETEL-orange?style=flat-square)](https://github.com/aiim-research/GRETEL)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green?style=flat-square&logo=python)](https://www.python.org/) -->
<!-- [![OpenReview](https://img.shields.io/badge/OpenReview-ICLR%202026-blue?style=flat-square)](https://openreview.net/forum?id=L4KJT9QpqE) -->

<img src="imgs\Interpretability_t-sne.jpg" alt="Photo" width="100%" />

**Gradient-guided counterfactual explanations for Graph Neural Networks via comprehensive input perturbation.**

*+56.3% validity · +52.8% fidelity · best on 17/18 benchmarks*

</div>

---

## Overview

**XPlore** is a novel counterfactual explanation method for Graph Neural Networks (GNNs). Given a graph classified by a GNN, XPlore finds the *minimal perturbation* that flips the model's prediction, revealing what the model truly relies on.

Unlike prior methods that restrict themselves to edge deletions, XPlore jointly optimizes over:

- **Edge additions** — inserting new structural connections
- **Edge removals** — deleting existing edges
- **Node feature perturbations** — modifying node attributes

This comprehensive perturbation space, guided entirely by oracle gradients, yields more coherent and minimal counterfactuals without any auxiliary training or surrogate models.

> *"What would this molecule need to change to be predicted non-toxic?"*
> *"Which edges, if added or removed, would flip the fraud classification?"*

XPlore answers these questions faithfully and efficiently.

---

## Key Contributions

1. **Comprehensive Perturbation Space** — Supports edge additions, edge removals, and node feature changes simultaneously under a single unified gradient-based framework.

2. **Oracle-Gradient Guidance** — Treats the target GNN as a fixed oracle. No auxiliary model, no surrogate, no additional training phase. Lightweight by design.

3. **Cosine Similarity Metric** — A new embedding-space distance metric for evaluating counterfactual quality, addressing limitations of traditional graph edit distance.

4. **State-of-the-Art Results** — Best validity and fidelity on 17 out of 18 benchmark datasets (9 real-world + 5 synthetic), with competitive runtime.

---

## Results at a Glance

| Metric | XPlore vs. 2nd Best |
|--------|-------------------|
| Validity | **+15.1%** average improvement |
| Fidelity | **+14.0%** average improvement |
| Best-on datasets | **17 / 18** |
| Max validity gain | **+56.3%** |
| Max fidelity gain | **+52.8%** |

---

## Quick Start

### Installation

**Option 1: pip**
```bash
pip install -r requirements.txt
```

**Option 2: conda**
```bash
conda env create -f environment.yml
conda activate gretel
```

<!-- 
**Option 3: Docker**
```bash
# CPU
docker build -f dockerfile -t xplore .

# GPU
docker build -f dockerfile.gpu -t xplore-gpu . 
```
-->

### Running XPlore

```bash
# Run XPlore on a specific dataset and classifier
python main.py XPlore_config/GRAPH/TCR/TCR-5000-28-0.3_GCN_XPlore++.jsonc
```

Configuration files follow the naming pattern:
```
XPlore_config/<task>/<dataset>/<dataset_params>_<classifier>_<explainer>.jsonc
```

Available task types: `GRAPH` (graph classification), `NODE` (node classification).

---

## Repository Structure

```
XPlore/
├── src/
│   └── explainer/
│       └── XPlore.py          ← XPlore algorithm implementation
├── XPlore_config/              ← Configuration files per dataset/classifier
│   └── GRAPH/
│       └── TCR/                ← TreeCyclesRand configs
├── config/                     ← General framework configurations
├── data/datasets/              ← Dataset storage
├── run_experiments/            ← Experiment launcher scripts
├── visualizations/             ← Output visualization utilities
├── launchers/                  ← Batch experiment launchers
├── main.py                     ← Entry point
├── environment.yml             ← Conda environment
└── requirements.txt            ← pip dependencies
```

The core algorithm is implemented in [XPlore](src/explainer/XPlore.py).

---

## How It Works

XPlore frames counterfactual explanation as a gradient-guided optimization over both the adjacency matrix **A** and node feature matrix **X**:

1. **Forward pass** — the input graph is classified by the frozen target GNN (the oracle).
2. **Gradient computation** — directional signals from the oracle's loss indicate which perturbations push the prediction toward the counterfactual class.
3. **Perturbation step** — adjacency values and node features are updated using the gradient signal, under a distance constraint to keep changes minimal.
4. **Termination** — the loop stops when the prediction flips, or a maximum number of oracle calls is reached.

This design is transparent by construction: there is no learned surrogate, no generative model, and no hidden intermediate representation.

---

<!-- ## Citing

If you use XPlore in your research, please cite:

```bibtex
@misc{desanctis2026edgedeletioncomprehensiveapproach,
      title={Beyond Edge Deletion: A Comprehensive Approach to Counterfactual Explanation in Graph Neural Networks}, 
      author={Matteo De Sanctis and Riccardo De Sanctis and Stefano Faralli and Paola Velardi and Bardh Prenkaj},
      year={2026},
      eprint={2603.04209},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2603.04209}, 
}
```

--- -->

<!-- ## Built On

XPlore is implemented within the [**GRETEL**](https://github.com/aiim-research/GRETEL) framework — a unified environment for developing and evaluating counterfactual explanation methods for graph classifiers.

--- -->

<!-- ## License

See [LICENSE](LICENSE) for details. -->
