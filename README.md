SDGFN: A sMRI-based Dual-branch Graph Fusion Network for MDD Recognition

Core implementation of SDGFN for major depressive disorder (MDD) classification using structural MRI-derived brain networks.

The experiments used the Beijing Anding Hospital clinical cohort and the publicly available multi-site Meta-MDD dataset; neither dataset is included in this repository, and access is subject to the requirements of their respective providers.

SDGFN integrates low-order individualized radiomics-based structural similarity networks (iRSSN) with their high-order counterparts (Ho-iRSSN). The low-order branch combines GATv2-style local attention with GraphGPS-style global modeling. The high-order branch uses signed edge-aware attention. Both branches are trained jointly through adaptive gated fusion, auxiliary classification losses, and an orthogonality constraint.

The low-order iRSSN represents pairwise similarity between the radiomic profiles of gray-matter regions, capturing regional structural correspondence in sMRI rather than direct anatomical connectivity. Ho-iRSSN compares the whole-brain connectivity profiles represented in iRSSN, characterizing how similarly regions are embedded in the broader structural network. It may reflect coordinated organization across distributed brain regions, but should not be interpreted as direct functional coupling or causal interaction. Together, the two representations provide complementary views of local structural similarity and whole-brain topological organization that may help characterize distributed structural patterns associated with MDD.

## Files

- `sdgfn\\\_low\\\_order.py`: Low-order network encoder.
- `sdgfn\\\_high\\\_order.py`: High-order network encoder.
- `train\\\_sdgfn.py`: Data loading, matrix preprocessing, fusion model, training, validation, and checkpoint saving.
- `requirements.txt`: Package dependencies.

This release starts from precomputed brain-network matrices. MRI preprocessing, radiomics extraction, network construction, and additional experimental scripts are not included.

## Installation

Install PyTorch, NumPy, pandas, scikit-learn, and openpyxl:

```bash

pip install -r requirements.txt

```

The supplied training implementation uses Windows keyboard handling through `msvcrt` and CUDA automatic mixed precision. It is intended for a Windows environment with a CUDA-capable GPU.

## Data preparation

Organize the input files as follows:

```text

data/

├── metadata.xlsx

├── irssn/

│   └── iRSSN\\\_S1-001.txt

└── high\\\_order/

\&#x20;   └── iRSSN\\\_S1-001.txt

```

The metadata spreadsheet must contain:

- `ID`: Subject identifier matching the network filenames.
- `group`: Diagnostic label, with `1` for MDD and `0` for HC.

Center identifiers are extracted from the part of `ID` before the first hyphen. For example, `S1-001` belongs to center `S1`.

Each network file must contain a 246 × 246 matrix of finite, whitespace-separated values. Region ordering must be consistent across subjects and network types. Both directories use the filename pattern `iRSSN\\\_<ID>.txt`.

Subjects missing either network file are skipped. Check the reported valid sample counts before interpreting the results. Missing or unexpected diagnostic labels must be resolved before training.

Participant-level data are not included. Dataset access is subject to the permissions and terms of the original providers.

## Preprocessing and configuration

Both network matrices are symmetrized, and diagonal entries are set to 1. Negative iRSSN weights are converted to half their absolute magnitude, whereas Ho-iRSSN retains edge signs. Each processed matrix is used as both node features and adjacency input.

The supplied configuration uses:

- Hidden dimension: 128.
- Low-order/high-order blocks: 2/3.
- Batch size: 8.
- Training epochs: 120 per split.
- Low-order/high-order dropout: 0.35/0.30.
- iRSSN/Ho-iRSSN thresholds: 0.04/0.03.
- AdamW learning rates: 1.1e-4, 1.3e-4, and 1.1e-4 for the low-order branch, high-order branch, and fusion components.
- Weight decay: 2e-5.
- Class weights: 1.0 for HC and 1.45 for MDD.
- Label smoothing: 0.01.

During the first five training epochs, the low-order fusion coefficient is fixed at 0.45. Subsequent training uses adaptive gated fusion with gate values bounded between 0.25 and 0.75. Validation always uses the adaptive gate.

## Training and validation

Update these paths in `train\\\_sdgfn.py`:

```text

EXCEL\\\_PATH

DATA\\\_DIR\\\_IRSSN

DATA\\\_DIR\\\_CO

SAVE\\\_DIR

```

Run:

```bash

python train\\\_sdgfn.py

```

Five predefined center-grouped splits are provided for the Meta-MDD data, whose centers differ in sample size and class composition. Within each split, validation centers are excluded from training. Validation-center membership may overlap across splits.

The checkpoint with the highest validation accuracy is saved for each split as `best\\\_fold\\\_<number>.pth`. Reported metrics correspond to that checkpoint and represent validation performance.

Pressing ESC interrupts the current split. Complete all splits without manual interruption when producing comparison results.

## Metrics and reproducibility

The original training script reports ACC, AUC, SEN, SPEC, and PRE as the mean and sample standard deviation across included splits.

F1 can be calculated for each selected checkpoint as:

```text

F1 = 2 × TP / (2 × TP + FP + FN)

```

Calculate F1 separately for each split before averaging; do not derive mean F1 from averaged SEN and PRE.

No random seed is fixed in the supplied implementation. Results may vary with initialization, dropout, batch ordering, and the computing environment.

Two original implementation details should be considered when reproducing results:

- Gradient clipping occurs before mixed-precision gradients are unscaled.
- AUC exceptions are assigned a fallback value of 0.5. This fallback must not be reported as a measured AUC when AUC is undefined, such as for single-class data.

## Citation

Please cite the associated SDGFN manuscript when using this implementation. Bibliographic details will be added when available.
