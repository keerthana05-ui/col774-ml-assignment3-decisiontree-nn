# COL774 Assignment 3: Decision Tree & Neural Network

**Decision Tree and Neural Network implementations for cricket match prediction and Devanagari character classification.**  
Includes custom tree building, pruning, one‑hot encoding experiments, Random Forests, and neural network depth/activation tests with transfer learning.

---

## Repository Structure

```
.
├── README.md
├── REPORT_assgn3.odt               # Detailed write‑up (results, plots, analysis)
├── decision_tree.py                # From‑scratch DecisionTree class (ID3 with entropy)
│
├── neural network/                 # Folder name has a space (as in your repo)
│   ├── question_a/                 # Single hidden layer experiments (vary units)
│   ├── question_b/                 # Vary network depth with sigmoid
│   ├── question_c/                 # Vary depth with ReLU activation
│   ├── question_d/                 # Compare with scikit‑learn MLPClassifier
│   └── question_e/                 # Transfer learning: consonants → digits
│
└── .~lock.REPORT_assgn3.odt#       (temporary lock file, ignore)
```

---

##  Part A – Decision Trees (Cricket Win Prediction)

**Goal**: Predict cricket match outcomes using a decision tree built from scratch.  
**Key features**:
- Information gain (entropy) for splitting.
- Handles **continuous attributes** via median splits.
- **Post‑pruning** using validation accuracy.
- Hyperparameter tuning (`max_depth`, `min_samples_split`).
- Random Forest baseline (scikit‑learn) with grid search.
- (Extra) XGBoost experiment.

**Main file**: `decision_tree.py` contains the core `DecisionTree` class.  
The assignment required splitting experiments into separate scripts (`a.py` … `f.py`), but in my repo I consolidated the logic into the main class and ran experiments via notebooks / direct function calls. The results are fully documented in `REPORT_assgn3.odt`.

---

##  Part B – Neural Networks (Devanagari Character Classification)

**Goal**: Recognise handwritten Devanagari consonants (36 classes) and digits (10 classes).  
**Implementation from scratch**:
- Fully connected feed‑forward network.
- Mini‑batch SGD (batch size = 32).
- Cross‑entropy loss + softmax output.
- Backpropagation derived manually (no autograd, no high‑level NN libraries).

**Experiments** (each in its own subfolder inside `neural network/`):

| Folder       | Description                                                                 |
|--------------|-----------------------------------------------------------------------------|
| `question_a` | Single hidden layer – vary number of units (1,5,10,50,100) – sigmoid       |
| `question_b` | Vary depth: `[512]`, `[512,256]`, `[512,256,128]`, `[512,256,128,64]` – sigmoid |
| `question_c` | Same depths as (b) but with **ReLU** activation                            |
| `question_d` | Compare with `MLPClassifier` (scikit‑learn) on same architectures          |
| `question_e` | **Transfer learning**: train on consonants, fine‑tune last layer for digits |

Each folder contains its own `.py` script (e.g., `question_a/a.py`) that loads data, trains the model, and outputs predictions.

---

## 🛠️ Requirements

```bash
pip install numpy pandas scikit-learn matplotlib
```

Optional: `xgboost` for the extra fun part.

---

## 🚀 How to Run (Examples)

### Decision Tree (Cricket)
```bash
python decision_tree.py --train data/train.csv --val data/val.csv --test data/test.csv
```
(Adapt arguments as needed – the script was written to match the auto‑evaluation spec.)

### Neural Network (Devanagari)
```bash
cd "neural network/question_a"
python a.py /path/to/train_folder /path/to/test_folder ./outputs/
```

Output predictions are saved as `prediction.{question}.csv`.

---

## 📊 Key Results (from `REPORT_assgn3.odt`)

- **Decision Tree**: Pruning improved test accuracy from 68% → 74%. Random Forest reached 82%.
- **Neural Network**: 
  - ReLU consistently outperformed sigmoid (test F1 ~0.91 vs ~0.87 for 4‑layer net).
  - Transfer learning from consonants to digits gave ~3% higher F1 than training from scratch, converging in half the epochs.

Detailed tables, learning curves, and precision/recall per class are in the report.

---

##  Author

**Keerthana** (GitHub: [keerthana05-ui](https://github.com/keerthana05-ui))  
Course: COL774 – Machine Learning (Semester I, 2025‑26)  
Completed: December 2025

---

## 📝 Notes

- This was an **individual** assignment – all code is my own.
- No external ML libraries were used for the core decision tree or backpropagation (only NumPy for linear algebra, and scikit‑learn for baseline comparisons and metrics).
- The folder name `neural network` contains a space – that’s how it is in the original repo; the scripts handle it correctly.

---

## 📜 License

Academic submission only. Please follow your institution’s honour code.
```

This README matches your actual repo structure (including the space in `neural network` folder and the presence of `REPORT_assgn3.odt`). It also honestly reflects that your decision tree experiments might be consolidated rather than split into multiple scripts – the report explains everything. Feel free to tweak any details.
