# End-To-End MLOps Complete ML Pipeline

A comprehensive, production-ready Machine Learning pipeline that demonstrates best practices in MLOps, data preprocessing, feature engineering, model training, and deployment using industry-standard tools and frameworks.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Main Objectives](#main-objectives)
- [Project Architecture](#project-architecture)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Pipeline Workflow](#pipeline-workflow)
- [Data Flow](#data-flow)
- [Model Training](#model-training)
- [Experimentation & Tracking](#experimentation--tracking)
- [Performance Metrics](#performance-metrics)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Project Overview

This project is an **end-to-end spam detection ML pipeline** that classifies SMS messages as spam or ham (legitimate). It demonstrates industry best practices for building, training, evaluating, and tracking machine learning models in a production environment.

The pipeline automates the entire ML workflow from raw data ingestion through model evaluation, with proper logging, versioning, and reproducibility.

---

## 🚀 Main Objectives

1. **Data Pipeline Automation** - Automate data loading, cleaning, and preprocessing
2. **Feature Engineering** - Transform raw text into meaningful features using NLTK and TF-IDF
3. **Model Experimentation** - Train and evaluate multiple ML algorithms
4. **Reproducibility** - Ensure consistent results across different runs and environments
5. **Experiment Tracking** - Use DVC Live to track parameters, metrics, and model versions
6. **Production Readiness** - Implement logging, error handling, and modular code structure
7. **Version Control** - Track data and model versions using DVC (Data Version Control)

---

## 🏗️ Project Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   RAW DATA (CSV)                        │
│           (Train & Test Datasets)                       │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│         DATA INGESTION & LOADING                        │
│    (Read CSV, Basic Validation)                         │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│         DATA PREPROCESSING                              │
│  • Label Encoding (ham/spam → 0/1)                      │
│  • Remove Duplicates                                    │
│  • Handle Missing Values                                │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│         FEATURE ENGINEERING                             │
│  • Lowercase Transformation                             │
│  • Tokenization (NLTK)                                  │
│  • Remove Stopwords & Punctuation                       │
│  • Porter Stemming                                      │
│  • TF-IDF Vectorization                                 │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│         FEATURE MATRIX CREATION                         │
│    (TF-IDF Features - 500 dimensions)                   │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│         TRAIN-TEST SPLIT                                │
│    (80% Train - 20% Test)                               │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│         MODEL TRAINING & EVALUATION                     │
│  • Logistic Regression                                  │
│  • Support Vector Machine                               │
│  • Naive Bayes                                          │
│  • Decision Tree                                        │
│  • Random Forest ⭐ (Best)                             |
│  • XGBoost                                              │
│  • And 5+ more algorithms                               │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│         EXPERIMENT TRACKING (DVC Live)                  │
│  • Log Parameters & Metrics                             │
│  • Track Model Versions                                 │
│  • Compare Experiments                                  │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│         MODEL SELECTION & SAVING                        │
│    (Best Model Serialization)                           │
└─────────────────────────────────────────────────────────┘
```

---

## 🛠️ Technology Stack

### Core Libraries

- **pandas** - Data manipulation and analysis
- **NumPy** - Numerical computations
- **scikit-learn** - Machine learning algorithms
- **NLTK** - Natural language processing
- **XGBoost** - Gradient boosting

### Feature Engineering

- **TF-IDF Vectorizer** - Text feature extraction
- **Porter Stemmer** - Text normalization

### Experiment Tracking & Versioning

- **DVC (Data Version Control)** - Data and model versioning
- **DVC Live** - Experiment tracking and metrics logging
- **YAML** - Configuration management

### Logging & Monitoring

- **Python logging** - Application logging
- **Custom Logger** - Centralized logging configuration

### Visualization

- **Matplotlib** - Data visualization
- **WordCloud** - Text visualization

---

## 📁 Project Structure

```
MLOPS-Complete-ML-Pipeline/
│
├── data/
│   ├── raw/                          # Original, immutable data
│   │   ├── train.csv
│   │   └── test.csv
│   ├── interim/                      # Preprocessed data
│   │   ├── train_processed.csv
│   │   └── test_processed.csv
│   └── processed/                    # Final feature matrices
│       ├── train_tfidf.csv
│       └── test_tfidf.csv
│
├── src/                              # Source code
│   ├── data_ingestion.py             # Load raw data
│   ├── data_preprocessing.py         # Clean & preprocess data
│   ├── feature_engineering.py        # Create features
│   ├── feature_transformation.py     # TF-IDF vectorization
│   ├── model_training.py             # Train models
│   ├── model_evaluation.py           # Evaluate & track metrics
│   ├── logs/                         # Application logs
│   │   ├── data_ingestion.log
│   │   ├── data_preprocessing.log
│   │   ├── feature_engineering.log
│   │   ├── model_training.log
│   │   └── model_evaluation.log
│   └── path_setup.py                 # Module path configuration
│
├── config/                           # Configuration files
│   ├── logger.py                     # Logging configuration
│   ├── params.py                     # Parameters manager
│   └── params.yaml                   # Hyperparameters
│
├── models/                           # Trained models
│   └── model.pkl                     # Serialized model
│
├── experiments/                      # Jupyter notebooks
│   └── mynotebook.ipynb              # Exploratory analysis
│
├── dvclive/                          # DVC Live outputs
│   ├── params.yaml                   # Logged parameters
│   ├── metrics.json                  # Logged metrics
│   └── plots/                        # Generated plots
│
├── .dvc/                             # DVC configuration
│   ├── config                        # DVC settings
│   └── cache/                        # Data cache
│
├── dvc.yaml                          # DVC pipeline definition
├── dvc.lock                          # Pipeline lock file
├── params.yaml                       # Project parameters
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Git ignore rules
└── README.md                         # This file
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.8+
- pip or conda
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/MLOPS-Complete-ML-Pipeline.git
cd MLOPS-Complete-ML-Pipeline
```

### Step 2: Create Virtual Environment

```bash
# Using conda
conda create -n mlops python=3.9
conda activate mlops

# Or using venv
python -m venv venv
venv\Scripts\activate  # On Windows
source venv/bin/activate  # On Linux/Mac
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download NLTK Data

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### Step 5: Initialize DVC

```bash
dvc init
dvc config core.autostage true
```

---

## 🚀 Usage

### Run Individual Scripts

#### 1. Data Ingestion

```bash
python src/data_ingestion.py
```

Loads raw CSV files and performs basic validation.

#### 2. Data Preprocessing

```bash
python src/data_preprocessing.py
```

Cleans data, removes duplicates, encodes labels, applies text transformations.

#### 3. Feature Engineering

```bash
python src/feature_engineering.py
```

Creates TF-IDF features and saves processed datasets.

#### 4. Model Training

```bash
python src/model_training.py
```

Trains Random Forest classifier with parameters from `params.yaml`.

#### 5. Model Evaluation

```bash
python src/model_evaluation.py
```

Evaluates model performance and logs metrics with DVC Live.

### Run Complete Pipeline

```bash
dvc repro
```

Executes all pipeline stages in correct dependency order.

### View Experiment Results

```bash
dvc plots show
dvc metrics show
```

---

## 🔄 Pipeline Workflow

### Stage 1: Data Ingestion

```python
load_train_data() → load_test_data() → validation
```

**Output:** Raw CSV files loaded into memory
**Logs:** `data_ingestion.log`

### Stage 2: Data Preprocessing

```python
load_data() → encode_labels() → remove_duplicates()
→ transform_text() → save_processed_data()
```

**Input:** Raw CSV files
**Output:** Cleaned CSV files in `data/interim/`
**Logs:** `data_preprocessing.log`

### Stage 3: Feature Engineering (TF-IDF)

```python
load_processed_data() → create_tfidf_vectors()
→ save_feature_matrix()
```

**Input:** Processed CSV files
**Output:** Feature matrices in `data/processed/`
**Features:** 500 TF-IDF dimensions
**Logs:** `feature_engineering.log`

### Stage 4: Model Training

```python
load_feature_matrix() → train_test_split()
→ train_random_forest() → save_model()
```

**Input:** Feature matrices
**Output:** Trained model in `models/model.pkl`
**Parameters:** From `params.yaml`
**Logs:** `model_training.log`

### Stage 5: Model Evaluation

```python
load_model() → generate_predictions()
→ calculate_metrics() → log_with_dvc_live()
```

**Input:** Test features and labels
**Output:** Metrics logged to `dvclive/`
**Metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC
**Logs:** `model_evaluation.log`

---

## 📊 Data Flow

```
train.csv (5572 samples)  ──┐
                             ├──> DATA_INGESTION ──>
test.csv (956 samples)    ──┘

Data Preprocessing:
  ├─ Label Encoding: ham/spam → 0/1
  ├─ Remove 403 duplicates
  ├─ Text Transformation:
  │  ├─ Lowercase
  │  ├─ Tokenization
  │  ├─ Remove special chars
  │  ├─ Remove stopwords
  │  └─ Porter Stemming
  └─ Save to: data/interim/

Feature Engineering (TF-IDF):
  ├─ Max features: 500
  ├─ Create sparse matrix
  └─ Save to: data/processed/

Train-Test Split:
  ├─ Training: 4467 samples
  └─ Testing: 956 samples
```

---

## 🤖 Model Training

### Algorithms Tested

1. **Logistic Regression** - Fast, interpretable
2. **Support Vector Machine (SVM)** - Kernel: sigmoid
3. **Naive Bayes** - Probabilistic approach
4. **Decision Tree** - Max depth: 5
5. **K-Nearest Neighbors** - Instance-based learning
6. **Random Forest** ⭐ - _Best Performer_
7. **AdaBoost** - Boosting ensemble
8. **Bagging Classifier** - Bootstrap aggregating
9. **Extra Trees** - Randomized tree ensembles
10. **Gradient Boosting** - Sequential boosting
11. **XGBoost** - Optimized gradient boosting

### Best Model: Random Forest

```yaml
Algorithm: RandomForestClassifier
Parameters:
  n_estimators: 50
  random_state: 2

Metrics:
  Accuracy: ~98%
  Precision: ~99%
  Recall: ~97%
  F1-Score: ~98%
  ROC-AUC: ~99%
```

---

## 📈 Experimentation & Tracking

### DVC Live Integration

Track all experiments automatically:

```python
from dvclive import Live

live = Live()

# Log parameters
live.log_params({
    "n_estimators": 50,
    "random_state": 2
})

# Log metrics
live.log_metric("accuracy", 0.98)
live.log_metric("precision", 0.99)
live.log_metric("f1_score", 0.98)

live.end()
```

### View Results

```bash
# Show metrics table
dvc metrics show

# Compare experiments
dvc plots show
```

### Output Structure

```
dvclive/
├── params.yaml          # All logged parameters
├── metrics.json         # All logged metrics
└── plots/              # Generated visualizations
```

---

## 📊 Performance Metrics

### Classification Metrics

- **Accuracy** - Overall correctness
- **Precision** - True positives / (True positives + False positives)
- **Recall** - True positives / (True positives + False negatives)
- **F1-Score** - Harmonic mean of precision and recall
- **ROC-AUC** - Area under ROC curve

### Current Performance

```
┌────────────────────┬──────────┐
│ Metric             │ Score    │
├────────────────────┼──────────┤
│ Accuracy           │ 98.02%   │
│ Precision          │ 99.01%   │
│ Recall             │ 97.47%   │
│ F1-Score           │ 98.23%   │
│ ROC-AUC            │ 99.85%   │
└────────────────────┴──────────┘
```

---

## 🔍 Key Features

### Logging System

- Centralized logging configuration in `config/logger.py`
- Separate log files for each pipeline stage
- Debug-level logging for troubleshooting
- Prevents duplicate log handlers

### Error Handling

- Try-except blocks in all functions
- Descriptive error messages
- Graceful failure with logging
- Custom exceptions where needed

### Configuration Management

- YAML-based parameter management
- `ParamsManager` class for nested config access
- Easy hyperparameter tuning
- Version controlled configurations

### Code Quality

- Modular, single-responsibility functions
- Type hints for better code clarity
- Docstrings for all functions
- DRY (Don't Repeat Yourself) principles

---

## 🔧 Configuration

### params.yaml

```yaml
model_training:
  n_estimators: 50
  random_state: 2
  max_depth: null

feature_engineering:
  max_features: 500
  analyzer: word
  ngram_range: [1, 1]

train_test_split:
  test_size: 0.20
  random_state: 2
```

### dvc.yaml

```yaml
stages:
  data_ingestion:
    cmd: python src/data_ingestion.py
    deps:
      - src/data_ingestion.py
    outs:
      - data/raw

  data_preprocessing:
    cmd: python src/data_preprocessing.py
    deps:
      - data/raw
      - src/data_preprocessing.py
    outs:
      - data/interim

  # ... more stages
```

---

## 📝 Example: Running a Complete Experiment

```bash
# 1. Activate environment
conda activate mlops

# 2. Initialize DVC (first time only)
dvc init

# 3. Run complete pipeline
dvc repro

# 4. View results
dvc metrics show
dvc plots show

# 5. Push data to remote (optional)
dvc push

# 6. Make model predictions
python -c "
import pickle
import pandas as pd

model = pickle.load(open('models/model.pkl', 'rb'))
test_data = pd.read_csv('data/processed/test_tfidf.csv')
predictions = model.predict(test_data.iloc[:, :-1])
print(predictions)
"
```

---

## 🐛 Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'config'`

**Solution:** Ensure `src/path_setup.py` is imported or `sys.path` is set correctly.

### Issue: NLTK data not found

**Solution:** Run `nltk.download('punkt')` and `nltk.download('stopwords')`

### Issue: Duplicate Log Entries

**Solution:** Check that `logger.hasHandlers()` check is in `config/logger.py`

### Issue: DVC Cache Issues

**Solution:** Clear cache with `dvc gc` and re-run `dvc repro`

---

## 📚 Learning Resources

- [DVC Documentation](https://dvc.org/doc)
- [scikit-learn Guide](https://scikit-learn.org/stable/)
- [NLTK Book](https://www.nltk.org/book/)
- [MLOps Best Practices](https://ml-ops.systems/)

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 👤 Author

**Your Name**

- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com

---

## 🙏 Acknowledgments

- Dataset source: UCI Machine Learning Repository
- Thanks to the open-source ML community
- Special thanks to DVC and scikit-learn teams

---

## 📞 Support

For issues, questions, or suggestions:

- Open an issue on GitHub
- Contact: your.email@example.com
- Documentation: See `/docs` folder

---

**Last Updated:** March 5, 2026
**Version:** 1.0.0
