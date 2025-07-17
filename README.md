<h1 align="center">🏮 LANTERN: A Machine Learning Framework for Lipid Nanoparticle Transfection Efficiency Prediction</h1>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License"/></a>
  <a href="https://github.com/AsalMehradfar/LANTERN/stargazers"><img src="https://img.shields.io/github/stars/AsalMehradfar/LANTERN?style=social" alt="GitHub Stars"/></a>
</p>

<p align="justify">
<strong>LANTERN</strong> (Lipid nANoparticle Transfection Efficiency pRedictioN) is a machine learning framework for predicting the transfection efficiency of ionizable lipids used in lipid nanoparticle (LNP)-mediated RNA delivery. LANTERN addresses key limitations in existing approaches by combining high-quality curated data, chemically meaningful molecular representations, and rigorous model evaluation.
</p>

<p align="justify">
The framework enables robust benchmarking of diverse ML models—including traditional regressors and neural architectures—based on interpretable fingerprint-based and learned molecular features. LANTERN models significantly outperform AGILE, a prior state-of-the-art model, achieving an R² of 0.8161 and Pearson r of 0.9053 using a multi-layer perceptron trained on combined Morgan fingerprints and expert descriptors.
</p>

LANTERN is designed as a modular and extensible platform for:
- Predicting transfection efficiency from SMILES-based lipid structures  
- Benchmarking ML models on general molecular regression tasks  
- Accelerating LNP design in RNA-based therapeutics

## 📖 Table of Contents

  * [Environment Setup](#%EF%B8%8F-environment-setup)
  * [Usage](#-usage)
  * [Citation](#-citation)
  * [Where to Ask for Help](#-where-to-ask-for-help)
    
## ⚙️ Environment Setup

We recommend using [Conda](https://docs.conda.io/en/latest/) to manage dependencies for LANTERN.

#### 📦 Install via `lantern.yml`

Clone the repository and create the environment:

```bash
# Clone the repository
git clone https://github.com/AsalMehradfar/LANTERN.git
cd LANTERN

# Create the environment from the YAML file
conda env create -f lantern.yml

# Activate the environment
conda activate lantern
```

#### 🔄 Optional: Update Environment

If you make changes to the YAML or add packages later:

```bash
conda env update -f lantern.yml --prune
```

## 🚀 Usage

This repository provides a streamlined pipeline for making predictions using pretrained models on molecular data.

### 1. Prepare Your Data

Place your dataset in the following format: `data/YOUR_DATASET.csv`. It must include at least these two columns:

- `SMILES`: the molecular structure in SMILES format  
- `Target`: *(optional)* ground truth value for evaluation

---

### 2. Extract Fingerprints

Run the following script **once for each fingerprint type**:

```bash
python scripts/extract_fingerprint.py --mode circular --data_name YOUR_DATASET --save_path data/fingerprints/YOUR_DATASET
python scripts/extract_fingerprint.py --mode expert --data_name YOUR_DATASET --save_path data/fingerprints/YOUR_DATASET
```

This will generate required fingerprint files in: `data/fingerprints/YOUR_DATASET/`

### 3. Run Inference

Once fingerprints are extracted, you can run inference using a pretrained model:

```bash
python scripts/run_inference.py --csv_path data/YOUR_DATASET.csv
```
**Notes:**
- `YOUR_DATASET.csv` should contain the same molecule entries used during fingerprint extraction
- No need to retrain or modify the model values
- If your dataset contains a `Target` column, evaluation metrics (e.g., RMSE, R²) will be computed and saved alongside the predictions
  
## 🎯 Citation 

If you use LANTERN in a research paper, please cite our [paper](https://arxiv.org/abs/2507.03209):

```bibtex
@article{Mehradfar2025LANTERN,
      title={LANTERN: A Machine Learning Framework for Lipid Nanoparticle Transfection Efficiency Prediction}, 
      author={Asal Mehradfar and Mohammad Shahab Sepehri and Jose Miguel Hernandez-Lobato and Glen S. Kwon and Mahdi Soltanolkotabi and Salman Avestimehr and Morteza Rasoulianboroujeni},
      year={2025},
      url={https://arxiv.org/abs/2507.03209}
}
```

## ❓ Where to Ask for Help

<p align="justify" > 
If you have any questions, feel free to open a <a href="https://github.com/AsalMehradfar/LANTERN/discussions">Discussion</a> and ask your question. You can also email <a href="mailto:mehradfa@usc.edu">mehradfa@usc.edu</a> (Asal Mehradfar) or <a href="mailto:sepehri@usc.edu">sepehri@usc.edu</a> (Mohammad Shahab Sepehri).
</p>
