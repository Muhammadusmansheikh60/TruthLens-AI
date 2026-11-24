
# TruthLens AI

### AI-Based Claim Credibility Classification

TruthLens AI is an experimental machine-learning project
that classifies short political statements into six credibility
categories using the LIAR dataset.

## Project Overview

| Component | Technology |
|---|---|
| Programming Language | Python |
| Dataset | LIAR |
| Machine Learning | Logistic Regression |
| Text Representation | TF-IDF |
| Explainable AI | LIME |
| Web Framework | Streamlit |

## Classification Categories

- Pants-fire
- False
- Barely-true
- Half-true
- Mostly-true
- True

## Application Features

- User-entered statement analysis
- Six-category probability distribution
- LIME-based local explanations
- Downloadable probability results
- Interactive web interface

## Model Evaluation

The final model achieved:

- Test Accuracy: 25.41%
- Macro F1-score: 25.26%
- Weighted F1-score: 25.39%

These results indicate modest classification performance.
The application is intended for academic experimentation,
not reliable real-world fact verification.

## Dataset

The project uses the LIAR dataset introduced by
William Yang Wang (2017).

Dataset and research paper:

https://aclanthology.org/P17-2067/

The dataset contains short political statements with
six fact-checking labels.

## Installation

Clone the repository:

    git clone YOUR_GITHUB_REPOSITORY_URL

Move into the project directory:

    cd TruthLens_AI

Install dependencies:

    pip install -r requirements.txt

Run the application:

    streamlit run app.py

## Project Structure

    TruthLens_AI/
    |
    |-- app.py
    |-- requirements.txt
    |-- README.md
    |-- .gitignore
    |
    |-- models/
        |-- final_model.pkl
        |-- tfidf_vectorizer.pkl
        |-- label_encoder.pkl

## Research Limitations

- The model learns patterns from the LIAR dataset.
- The dataset consists of political claims, not complete news articles.
- The model does not independently verify factual accuracy.
- Predictions can be incorrect.
- LIME provides local explanations of model behaviour,
  not proof of a statement's credibility.

## Disclaimer

TruthLens AI is an experimental academic research project.
Its outputs should not be treated as authoritative fact checks.
