# E-Commerce Recommendation Engine: Mitigating Choice Overload
**An End-to-End Machine Learning Capstone Project**

## Project Overview
This repository contains a Collaborative Filtering recommendation system built to address "choice overload" in high-volume online retail environments. Utilizing the Olist Brazilian E-Commerce dataset, the model maps historical user-item interactions to generate personalized product recommendations, optimizing for Average Order Value (AOV) uplift and engagement.

## Dataset
https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

## Technical Architecture
*   **Algorithm:** Singular Value Decomposition (SVD) Matrix Factorization
*   **Hyperparameter Tuning:** `GridSearchCV` (192 combinations)
*   **Explainability & Fairness:** SHAP (Surrogate Model), PCA Dimensionality Reduction, and Equalized Odds bias auditing across engineered socioeconomic proxies.
*   **Deployment:** RESTful API built with FastAPI.
*   **Hardware Profile:** Optimized for local GPU execution (e.g., NVIDIA RTX 3060) during hyperparameter grid search.

## Repository Structure
*   `data/`: Directory for raw interaction datasets (CSVs ignored in version control).
*   `notebooks/`: Jupyter notebooks detailing EDA, preprocessing, model training, and bias audits.
*   `models/`: Directory for serialized `.pkl` files generated during model training (ignored in version control due to file size).
*   `src/`: Python scripts for deployment (`app.py`).
*   `powerpoint/`: Presentation assets and final reports.

## Local Installation & Execution
**1. Clone the repository and install dependencies:**
```bash
git clone <your-github-repo-url>
pip install -r requirements.txt
