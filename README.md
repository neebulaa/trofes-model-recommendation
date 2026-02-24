<p align="center">
  <img src="./logo-transparent.png" alt="Trofes logo" width="160" />
</p>

<h1 align="center">🥗 Trofes: Machine Learning Recommendation API</h1>

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/) [![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688)](https://fastapi.tiangolo.com/) [![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3-orange)](https://scikit-learn.org/) [![Pandas](https://img.shields.io/badge/Pandas-Data_Processing-150458)](https://pandas.pydata.org/) [![Hugging Face](https://img.shields.io/badge/Hugging_Face-Deployment-FFD21E)](https://huggingface.co/)

<p align='center'>Trofes intelligent backend ecosystem serving Machine Learning models for recipe recommendations and precise nutritional calculations. This project implements two distinct clustering approaches to handle user preferences and absolute macronutrient targets.</p>

---

## 🚀 Machine Learning Architecture

This repository contains two core models, each architected with specific algorithmic trade-offs to handle different business logic:

### 1. Model A: Preference-Based Recommendation
* **Algorithm:** K-Means Clustering + Cosine Similarity
* **Location:** `notebooks/FoodRecomendation/final_kmeans.ipynb`
* **Use Case:** Recommending recipes based on user "likes", ingredients, and historical preferences.
* **Why this approach?** K-Means provides highly efficient data partitioning. Combined with Cosine Similarity, it perfectly captures the "pattern" or "direction" of user preferences regardless of the absolute magnitude. Highly optimized for real-time inference (Latency: ~30ms).

### 2. Model B: Nutrient Calculator-Based Recommendation
* **Algorithm:** Agglomerative Hierarchical Clustering (Ward Linkage) + Euclidean Distance
* **Location:** `notebooks/NutrientsCalculator/final_agglomerative...`
* **Use Case:** High-precision recipe matching based on strict macronutrient calculations (Calories, Protein, Fat, Carbs).
* **Why this approach?** Unlike Model A, nutritional values require absolute precision. Agglomerative Clustering works **bottom-up**, and combined with **Ward Linkage**, it minimizes the variance within clusters. This creates a high-precision, highly homogenous "neighborhood" filter before applying Euclidean Distance to find the exact nutrient match.

---

## 📂 Project Structure

The repository is organized to separate data processing, model training, and API deployment clearly:
```text
📦 TROFES-MODEL-RECOMMENDATION
 ┣ 📂 .github/workflows      # CI/CD pipelines (e.g., automated deployment to Hugging Face)
 ┣ 📂 data
 ┃ ┣ 📂 output               # Relational database-ready CSVs (recipe.csv, ingredient.csv, etc.)
 ┃ ┣ 📂 processed            # Intermediate cleaned datasets
 ┃ ┗ 📂 raw                  # Original raw datasets (epi_r.csv, etc.)
 ┣ 📂 models                 # Exported model artifacts (e.g., .pkl, .joblib)
 ┣ 📂 notebooks              # Jupyter notebooks for EDA and Model Training
 ┃ ┣ 📂 FoodRecomendation    # K-Means clustering development
 ┃ ┣ 📂 NutrientsCalculator  # Agglomerative clustering development
 ┃ ┗ 📜 data_preparation.ipynb # Master data cleaning and preprocessing pipeline
 ┣ 📜 .gitignore
 ┣ 📜 README.md              # Project documentation
 ┣ 📜 main.py                # FastAPI application entry point
 ┗ 📜 requirements.txt       # Python dependencies
```

---

## ⚙️ Installation & Setup

Follow these steps to get the API running on your local machine.

### Prerequisites

Make sure you have the following installed:
- **Python 3.8+** — [Download here](https://www.python.org/downloads/)
- **Git** — [Download here](https://git-scm.com/)

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/TROFES-MODEL-RECOMMENDATION.git
cd TROFES-MODEL-RECOMMENDATION
```

### 2. Create a Virtual Environment

It is strongly recommended to use a virtual environment to avoid dependency conflicts.
```bash
# Create virtual environment
python -m venv venv

# Activate — macOS/Linux
source venv/bin/activate

# Activate — Windows
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Model Artifacts

Before running the API, make sure the trained model files (`.pkl` / `.joblib`) are present inside the `models/` directory. If they are missing, re-run the notebooks in order:
```
1. notebooks/data_preparation.ipynb
2. notebooks/FoodRecomendation/final_kmeans.ipynb
3. notebooks/NutrientsCalculator/final_agglomerative.ipynb
```

### 5. Run the API
```bash
uvicorn main:app --reload
```

The API will be live at **`http://127.0.0.1:8000`**

### 6. Explore the Interactive Docs

FastAPI provides automatic interactive documentation out of the box:

| Interface | URL |
|-----------|-----|
| Swagger UI | `http://127.0.0.1:8000/docs` |
| ReDoc | `http://127.0.0.1:8000/redoc` |

---

## 🌐 Deployment

This project is deployed on **Hugging Face Spaces** and uses a GitHub Actions CI/CD pipeline (`.github/workflows/`) for automated deployment on every push to `main`.

> 🔗 **Live API:** `https://huggingface.co/spaces/ArNight/trofes-api`

---

## 📡 API Endpoints

| Method | Endpoint | Model Used | Description |
|--------|----------|------------|-------------|
| `POST` | `/recommend` | Model A (K-Means) | Get recipes based on user preferences & ingredients |
| `POST` | `/recommendCalculator` | Model B (Agglomerative) | Get recipes based on target macronutrients |

> Full request/response schema is available on the Swagger UI at `/docs`.

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---
