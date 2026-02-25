# 🥗 TROFES: Machine Learning Recommendation API

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3-orange)
![Pandas](https://img.shields.io/badge/Pandas-Data_Processing-150458)
![Hugging Face](https://img.shields.io/badge/Hugging_Face-Deployment-FFD21E)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-2088FF)

An intelligent backend ecosystem serving Machine Learning models for recipe recommendations and precise nutritional calculations. This project implements two distinct clustering approaches to handle user preferences and absolute macronutrient targets.
<p align="center">
  <img src="./logo-transparent.png" alt="Trofes logo" width="160" />
</p>

<h1 align="center">Trofes: Machine Learning Recommendation API</h1>

<p align="center">
  <a href="https://www.python.org/" target="_blank" rel="noopener noreferrer">
    <img alt="Python" src="https://img.shields.io/badge/Python-3.8%2B-blue" />
  </a>
  <a href="https://fastapi.tiangolo.com/" target="_blank" rel="noopener noreferrer">
    <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-0.100%2B-009688" />
  </a>
  <a href="https://scikit-learn.org/" target="_blank" rel="noopener noreferrer">
    <img alt="Scikit-Learn" src="https://img.shields.io/badge/scikit--learn-1.3-orange" />
  </a>
  <a href="https://pandas.pydata.org/" target="_blank" rel="noopener noreferrer">
    <img alt="Pandas" src="https://img.shields.io/badge/Pandas-Data_Processing-150458" />
  </a>
  <a href="https://huggingface.co/" target="_blank" rel="noopener noreferrer">
    <img alt="Hugging Face" src="https://img.shields.io/badge/Hugging_Face-Deployment-FFD21E" />
  </a>
</p>

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
* **Location:** `notebooks/NutrientsCalculator/final_agglomerative_focusNutrient.ipynb`
* **Use Case:** High-precision recipe matching based on strict macronutrient calculations (Calories, Protein, Fat, Carbs).
* **Why this approach?** Unlike Model A, nutritional values require absolute precision. Agglomerative Clustering works **bottom-up**, and combined with **Ward Linkage**, it minimizes the variance within clusters. This creates a high-precision, highly homogenous "neighborhood" filter before applying Euclidean Distance to find the exact nutrient match.

---

## 📂 Project Structure
```text
📦 TROFES-MODEL-RECOMMENDATION
 ┣ 📂 .github
 ┃ ┗ 📂 workflows            # CI/CD pipelines (GitHub Actions)
 ┣ 📂 API                    # FastAPI application (deployed to Hugging Face Spaces)
 ┃ ┗ 📜 app.py               # Main API entry point & all route definitions
 ┣ 📂 data
 ┃ ┣ 📂 output               # Relational database-ready CSVs (recipe.csv, etc.)
 ┃ ┣ 📂 processed            # Intermediate cleaned datasets
 ┃ ┗ 📂 raw                  # Original raw datasets (epi_r.csv, etc.)
 ┣ 📂 models                 # Exported model artifacts (.pkl, .joblib) — auto-generated
 ┣ 📂 notebooks              # Jupyter notebooks for EDA & Model Training
 ┃ ┣ 📂 FoodRecomendation    # K-Means clustering development
 ┃ ┣ 📂 NutrientsCalculator  # Agglomerative clustering development
 ┃ ┗ 📜 data_preparation.ipynb # Master data cleaning and preprocessing pipeline
 ┣ 📜 .gitignore
 ┣ 📜 Dockerfile             # Container config for Hugging Face Spaces deployment
 ┣ 📜 main.py                # CI/CD automation script — executes training notebooks
 ┣ 📜 README.md
 ┗ 📜 requirements.txt       # Python dependencies
```

---

## ⚙️ CI/CD Pipeline & Automation

This project uses a fully automated ML pipeline triggered via **GitHub Actions**.

### How It Works
```
Push to main branch
       │
       ▼
GitHub Actions triggered
       │
       ▼
main.py executes training notebooks in order:
  1. notebooks/FoodRecomendation/final_kmeans.ipynb
  2. notebooks/NutrientsCalculator/final_agglomerative_focusNutrient.ipynb
       │
       ▼
Model artifacts (.pkl / .joblib) saved to /models
       │
       ▼
Artifacts pushed to Hugging Face Hub
(ArNight/Trofes_Recipe_Recomendation_Model)
       │
       ▼
Hugging Face Spaces pulls latest artifacts at runtime
via hf_hub_download()
```

### `main.py` — Automation Script

The `main.py` at the project root is **not** the API entry point. It is a CI/CD automation script that:

- Automatically creates the `models/` directory if it doesn't exist
- Sequentially executes both training notebooks using `nbconvert`
- Sets the execution path to the project root so all relative paths in notebooks resolve correctly
- Raises errors immediately if any notebook fails, preventing broken artifacts from being deployed

---

## 🐳 Docker & Deployment

### Dockerfile

The included `Dockerfile` containerizes the `API/app.py` for deployment on **Hugging Face Spaces**:
```dockerfile
FROM python:3.10-slim
WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
COPY . .
CMD ["uvicorn", "API.app:app", "--host", "0.0.0.0", "--port", "7860"]
```

> ⚠️ Hugging Face Spaces requires the `Dockerfile` to be at the **repository root** and exposes port `7860` by default.

### Deployment Architecture
```
GitHub Repository
 ┣ Source code + Dockerfile (root)
 ┗ main.py triggers notebook training on push
        │
        ▼ GitHub Actions CI/CD
        │
        ├──► Hugging Face Hub
        │    └── Model artifacts stored here
        │         (ArNight/Trofes_Recipe_Recomendation_Model)
        │
        └──► Hugging Face Spaces (Docker)
             └── API/app.py running on port 7860
                  └── pulls model artifacts from HF Hub at runtime
                       via hf_hub_download()
```

> 🔗 **Live API:** `https://huggingface.co/spaces/ArNight/trofes-api`

---

## 💻 Local Development

### Prerequisites

- **Python 3.10+** — [Download here](https://www.python.org/downloads/)
- **Git** — [Download here](https://git-scm.com/)
- **Docker** *(optional, for container testing)* — [Download here](https://www.docker.com/)

### 1. Clone the Repository
```bash
git clone https://github.com/neebulaa/trofes-model-recommendation.git
cd TROFES-MODEL-RECOMMENDATION
```

### 2. Create a Virtual Environment
```bash
# Create
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

### 4. (Optional) Run Training Pipeline Manually

If you want to retrain the models locally without pushing to GitHub:
```bash
python main.py
```

This will execute both training notebooks and save the artifacts to `models/`.

### 5. Run the API Locally
```bash
uvicorn API.app:app --reload
```

The API will be live at **`http://127.0.0.1:8000`**

> 💡 On first run, the app will automatically download model artifacts from  
> **[ArNight/Trofes_Recipe_Recomendation_Model](https://huggingface.co/ArNight/Trofes_Recipe_Recomendation_Model)**  
> via Hugging Face Hub. Make sure you have a stable internet connection.

### 6. Interactive API Docs

| Interface | URL |
|-----------|-----|
| Swagger UI | `http://127.0.0.1:8000/docs` |
| ReDoc | `http://127.0.0.1:8000/redoc` |

---

### Run with Docker (Local Container)

To replicate the exact production environment:
```bash
# Build the image
docker build -t trofes-api .

# Run the container
docker run -p 7860:7860 trofes-api
```

API available at **`http://localhost:7860`**

---

## 📡 API Endpoints

| Method | Endpoint | Model Used | Description |
|--------|----------|------------|-------------|
| `POST` | `/recommend` | Model A (K-Means) | Recommendations based on user preferences & ingredients |
| `POST` | `/recommendCalculator` | Model B (Agglomerative) | Recommendations based on target macronutrients |

> Full request/response schema available on Swagger UI at `/docs`.

---

## 🤝 Contributing

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'feat: add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---