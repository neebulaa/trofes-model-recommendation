import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os

def run_notebook(notebook_path):
    print(f"Executing {notebook_path}...")
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)
    
    # Menjalankan notebook
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    ep.preprocess(nb, {'metadata': {'path': os.path.dirname(notebook_path)}})
    print(f"Finished {notebook_path}")

if __name__ == "__main__":
    # Jalankan kedua notebook sesuai folder kamu
    try:
        run_notebook('notebooks/FoodRecomendation/final_kmeans.ipynb')
        run_notebook('notebooks/NutrientsCalculator/final_agglomerative_focusNutrient.ipynb')
        print("All training completed successfully!")
    except Exception as e:
        print(f"Error during execution: {e}")
        exit(1)