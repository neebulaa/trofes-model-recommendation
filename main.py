import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os

def run_notebook(notebook_path):
    # Ambil lokasi folder root project secara absolut
    project_root = os.path.abspath(os.path.dirname(__file__))
    print(f"--- Working Directory: {project_root} ---")
    
    # KUNCIAN: Pastikan folder 'models' ada sebelum running
    models_dir = os.path.join(project_root, 'models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print("Folder 'models' berhasil dibuat secara otomatis!")

    print(f"Executing {notebook_path}...")
    
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)
    
    # Set path ke project_root agar notebook menganggap dirinya berjalan di folder utama
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    
    try:
        ep.preprocess(nb, {'metadata': {'path': project_root}})
        print(f"Selesai menjalankan {notebook_path}")
    except Exception as e:
        print(f"Error pada notebook {notebook_path}: {e}")
        raise e

if __name__ == "__main__":
    notebooks = [
        'notebooks/FoodRecomendation/final_kmeans.ipynb',
        'notebooks/NutrientsCalculator/final_agglomerative_focusNutrient.ipynb'
    ]
    
    for nb in notebooks:
        if os.path.exists(nb):
            run_notebook(nb)
        else:
            print(f"File notebook tidak ditemukan: {nb}")