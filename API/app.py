from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
from sklearn.metrics import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
import random
from collections import Counter
from huggingface_hub import hf_hub_download

app = FastAPI(title="Trofes Recipe Engine")

# 1. Definisikan fungsi
def apply_weight(X, weight=1.0):
    return X * weight

# 2. Registrasi fungsi ke modul __main__
import __main__
__main__.apply_weight = apply_weight

# --- 1. SETUP MODEL DARI HUB ---
REPO_ID = "ArNight/Trofes_Recipe_Recomendation_Model"

def download_model(filename):
    # Ini akan mendownload file dari Gudang Model kamu ke cache Space
    return hf_hub_download(repo_id=REPO_ID, filename=filename)

ALLERGY_MAP = {
    1: 'has_dairy',
    2: 'has_egg',
    3: 'has_fish',
    4: 'has_shellfish',
    5: 'has_soy',
    6: 'has_sesame',
    7: 'has_wheat',
    8: 'has_peanut',
    9: 'has_treenut'
}

DIET_MAP = {
    1: 'halal',
    2: 'is_lactose_free',
    3: 'low_carb',
    4: 'weight_loss',
    5: 'high_protein',
    6: 'gluten_free',
    7: 'dairy_free',
    8: 'is_spicy',
    9: 'is_not_fried'
}

print("Memuat data model...")

# Load Model Kmeans
try:
    path_kmeans = download_model('recipe_engine_kmeans.pkl')
    with open(path_kmeans, 'rb') as f:
        data_package = pickle.load(f)
    
    df = data_package['metadata']
    matrix = data_package['matrix']
    if df.index.name != 'recipe_id':
        df = df.set_index('recipe_id')
    print("Model K-Means siap!")
except Exception as e:
    print(f"ERROR Load K-Means: {e}")
    df, matrix = pd.DataFrame(), None

# Load Model Agglomerative
try:
    path_agglo = download_model('nutri_engine_agglo.pkl')
    with open(path_agglo, 'rb') as f:
        pkg_agglo = pickle.load(f)
    df_agg = pkg_agglo['metadata'].set_index('recipe_id') if 'recipe_id' in pkg_agglo['metadata'].columns else pkg_agglo['metadata']
    matrix_agg = pkg_agglo['matrix_a'] 
    pipe_agg = pkg_agglo['pipeline_a']
    print("Model Agglo siap!")
except Exception as e:
    print(f"Error Load Agglo: {e}")
    df_agg, matrix_agg, pipe_agg = pd.DataFrame(), None, None


# --- 2. INPUT REQUEST ---
class RecommendRequest(BaseModel):
    liked_ids: List[int]
    top_k: Optional[int] = 5
    is_start_from_zero: bool = True

@app.get("/")
def home():
    return {"message": "Trofes API is Running"}

@app.post("/recommend")
def get_recommendations(req: RecommendRequest):
    if df.empty or matrix is None:
        raise HTTPException(status_code=500, detail="Model belum dimuat.")
    
    # ---------------------------------------------------------
    # LANGKAH 1: Normalisasi Input ke Index 0
    # ---------------------------------------------------------
    processed_ids = []
    if req.is_start_from_zero:
        processed_ids = req.liked_ids
    else:
        processed_ids = [x - 1 for x in req.liked_ids]
    
    valid_ids = [uid for uid in processed_ids if uid in df.index]
    if not valid_ids:
        return {"status": "empty", "recommended_ids": []}

    # ---------------------------------------------------------
    # LANGKAH 2: Cari Dominant Cluster (CERDAS & RECENCY AWARE)
    # ---------------------------------------------------------
    user_clusters = df.loc[valid_ids, 'cluster'].tolist()
    
    # HITUNG FREKUENSI
    counts = Counter(user_clusters)
    max_freq = max(counts.values())
    
    dominant_cluster = None

    if max_freq == 1:
        # KASUS 1: (Semua muncul 1x) -> [5, 6, 7]
        # Ambil cluster dari resep TERAKHIR yang diklik user
        dominant_cluster = user_clusters[-1]
    else:
        # KASUS 2: Ada Mayoritas / Seri -> [5, 5, 8, 8, 10]
        # Cari kandidat yang jumlahnya == max_freq (Kandidat: 5 dan 8)
        top_candidates = [k for k, v in counts.items() if v == max_freq]
        
        # Loop dari BELAKANG list user_clusters (Recency Check)
        # Cari kandidat mana yang muncul paling akhir
        for cluster in reversed(user_clusters):
            if cluster in top_candidates:
                dominant_cluster = cluster
                break
    # ---------------------------------------------------------
    # LANGKAH 3: Strategi Rekomendasi (Strict Cluster + Anchor)
    # ---------------------------------------------------------
    
    # 1. Hitung jumlah porsi
    num_main = int(req.top_k * 0.7) # 70% dari cluster utama
    num_diverse = req.top_k - num_main # 30% dari cluster lain

    anchors = [uid for uid in valid_ids if df.loc[uid, 'cluster'] == dominant_cluster]
    anchor_id = anchors[-1]
    strategy = "hybrid_exploration_70_30"
    recommended_internal_ids = []

    # --- BAGIAN A: Ambil dari Dominant Cluster (Exploitation) ---
    main_candidates = df[df['cluster'] == dominant_cluster].index.tolist()
    if main_candidates:
        target_vec = matrix[anchor_id]
        candidate_vecs = matrix[main_candidates]
        scores = cosine_similarity(target_vec.reshape(1, -1), candidate_vecs).flatten()
        
        sorted_indices = scores.argsort()[::-1]
        for idx in sorted_indices:
            rid = main_candidates[idx]
            if rid not in valid_ids:
                recommended_internal_ids.append(rid)
            if len(recommended_internal_ids) >= num_main:
                break
    
    # --- BAGIAN B: Ambil dari Neighboring Cluster (Exploration / Serendipity) ---
    other_clusters = [c for c in df['cluster'].unique() if c != dominant_cluster]
    
    if other_clusters and num_diverse > 0:
        # Ambil resep dari cluster selain dominant yang paling mirip dengan anchor_id
        diverse_candidates = df[df['cluster'] != dominant_cluster].index.tolist()
        
        div_target_vec = matrix[anchor_id]
        div_candidate_vecs = matrix[diverse_candidates]
        div_scores = cosine_similarity(div_target_vec.reshape(1, -1), div_candidate_vecs).flatten()
        sorted_div_indices = div_scores.argsort()[::-1]
        
        added_diverse = 0
        for idx in sorted_div_indices:
            rid = diverse_candidates[idx]
            if rid not in valid_ids and rid not in recommended_internal_ids:
                recommended_internal_ids.append(rid)
                added_diverse += 1
            if added_diverse >= num_diverse:
                break
    
    random.shuffle(recommended_internal_ids)

    # ---------------------------------------------------------
    # LANGKAH 4: Output
    # ---------------------------------------------------------
    final_output_ids = []
    if req.is_start_from_zero:
        final_output_ids = recommended_internal_ids
    else:
        final_output_ids = [x + 1 for x in recommended_internal_ids]
        
    return {
        "status": "success",
        "strategy": strategy,
        "dominant_cluster": int(dominant_cluster),
        "anchor_used": int(anchor_id),
        "recommended_ids": [int(x) for x in final_output_ids]
    }

class CalculatorRequest(BaseModel):
    calories: float
    protein: float
    fat: float
    carbs: float
    top_k: Optional[int] = 10
    is_start_from_zero: bool = True
    is_login: bool = False
    allergy_ids: Optional[List[int]] = []
    dietary_ids: Optional[List[int]] = []

@app.post("/recommendCalculator")
def calculate_recommendations(req: CalculatorRequest):
    if df_agg.empty or matrix_agg is None or pipe_agg is None:
        raise HTTPException(status_code=500, detail="Model Agglo belum dimuat.")
    
    try:
        pipeline = data_package.get('pipeline')
        # candidate_indices = df.index.tolist()
        filtered_df = df.copy()

        filtered_df = df_agg.copy()

        # 1. Filter Alergi & Diet (Sama seperti sebelumnya)
        if req.is_login:
            if req.allergy_ids:
                for a_id in req.allergy_ids:
                    col = ALLERGY_MAP.get(a_id)
                    if col in filtered_df.columns:
                        filtered_df = filtered_df[filtered_df[col] == 0]
            if req.dietary_ids:
                for d_id in req.dietary_ids:
                    col = DIET_MAP.get(d_id)
                    if col in filtered_df.columns:
                        filtered_df = filtered_df[filtered_df[col] == 1]

        if filtered_df.empty:
            return {"status": "no_match", "recommended_ids": []}
        
        # 3. Transform Input User ke Vektor
        # Kita buat baris dummy dengan kolom lengkap sesuai ekspektasi preprocessor
        input_dict = {
            'calories': req.calories,
            'protein': req.protein,
            'fat': req.fat,
            'carbohydrate': req.carbs,
            'sodium': 0, 
            'text_feature': '', 
            'cooking_time': 30,
        }
        
        # Tambahkan dummy 0 untuk semua kolom allergen & diet agar pipeline tidak error
        # Kita ambil nama kolomnya langsung dari dataframe metadata
        for col in df_agg.columns:
            if col not in input_dict:
                input_dict[col] = 0
        
        input_df = pd.DataFrame([input_dict])[df_agg.columns.tolist()]
        user_vector = pipe_agg.transform(input_df)

        # 4. Hitung Similarity dengan EUCLIDEAN DISTANCE
        candidate_indices = filtered_df.index.tolist()
        pos_indices = [df_agg.index.get_loc(idx) for idx in candidate_indices]
        candidate_matrix = matrix_agg[pos_indices]
        
        # Euclidean: Semakin kecil jarak, semakin mirip.
        dists = euclidean_distances(user_vector, candidate_matrix).flatten()
        # 5. Ambil kandidat lebih banyak (misal top 20) lalu acak (Shuffle)
        # Agar tidak monoton jika user memasukkan angka yang sama
        num_candidates = min(len(dists), 20)
        top_local_indices = dists.argsort()[:num_candidates]
        
        recommended_pool = [candidate_indices[i] for i in top_local_indices]
        
        # Ambil secara acak dari pool terbaik
        final_sample = random.sample(recommended_pool, min(len(recommended_pool), req.top_k))

        # 6. Output
        if not req.is_start_from_zero:
            final_ids = [int(x) + 1 for x in final_sample]
        else:
            final_ids = [int(x) for x in final_sample]

        return {
            "status": "success",
            "is_login": req.is_login,
            "recommended_ids": final_ids
        }
    except Exception as e:
        print(f"Error Calculator: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
