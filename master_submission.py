#!/usr/bin/env python3
"""
🏆 Numerai Master Pipeline
Combines Standard Models (LGBM, CatBoost, XGB) and Enhanced Models (Neural Net, Ensembles)
"""

import pandas as pd
import numpy as np
import os
import gc
import sys
import time
import requests
import warnings
import pickle
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from numerapi import NumerAPI

# Machine Learning Imports
import lightgbm as lgb
from xgboost import XGBRegressor

# Try imports for optional libraries
try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("⚠️ CatBoost not found. Skipping CatBoost models.")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    PYTORCH_AVAILABLE = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
except ImportError:
    PYTORCH_AVAILABLE = False
    print("⚠️ PyTorch not found. Skipping Neural Network models.")

# Try to import your custom feature engineer
try:
    from numerai_feature_engineering import NumeraiFeatureEngineer, get_all_feature_columns
    FEATURE_ENG_AVAILABLE = True
except ImportError:
    FEATURE_ENG_AVAILABLE = False
    print("⚠️ 'numerai_feature_engineering.py' not found. Skipping Enhanced models.")

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

# API Keys (Priority: Env Vars -> Manual Fallback)
PUBLIC_ID = os.getenv("NUMERAI_PUBLIC_ID")
SECRET_KEY = os.getenv("NUMERAI_SECRET_KEY")

# Model Definitions
MODELS = {
    # Standard Models (Raw Features)
    'lightgbm': {'id': 'bd2f8540-d90a-4206-b1c5-4e28f2865cba', 'name': 'jewellzilla_std'},
    'catboost': {'id': '9e253cd6-6b6b-4178-a641-c9738f21eb11', 'name': 'jewellzilla_cat'},
    'xgboost':  {'id': 'a65acf61-b5ba-4982-a7c8-7339be001a13', 'name': 'jewellzilla_xg'},
    
    # Enhanced Models (Engineered Features)
    'enh_lgbm': {'id': 'e02eda5d-1760-4984-a6c2-526a876621fb', 'name': 'jewellzilla_std_enh'},
    'enh_xgboost': {'id': '98639631-99f8-4139-b069-21fdc094a074', 'name': 'jewellzilla_xg_enh'},
    'enh_nn': {'id': '548df016-74d2-4e9b-9164-ee4165e19a7e', 'name': 'jewellzilla_nn'},
    'enh_rank_ens': {'id': 'ed68de39-57f8-43d2-92ce-3a6e375f4b8a', 'name': 'jewellzilla_rank_ens'}
}

# Settings
DATA_VERSION = "v5.2"
TRAINING_FILE = f"{DATA_VERSION}/train.parquet"
LIVE_FILE = f"{DATA_VERSION}/live.parquet"
SUBMISSIONS_DIR = Path("submissions")
SUBMISSIONS_DIR.mkdir(exist_ok=True)
DATE_STR = datetime.now().strftime("%Y%m%d_%H%M%S")

# Memory & Training Settings
SAMPLE_FRACTION = 0.08  # For Standard Models
ENHANCED_ROWS = 200000  # For Enhanced Models
MAX_FEATURES = 750      # For Standard Models

# ============================================================
# HELPER CLASSES & FUNCTIONS
# ============================================================

def download_with_retry(napi, filename, max_retries=10, chunk_size=1024*1024):
    """Robust download with resume support."""
    dest_path = Path(filename)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    query = "query($filename: String!) { dataset(filename: $filename) }"
    args = {'filename': filename}
    url = napi.raw_query(query, args)['data']['dataset']
    
    print(f"\n📥 Downloading {filename}...")
    for attempt in range(max_retries):
        try:
            resume_pos = 0
            mode = 'wb'
            if dest_path.exists():
                resume_pos = dest_path.stat().st_size
                mode = 'ab'
            
            headers = {'Range': f'bytes={resume_pos}-'} if resume_pos > 0 else {}
            response = requests.get(url, headers=headers, stream=True, timeout=30)
            
            total_size = int(response.headers.get('content-length', 0)) + resume_pos
            if 'content-range' in response.headers:
                total_size = int(response.headers['content-range'].split('/')[-1])

            with open(dest_path, mode) as f:
                with tqdm(total=total_size, initial=resume_pos, unit='B', unit_scale=True, disable=None) as pbar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            if dest_path.stat().st_size >= total_size:
                print("   ✅ Download complete!")
                return True
        except Exception as e:
            print(f"   ⚠️ Attempt {attempt+1} failed: {e}")
            time.sleep(5)
    return False

if PYTORCH_AVAILABLE:
    class NumeraiDataset(Dataset):
        def __init__(self, features, targets):
            self.features = torch.FloatTensor(features)
            self.targets = torch.FloatTensor(targets)
        def __len__(self): return len(self.features)
        def __getitem__(self, idx): return self.features[idx], self.targets[idx]

    class NumeraiNN(nn.Module):
        def __init__(self, input_dim):
            super(NumeraiNN, self).__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(64, 1), nn.Sigmoid()
            )
        def forward(self, x): return self.network(x)

# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("="*70)
    print("🏆 NUMERAI MASTER PIPELINE")
    print("="*70)

    # 1. Connect
    if not PUBLIC_ID or not SECRET_KEY:
        raise ValueError("❌ API keys missing! Set NUMERAI_PUBLIC_ID and NUMERAI_SECRET_KEY.")
    napi = NumerAPI(PUBLIC_ID, SECRET_KEY)
    print(f"✅ Connected as {napi.get_account()['username']}")

    # 2. Download Data
    if not Path(TRAINING_FILE).exists():
        download_with_retry(napi, TRAINING_FILE)
    
    # Always get fresh live data
    if Path(LIVE_FILE).exists(): os.remove(LIVE_FILE)
    download_with_retry(napi, LIVE_FILE)

    # Dictionary to store file paths for submission
    submission_files = {}

    # ========================================================
    # PHASE 1: STANDARD MODELS (Raw Features)
    # ========================================================
    print("\n" + "="*70)
    print("🚀 PHASE 1: STANDARD MODELS (Raw Features)")
    print("="*70)

    import pyarrow.parquet as pq
    pq_file = pq.ParquetFile(TRAINING_FILE)
    all_cols = pq_file.schema.names
    feature_cols = [c for c in all_cols if c.startswith("feature_")][:MAX_FEATURES]
    cols_to_load = feature_cols + ["target"]

    # Load Data (Standard)
    print("🔄 Loading standard training data...")
    total_rows = pq_file.metadata.num_rows
    sample_size = int(total_rows * SAMPLE_FRACTION)
    
    df_train = pd.read_parquet(TRAINING_FILE, columns=cols_to_load).sample(n=sample_size, random_state=42)
    df_live = pd.read_parquet(LIVE_FILE, columns=feature_cols)
    
    X_train = df_train[feature_cols]
    y_train = df_train["target"]
    
    # --- LightGBM Standard ---
    print("\n🌲 Training Standard LightGBM...")
    model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, max_depth=4, num_leaves=16, colsample_bytree=0.1, n_jobs=-1, verbose=-1)
    model.fit(X_train, y_train)
    preds = model.predict(df_live)
    preds_norm = (preds - preds.min()) / (preds.max() - preds.min())
    
    fname = SUBMISSIONS_DIR / f"std_lgbm_{DATE_STR}.csv"
    pd.DataFrame({"id": df_live.index, "prediction": preds_norm}).to_csv(fname, index=False)
    submission_files['lightgbm'] = fname
    print(f"   ✅ Saved {fname}")

    # --- XGBoost Standard ---
    print("\n🚀 Training Standard XGBoost...")
    model = XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=4, colsample_bytree=0.1, n_jobs=-1, verbosity=0)
    model.fit(X_train, y_train)
    preds = model.predict(df_live)
    preds_norm = (preds - preds.min()) / (preds.max() - preds.min())
    
    fname = SUBMISSIONS_DIR / f"std_xgb_{DATE_STR}.csv"
    pd.DataFrame({"id": df_live.index, "prediction": preds_norm}).to_csv(fname, index=False)
    submission_files['xgboost'] = fname
    print(f"   ✅ Saved {fname}")

    # --- CatBoost Standard ---
    if CATBOOST_AVAILABLE:
        print("\n🐱 Training Standard CatBoost...")
        model = CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=4, verbose=False)
        model.fit(X_train, y_train)
        preds = model.predict(df_live)
        preds_norm = (preds - preds.min()) / (preds.max() - preds.min())
        
        fname = SUBMISSIONS_DIR / f"std_cat_{DATE_STR}.csv"
        pd.DataFrame({"id": df_live.index, "prediction": preds_norm}).to_csv(fname, index=False)
        submission_files['catboost'] = fname
        print(f"   ✅ Saved {fname}")

    # Clean up memory before Phase 2
    del df_train, X_train, y_train, model, preds
    gc.collect()

    # ========================================================
    # PHASE 2: ENHANCED MODELS (Engineered Features)
    # ========================================================
    if FEATURE_ENG_AVAILABLE:
        print("\n" + "="*70)
        print("🧪 PHASE 2: ENHANCED MODELS (Feature Engineering)")
        print("="*70)

        # Reload data for Feature Engineering (might need more cols or different sample)
        print("🔄 Loading data for engineering...")
        df_train = pd.read_parquet(TRAINING_FILE).sample(n=ENHANCED_ROWS, random_state=42)
        df_live = pd.read_parquet(LIVE_FILE)

        print("🔧 Running Feature Engineering (this takes time)...")
        if os.path.exists('feature_engineer.pkl'):
            fe = NumeraiFeatureEngineer.load('feature_engineer.pkl')
            df_train = fe.transform(df_train)
            df_live = fe.transform(df_live)
        else:
            fe = NumeraiFeatureEngineer()
            df_train = fe.fit_transform(df_train)
            df_live = fe.transform(df_live)
            fe.save('feature_engineer.pkl')

        # Select Features
        all_feats = get_all_feature_columns(df_train)
        corrs = df_train[all_feats].corrwith(df_train['target']).abs()
        top_features = corrs.nlargest(2000).index.tolist()
        
        X_train_eng = df_train[top_features].values
        y_train_eng = df_train['target'].values
        X_live_eng = df_live[top_features].values
        
        live_preds = pd.DataFrame(index=df_live.index)

        # --- Enhanced LGBM ---
        print("\n🌲 Training Enhanced LightGBM...")
        model = lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.01, max_depth=5, num_leaves=32, colsample_bytree=0.1, n_jobs=-1, verbose=-1)
        model.fit(X_train_eng, y_train_eng)
        live_preds['lgbm'] = model.predict(X_live_eng)
        
        fname = SUBMISSIONS_DIR / f"enh_lgbm_{DATE_STR}.csv"
        live_preds[['lgbm']].rename(columns={'lgbm': 'prediction'}).to_csv(fname)
        submission_files['enh_lgbm'] = fname

        # --- Enhanced XGBoost ---
        print("\n🚀 Training Enhanced XGBoost...")
        model = XGBRegressor(n_estimators=2000, learning_rate=0.01, max_depth=5, colsample_bytree=0.1, n_jobs=-1, verbosity=0)
        model.fit(X_train_eng, y_train_eng)
        live_preds['xgb'] = model.predict(X_live_eng)
        
        fname = SUBMISSIONS_DIR / f"enh_xgb_{DATE_STR}.csv"
        live_preds[['xgb']].rename(columns={'xgb': 'prediction'}).to_csv(fname)
        submission_files['enh_xgboost'] = fname

        # --- Enhanced Neural Net ---
        if PYTORCH_AVAILABLE:
            print("\n🧠 Training Neural Network...")
            train_ds = NumeraiDataset(X_train_eng, y_train_eng)
            train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True)
            
            nn_model = NumeraiNN(len(top_features)).to(device)
            optimizer = optim.Adam(nn_model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
            nn_model.train()
            for epoch in range(50): # Reduced epochs for daily speed
                for batch_x, batch_y in train_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    optimizer.zero_grad()
                    loss = criterion(nn_model(batch_x).squeeze(), batch_y)
                    loss.backward()
                    optimizer.step()
            
            nn_model.eval()
            with torch.no_grad():
                live_tensor = torch.FloatTensor(X_live_eng).to(device)
                live_preds['nn'] = nn_model(live_tensor).cpu().numpy().flatten()
            
            fname = SUBMISSIONS_DIR / f"enh_nn_{DATE_STR}.csv"
            live_preds[['nn']].rename(columns={'nn': 'prediction'}).to_csv(fname)
            submission_files['enh_nn'] = fname

        # --- Enhanced Ensemble ---
        print("\n⚖️ Creating Ensemble...")
        cols = [c for c in ['lgbm', 'xgb', 'nn'] if c in live_preds.columns]
        for c in cols:
            live_preds[f'{c}_rank'] = live_preds[c].rank(pct=True)
        
        live_preds['ensemble'] = live_preds[[f'{c}_rank' for c in cols]].mean(axis=1)
        
        fname = SUBMISSIONS_DIR / f"enh_ensemble_{DATE_STR}.csv"
        live_preds[['ensemble']].rename(columns={'ensemble': 'prediction'}).to_csv(fname)
        submission_files['enh_rank_ens'] = fname

    # ========================================================
    # PHASE 3: SUBMISSIONS
    # ========================================================
    print("\n" + "="*70)
    print("📤 PHASE 3: UPLOADING SUBMISSIONS")
    print("="*70)

    for key, fpath in submission_files.items():
        if key in MODELS:
            model_conf = MODELS[key]
            print(f"Uploading {key} -> {model_conf['name']}...")
            try:
                submission_id = napi.upload_predictions(fpath, model_id=model_conf['id'])
                print(f"   ✅ Success! ID: {submission_id}")
            except Exception as e:
                print(f"   ❌ Failed: {e}")
        else:
            print(f"⚠️  Skipping upload for {key} (No ID found in configuration)")

    print("\n✅ MASTER PIPELINE COMPLETE")
