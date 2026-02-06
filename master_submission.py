#!/usr/bin/env python3
"""
🏆 Numerai Master Pipeline (Memory Optimized + Immediate Uploads)
"""

import pandas as pd
import numpy as np
import os
import gc
import sys
import time
import requests
import warnings
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from numerapi import NumerAPI

# Machine Learning Imports
import lightgbm as lgb
from xgboost import XGBRegressor

# Optional Imports
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
    device = torch.device('cpu') # Force CPU for GitHub Actions stability
except ImportError:
    PYTORCH_AVAILABLE = False
    print("⚠️ PyTorch not found. Skipping Neural Network models.")

# Custom Feature Engineering
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

# API Keys
public_id = os.getenv("NUMERAI_PUBLIC_ID")
secret_key = os.getenv("NUMERAI_SECRET_KEY")

# Model Definitions
MODELS = {
    # Standard Models
    'lightgbm': {'id': 'bd2f8540-d90a-4206-b1c5-4e28f2865cba', 'name': 'jewellzilla_std'},
    'catboost': {'id': '9e253cd6-6b6b-4178-a641-c9738f21eb11', 'name': 'jewellzilla_cat'},
    'xgboost':  {'id': 'a65acf61-b5ba-4982-a7c8-7339be001a13', 'name': 'jewellzilla_xg'},
    
    # Enhanced Models
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

# ⚠️ MEMORY SETTINGS (Optimized for GitHub Free Runners)
SAMPLE_FRACTION = 0.03  # 3% Sample (Safe for 7GB RAM)
ENHANCED_ROWS = 100000  # Reduced for Feature Engineering safety
MAX_FEATURES = 500      # Reduce features slightly for Standard models

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def download_with_retry(napi, filename, max_retries=5, chunk_size=1024*1024):
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

            # Disable tqdm on non-interactive terminals to keep logs clean
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

def upload_immediately(napi, file_path, model_key):
    """Uploads a single prediction file immediately to Numerai."""
    
    # 1. Check if model exists in config
    if model_key not in MODELS:
        print(f"   ⚠️ Configuration for {model_key} not found. Skipping upload.")
        return

    model_conf = MODELS[model_key]
    print(f"   📤 UPLOADING {model_key} -> {model_conf['name']}...")
    
    # 2. 🛑 SANITY CHECK (Fixed)
    # Use the 'file_path' argument passed to the function!
    if os.path.exists(file_path):
        # Calculate size in MB
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        
        print(f"      📊 Stats: {file_size_mb:.2f} MB")

        # SAFETY LIMIT: 50MB
        # (Live files are usually < 5MB. If it's 50MB+, you likely predicted on training data)
        if file_size_mb > 50:
            print(f"      ❌ CRITICAL WARNING: File is {file_size_mb:.2f} MB.")
            print("         This is suspiciously large for a live submission.")
            print("         Aborting upload to prevent timeout.")
            return  # Stop here, do not upload
    else:
        print(f"      ❌ Error: File not found at {file_path}")
        return

    # 3. Attempt Upload
    try:
        submission_id = napi.upload_predictions(file_path, model_id=model_conf['id'])
        print(f"      ✅ SUCCESS! Submission ID: {submission_id}")
    except Exception as e:
        print(f"      ❌ UPLOAD FAILED: {e}")

# ============================================================
# MAIN PIPELINE
# ============================================================

if __name__ == "__main__":
    print("="*70)
    print("🏆 NUMERAI MASTER PIPELINE (IMMEDIATE SUBMISSION MODE)")
    print("="*70)

    # 1. Connect
    if not public_id or not secret_key:
        raise ValueError("❌ API keys missing! Set NUMERAI_PUBLIC_ID and NUMERAI_SECRET_KEY.")
    napi = NumerAPI(public_id, secret_key)
    print(f"✅ Connected as {napi.get_account()['username']}")

    # 2. Download Data
    if not Path(TRAINING_FILE).exists():
        download_with_retry(napi, TRAINING_FILE)
    
    if Path(LIVE_FILE).exists(): os.remove(LIVE_FILE)
    download_with_retry(napi, LIVE_FILE)

    # ========================================================
    # PHASE 1: STANDARD MODELS (Raw Features)
    # ========================================================
    print("\n" + "="*70)
    print("🚀 PHASE 1: STANDARD MODELS")
    print("="*70)

    import pyarrow.parquet as pq
    pq_file = pq.ParquetFile(TRAINING_FILE)
    all_cols = pq_file.schema.names
    feature_cols = [c for c in all_cols if c.startswith("feature_")][:MAX_FEATURES]
    cols_to_load = feature_cols + ["target"]

    print("🔄 Loading standard training data (Float32)...")
    total_rows = pq_file.metadata.num_rows
    sample_size = int(total_rows * SAMPLE_FRACTION)
    
    # Load as float32 to save memory
    df_train = pd.read_parquet(TRAINING_FILE, columns=cols_to_load).sample(n=sample_size, random_state=42).astype("float32")
    df_live = pd.read_parquet(LIVE_FILE, columns=feature_cols).astype("float32")
    
    X_train = df_train[feature_cols]
    y_train = df_train["target"]
    
    # --- LightGBM Standard ---
    print("\n🌲 Standard LightGBM: Training...")
    model = lgb.LGBMRegressor(n_estimators=800, learning_rate=0.05, max_depth=4, num_leaves=16, colsample_bytree=0.1, n_jobs=-1, verbose=-1)
    model.fit(X_train, y_train)
    preds = model.predict(df_live)
    preds_norm = (preds - preds.min()) / (preds.max() - preds.min())
    
    fname = SUBMISSIONS_DIR / f"std_lgbm_{DATE_STR}.csv"
    pd.DataFrame({"id": df_live.index, "prediction": preds_norm}).to_csv(fname, index=False)
    upload_immediately(napi, fname, 'lightgbm') # <--- IMMEDIATE UPLOAD
    
    del model, preds, preds_norm
    gc.collect()

    # --- XGBoost Standard ---
    print("\n🚀 Standard XGBoost: Training...")
    model = XGBRegressor(n_estimators=800, learning_rate=0.05, max_depth=4, colsample_bytree=0.1, n_jobs=-1, verbosity=0)
    model.fit(X_train, y_train)
    preds = model.predict(df_live)
    preds_norm = (preds - preds.min()) / (preds.max() - preds.min())
    
    fname = SUBMISSIONS_DIR / f"std_xgb_{DATE_STR}.csv"
    pd.DataFrame({"id": df_live.index, "prediction": preds_norm}).to_csv(fname, index=False)
    upload_immediately(napi, fname, 'xgboost') # <--- IMMEDIATE UPLOAD

    del model, preds, preds_norm
    gc.collect()

    # --- CatBoost Standard ---
    if CATBOOST_AVAILABLE:
        print("\n🐱 Standard CatBoost: Training...")
        model = CatBoostRegressor(iterations=800, learning_rate=0.05, depth=4, verbose=False, allow_writing_files=False)
        model.fit(X_train, y_train)
        preds = model.predict(df_live)
        preds_norm = (preds - preds.min()) / (preds.max() - preds.min())
        
        fname = SUBMISSIONS_DIR / f"std_cat_{DATE_STR}.csv"
        pd.DataFrame({"id": df_live.index, "prediction": preds_norm}).to_csv(fname, index=False)
        upload_immediately(napi, fname, 'catboost') # <--- IMMEDIATE UPLOAD

        del model, preds, preds_norm
        gc.collect()

    # Cleanup Phase 1 Data
    print("\n🧹 Cleaning up Phase 1 memory...")
    del df_train, X_train, y_train
    gc.collect()

    # ========================================================
    # PHASE 2: ENHANCED MODELS (Engineered Features)
    # ========================================================
    if FEATURE_ENG_AVAILABLE:
        print("\n" + "="*70)
        print("🧪 PHASE 2: ENHANCED MODELS")
        print("="*70)

        print("🔄 Loading data for engineering...")
        # Re-load smaller chunk for engineering
        df_train = pd.read_parquet(TRAINING_FILE).sample(n=ENHANCED_ROWS, random_state=42)
        df_live = pd.read_parquet(LIVE_FILE) # Reload raw live

        print("🔧 Engineering Features...")
        if os.path.exists('feature_engineer.pkl'):
            fe = NumeraiFeatureEngineer.load('feature_engineer.pkl')
            df_train = fe.transform(df_train)
            df_live = fe.transform(df_live)
        else:
            fe = NumeraiFeatureEngineer()
            df_train = fe.fit_transform(df_train)
            df_live = fe.transform(df_live)
            fe.save('feature_engineer.pkl')

        # Feature Selection
        all_feats = get_all_feature_columns(df_train)
        corrs = df_train[all_feats].corrwith(df_train['target']).abs()
        top_features = corrs.nlargest(2000).index.tolist()
        
        # Convert to numpy and clear dataframe to save RAM
        X_train_eng = df_train[top_features].values.astype('float32')
        y_train_eng = df_train['target'].values.astype('float32')
        X_live_eng = df_live[top_features].values.astype('float32')
        live_index = df_live.index
        
        del df_train, df_live, corrs
        gc.collect()

        live_preds = pd.DataFrame(index=live_index)

        # --- Enhanced LGBM ---
        print("\n🌲 Enhanced LightGBM: Training...")
        model = lgb.LGBMRegressor(n_estimators=1500, learning_rate=0.01, max_depth=5, num_leaves=32, colsample_bytree=0.1, n_jobs=-1, verbose=-1)
        model.fit(X_train_eng, y_train_eng)
        pred = model.predict(X_live_eng)
        live_preds['lgbm'] = pred
        
        fname = SUBMISSIONS_DIR / f"enh_lgbm_{DATE_STR}.csv"
        pd.DataFrame({"id": live_index, "prediction": pred}).to_csv(fname, index=False)
        upload_immediately(napi, fname, 'enh_lgbm') # <--- IMMEDIATE UPLOAD
        
        del model
        gc.collect()

        # --- Enhanced XGBoost ---
        print("\n🚀 Enhanced XGBoost: Training...")
        model = XGBRegressor(n_estimators=1500, learning_rate=0.01, max_depth=5, colsample_bytree=0.1, n_jobs=-1, verbosity=0)
        model.fit(X_train_eng, y_train_eng)
        pred = model.predict(X_live_eng)
        live_preds['xgb'] = pred
        
        fname = SUBMISSIONS_DIR / f"enh_xgb_{DATE_STR}.csv"
        pd.DataFrame({"id": live_index, "prediction": pred}).to_csv(fname, index=False)
        upload_immediately(napi, fname, 'enh_xgboost') # <--- IMMEDIATE UPLOAD
        
        del model
        gc.collect()

        # --- Enhanced Neural Net ---
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

            print("\n🧠 Enhanced NN: Training...")
            train_ds = NumeraiDataset(X_train_eng, y_train_eng)
            train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True)
            
            nn_model = NumeraiNN(len(top_features)).to(device)
            optimizer = optim.Adam(nn_model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
            nn_model.train()
            for epoch in range(30): # Reduced epochs
                for batch_x, batch_y in train_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    optimizer.zero_grad()
                    loss = criterion(nn_model(batch_x).squeeze(), batch_y)
                    loss.backward()
                    optimizer.step()
            
            nn_model.eval()
            with torch.no_grad():
                live_tensor = torch.FloatTensor(X_live_eng).to(device)
                pred = nn_model(live_tensor).cpu().numpy().flatten()
                live_preds['nn'] = pred
            
            fname = SUBMISSIONS_DIR / f"enh_nn_{DATE_STR}.csv"
            pd.DataFrame({"id": live_index, "prediction": pred}).to_csv(fname, index=False)
            upload_immediately(napi, fname, 'enh_nn') # <--- IMMEDIATE UPLOAD
            
            del nn_model, train_ds, train_loader
            gc.collect()

        # --- Enhanced Ensemble ---
        print("\n⚖️ Enhanced Ensemble: Calculating...")
        cols = [c for c in ['lgbm', 'xgb', 'nn'] if c in live_preds.columns]
        for c in cols:
            live_preds[f'{c}_rank'] = live_preds[c].rank(pct=True)
        
        live_preds['ensemble'] = live_preds[[f'{c}_rank' for c in cols]].mean(axis=1)
        
        fname = SUBMISSIONS_DIR / f"enh_ensemble_{DATE_STR}.csv"
        live_preds[['ensemble']].rename(columns={'ensemble': 'prediction'}).to_csv(fname)
        upload_immediately(napi, fname, 'enh_rank_ens') # <--- IMMEDIATE UPLOAD

    print("\n✅ MASTER PIPELINE COMPLETE")
