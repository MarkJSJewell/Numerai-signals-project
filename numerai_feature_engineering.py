"""
Numerai Feature Engineering Pipeline
=====================================
Complete feature engineering system for improving Numerai model performance.
Integrates with existing automated submission pipeline.

Author: Mark
Date: 2025
"""

import pandas as pd
import numpy as np
from numerapi import NumerAPI
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
import pickle
import os
from datetime import datetime


class NumeraiFeatureEngineer:
    """
    Feature engineering class for Numerai tournament.
    Handles creation, fitting, and transformation of engineered features.
    """
    
    def __init__(self):
        self.fitted_objects = {}
        self.feature_names = []
        
    def create_charisma_interactions(self, df):
        """Create powerful interaction features from charisma group"""
        print("  - Creating charisma interactions...")
        
        # Get top charisma features based on correlation analysis
        top_charisma = [
            'feature_charisma_large_48',
            'feature_charisma_large_17', 
            'feature_charisma_small_17',
            'feature_charisma_large_14',
            'feature_charisma_large_32'
        ]
        
        # Only use features that exist in the dataframe
        top_charisma = [f for f in top_charisma if f in df.columns]
        
        if len(top_charisma) >= 2:
            # 1. Multiplicative interactions (captures joint effects)
            df['charisma_interaction_1'] = (
                df[top_charisma[0]] * df[top_charisma[1]]
            )
            
            # 2. Ratio features (captures relative strength)
            df['charisma_ratio_1'] = (
                df[top_charisma[0]] / (df[top_charisma[1]].abs() + 0.001)
            )
            
            # 3. Sum of top charisma features (aggregate signal)
            df['charisma_sum_top5'] = df[top_charisma].sum(axis=1)
            
            # 4. Mean of top charisma features
            df['charisma_mean_top5'] = df[top_charisma].mean(axis=1)
            
            # 5. Standard deviation (captures volatility)
            df['charisma_std_top5'] = df[top_charisma].std(axis=1)
            
            # 6. Min/Max ratio (captures range)
            df['charisma_minmax_ratio'] = (
                df[top_charisma].min(axis=1) / 
                (df[top_charisma].max(axis=1).abs() + 0.001)
            )
        
        return df
    
    def create_cross_group_features(self, df):
        """Combine signals from different feature groups"""
        print("  - Creating cross-group features...")
        
        # Define top features from different groups
        top_features = {
            'charisma': 'feature_charisma_large_48',
            'constitution': 'feature_constitution_large_17',
            'dexterity': 'feature_dexterity_small_40',
            'intelligence': 'feature_intelligence_small_24',
            'strength': 'feature_strength_large_23'
        }
        
        # Filter to only features that exist
        available_features = {k: v for k, v in top_features.items() if v in df.columns}
        
        if len(available_features) >= 2:
            feature_list = list(available_features.values())
            
            # 1. Multiplicative interactions
            if len(feature_list) >= 2:
                df['cross_interaction_1'] = df[feature_list[0]] * df[feature_list[1]]
            
            if len(feature_list) >= 4:
                df['cross_interaction_2'] = df[feature_list[2]] * df[feature_list[3]]
            
            # 2. Weighted combination of top groups (based on group size)
            if 'charisma' in available_features and 'constitution' in available_features:
                df['cross_weighted_top'] = (
                    0.38 * df[available_features['charisma']] +
                    0.21 * df[available_features['constitution']]
                )
                
                if 'strength' in available_features:
                    df['cross_weighted_top'] += 0.16 * df[available_features['strength']]
                
                if 'dexterity' in available_features:
                    df['cross_weighted_top'] += 0.13 * df[available_features['dexterity']]
                
                if 'intelligence' in available_features:
                    df['cross_weighted_top'] += 0.12 * df[available_features['intelligence']]
            
            # 3. Ratio features
            if len(feature_list) >= 2:
                df['cross_ratio_1'] = (
                    df[feature_list[0]] / (df[feature_list[1]].abs() + 0.001)
                )
        
        return df
    
    def create_rank_features(self, df):
        """Create rank-based features that are robust to outliers"""
        print("  - Creating rank features...")
        
        top_features = [
            'feature_charisma_large_48',
            'feature_charisma_large_17',
            'feature_constitution_large_17',
            'feature_charisma_small_17',
            'feature_charisma_large_14'
        ]
        
        # Filter to existing features
        top_features = [f for f in top_features if f in df.columns]
        
        if len(top_features) > 0:
            # 1. Percentile ranks (0-1 scale)
            for feat in top_features[:5]:  # Limit to top 5 to avoid too many features
                df[f'{feat}_rank'] = df[feat].rank(pct=True)
            
            # 2. Rank-based interactions
            if len(top_features) >= 2:
                df['rank_interaction_1'] = (
                    df[f'{top_features[0]}_rank'] * df[f'{top_features[1]}_rank']
                )
            
            # 3. Rank sum (composite indicator)
            rank_cols = [f'{feat}_rank' for feat in top_features[:5]]
            rank_cols = [c for c in rank_cols if c in df.columns]
            if len(rank_cols) > 0:
                df['rank_sum_top'] = df[rank_cols].sum(axis=1)
            
            # 4. Rank divergence (captures relative positioning)
            if len(top_features) >= 2:
                df['rank_divergence'] = (
                    df[f'{top_features[0]}_rank'] - df[f'{top_features[1]}_rank']
                ).abs()
        
        return df
    
    def create_polynomial_features(self, df):
        """Create polynomial features from top predictors"""
        print("  - Creating polynomial features...")
        
        top_5 = [
            'feature_charisma_large_48',
            'feature_charisma_large_17',
            'feature_constitution_large_17',
            'feature_charisma_small_17',
            'feature_charisma_large_14'
        ]
        
        # Filter to existing features and limit to top 3 to avoid explosion
        top_5 = [f for f in top_5 if f in df.columns][:3]
        
        for feat in top_5:
            # 1. Squared terms
            df[f'{feat}_squared'] = df[feat] ** 2
            
            # 2. Square root (for diminishing returns patterns)
            df[f'{feat}_sqrt'] = np.sign(df[feat]) * np.sqrt(np.abs(df[feat]))
            
            # 3. Log transform (for exponential relationships)
            df[f'{feat}_log'] = np.sign(df[feat]) * np.log1p(np.abs(df[feat]))
        
        return df
    
    def create_era_features(self, df):
        """Create features that account for era-specific patterns"""
        print("  - Creating era-aware features...")
        
        if 'era' not in df.columns:
            print("    (No era column found, skipping era features)")
            return df
        
        top_features = [
            'feature_charisma_large_48',
            'feature_charisma_large_17',
            'feature_constitution_large_17'
        ]
        
        # Filter to existing features
        top_features = [f for f in top_features if f in df.columns]
        
        for feat in top_features:
            # 1. Era-relative positioning (z-score within era)
            df[f'{feat}_era_zscore'] = df.groupby('era')[feat].transform(
                lambda x: (x - x.mean()) / (x.std() + 0.001)
            )
            
            # 2. Era percentile rank
            df[f'{feat}_era_rank'] = df.groupby('era')[feat].transform(
                lambda x: x.rank(pct=True)
            )
            
            # 3. Distance from era median
            df[f'{feat}_era_median_diff'] = df.groupby('era')[feat].transform(
                lambda x: x - x.median()
            )
        
        return df
    
    def create_binned_features(self, df):
        """Create binned versions for capturing threshold effects"""
        print("  - Creating binned features...")
        
        top_features = [
            'feature_charisma_large_48',
            'feature_charisma_large_17',
            'feature_constitution_large_17'
        ]
        
        # Filter to existing features
        top_features = [f for f in top_features if f in df.columns]
        
        for feat in top_features:
            try:
                # Create quintile bins
                df[f'{feat}_quintile'] = pd.qcut(
                    df[feat], 
                    q=5, 
                    labels=False, 
                    duplicates='drop'
                )
            except ValueError:
                # If qcut fails (not enough unique values), use regular cut
                df[f'{feat}_quintile'] = pd.cut(
                    df[feat], 
                    bins=5, 
                    labels=False
                )
        
        # Interaction between binned features
        if len(top_features) >= 2:
            df['binned_interaction_1'] = (
                df[f'{top_features[0]}_quintile'] * 10 +
                df[f'{top_features[1]}_quintile']
            )
        
        return df
    
    def create_group_aggregates(self, df):
        """Create statistical summaries by feature group"""
        print("  - Creating group aggregates...")
        
        # Get all feature groups
        feature_groups = {}
        for col in df.columns:
            if col.startswith('feature_'):
                parts = col.split('_')
                if len(parts) >= 2:
                    group = parts[1]
                    if group not in feature_groups:
                        feature_groups[group] = []
                    feature_groups[group].append(col)
        
        # Only process the top 3 groups to avoid too many features
        # Sort by group size (most features = most important)
        top_groups = sorted(feature_groups.items(), 
                          key=lambda x: len(x[1]), 
                          reverse=True)[:3]
        
        for group, features in top_groups:
            if len(features) > 1:
                # Mean
                df[f'{group}_mean'] = df[features].mean(axis=1)
                
                # Median (robust to outliers)
                df[f'{group}_median'] = df[features].median(axis=1)
                
                # Standard deviation (volatility)
                df[f'{group}_std'] = df[features].std(axis=1)
                
                # Min/Max
                df[f'{group}_min'] = df[features].min(axis=1)
                df[f'{group}_max'] = df[features].max(axis=1)
                
                # Range
                df[f'{group}_range'] = df[f'{group}_max'] - df[f'{group}_min']
        
        return df
    
    def create_pca_features(self, df, n_components=5, fit=True):
        """Create PCA components from highly correlated features"""
        print("  - Creating PCA features...")
        
        # Get charisma features (largest and most predictive group)
        charisma_features = [f for f in df.columns if 'charisma' in f and f.startswith('feature_')]
        
        if len(charisma_features) > n_components:
            if fit:
                pca_charisma = PCA(n_components=n_components)
                charisma_pca = pca_charisma.fit_transform(df[charisma_features])
                self.fitted_objects['pca_charisma'] = pca_charisma
                print(f"    Charisma PCA explained variance: {pca_charisma.explained_variance_ratio_.sum():.3f}")
            else:
                pca_charisma = self.fitted_objects['pca_charisma']
                charisma_pca = pca_charisma.transform(df[charisma_features])
            
            for i in range(n_components):
                df[f'pca_charisma_{i}'] = charisma_pca[:, i]
        
        # Get constitution features (second largest group)
        const_features = [f for f in df.columns if 'constitution' in f and f.startswith('feature_')]
        
        if len(const_features) > n_components:
            if fit:
                pca_const = PCA(n_components=n_components)
                const_pca = pca_const.fit_transform(df[const_features])
                self.fitted_objects['pca_constitution'] = pca_const
                print(f"    Constitution PCA explained variance: {pca_const.explained_variance_ratio_.sum():.3f}")
            else:
                pca_const = self.fitted_objects['pca_constitution']
                const_pca = pca_const.transform(df[const_features])
            
            for i in range(n_components):
                df[f'pca_constitution_{i}'] = const_pca[:, i]
        
        return df
    
    def fit_transform(self, df):
        """
        Fit the feature engineering pipeline on training data and transform it.
        
        Args:
            df: Training dataframe with original features
            
        Returns:
            Transformed dataframe with engineered features
        """
        print("\n" + "="*60)
        print("FITTING AND TRANSFORMING FEATURE ENGINEERING PIPELINE")
        print("="*60)
        
        # Store original feature count
        original_cols = df.columns.tolist()
        original_feature_count = len([c for c in original_cols if c.startswith('feature_')])
        
        # Apply all transformations
        df = self.create_charisma_interactions(df)
        df = self.create_cross_group_features(df)
        df = self.create_rank_features(df)
        df = self.create_polynomial_features(df)
        df = self.create_era_features(df)
        df = self.create_binned_features(df)
        df = self.create_group_aggregates(df)
        df = self.create_pca_features(df, n_components=5, fit=True)
        
        # Store feature names for later use
        self.feature_names = [c for c in df.columns if c not in original_cols]
        
        # Summary
        new_feature_count = len(self.feature_names)
        total_feature_count = len([c for c in df.columns if c.startswith('feature_') or 
                                   c in self.feature_names])
        
        print("\n" + "="*60)
        print("FEATURE ENGINEERING SUMMARY")
        print("="*60)
        print(f"Original features:      {original_feature_count:,}")
        print(f"New features created:   {new_feature_count:,}")
        print(f"Total features:         {total_feature_count:,}")
        print(f"Feature increase:       {(new_feature_count/original_feature_count*100):.1f}%")
        print("="*60 + "\n")
        
        return df
    
    def transform(self, df):
        """
        Transform new data using fitted feature engineering pipeline.
        
        Args:
            df: New dataframe to transform (validation/test/live data)
            
        Returns:
            Transformed dataframe with engineered features
        """
        print("\n" + "="*60)
        print("TRANSFORMING DATA WITH FITTED PIPELINE")
        print("="*60)
        
        # Apply all transformations (without fitting)
        df = self.create_charisma_interactions(df)
        df = self.create_cross_group_features(df)
        df = self.create_rank_features(df)
        df = self.create_polynomial_features(df)
        df = self.create_era_features(df)
        df = self.create_binned_features(df)
        df = self.create_group_aggregates(df)
        df = self.create_pca_features(df, n_components=5, fit=False)
        
        print("✓ Transformation complete\n")
        
        return df
    
    def save(self, filepath='feature_engineer.pkl'):
        """Save fitted feature engineer to disk"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"✓ Feature engineer saved to {filepath}")
    
    @staticmethod
    def load(filepath='feature_engineer.pkl'):
        """Load fitted feature engineer from disk"""
        with open(filepath, 'rb') as f:
            engineer = pickle.load(f)
        print(f"✓ Feature engineer loaded from {filepath}")
        return engineer


def get_all_feature_columns(df):
    """Get all feature columns including engineered ones"""
    return [c for c in df.columns if c.startswith('feature_') or 
            'rank' in c or 'pca' in c or 'cross' in c or 
            'charisma' in c or 'binned' in c or 
            any(suffix in c for suffix in ['_mean', '_std', '_median', '_min', '_max', '_range',
                                           '_squared', '_sqrt', '_log', '_quintile',
                                           '_era_zscore', '_era_rank', '_era_median_diff'])]


# Example usage function
def example_usage():
    """
    Example of how to use the feature engineering pipeline
    """
    print("\n" + "="*60)
    print("NUMERAI FEATURE ENGINEERING - EXAMPLE USAGE")
    print("="*60 + "\n")
    
    # Initialize API
    napi = NumerAPI()
    
    # Download data
    print("Downloading Numerai data...")
    napi.download_dataset("v5.2/train.parquet", "train.parquet")
    napi.download_dataset("v5.2/validation.parquet", "validation.parquet")
    
    # Load data
    print("Loading data...")
    train_data = pd.read_parquet("train.parquet")
    val_data = pd.read_parquet("validation.parquet")
    
    print(f"Training data shape: {train_data.shape}")
    print(f"Validation data shape: {val_data.shape}")
    
    # Initialize feature engineer
    engineer = NumeraiFeatureEngineer()
    
    # Fit and transform training data
    train_data_enhanced = engineer.fit_transform(train_data)
    
    # Transform validation data
    val_data_enhanced = engineer.transform(val_data)
    
    # Save the fitted engineer
    engineer.save('feature_engineer.pkl')
    
    # Get all feature columns
    feature_cols = get_all_feature_columns(train_data_enhanced)
    
    print(f"\nTotal feature columns available for modeling: {len(feature_cols)}")
    
    # Example: Train a model with enhanced features
    print("\nExample: Training LightGBM with enhanced features...")
    import lightgbm as lgb
    
    model = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.01,
        max_depth=5,
        num_leaves=2**5,
        colsample_bytree=0.1,
        random_state=42
    )
    
    X_train = train_data_enhanced[feature_cols]
    y_train = train_data_enhanced['target']
    
    X_val = val_data_enhanced[feature_cols]
    y_val = val_data_enhanced['target']
    
    print("Training model...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='l2',
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )
    
    print("\n✓ Example complete!")
    print("\nNext steps:")
    print("1. Integrate this into your automated pipeline")
    print("2. Compare performance with your existing models")
    print("3. Adjust feature engineering based on results")
    

if __name__ == "__main__":
    example_usage()
