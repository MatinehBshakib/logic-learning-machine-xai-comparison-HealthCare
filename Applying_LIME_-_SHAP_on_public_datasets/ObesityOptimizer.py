import pandas as pd
import numpy as np

class ObesityOptimizer:
    def __init__(self):
        # Added to ensure train/test sets have the exact same columns after one-hot encoding
        self.train_columns = None 

    def optimize(self, X, y=None, target_name="NObeyesdad", is_train=True):
        print("\n>>> Starting Obesity Dataset Optimization...")
        X = X.copy()
        
        # A. Binary Columns
        binary_mapping = {'yes': 1, 'no': 0, 'Male': 1, 'Female': 0}
        binary_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']
        
        for col in binary_cols:
            if col in X.columns:
                X[col] = X[col].map(binary_mapping)

        # B. Ordinal Columns
        ordinal_mapping = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
        ordinal_cols = ['CAEC', 'CALC']
        
        for col in ordinal_cols:
            if col in X.columns:
                X[col] = X[col].map(ordinal_mapping)

        # C. Nominal Columns (One-Hot)
        if 'MTRANS' in X.columns:
            dummies = pd.get_dummies(X['MTRANS'], prefix='Transport', dtype=int)
            X = pd.concat([X, dummies], axis=1)
            X = X.drop(columns=['MTRANS'])

        # D. Drop Leakage Columns (Height/Weight directly calculate BMI/Obesity)
        leakage_cols = ['Height', 'Weight']
        cols_to_drop = [c for c in leakage_cols if c in X.columns]
        if cols_to_drop:
            X = X.drop(columns=cols_to_drop)

        # E. Target Transformation
        y_final = None
        if y is not None:
            obesity_mapping = {
                'Insufficient_Weight': 0, 'Normal_Weight': 0,
                'Overweight_Level_I': 0, 'Overweight_Level_II': 0,
                'Obesity_Type_I': 1, 'Obesity_Type_II': 1, 'Obesity_Type_III': 1
            }

            # Safely extract target
            if isinstance(y, pd.DataFrame) and target_name in y.columns:
                raw_target = y[target_name]
            elif isinstance(y, pd.DataFrame):
                raw_target = y.iloc[:, 0]
            else:
                raw_target = y

            # Strip spaces from strings to ensure the mapping works perfectly
            clean_target = raw_target.astype(str).str.strip()
            y_new = clean_target.map(obesity_mapping)
            
            # The Brute-Force Rename Fix
            y_final = pd.DataFrame(y_new.values, index=raw_target.index, columns=['Obesity'])

            if y_final.isnull().any().any():
                print("Warning: Found unmapped target labels. Filling with 0.")
                y_final = y_final.fillna(0)

        # Drop original target if it somehow snuck into X
        if target_name in X.columns:
            X = X.drop(columns=[target_name])

        # F. Feature Alignment (Crucial for XGBoost & MTRANS One-Hot)
        if is_train:
            self.train_columns = X.columns
        else:
            X = X.reindex(columns=self.train_columns, fill_value=0)

        # Final numeric conversion safety check
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        return X, y_final