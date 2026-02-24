import pandas as pd
import numpy as np

class DiabeticRetinopathyOptimizer:
    def __init__(self):
        self.train_columns = None
        self.redundant_cols = [] # Remembers which correlated columns to drop

    def optimize(self, X, y=None, target_name='Class', is_train=True):
        print("\n>>> Starting Diabetic Retinopathy Optimization...")
        X = X.copy()
        
        # 1. Clean Column Names
        X.columns = X.columns.astype(str).str.strip()

        # 2. Automated Correlation Filter (The "Echo Chamber" Fix)
        # We only calculate the correlation matrix on the TRAINING set
        if is_train:
            corr_matrix = X.corr().abs()
            upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            # Find features that are >90% identical
            self.redundant_cols = [col for col in upper_triangle.columns if any(upper_triangle[col] > 0.90)]
        
        # Drop the redundant columns from both Train and Test sets
        if self.redundant_cols:
            cols_to_drop = [c for c in self.redundant_cols if c in X.columns]
            X = X.drop(columns=cols_to_drop)
            if is_train:
                print(f"Dropped {len(cols_to_drop)} highly correlated features to protect SHAP/LIME.")

        # 3. Transform Target 
        y_final = None
        if y is not None:
            # Safely extract target
            if isinstance(y, pd.DataFrame):
                y.columns = y.columns.astype(str).str.strip()
            if isinstance(y, pd.DataFrame) and target_name in y.columns:
                raw_target = y[target_name]
            elif isinstance(y, pd.DataFrame):
                raw_target = y.iloc[:, 0]
            else:
                raw_target = y
            
            # Prevent string/integer trap
            target_series = pd.to_numeric(raw_target, errors='coerce').fillna(0)
            
            # Binary Logic: 1 = DR, 0 = No DR
            y_new = target_series.apply(lambda val: 1 if val == 1 else 0)
            
            # Brute-force rename
            y_final = pd.DataFrame(y_new.values, index=raw_target.index, columns=['DR_Class'])

        # Drop original target from features if present to prevent leakage
        if target_name in X.columns:
            X = X.drop(columns=[target_name])

        # 4. Feature Alignment (Crucial for XGBoost)
        if is_train:
            self.train_columns = X.columns
        else:
            X = X.reindex(columns=self.train_columns, fill_value=0)

        # 5. Final Numeric Conversion
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        return X, y_final