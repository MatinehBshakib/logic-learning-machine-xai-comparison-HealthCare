import pandas as pd
import numpy as np

class HCVOptimizer:
    def __init__(self):
        self.train_columns = None
        # Define the binary columns that need 1/2 -> 0/1 correction
        self.binary_cols = [
            'Gender', 'Fever', 'Nausea/Vomting', 'Headache', 
            'Diarrhea', 'Fatigue & generalized bone ache', 
            'Jaundice', 'Epigastric pain'
        ]

    def optimize(self, X, y=None, target_name='Baseline histological fibroses', is_train=True):
        print("\n>>> Starting HCV Specific Optimization...")
        X = X.copy()
        # 1. Clean Column Names (Remove trailing spaces)
        X.columns = X.columns.str.strip()
        
        # 2. Drop Leaky Columns
        # We remove 'Baseline histological Grading' because it reveals the answer
        leaky_cols = ['Baseline histological Grading']
        X = X.drop(columns=[c for c in leaky_cols if c in X.columns], errors='ignore')

        # 3. Fix Binary Columns (1/2 -> 0/1)
        # Your specific logic: shift values if max is 2
        for col in self.binary_cols:
            if col in X.columns:
                # Check if values are mostly 1 and 2
                if X[col].max() == 2:
                    X[col] = X[col] - 1
        # 4. RNA Log Transformation (Normalizing Viral Load)
        rna_cols = ['RNA Base', 'RNA 4', 'RNA 12', 'RNA EOT', 'RNA EF']
        for col in rna_cols:
            if col in X.columns:
                X[col] = np.log1p(X[col])            
        # 5. Transform Target
        # Logic: 3, 4 -> 1 (Hazardous); Others -> 0
        y_final = None
        if y is not None:
            if isinstance(y, pd.DataFrame): 
                y.columns = y.columns.str.strip()
            if isinstance(y, pd.DataFrame) and target_name in y.columns:
                target_series = y[target_name]
            elif isinstance(y, pd.DataFrame):
                target_series = y.iloc[:, 0]
            else:
                target_series = y
            y_new = target_series.apply(lambda val: 1 if val in [3, 4] else 0)
            y_final = y_new.to_frame(name='Stage')

        # 6. Feature Alignment
        if is_train:
            self.train_columns = X.columns
        else:
            X = X.reindex(columns=self.train_columns, fill_value=0)

        X = X.apply(pd.to_numeric, errors='coerce')
        return X, y_final