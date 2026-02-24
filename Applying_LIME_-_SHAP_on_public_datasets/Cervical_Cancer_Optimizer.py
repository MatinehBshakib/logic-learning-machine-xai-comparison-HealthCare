import pandas as pd
import numpy as np

class CervicalCancerOptimizer:
    def __init__(self):
        self.train_columns = None

    def optimize(self, X, y=None, target_name='Biopsy', is_train=True):
        print("\n>>> Starting Cervical Cancer Dataset Optimization...")
        X = X.copy()
        
        # 1. Handle missing values specific to this dataset
        X = X.replace('?', np.nan)
        
        # 2. Drop severely sparse columns that provide no predictive value
        sparse_cols = ['STDs: Time since first diagnosis', 'STDs: Time since last diagnosis']
        X = X.drop(columns=[c for c in sparse_cols if c in X.columns], errors='ignore')

        # 3. Transform Target
        y_final = None
        if y is not None:
            # Safely extract target
            if isinstance(y, pd.DataFrame) and target_name in y.columns:
                raw_target = y[target_name]
            elif isinstance(y, pd.DataFrame):
                raw_target = y.iloc[:, 0]
            else:
                raw_target = y

            # Prevent string/integer trap
            target_series = pd.to_numeric(raw_target, errors='coerce').fillna(0)
            
            # Biopsy: 1 = Cancer/Pre-cancer, 0 = Healthy
            y_new = target_series.apply(lambda val: 1 if val == 1 else 0)
            
            # Brute-force rename
            y_final = pd.DataFrame(y_new.values, index=raw_target.index, columns=['Cancer_Diagnosis'])
            
        # Drop target from features to prevent leakage
        if target_name in X.columns:
            X = X.drop(columns=[target_name])

        # 4. Feature Alignment
        if is_train:
            self.train_columns = X.columns
        else:
            X = X.reindex(columns=self.train_columns, fill_value=0)

        # 5. Numeric conversion (XGBoost handles the remaining NaNs automatically!)
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        return X, y_final