import pandas as pd
import numpy as np

class DiabetesOptimizer: #Diabetes 130-US Hospitals for years 1999-2008 Data Set
    def __init__(self):
        self.train_columns = None
        self.med_cols = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 
            'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 
            'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 
            'miglitol', 'troglitazone', 'tolazamide', 'examide', 
            'citoglipton', 'insulin', 'glyburide-metformin', 
            'glipizide-metformin', 'glimepiride-pioglitazone', 
            'metformin-rosiglitazone', 'metformin-pioglitazone']
        
    def optimize(self, X, y=None, target_name='readmitted', is_train=True):
        print("\n>>> Starting Diabetes Dataset Optimization...")
        if y is not None:
            X = X.copy()
            if isinstance(y, pd.DataFrame):
                X[target_name] = y[target_name]
            else:
                X[target_name] = y
        # Replace ? with NaN
        X = X.replace('?', np.nan)

        # Drop columns with too many missing values
        drop_cols = ['weight', 'payer_code', 'encounter_id', 'patient_nbr', 'medical_specialty']
        X = X.drop(columns=[c for c in drop_cols if c in X.columns], errors='ignore')
        if is_train: 
            # Drop rows with missing critical info
            subset_cols = ['race', 'diag_1', 'diag_2', 'diag_3', 'gender']
            existing_subset = [c for c in subset_cols if c in X.columns]
            X = X.dropna(subset=existing_subset)
        
            # Drop Invalid Gender
            if 'gender' in X.columns:
                X = X[X['gender'] != 'Unknown/Invalid']

            # Remove "Dead" Patients (cannot be readmitted)
            if 'discharge_disposition_id' in X.columns:
                dead_ids = [11, 13, 14, 19, 20, 21]
                # Ensure numeric comparison
                X['discharge_disposition_id'] = pd.to_numeric(X['discharge_disposition_id'], errors='coerce')
                X = X[~X['discharge_disposition_id'].isin(dead_ids)]
        
        # 4. Target Transformation
        y_final = None
        if y is not None:
            if target_name in X.columns:# If target is still in X (train), use it; else use y
                y_series = X[target_name]
            else:
            # Logic: <30 days = 1 (Early Readmission), All else = 0
                 y_series = y[target_name] if isinstance(y, pd.DataFrame) else y
                 
            y_new = y_series.apply(lambda x: 1 if str(x) == '<30' else 0)
            y_final = y_new.to_frame(name='Readmitted')
            # Remove original target from features to prevent leakage
            X = X.drop(columns=[target_name], errors='ignore')
        # A. Map Age
        if 'age' in X.columns:
            age_map = {f'[{i*10}-{i*10+10})': i for i in range(10)}
            X['age'] = X['age'].map(age_map)

        # B. Map Medications (No/Steady/Up/Down -> 0/1/2)
        med_map = {'No': 0, 'Steady': 1, 'Up': 2, 'Down': 2}
        for col in self.med_cols:
            if col in X.columns:
                X[col] = X[col].map(med_map).fillna(0)

        # C. Binary Columns
        binary_cols = ['change', 'diabetesMed']
        bin_map = {'No': 0, 'Yes': 1, 'Ch': 1}
        for col in binary_cols:
            if col in X.columns:
                X[col] = X[col].map(bin_map).fillna(0)

        # D. ICD-9 Diagnosis Grouping
        def get_diag_category(code):
            try:
                if str(code).startswith(('V', 'E')): return 'Other'
                n = float(code)
                if 390 <= n <= 459 or n == 785: return 'Circulatory'
                if 460 <= n <= 519 or n == 786: return 'Respiratory'
                if 520 <= n <= 579 or n == 787: return 'Digestive'
                if str(n).startswith('250'): return 'Diabetes'
                if 800 <= n <= 999: return 'Injury'
                if 710 <= n <= 739: return 'Musculoskeletal'
                if 580 <= n <= 629 or n == 788: return 'Genitourinary'
                if 140 <= n <= 239: return 'Neoplasms'
                return 'Other'
            except:
                return 'Other'

        diag_cols = ['diag_1', 'diag_2', 'diag_3']
        for col in diag_cols:
            if col in X.columns:
                X[col] = X[col].apply(get_diag_category)
        # E. One-Hot Encoding for Nominal Variables
        nominal_cols = ['race', 'gender', 'max_glu_serum', 'A1Cresult', 'diag_1', 'diag_2', 'diag_3']
        X = pd.get_dummies(X, columns=[c for c in nominal_cols if c in X.columns], dtype=int)
        # F. Ensure consistent columns between train and test
        if is_train:
                    self.train_columns = X.columns
        else:
            X = X.reindex(columns=self.train_columns, fill_value=0)
            
        #Final conversion (keep NaNs for XGBoost)
        X = X.apply(pd.to_numeric, errors='coerce')
        
        return X, y_final