import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.datasets import fetch_openml
from imblearn.over_sampling import SMOTE
from scipy.io import arff
import json

class LoadData:

    def __init__(self):
        self._num_imputer = None
        self._cat_imputer = None
        self._drop_threshold = 0.5
        self._cols_to_keep = None   # Columns that survived the drop-threshold on training set

    def fit_imputer(self, x_train, drop_threshold=0.5):
        x = x_train.copy()
        self._drop_threshold = drop_threshold

        # 1. Try to coerce object columns that are really numeric
        for col in x.columns:
            if x[col].dtype == 'object':
                temp = pd.to_numeric(x[col], errors='coerce')
                if temp.isna().mean() < drop_threshold:
                    x[col] = temp

        # 2. Drop columns with too many missing values (fit on train only)
        x = x.drop(columns=x.columns[x.isnull().mean() > drop_threshold])
        self._cols_to_keep = x.columns.tolist()

        # 3. Separate numeric / categorical
        num_cols = x.select_dtypes(include=[np.number]).columns
        cat_cols = x.select_dtypes(exclude=[np.number]).columns

        # 4. Fit and transform numeric columns
        if len(num_cols) > 0:
            self._num_imputer = SimpleImputer(strategy='mean')
            x[num_cols] = pd.DataFrame(
                self._num_imputer.fit_transform(x[num_cols]),
                columns=num_cols,
                index=x.index
            )

        # 5. Fit and transform categorical columns
        if len(cat_cols) > 0:
            self._cat_imputer = SimpleImputer(strategy='most_frequent')
            x[cat_cols] = pd.DataFrame(
                self._cat_imputer.fit_transform(x[cat_cols]),
                columns=cat_cols,
                index=x.index
            )

        return x

    def apply_imputer(self, x):
        if self._cols_to_keep is None:
            raise RuntimeError("fit_imputer() must be called before apply_imputer().")

        x = x.copy()

        # 1. Same numeric coercion applied at fit time
        for col in x.columns:
            if x[col].dtype == 'object':
                temp = pd.to_numeric(x[col], errors='coerce')
                if temp.isna().mean() < self._drop_threshold:
                    x[col] = temp

        # 2. Keep only the columns that survived the training-set drop
        x = x.reindex(columns=self._cols_to_keep)

        num_cols = x.select_dtypes(include=[np.number]).columns
        cat_cols = x.select_dtypes(exclude=[np.number]).columns

        if self._num_imputer is not None and len(num_cols) > 0:
            x[num_cols] = pd.DataFrame(
                self._num_imputer.transform(x[num_cols]),
                columns=num_cols,
                index=x.index
            )

        if self._cat_imputer is not None and len(cat_cols) > 0:
            x[cat_cols] = pd.DataFrame(
                self._cat_imputer.transform(x[cat_cols]),
                columns=cat_cols,
                index=x.index
            )

        return x
  
    @staticmethod
    def _process_target(df, target_cols):
        missing = [col for col in target_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Target columns not found in dataset: {missing}")

        y = df[target_cols].copy()

        for col in y.columns:
            try:
                y[col] = pd.to_numeric(y[col])
            except (ValueError, TypeError):
                pass  # Leave text labels as-is (e.g. Obesity, Grade)

            if pd.api.types.is_numeric_dtype(y[col]):
                y[col] = y[col].fillna(0).astype(int)

        X = df.drop(columns=target_cols)
        return X, y

    def load_link(self, data_id, target_cols=None):
        data = fetch_openml(data_id=data_id, version='active', as_frame=True, parser='auto')
        df = data.frame.copy()

        if 'id' in df.columns:
            df.drop(columns=['id'], inplace=True)

        df.replace(['?', 'NA', '', 'null'], np.nan, inplace=True)

        if target_cols:
            X, y = self._process_target(df, target_cols)
        else:
            target_name = data.default_target_attribute
            X = df.drop(columns=[target_name])
            y = df[target_name]
        return X, y

    def load_file(self, file_path, target_cols=None):
        if file_path.endswith('.arff'):
            data, _ = arff.loadarff(file_path)
            df = pd.DataFrame(data)
            for col in df.select_dtypes([object]):
                df[col] = df[col].str.decode('utf-8')
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported format. Use .csv, .xlsx, .xls, or .arff.")

        # Ensure a row-id column exists (used by export_data_for_rulex)
        if 'id' not in df.columns:
            df.insert(0, 'id', range(len(df)))

        df.replace(['?', 'NA', '', 'null'], np.nan, inplace=True)

        if not target_cols:
            raise ValueError("target_cols must be specified when loading from a file.")

        X, y = self._process_target(df, target_cols)
        X = X.drop(columns=['id'], errors='ignore')
        return X, y

    def export_data_for_rulex(self, x_train, x_test, y_train, y_test,
                               dataset_name="Dataset",
                               filename="rulex_ready_data.csv"):
          
        full_filename = f"{dataset_name}_{filename}"
        # Combine train and test back together with a 'Set_Type' column to indicate origin
        y_train_df = y_train.to_frame(name='Target') if isinstance(y_train, pd.Series) else y_train.copy()
        y_test_df  = y_test.to_frame(name='Target')  if isinstance(y_test,  pd.Series) else y_test.copy()

        train_df = pd.concat([x_train, y_train_df], axis=1)
        test_df  = pd.concat([x_test,  y_test_df],  axis=1)

        train_df['Set_Type'] = 'Train'
        test_df['Set_Type']  = 'Test'

        full_df = pd.concat([train_df, test_df])
        full_df.index.name = 'id'
        full_df.to_csv(full_filename, index=True)

        print(f"Data saved to '{full_filename}'. Train: {len(train_df)}, Test: {len(test_df)}")
        
    def load_dataset_config(self, config_path="datasets_config.json"):
        with open(config_path, "r") as f:
            config = json.load(f)
            
        for dataset_name, entry in config.items():
            if dataset_name == "active":
                continue
            
            # Dynamically check if the dataset uses an OpenML 'id' or a file 'url'
            if "id" in entry:
                source = str(entry["id"])  # Convert to string so .isdigit() works in main
            elif "url" in entry:
                source = str(entry["url"])
            else:
                raise KeyError(f"Dataset '{dataset_name}' must have either an 'id' or 'url' key in your JSON.")
            
            yield dataset_name, source, entry["target_list"]
    
    def validate_numeric(self, dataset_name, **kwargs):
        """Ensures all provided dataframes/series are strictly numeric."""
        for name, data in kwargs.items():
            df_check = data.to_frame() if isinstance(data, pd.Series) else data
            bad_cols = df_check.select_dtypes(exclude=['number', 'bool']).columns.tolist()
            
            if bad_cols:
                raise ValueError(f"CRITICAL ERROR: {name} in '{dataset_name}' contains non-numeric columns: {bad_cols}.")
    
    def drop_correlated(self, x_train, x_test, threshold=0.90):
        """Drops features that are more than 'threshold' correlated.
        Learns which columns to drop from train, then applies same to test."""
        corr_matrix = x_train.corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        cols_to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
        x_train = x_train.drop(columns=cols_to_drop)
        x_test  = x_test.drop(columns=cols_to_drop, errors='ignore')
        print(f"Dropped {len(cols_to_drop)} correlated columns: {cols_to_drop}")
        return x_train, x_test
    
    def export_balanced_for_rulex(self, x_train, y_train, x_test, y_test,
                               dataset_name="Dataset"):
        """Balances training data with SMOTE before exporting for Rulex."""
        target_col = y_train.columns[0] if isinstance(y_train, pd.DataFrame) else 'Target'
        y_tr = y_train.iloc[:, 0] if isinstance(y_train, pd.DataFrame) else y_train

        sm = SMOTE(random_state=42)
        x_bal, y_bal = sm.fit_resample(x_train, y_tr)

        x_bal_df = pd.DataFrame(x_bal, columns=x_train.columns)
        y_bal_df  = pd.DataFrame(y_bal, columns=[target_col])
        x_bal_df['Set_Type'] = 'Train'
        
        y_te = y_test.copy() if isinstance(y_test, pd.DataFrame) else y_test.to_frame(name=target_col)
        x_test_df = x_test.copy()
        x_test_df['Set_Type'] = 'Test'
        
        train_export = pd.concat([x_bal_df, y_bal_df], axis=1)
        test_export  = pd.concat([x_test_df, y_te], axis=1)
        full_df = pd.concat([train_export, test_export])
        full_df.index.name = 'id'
        filename = f"{dataset_name}_rulex_BALANCED_data.csv"
        full_df.to_csv(filename, index=True)
        print(f"Balanced Rulex export saved to '{filename}'.")
        
    def export_raw_for_rulex(self, x_train_raw, x_test_raw, y_train, y_test,
                          dataset_name="Dataset"):
        """Exports data BEFORE one-hot encoding — for Rulex categorical rule mining."""
        # Reuse the exact same saving logic, just with a different filename
        self.export_data_for_rulex(
            x_train_raw, x_test_raw,
            y_train, y_test,
            dataset_name=dataset_name,
            filename="rulex_RAW_data.csv"   # different filename so it doesn't overwrite
    )
        
    def discretize_for_rulex(self, df):     
        df_disc = df.copy()
        num_cols = df_disc.select_dtypes(include='number').columns
        all_labels = ['Low', 'Mid', 'High', 'VeryHigh']
        
        for col in num_cols:
            if df_disc[col].nunique() > 10:  # only bin truly continuous columns
                try:
                    # First figure out how many unique bins actually result
                    _, bins = pd.qcut(df_disc[col], q=4, retbins=True, duplicates='drop')
                    n_bins = len(bins) - 1  # number of actual bins created
                    
                    if n_bins < 2:
                        # Too few unique values to bin meaningfully — skip this column
                        continue
                    
                    # Use only as many labels as there are actual bins
                    labels = all_labels[:n_bins]
                    df_disc[col] = pd.qcut(df_disc[col], q=4,
                                        labels=labels,
                                        duplicates='drop')
                except Exception as e:
                    print(f"Could not discretize column '{col}': {e} — skipping.")
                    continue
        return df_disc