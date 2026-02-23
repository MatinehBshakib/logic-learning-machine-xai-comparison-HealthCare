import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from scipy.io import arff
      
class LoadData:
      def advanced_imputation(self, x, drop_threshold=0.5):
          x = x.copy()
          for col in x.columns:
            if x[col].dtype == 'object':
                  # Try to convert, if it fails, check if it was mostly numbers
                  temp_col = pd.to_numeric(x[col], errors='coerce')
                  if temp_col.isna().mean() < drop_threshold: # If valid numbers remain
                        x[col] = temp_col
          # Drop columns with missing value percentage above the threshold
          x=x.drop(columns=x.columns[x.isnull().mean()>drop_threshold])
          # Separate numerical and categorical columns
          num_cols = x.select_dtypes(include=[np.number]).columns
          cat_cols = x.select_dtypes(exclude=[np.number]).columns
          # Impute numerical columns with mean
          if len(num_cols) > 0:
                  imputer_num = SimpleImputer(strategy='mean')
                  x[num_cols] = pd.DataFrame(
                        imputer_num.fit_transform(x[num_cols]),
                        columns=num_cols,
                        index=x.index
                  )
          # Impute categorical columns with mode
          if len(cat_cols) > 0:
                  imputer_cat = SimpleImputer(strategy='most_frequent')
                  x[cat_cols] = pd.DataFrame(
                        imputer_cat.fit_transform(x[cat_cols]),
                        columns=cat_cols,
                        index=x.index
                  )        
          return x
    
      def load_link(self, data_id, target_cols=None):
            #Load the dataset from OpenML
            data = fetch_openml(data_id=data_id, version='active', as_frame=True) 
            df = data.frame.copy()
            #Drop id column
            if 'id' in df.columns:
               df.drop(columns=['id'], inplace=True)
              
            df.replace(['?', 'NA', '', 'null'], np.nan, inplace=True)
            if target_cols:
                  missing_targets = [col for col in target_cols if col not in df.columns]
                  if missing_targets:
                      raise ValueError(f"Requested target columns not found in dataset: {missing_targets}")
                  y = df[target_cols].copy()
                  X = df.drop(columns=target_cols)
                  y = y.apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
            else:
                  #Get the exact target column name
                  target_name = data.default_target_attribute
                  #Separate features and target
                  X = df.drop(columns=[target_name])
                  y = df[target_name]
            return self.advanced_imputation(X), y
            
      def load_file(self, file_path, target_cols=None):
            if file_path.endswith('.arff'):
                  data, meta = arff.loadarff(file_path)
                  df = pd.DataFrame(data)
                  # ARFF strings load as bytes (e.g., b'string'), we need to decode them
                  for col in df.select_dtypes([object]):
                        df[col] = df[col].str.decode('utf-8')
            elif file_path.endswith('.csv'):
                  df = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx', '.xls'):
                  df = pd.read_excel(file_path)
            else:
                  raise ValueError("Unsupported file format. Please provide a .csv, .xlsx, .xls, or .arff file.")
            #Drop id column
            
            if 'id' not in df.columns:
               df.insert(0, 'id', range(len(df)))
            df.replace(['?', 'NA', '', 'null'], np.nan, inplace=True)
            if target_cols:
                  missing_targets = [col for col in target_cols if col not in df.columns]
                  if missing_targets:
                      raise ValueError(f"Requested target columns not found in dataset: {missing_targets}")
                  y = df[target_cols].copy()
                  cols_to_drop = target_cols + ['id']  
                  X = df.drop(columns=cols_to_drop)
                  for col in y.columns:
                      # 1. Safely try to convert the column to numbers
                      try:
                          y[col] = pd.to_numeric(y[col])
                      except (ValueError, TypeError):
                          # If it fails (because it's real text), just leave it alone
                          pass
                      
                      # 2. If it is now numeric, fill the NaNs and convert to integer
                      if pd.api.types.is_numeric_dtype(y[col]): 
                          y[col] = y[col].fillna(0).astype(int)          
            else:
                  raise ValueError("target_cols must be specified when loading from CSV.")
            return self.advanced_imputation(X), y
      
      def export_data_for_rulex(self, x_train, x_test, y_train, y_test, dataset_name="Dataset", filename="rulex_ready_data.csv"):
            full_filename = f"{dataset_name}_{filename}"
            
            # Prepare export DataFrames
            y_train_df = y_train.to_frame(name='Target') if isinstance(y_train, pd.Series) else y_train.copy()
            y_test_df = y_test.to_frame(name='Target') if isinstance(y_test, pd.Series) else y_test.copy()

            # Join features and targets
            train_df = pd.concat([x_train.reset_index(drop=True), y_train_df.reset_index(drop=True)], axis=1)
            test_df = pd.concat([x_test.reset_index(drop=True), y_test_df.reset_index(drop=True)], axis=1)

            # Add labels
            train_df['Set_Type'] = 'Train'
            test_df['Set_Type'] = 'Test'
            
            # Save to CSV
            full_df = pd.concat([train_df, test_df], ignore_index=True)
            full_df.index.name = 'id'
            full_df.to_csv(full_filename, index=True)
            
            print(f"Data saved to {full_filename}. Train: {len(train_df)}, Test: {len(test_df)}")