import random
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from Load import LoadData
from sklearn.preprocessing import LabelEncoder

from Strategy import SingleOutput
from HCVOptimizer import HCVOptimizer as HCVOpt
from ObesityOptimizer import ObesityOptimizer as ObesityOpt
from DiabetesOptimizer import DiabetesOptimizer as DiabetesOpt
from Diabetic_Retinopathy_Optimizer import DiabeticRetinopathyOptimizer as DROpt
from PerformanceMetrics import save_performance_metrics
from PostProcessor import PostProcessor


np.random.seed(42)
random.seed(42)

def main():
    loader = LoadData()
    for dataset_name, source, target_list in loader.load_dataset_config(os.path.join(os.path.dirname(__file__), "dataset_config.json")):
        print(f"\n>>> Processing: {dataset_name}")
        # 1. Load file (Dynamic Checking)
        if str(source).isdigit():
            # If the source is just numbers treat it as an OpenML ID
            print(f"Loading via OpenML link (ID: {source})...")
            X_raw, y_raw = loader.load_link(data_id=int(source), target_cols=target_list)
        else:
            # Otherwise, treat it as a local file path
            print(f"Loading via local file ({source})...")
            X_raw, y_raw = loader.load_file(file_path=source, target_cols=target_list)
        
        # 2. Downsample FIRST 
        SAMPLE_SIZE = 2500  
        if len(X_raw) > SAMPLE_SIZE:
            print(f"\n>>> Downsampling dataset from {len(X_raw)} to {SAMPLE_SIZE} rows...")
            X_raw, _, y_raw, _ = train_test_split(
                X_raw, y_raw, 
                train_size=SAMPLE_SIZE, 
                random_state=42
            )

        # 3. Split into Train/Test BEFORE optimization to prevent Data Leakage
        x_train_raw, x_test_raw, y_train_raw, y_test_raw = train_test_split(
            X_raw, y_raw, test_size=0.3,stratify=y_raw, random_state=42
        )
        x_train_raw = loader.fit_imputer(x_train_raw)   # Learns mean/mode from train ONLY
        x_test_raw  = loader.apply_imputer(x_test_raw)  # Applies those same statistics to test
        
        # Re-number Train indices to start at 0
        x_train_raw = x_train_raw.reset_index(drop=True)
        y_train_raw = y_train_raw.reset_index(drop=True)
        
        # Make Test indices start exactly where Train left off!
        start_idx = len(x_train_raw)
        x_test_raw.index = range(start_idx, start_idx + len(x_test_raw))
        y_test_raw.index = range(start_idx, start_idx + len(y_test_raw))
        
        # 4. Initialize the correct Optimizer
        optimizer = None
        if dataset_name == "Hepatitis":
            optimizer = HCVOpt()
        elif dataset_name == "Obesity_level":
            optimizer = ObesityOpt()
        elif dataset_name == "Diabetes_130_US":
            optimizer = DiabetesOpt()
        elif dataset_name == "Diabetic_Retinopathy":
            optimizer = DROpt()

        # Export the raw data before one-hot encoding for Rulex's categorical rule mining
        #loader.export_raw_for_rulex(x_train_raw, x_test_raw, y_train_raw, y_test_raw,dataset_name=dataset_name)
        # 5. Optimize Train and Test Separately
        if optimizer:
            print("\n--- Optimizing Training Set ---")
            x_train, y_train = optimizer.optimize(x_train_raw, y_train_raw, target_name=target_list[0], is_train=True)
            
            print("\n--- Optimizing Testing Set ---")
            # is_train=False forces the test set to match the train set's columns perfectly!
            x_test, y_test = optimizer.optimize(x_test_raw, y_test_raw, target_name=target_list[0], is_train=False)
        else:
            # Fallback for datasets without custom optimizers
            le = LabelEncoder()
            y_train_encoded = le.fit_transform(y_train_raw.values.ravel())
            y_test_encoded = le.transform(y_test_raw.values.ravel())
        
            y_train = pd.DataFrame(y_train_encoded, index=y_train_raw.index, columns=target_list)
            y_test = pd.DataFrame(y_test_encoded, index=y_test_raw.index, columns=target_list)
            
            # Automatically converts text columns into binary numeric columns (0 and 1)
            x_train = pd.get_dummies(x_train_raw, drop_first=True, dtype=int)
            x_test = pd.get_dummies(x_test_raw, drop_first=True, dtype=int)
            
            # Ensure the test set has the exact same columns as the train set
            # (In case a category existed in train but not test, or vice versa)
            x_train, x_test = x_train.align(x_test, join='left', axis=1, fill_value=0)
            
            x_train = x_train.apply(pd.to_numeric, errors='coerce').fillna(0)
            x_test = x_test.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        x_train, x_test = loader.drop_correlated(x_train, x_test, threshold=0.90)
        # Validate that all data is numeric before proceeding    
        loader.validate_numeric(dataset_name, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
        #Discretize for Rulex (bins numbers into groups)
        x_train_disc, x_test_disc = loader.discretize_for_rulex(x_train, x_test)

        loader.export_data_for_rulex(x_train_disc, x_test_disc, y_train, y_test, dataset_name=dataset_name)
        # For imbalanced datasets, also export a balanced version for Rulex to mine balanced rules
        imbalanced_datasets = ['CDC_Diabetes', 'Cervical_Cancer']
        if dataset_name in imbalanced_datasets:
            loader.export_balanced_for_rulex(x_train, y_train, x_test, y_test,
                                             dataset_name=dataset_name)
        # 7. Execute Strategy
        counts = y_train.iloc[:, 0].value_counts()
        imbalance_ratio = counts[0] / counts[1] if 1 in counts and counts[1] > 0 else 1
        print(f"\n>>> Recommended Random Forest scale_pos_weight: {imbalance_ratio:.2f}")
        
        if dataset_name == "Glioma_Grading":
            strategy = SingleOutput(algo='rf')
        elif dataset_name == "Breast_Cancer":
            strategy = SingleOutput(algo='rf')
        else:
            strategy = SingleOutput(algo='xgb', scale_pos_weight=imbalance_ratio) 
        clf = strategy.execute(x_train, x_test, y_train, y_test)
        
        # 8. Save Performance Metrics
        target_col = y_test.columns[0] if isinstance(y_test, pd.DataFrame) else target_list[0]
        y_test_1d  = y_test.iloc[:, 0] if isinstance(y_test, pd.DataFrame) else y_test
        if clf is not None:
            save_performance_metrics(
                clf           = clf,
                x_test        = x_test,
                y_test        = y_test_1d,
                dataset_name  = dataset_name,
                target_name   = target_col,
                n_train       = len(x_train),  
                output_folder = 'outputs',
            )
            
        # 9. Post Processing
        aggregator = PostProcessor()
        aggregator.aggregate_and_clean(database_name=dataset_name)

if __name__ == "__main__":
    main()