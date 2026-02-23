from sklearn.model_selection import train_test_split
from Load import LoadData
from sklearn.preprocessing import LabelEncoder
from Strategy import SingleOutput
from HCVOptimizer import HCVOptimizer as HCVOpt
from ObesityOptimizer import ObesityOptimizer as ObesityOpt
from DiabetesOptimizer import DiabetesOptimizer as DiabetesOpt
from PostProcessor import PostProcessor
import pandas as pd

def main():
    loader = LoadData()
    target_list = ["Baselinehistological staging"]  
    dataset_name = "HCV-Egy-Data"  
    url = "HCV-Egy-Data.csv" 
    
    # 1. Load Raw Data
    X_raw, y_raw = loader.load_file(file_path=url, target_cols=target_list)
    
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
        X_raw, y_raw, test_size=0.3, random_state=42
    )

    # 4. Initialize the correct Optimizer
    optimizer = None
    if dataset_name == "Hepatitis":
        optimizer = HCVOpt()
    elif dataset_name == "Obesity_level":
        optimizer = ObesityOpt()
    elif dataset_name == "Diabetes_130_US":
        optimizer = DiabetesOpt()

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
        x_train, x_test = x_train_raw, x_test_raw

    # 6. Export the cleanly processed data 
    loader.export_data_for_rulex(x_train, x_test, y_train, y_test, dataset_name=dataset_name)

    # 7. Execute Strategy
    strategy = SingleOutput(algo='xgb') 
    strategy.execute(x_train, x_test, y_train, y_test)
    
    # 8. Post Processing
    aggregator = PostProcessor()
    aggregator.aggregate_and_clean(database_name=dataset_name)

if __name__ == "__main__":
    main()