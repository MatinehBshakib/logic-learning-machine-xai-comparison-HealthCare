import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 

from Load import LoadData
from Strategy import MultiLabelStrategy
from PostProcessor import PostProcessor
from PerformanceMetrics import save_performance_metrics_multilabel

np.random.seed(42)
random.seed(42)

def main():
    loader = LoadData()
    dataset_name = "Cervical_Cancer"
    url = "risk_factors_cervical_cancer.csv"  
    target_list = ["Hinselmann", "Schiller", "Citology", "Biopsy"]
    
    # 1. Load and Shuffle Data
    X, y = loader.load_file(file_path=url, target_cols=target_list)
    
    # 2. Split Data 
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    x_train = loader.fit_imputer(x_train)   # Learns mean/mode from train ONLY
    x_test = loader.apply_imputer(x_test)  # Applies those same statistics to test
    
    # Re-number Train indices to start at 0
    x_train = x_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    
    # Make Test indices start exactly where Train left off!
    start_idx = len(x_train)
    x_test.index = range(start_idx, start_idx + len(x_test))
    y_test.index = range(start_idx, start_idx + len(y_test)) 
    
    # 3. Robust Categorical Encoding & Safety Check
    print(f"\n>>> Applying robust categorical encoding for {dataset_name}...")
    x_train = pd.get_dummies(x_train, drop_first=True, dtype=int)
    x_test = pd.get_dummies(x_test, drop_first=True, dtype=int)
    
    # Align train and test to ensure identical columns
    x_train, x_test = x_train.align(x_test, join='left', axis=1, fill_value=0)
    
    x_train = x_train.apply(pd.to_numeric, errors='coerce').fillna(0)
    x_test = x_test.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    #drop correlated features
    x_train, x_test = loader.drop_correlated(x_train, x_test, threshold=0.90)
    
    # Validate that all data is perfectly numeric before proceeding
    loader.validate_numeric(dataset_name, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)

    # 4. Discretize and Export Cleaned Data for Rulex
    x_train_disc, x_test_disc = loader.discretize_for_rulex(x_train, x_test)
    loader.export_data_for_rulex(x_train_disc, x_test_disc, y_train, y_test, dataset_name=dataset_name)

    # 5. Strategy Execution
    strategy = MultiLabelStrategy(algo='xgb')
    clf = strategy.execute(x_train, x_test, y_train, y_test)
    
    # 6. Save Performance Metrics
    save_performance_metrics_multilabel(
        clf          = clf,
        x_test       = x_test,
        y_test       = y_test,
        dataset_name = dataset_name,
        target_cols  = target_list,
        n_train      = len(x_train),
        output_folder= 'outputs',
    )
    
    # 7. Post Processing
    aggregator = PostProcessor()
    aggregator.aggregate_and_clean(database_name=dataset_name)

if __name__ == "__main__":
    main()