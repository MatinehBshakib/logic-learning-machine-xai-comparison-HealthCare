from Load import LoadData
from sklearn.model_selection import train_test_split 
from Strategy import MultiLabelStrategy
from sklearn.utils import shuffle
import pandas as pd
from PostProcessor import PostProcessor
from Cervical_Cancer_Optimizer import CervicalCancerOptimizer as CervicalOpt

def main():
    loader = LoadData()
    dataset_name = "Cervical_Cancer"
    url = "risk_factors_cervical_cancer.csv"  
    target_list = ["Hinselmann", "Schiller", "Citology", "Biopsy"]
    
    # 1. Load and Shuffle Data
    X, y = loader.load_file(file_path=url, target_cols=target_list)
    X, y = shuffle(X, y, random_state=42)
    
    # 2. Split Data 
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print(f"\n>>> Routing to optimizer for: {dataset_name}")
    
    # 3. Optimization
    if dataset_name == "Cervical_Cancer":
        opt = CervicalOpt()
        # Pass y=None so it only cleans the features (X) and ignores the multi-targets
        x_train, _ = opt.optimize(x_train, y=None, is_train=True)
        x_test, _ = opt.optimize(x_test, y=None, is_train=False)
        
    else:
        print(f"No specific optimizer found for {dataset_name}. Using raw data.")
        # Ensure fallback safety
        x_train = x_train.apply(pd.to_numeric, errors='coerce').fillna(0)
        x_test = x_test.apply(pd.to_numeric, errors='coerce').fillna(0)

    # 4. Export Cleaned Data for Rulex
    loader.export_data_for_rulex(x_train, x_test, y_train, y_test, dataset_name=dataset_name)

    # 5. Strategy Execution
    strategy = MultiLabelStrategy(algo='xgb')
    strategy.execute(x_train, x_test, y_train, y_test)
    
    # 6. Post Processing
    aggregator = PostProcessor()
    aggregator.aggregate_and_clean(database_name=dataset_name)

if __name__ == "__main__":
    main()