from Load import LoadData
from sklearn.model_selection import train_test_split 
from Strategy import MultiLabelStrategy
from sklearn.utils import shuffle
import pandas as pd
from PostProcessor import PostProcessor

def main():
    loader = LoadData()
    dataset_name = "Cervical_Cancer"
    url = "risk_factors_cervical_cancer.csv"  
    target_list = ["Hinselmann", "Schiller", "Citology", "Biopsy"]
    
    # 1. Load and Shuffle Data
    X, y = loader.load_file(file_path=url, target_cols=target_list)
    
    # 2. Split Data 
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    x_train = x_train.reset_index(drop=True)
    x_test= x_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    
    # 3. Ensure fallback safety (Converts everything to numbers safely)
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