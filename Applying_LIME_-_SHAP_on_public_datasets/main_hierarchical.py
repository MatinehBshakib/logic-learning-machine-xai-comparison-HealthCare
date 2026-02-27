from Load import LoadData
from sklearn.model_selection import train_test_split 
import pandas as pd
from Strategy import HierarchicalStrategy
from Config import MycordinalConfig as config
from PostProcessor import PostProcessor

def main():
    loader = LoadData()
    dataset_name = "Myocardial_Infarction"
    target_list = config.get_all_target_cols()
    
    # 1. Load Data
    X, y = loader.load_link(data_id=46943, target_cols=target_list)
    
    # 2. Split Data (Using train_test_split)
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
    
    # 4. Export Cleaned Data for Rulex
    loader.export_data_for_rulex(x_train, x_test, y_train, y_test, dataset_name=dataset_name)

    # 5. Strategy Execution
    strategy = HierarchicalStrategy(
        group_mapping=config.Hierarchy_mapping,
        algo='xgb'
    )
    strategy.execute(x_train, x_test, y_train, y_test)
    
    # 6. Post Processing
    aggregator = PostProcessor()
    # Explicitly pass the database name so the final file isn't just "None_final_explanation_results.csv"
    aggregator.aggregate_and_clean(database_name=dataset_name) 

if __name__ == "__main__":
    main()