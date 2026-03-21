import pandas as pd
import os
import glob

class PostProcessor:
    def aggregate_and_clean(self, database_name=None, base_filename="final_explanation_results.csv"):
        print("\n>>> Starting Aggregation and Cleanup...")
        output_filename = f"{database_name}_{base_filename}"
        # 1. Find all SHAP files (we use these as the anchor)
        shap_files = glob.glob("shap_*.csv")
        
        if not shap_files:
            print("No explanation files found to aggregate.")
            return

        all_data = []
        files_to_delete = []

        for shap_path in shap_files:
            # Determine the corresponding LIME filename
            # Example: shap_mech_cols_OTEK_LANC.csv -> lime_mech_cols_OTEK_LANC.csv
            lime_path = shap_path.replace("shap_", "lime_")
            ablation_path = shap_path.replace("shap_", "ablation_")
            cum_ablation_path = shap_path.replace("shap_", "cum_ablation_")
            
            try:
                # 2. Load Data
                shap_df = pd.read_csv(shap_path)
                
                # Check if corresponding LIME file exists
                if os.path.exists(lime_path):
                    lime_df = pd.read_csv(lime_path)
                    
                    # 3. Merge SHAP and LIME
                    # We merge on 'id' and 'feature'. 
                    # We only take 'lime_value' and 'base_value' (renamed) from the LIME file.
                    merged_df = pd.merge(
                        shap_df, 
                        lime_df[['id', 'feature', 'lime_value', 'base_value']], 
                        on=['id', 'feature'], 
                        how='left',
                        suffixes=('_shap', '_lime')
                    )
                else:
                    print(f"Warning: LIME file not found for {shap_path}. Keeping SHAP only.")
                    merged_df = shap_df
                    merged_df['lime_value'] = 0 # Fill missing LIME with 0
                    merged_df.rename(columns={'base_value': 'base_value_shap'}, inplace=True)
                    
                # Check if corresponding Ablation file exists 
                if os.path.exists(ablation_path):
                    ablation_df = pd.read_csv(ablation_path)
                    merged_df = pd.merge(
                        merged_df, 
                        ablation_df[['id', 'feature', 'ablation_value']], 
                        on=['id', 'feature'], 
                        how='left'
                    )
                else:
                    print(f"Warning: Ablation file not found for {shap_path}.")
                    merged_df['ablation_value'] = 0
                
                # Check if corresponding Cumulative Ablation file exists 
                if os.path.exists(cum_ablation_path):
                    cum_ablation_df = pd.read_csv(cum_ablation_path)
                    merged_df = pd.merge(
                        merged_df, 
                        cum_ablation_df[['id', 'feature', 'cum_ablation_value', "original_prediction", "current_prediction"]], 
                        on=['id', 'feature'], 
                        how='left'
                    )
                    files_to_delete.append(cum_ablation_path) # Mark for deletion
                else:
                    print(f"Warning: Cum Ablation file not found for {shap_path}.")
                    merged_df['cum_ablation_value'] = 0
                    
                # 4. Extract Context/Target Name from filename
                # Remove "shap_" prefix and ".csv" suffix
                context_name = os.path.basename(shap_path).replace("shap_", "").replace(".csv", "")
                merged_df['Target_Context'] = context_name
                
                all_data.append(merged_df)

                # 5. Delete Original Files
                files_to_delete.append(shap_path)
                if os.path.exists(lime_path):
                    files_to_delete.append(lime_path)
                
                if os.path.exists(ablation_path): 
                    files_to_delete.append(ablation_path)
                print(f"Processed and marked for deletion: {context_name}")
            except Exception as e:
                print(f"Error processing {shap_path}: {e}")

        # 6. Save Final Master File
        if all_data:
            final_df = pd.concat(all_data, ignore_index=True)
            
            # Reorder columns for readability
            cols = ['id', 'Target_Context', 'feature', 'feature_value', 
                    'shap_value', 'lime_value', 'ablation_value', 'cum_ablation_value', 
                    'original_prediction', 'current_prediction', 'base_value_shap', 'base_value_lime']
            # Only select columns that actually exist in the dataframe
            final_cols = [c for c in cols if c in final_df.columns]
            final_df = final_df[final_cols]
            
            final_df.to_csv(output_filename, index=False)
            print(f"\nSUCCESS: All explanations merged into '{output_filename}'.")
            
            print("Cleaning up temporary files...")
            for file_path in files_to_delete:
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Could not delete {file_path}: {e}")
            print("Cleanup complete.")
        else:
            print("No data collected.")