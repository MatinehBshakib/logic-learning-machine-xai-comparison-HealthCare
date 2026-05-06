import pandas as pd
import numpy as np
from scipy.stats import spearmanr, kendalltau
import os
from Visualization import plot_cumulative_ablation_per_dataset

class XAIComparativeAnalysis:
    def __init__(self, figures_folder: str = 'outputs/figures'):
        # This list will hold exactly ONE row per dataset (the aggregated result)
        self.dataset_summaries = []
        self.figures_folder = figures_folder

    def execute_analysis(self, file_path, top_k=5):
        print(f"\n>>> Processing Analysis for File: {file_path}")
        
        # 1. Load Data
        if not os.path.exists(file_path):
            print(f"Error: File '{file_path}' not found.")
            return None
            
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error reading file: {e}")
            return None

        # 2. Strict Column Verification
        required_cols = ['Attribute', 'Rulex', 'SHAP', 'LIME', 'Ablation']
        missing = [col for col in required_cols if col not in df.columns]
        
        if missing:
            print(f"Error: Missing required columns: {missing}")
            return None

        # 3. Data Mapping (Trusts your values)
        try:
            df['Rulex_Val'] = df['Rulex']
            df['SHAP_Val']  = df['SHAP']
            df['LIME_Val']  = df['LIME']
            df['Ablation_Val'] = df['Ablation']

        except Exception as e:
            print(f"Error during column mapping: {e}")
            return None

        # 4. Target Setup
        if 'Target' in df.columns:
            df['Target_Group'] = df['Target']
        else:
            df['Target_Group'] = 'Single_Target'

        # 5. Analysis Loop (Calculate metrics for each target)
        target_results = []
        for target in df['Target_Group'].unique():
            subset = df[df['Target_Group'] == target].copy()
            
            # A. Prepare Data 
            global_imp = subset[['Attribute', 'Rulex_Val', 'SHAP_Val', 'LIME_Val', 'Ablation_Val']].set_index('Attribute')
        
            # B. Ranking (Add Epsilon to break ties deterministically)
            np.random.seed(42) # Ensure reproducible tie-breaking
            epsilon = 1e-12
            
            # Add tiny noise to break ties
            rulex_noisy = global_imp['Rulex_Val'] + np.random.rand(len(global_imp)) * epsilon
            shap_noisy  = global_imp['SHAP_Val']  + np.random.rand(len(global_imp)) * epsilon
            lime_noisy  = global_imp['LIME_Val']  + np.random.rand(len(global_imp)) * epsilon
            ablation_noisy = global_imp['Ablation_Val'] + np.random.rand(len(global_imp)) * epsilon
                  
            # Rank with method='first' to guarantee entirely unique integer ranks
            global_imp['Rulex_Rank'] = rulex_noisy.rank(method='first', ascending=False)
            global_imp['SHAP_Rank']  = shap_noisy.rank(method='first', ascending=False)
            global_imp['LIME_Rank']  = lime_noisy.rank(method='first', ascending=False)
            global_imp['Ablation_Rank'] = ablation_noisy.rank(method='first', ascending=False)
            
            # C. Metrics: Spearman
            rho_rs, _ = spearmanr(global_imp['Rulex_Rank'], global_imp['SHAP_Rank'])
            rho_rl, _ = spearmanr(global_imp['Rulex_Rank'], global_imp['LIME_Rank'])
            rho_sl, _ = spearmanr(global_imp['SHAP_Rank'],  global_imp['LIME_Rank']) 
            
            rho_ra, _ = spearmanr(global_imp['Rulex_Rank'], global_imp['Ablation_Rank'])
            rho_sa, _ = spearmanr(global_imp['SHAP_Rank'],  global_imp['Ablation_Rank']) 
            rho_la, _ = spearmanr(global_imp['LIME_Rank'],  global_imp['Ablation_Rank'])
 
            # D. Metrics: Kendall Tau-b (Uses RAW values to utilize native tie-penalties)
            tau_rs, _ = kendalltau(global_imp['Rulex_Val'], global_imp['SHAP_Val'])
            tau_rl, _ = kendalltau(global_imp['Rulex_Val'], global_imp['LIME_Val'])
            tau_sl, _ = kendalltau(global_imp['SHAP_Val'],  global_imp['LIME_Val'])
            
            tau_ra, _ = kendalltau(global_imp['Rulex_Val'], global_imp['Ablation_Val'])
            tau_sa, _ = kendalltau(global_imp['SHAP_Val'],  global_imp['Ablation_Val'])
            tau_la, _ = kendalltau(global_imp['LIME_Val'],  global_imp['Ablation_Val'])
            

            tau_ra = 0 if np.isnan(tau_ra) else tau_ra
            tau_sa = 0 if np.isnan(tau_sa) else tau_sa
            tau_la = 0 if np.isnan(tau_la) else tau_la
            
            # Safety checks for NaNs (occurs if an algorithm gives 0 to literally every feature)
            tau_rs = 0 if np.isnan(tau_rs) else tau_rs
            tau_rl = 0 if np.isnan(tau_rl) else tau_rl
            tau_sl = 0 if np.isnan(tau_sl) else tau_sl
            
            
            #We sort by index (Attribute name) to ensure the 'Top K' is deterministic during ties
            def get_top_k_set(df, rank_col, k): 
                return set(df.sort_values(by=[rank_col, 'Attribute'], ascending=[True, True]).head(k)['Attribute'])
            # E. Metrics: Jaccard
            top_rulex = get_top_k_set(global_imp.reset_index(), 'Rulex_Rank', top_k)
            top_shap  = get_top_k_set(global_imp.reset_index(), 'SHAP_Rank', top_k)
            top_lime  = get_top_k_set(global_imp.reset_index(), 'LIME_Rank', top_k)
            top_ablation = get_top_k_set(global_imp.reset_index(), 'Ablation_Rank', top_k)
            
            def calc_jaccard(s1, s2):
                return len(s1.intersection(s2)) / len(s1.union(s2)) if len(s1.union(s2)) > 0 else 0
            
            target_results.append({
                'Spearman (Rulex-SHAP)': rho_rs,
                'Spearman (Rulex-LIME)': rho_rl,
                'Spearman (SHAP-LIME)': rho_sl,
                'Spearman (Rulex-Ablat)': rho_ra,
                'Spearman (SHAP-Ablat)': rho_sa,
                'Spearman (LIME-Ablat)': rho_la,
                'Kendall (Rulex-SHAP)': tau_rs,     
                'Kendall (Rulex-LIME)': tau_rl,     
                'Kendall (SHAP-LIME)': tau_sl, 
                'Kendall (Rulex-Ablat)': tau_ra,     
                'Kendall (SHAP-Ablat)': tau_sa, 
                'Kendall (LIME-Ablat)': tau_la, 
                'Jaccard (Rulex-SHAP)': calc_jaccard(top_rulex, top_shap),
                'Jaccard (Rulex-LIME)': calc_jaccard(top_rulex, top_lime),
                'Jaccard (SHAP-LIME)': calc_jaccard(top_shap, top_lime),
                'Jaccard (Rulex-Ablat)': calc_jaccard(top_rulex, top_ablation),
                'Jaccard (SHAP-Ablat)': calc_jaccard(top_shap, top_ablation),
                'Jaccard (LIME-Ablat)': calc_jaccard(top_lime, top_ablation),
            })

        # 6. INNER AVERAGING (The "Final Output" for this dataset)
        results_df = pd.DataFrame(target_results)
        
        if not results_df.empty:
            # Calculate the mean across all targets for this dataset
            dataset_avg = results_df.mean().to_frame().T.round(2)
            
            # Add the dataset name
            dataset_avg.insert(0, 'Dataset', os.path.basename(file_path))
            
            # Store ONLY this aggregated row for the final table
            self.dataset_summaries.append(dataset_avg)
            
            # Print specifically this dataset's result
            print(dataset_avg.to_string(index=False))
        else:
            print("No results generated.")
        
        # 7. Visualization: Cumulative Ablation Plot for this dataset
        explanation_csv = file_path.replace('.csv', '_final_explanation_results.csv')
        dataset_label   = (
            os.path.basename(file_path)
              .replace('.csv', '')
              .replace('_', ' ')
        )
        plot_cumulative_ablation_per_dataset(
            explanation_csv = explanation_csv,
            dataset_name    = dataset_label,
            output_folder   = self.figures_folder,
            max_display     = 20,   # show at most 20 features (most important ones)
        )

    def save_final_table(self, output_folder='outputs', filename='final_xai_summary.csv'):
        if not self.dataset_summaries:
            print("\nNo datasets processed successfully.")
            return

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        print("\n\n=== FINAL SUMMARY TABLE ===")
        # 1. Combine all dataset rows
        summary_df = pd.concat(self.dataset_summaries, ignore_index=True)
        
        # 2. Calculate the Total Average across all datasets
        total_avg = summary_df.select_dtypes(include=[np.number]).mean().to_frame().T.round(2)
        total_avg['Dataset'] = 'TOTAL AVERAGE'
        
        # 3. Combine into one final table
        final_table = pd.concat([summary_df, total_avg], ignore_index=True)
        
        # 4. Organize Columns
        cols = ['Dataset'] + [c for c in final_table.columns if c != 'Dataset']
        
        # 5. Print and Save
        print(final_table[cols].to_string(index=False))
        
        output_path = os.path.join(output_folder, filename)
        final_table[cols].to_csv(output_path, index=False)
        print(f"\n>>> Table saved to: {output_path}")

