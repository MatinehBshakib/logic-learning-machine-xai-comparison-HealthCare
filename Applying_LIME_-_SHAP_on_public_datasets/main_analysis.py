from Analysis import XAIComparativeAnalysis
from Visualization import generate_all_plots
import os

def main():
      analysis = XAIComparativeAnalysis()
      base_folder = 'outputs'
      files_to_process = [
      'Breast_Cancer.csv', 
      'CDC_Diabetes.csv',
      'Cervical_Cancer.csv', 
      'Diabetes_130_US.csv',
      'Diabetic_Retinopathy_Debrecen.csv',
      'Glioma_Grading_Clinical.csv',
      'Hepatit_C_Virus.csv',
      'Myocardial_Infarction.csv',
      'Obesity_level.csv',
      ]
      files_to_process = [os.path.join(base_folder, f) for f in files_to_process]
      # 2. Run the Loop
      for f in files_to_process:
            analysis.execute_analysis(f) 

      # 3. Save the Final Table
      analysis.save_final_table(output_folder=base_folder)

      # 4. Generate all visual plots
      generate_all_plots(
          summary_csv='outputs/final_xai_summary.csv',
          output_folder='outputs/figures'
      )

if __name__ == "__main__":
    main()