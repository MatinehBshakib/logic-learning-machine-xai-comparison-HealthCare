from Analysis import XAIComparativeAnalysis
import os

def main():
      analysis = XAIComparativeAnalysis()
      base_folder = 'outputs'
      files_to_process = [
      'Breast_Cancer.csv', 
      'CDC_Diabetes.csv',
      'Cervical_Cancer.csv', 
      'Diabetes1999-2008.csv',
      'Diabetic_Retinopathy_Debrecen.csv',
      'Glioma_Grading_Clinical.csv',
      'Hepatit_C_Virus.csv',
      'Myocardial_Infraction.csv',
      'Obesity_level.csv',
      ]
      files_to_process = [os.path.join(base_folder, f) for f in files_to_process]
      # 2. Run the Loop
      for f in files_to_process:
            analysis.execute_analysis(f) 

      # 3. Save the Final Table
      analysis.save_final_table(output_folder=base_folder)

if __name__ == "__main__":
    main()