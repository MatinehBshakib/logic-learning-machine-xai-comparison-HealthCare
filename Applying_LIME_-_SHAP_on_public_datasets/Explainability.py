from lime import lime_tabular
import shap
import pandas as pd
import numpy as np

class Explainability:
      
      def run_lime(self,clf, x_train, x_test, class_names, output_filename="lime_explanation_results.csv"):
            def predict_fn_wrapper(data_numpy):
                  if len(data_numpy.shape) == 1:
                        data_numpy = data_numpy.reshape(1, -1)
                  data_df = pd.DataFrame(data_numpy, columns=x_train.columns)
                  return clf.predict_proba(data_df)
            
            #initialize LIME explainer
            explainer = lime_tabular.LimeTabularExplainer(
                  training_data=x_train.values,
                  feature_names=x_train.columns.tolist(),
                  class_names=class_names,
                  mode='classification',
                  discretize_continuous=True
            )
            #generate LIME explanations for each instance in the test set
            lime_rows = []
            n_features = len(x_train.columns)
            total_instances = len(x_test)
            
            #Iterate through each instance in the test set
            for i in range(total_instances):
                  try:
                        exp= explainer.explain_instance(
                              data_row=x_test.values[i],
                              predict_fn=predict_fn_wrapper,
                              labels=[1], # Assuming binary classification with positive class as 1
                              num_features=n_features,
                              random_state=42
                        )  
                        #extract values
                        base_value = exp.intercept[1]  # Base value for positive class
                        local_weight = dict(exp.local_exp[1])  # Local weights for positive class
                        current_id = x_test.index[i]  # Get the original index of the instance
                  
                        for feat_idx, feat_name in enumerate(x_train.columns):
                              feat_val = x_test.values[i][feat_idx]
                              lime_value = local_weight.get(feat_idx, 0.0)
                              lime_rows.append({
                                    "id": current_id,
                                    "feature": feat_name,
                                    "feature_value": feat_val,
                                    "base_value": base_value,
                                    "lime_value": lime_value
                              })
                  except Exception as e:
                        print(f"LIME failed for instance {i}: {e}")
                        continue
            #convert to dataframe and save to csv
            lime_df = pd.DataFrame(lime_rows)
            if not lime_df.empty:
                  lime_df["feature_lower"] = lime_df["feature"].str.lower() # Create a temporary column that is all lowercase just for sorting
                  sort_lime_df = lime_df.sort_values(by=["id", "feature_lower"], ascending=[True, True])
                  sort_lime_df = sort_lime_df.drop(columns=["feature_lower"])
                  sort_lime_df.to_csv(output_filename, index=False)
                  print(f"LIME explanations saved to {output_filename}")
                  return sort_lime_df
            else:
                  print("No LIME explanations were generated.")
                  return pd.DataFrame()  # Return empty DataFrame if no explanations were generated
                        
      def run_shap(self,clf, x_train, x_test, output_filename="shap_explanation_results.csv"):
            explainer= shap.TreeExplainer(clf) # Use TreeExplainer for tree-based models
            shap_values_all= explainer(x_test)
            # The output shape is typically (rows, features, classes). 
            # We select [:, :, 1] to get the explanation for the "Positive" class.
            if len(shap_values_all.values.shape) == 2: #handle xgboost single output case
                  shap_values = shap_values_all.values
                  base_values = shap_values_all.base_values
            else:
                  # handle random forest multi-class case
                  shap_values = shap_values_all.values[:,:,1]
                  base_values = shap_values_all.base_values[:,1] # Base values for positive class
            # Normalize base_values to be a list/array of length N
            if not hasattr(base_values, "__iter__"):
                  base_values = np.repeat(base_values, len(x_test))
            
            shap_rows = []
            feature_names = x_train.columns.tolist()
            
            for i in range(len(x_test)):
                  current_id = x_test.index[i]  # Get the original index of the instance
                  current_base_value = base_values[i]
                  # Iterate through each feature
                  for feat_idx, feat_name in enumerate(feature_names):
                        shap_rows.append({
                              "id": current_id,
                              "feature": feat_name,
                              "feature_value": x_test.iloc[i, feat_idx],
                              "base_value": current_base_value,
                              "shap_value": shap_values[i,feat_idx]
                        })
            #convert to dataframe
            shap_df = pd.DataFrame(shap_rows)
            shap_df["feature_lower"] = shap_df["feature"].str.lower() # Create a temporary column that is all lowercase just for sorting
            sort_shap_df = shap_df.sort_values(by=["id", "feature_lower"], ascending=[True, True])
            sort_shap_df = sort_shap_df.drop(columns=["feature_lower"])
            sort_shap_df.to_csv(output_filename, index=False)
            print(f"SHAP explanations saved to {output_filename}.")
            return sort_shap_df