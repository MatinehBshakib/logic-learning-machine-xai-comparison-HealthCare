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
                  # Ensure the input data is in the same format as the training data
                  data_df = data_df.astype(x_train.dtypes.to_dict()) 
                  return clf.predict_proba(data_df)
            
            #initialize LIME explainer
            explainer = lime_tabular.LimeTabularExplainer(
                  training_data=x_train.values,
                  feature_names=x_train.columns.tolist(),
                  class_names=class_names,
                  mode='classification',
                  discretize_continuous=True,
                  random_state=42           
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
                              num_features=n_features
                        )  
                        #extract values
                        # If 1 is missing, base_value becomes 0.0
                        base_value = exp.intercept.get(1, 0.0) 
                        # If 1 is missing, it returns an empty list [], which dict() turns into an empty {}
                        local_weight = dict(exp.local_exp.get(1, []))
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
                  raw_base_values = shap_values_all.base_values
            else:
                  # handle random forest multi-class case
                  shap_values = shap_values_all.values[:,:,1]
                  bv = shap_values_all.base_values
                  # Normalize base_values to be a list/array of length N
                  raw_base_values = bv[:, 1] if len(bv.shape) == 2 else bv[1] 
            if np.isscalar(raw_base_values) or (isinstance(raw_base_values, np.ndarray) and raw_base_values.size == 1):
                  # Extract the single number and repeat it N times
                  base_values = np.repeat(np.squeeze(raw_base_values), len(x_test))
            else:
                  base_values = raw_base_values       
            
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
      
      def run_ablation(self, clf, x_train, x_test, output_filename="ablation_explanation_results.csv", n_samples=50):
            """
            Perturbation-based explainer using Marginal Background Sampling. 
            Measures the net expected impact by averaging signed differences 
            against random background samples, then taking the absolute magnitude.
            """
            print(f"Running Ablation with {n_samples} background samples per feature...")
            
            try:
                  original_preds = clf.predict_proba(x_test)[:, 1]
            except AttributeError:
                  print("Warning: Model does not support predict_proba. Using predict() instead.")
                  original_preds = clf.predict(x_test)
                  
            ablation_rows = []
            feature_names = x_train.columns.tolist()
            
            try:
                  train_base_value = float(clf.predict_proba(x_train)[:, 1].mean())
            except AttributeError:
                  train_base_value = float(clf.predict(x_train).mean())
            
            for feat_idx, feat_name in enumerate(feature_names):
                  
                  # Array to accumulate SIGNED impacts
                  accumulated_impacts = np.zeros(len(x_test))
                  
                  for _ in range(n_samples):
                        x_test_perturbed = x_test.copy()
                        
                        # Sample random values from the training distribution
                        random_background = x_train[feat_name].sample(n=len(x_test), replace=True).values
                        x_test_perturbed[feat_name] = random_background
                        
                        try:
                              perturbed_preds = clf.predict_proba(x_test_perturbed)[:, 1]
                        except AttributeError:
                              perturbed_preds = clf.predict(x_test_perturbed)
                              
                        # Accumulate signed differences (original - perturbed)
                        accumulated_impacts += (original_preds - perturbed_preds)
                  
                  # Take the absolute value of the AVERAGE impact
                  avg_impacts = np.abs(accumulated_impacts / n_samples)
                  
                  for i in range(len(x_test)):
                        ablation_rows.append({
                              "id": x_test.index[i],
                              "feature": feat_name,
                              "feature_value": x_test.iloc[i, feat_idx],
                              "base_value": train_base_value, 
                              "ablation_value": avg_impacts[i]
                        })
                        
            ablation_df = pd.DataFrame(ablation_rows)
            ablation_df["feature_lower"] = ablation_df["feature"].str.lower() 
            sort_ablation_df = ablation_df.sort_values(by=["id", "feature_lower"], ascending=[True, True])
            sort_ablation_df = sort_ablation_df.drop(columns=["feature_lower"])
            
            sort_ablation_df.to_csv(output_filename, index=False)
            print(f"Ablation explanations saved to {output_filename}.")
            
            return sort_ablation_df
      
      def run_cumulative_ablation(self, clf, x_train, x_test, ablation_df, output_filename="cum_ablation_results.csv", n_samples=50):
            """
            Cumulative Ablation: Takes the results from standard ablation, 
            calculates the global average importance to sort features from least to most,
            and cumulatively masks them to measure the drop jumps.
            """
            print(f"Running Cumulative Ablation with {n_samples} background samples...")
            
            # 1. Determine "Least to Most" Order using STANDARD ABLATION results
            # Group by feature and get the average importance across all rows
            global_ablation = ablation_df.groupby('feature')['ablation_value'].mean()
            
            # Sort ascending (least important first) and extract the feature names as a list
            least_to_most_features = global_ablation.sort_values(ascending=True).index.tolist()
            
            # 2. Get original predictions and base value
            try:
                  original_preds = clf.predict_proba(x_test)[:, 1]
                  train_base_value = float(clf.predict_proba(x_train)[:, 1].mean())
            except AttributeError:
                  print("Warning: Model does not support predict_proba. Using predict() instead.")
                  original_preds = clf.predict(x_test)
                  train_base_value = float(clf.predict(x_train).mean())
                  
            cumulative_rows = []
            features_to_mask = []
            
            # Track the 'previous' prediction to measure the jump caused by the NEW feature
            prev_preds = original_preds.copy()
            
            # 3. The Cumulative Loop
            for feat_name in least_to_most_features:
                  features_to_mask.append(feat_name) # Add feature to the cumulative mask list
                  
                  accumulated_preds = np.zeros(len(x_test))
                  
                  # Background sampling for the masked subset
                  for _ in range(n_samples):
                        x_test_perturbed = x_test.copy()
                        
                        # Mask ALL features currently in the list
                        for f in features_to_mask:
                              x_test_perturbed[f] = x_train[f].sample(n=len(x_test), replace=True).values
                              
                        try:
                              perturbed_preds = clf.predict_proba(x_test_perturbed)[:, 1]
                        except AttributeError:
                              perturbed_preds = clf.predict(x_test_perturbed)
                              
                        accumulated_preds += perturbed_preds
                        
                  # Average the predictions across the samples
                  avg_perturbed_preds = accumulated_preds / n_samples
                  
                  # The "Importance" is how much the prediction dropped specifically 
                  # at the moment THIS feature was added to the masking list.
                  jump_in_prediction = np.abs(prev_preds - avg_perturbed_preds)
                  
                  # Update the previous predictions for the next loop
                  prev_preds = avg_perturbed_preds
                  
                  # 4. Store results
                  for i in range(len(x_test)):
                        cumulative_rows.append({
                              "id": x_test.index[i],
                              "feature": feat_name,
                              "feature_value": x_test.iloc[i, x_train.columns.get_loc(feat_name)],
                              "base_value": train_base_value, 
                              "cum_ablation_value": jump_in_prediction[i] # Saved as cum_ablation_value
                        })
                        
            # 5. Format and Save
            cumulative_df = pd.DataFrame(cumulative_rows)
            cumulative_df["feature_lower"] = cumulative_df["feature"].str.lower() 
            sort_cumulative_df = cumulative_df.sort_values(by=["id", "feature_lower"], ascending=[True, True])
            sort_cumulative_df = sort_cumulative_df.drop(columns=["feature_lower"])
            
            sort_cumulative_df.to_csv(output_filename, index=False)
            print(f"Cumulative Ablation explanations saved to {output_filename}.")
            
            return sort_cumulative_df