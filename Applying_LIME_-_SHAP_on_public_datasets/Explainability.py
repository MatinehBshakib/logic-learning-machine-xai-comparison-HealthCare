from lime import lime_tabular
import shap
import pandas as pd
import numpy as np

class Explainability:
      
      def run_lime(self,clf, x_train, x_test, class_names, output_filename="lime_explanation_results.csv"):
            # This wrapper is needed because LIME expects a function that takes a 2D numpy array and returns a 2D array of probabilities.
            def predict_fn_wrapper(data_numpy):
                  if len(data_numpy.shape) == 1:             
                        data_numpy = data_numpy.reshape(1, -1)
                  data_df = pd.DataFrame(data_numpy, columns=x_train.columns)
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
            
            np.random.seed(42)
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
            
            x_test_values = x_test.values 
            x_test_indices = x_test.index.tolist()
            
            for i in range(len(x_test)):
                  current_id = x_test_indices[i]
                  current_base_value = base_values[i]
                  # Iterate through each feature
                  for feat_idx, feat_name in enumerate(feature_names):
                        shap_rows.append({
                              "id": current_id,
                              "feature": feat_name,
                              "feature_value": x_test_values[i, feat_idx],
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
            print(f"Running Optimized Ablation with {n_samples} background samples per feature...")
            
            try:
                  original_preds = clf.predict_proba(x_test)[:, 1]
                  train_base_value = float(clf.predict_proba(x_train)[:, 1].mean())
            except AttributeError:
                  print("Warning: Model does not support predict_proba. Using predict() instead.")
                  original_preds = clf.predict(x_test)
                  train_base_value = float(clf.predict(x_train).mean())
                  
            ablation_rows = []
            feature_names = x_train.columns.tolist()
            num_test_instances = len(x_test)
            
            # --- THE SPEED OPTIMIZATION: Pre-allocate the batch dataset ---
            # We copy the patients n_samples times into one giant dataframe
            x_test_repeated = pd.DataFrame(
                  np.tile(x_test.values, (n_samples, 1)), 
                  columns=x_test.columns
            )
            
            for feat_idx, feat_name in enumerate(feature_names):
                  x_test_batch = x_test_repeated.copy()
                  
                  # Sample random values for the entire batch at once
                  bg_samples = x_train[feat_name].sample(n=len(x_test_batch), replace=True, random_state=42).values
                  x_test_batch[feat_name] = bg_samples
                  
                  # Predict the entire batch in ONE model call
                  try:
                        batch_preds = clf.predict_proba(x_test_batch)[:, 1]
                  except AttributeError:
                        batch_preds = clf.predict(x_test_batch)
                        
                  # Reshape and get the average perturbed prediction for each patient
                  avg_perturbed_preds = batch_preds.reshape(n_samples, num_test_instances).mean(axis=0)
                  
                  # The impact is the absolute difference between original and the new average
                  avg_impacts = np.abs(original_preds - avg_perturbed_preds)
                  
                  for i in range(num_test_instances):
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
            print(f"Running Optimized Cumulative Ablation with {n_samples} background samples...")
            
            # 1. Determine "Least to Most" Order
            global_ablation = ablation_df.groupby('feature')['ablation_value'].mean()
            least_to_most_features = global_ablation.sort_values(ascending=True).index.tolist()
            
            # 2. Get original predictions and base value
            try:
                  original_preds = clf.predict_proba(x_test)[:, 1]
                  train_base_value = float(clf.predict_proba(x_train)[:, 1].mean())
            except AttributeError:
                  print("Warning: Model does not support predict_proba. Using predict() instead.")
                  original_preds = clf.predict(x_test)
                  train_base_value = float(clf.predict(x_train).mean())
                  
            # Instead of a loop, we tile x_test to create a single batch matrix of size (N * n_samples)
            num_test_instances = len(x_test)
            x_test_repeated = pd.DataFrame(
                  np.tile(x_test.values, (n_samples, 1)), 
                  columns=x_test.columns
            )

            cumulative_rows = []
            features_to_mask = []
            prev_preds = original_preds.copy()
            
            # 3. The Cumulative Loop
            for feat_name in least_to_most_features:
                  features_to_mask.append(feat_name) 
                  
                  # --- NEW SPEED OPTIMIZATION: Vectorized Masking & Batch Prediction ---
                  x_test_batch = x_test_repeated.copy()
                  
                  # Sample enough random background rows for the entire batch in one go
                  bg_samples = x_train[features_to_mask].sample(n=len(x_test_batch), replace=True, random_state=42).values
                  
                  # Overwrite all masked features simultaneously 
                  x_test_batch[features_to_mask] = bg_samples
                  
                  # Predict the entire batch of (N * n_samples) in a single model call!
                  try:
                        batch_preds = clf.predict_proba(x_test_batch)[:, 1]
                  except AttributeError:
                        batch_preds = clf.predict(x_test_batch)
                        
                  # Reshape the flat predictions array into a grid of (n_samples, N) and calculate the mean column-wise
                  avg_perturbed_preds = batch_preds.reshape(n_samples, num_test_instances).mean(axis=0)
                  
                  # The jump caused by this specific feature
                  jump_in_prediction = np.abs(prev_preds - avg_perturbed_preds)
                  prev_preds = avg_perturbed_preds
                  
                  # 4. Store results
                  for i in range(num_test_instances):
                        cumulative_rows.append({
                              "id": x_test.index[i],
                              "feature": feat_name,
                              "feature_value": x_test.iloc[i, x_train.columns.get_loc(feat_name)],
                              "base_value": train_base_value, 
                              "original_prediction": original_preds[i],
                              "current_prediction": avg_perturbed_preds[i],
                              "cum_ablation_value": jump_in_prediction[i]
                        })
                        
            # 5. Format and Save
            cumulative_df = pd.DataFrame(cumulative_rows)
            cumulative_df["feature_lower"] = cumulative_df["feature"].str.lower() 
            sort_cumulative_df = cumulative_df.sort_values(by=["id", "feature_lower"], ascending=[True, True])
            sort_cumulative_df = sort_cumulative_df.drop(columns=["feature_lower"])
            
            sort_cumulative_df.to_csv(output_filename, index=False)
            print(f"Cumulative Ablation explanations saved to {output_filename}.")
            
            return sort_cumulative_df