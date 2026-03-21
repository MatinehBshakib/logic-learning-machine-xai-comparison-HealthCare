import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from Explainability import Explainability
from sklearn.metrics import accuracy_score
from sklearn.multioutput import MultiOutputClassifier
import pandas as pd
import numpy as np

class BaseStrategy(Explainability):
      def execute(self, x_train, x_test, y_train, y_test):
            raise NotImplementedError()
class SingleOutput(BaseStrategy):
      def __init__(self, algo='rf', scale_pos_weight=1.0):
            self.algo = algo
            self.scale_pos_weight = scale_pos_weight
            
      def execute(self, x_train, x_test, y_train, y_test):
            # Ensure y is 1D for Single Output
            if isinstance(y_train, pd.DataFrame):
                  y_train = y_train.iloc[:, 0]
            if isinstance(y_test, pd.DataFrame):
                  y_test = y_test.iloc[:, 0]
            class_names = ["0", "1"]
            #Train the model 
            if self.algo == 'xgb':
                  clf = xgb.XGBClassifier(eval_metric='logloss', random_state=42, scale_pos_weight=self.scale_pos_weight)
            else:
                  clf = RandomForestClassifier(class_weight='balanced', random_state=42)
            clf.fit(x_train, y_train)
            print(f"Training Features: {x_train.columns.tolist()}")
            print(f"Model Accuracy: {clf.score(x_test, y_test):.4f}")
            target_col = y_train.name if hasattr(y_train, 'name') and y_train.name else "target"
            self.run_shap(clf, x_train, x_test, output_filename=f"shap_{target_col}.csv")
            self.run_lime(clf, x_train, x_test, class_names, output_filename=f"lime_{target_col}.csv")
            ablation_df = self.run_ablation(clf, x_train, x_test, output_filename=f"ablation_{target_col}.csv")
            self.run_cumulative_ablation(clf, x_train, x_test, ablation_df, output_filename=f"cum_ablation_{target_col}.csv")
            return clf
class HierarchicalStrategy(BaseStrategy):
      def __init__(self, group_mapping, algo='xgb'):
            self.group_mapping = group_mapping
            self.algo = algo
            
      def execute(self, x_train, x_test, y_train, y_test):
            if not isinstance(y_train, pd.DataFrame):
                  raise ValueError("Target y must be a DataFrame for Hierarchical Strategy")    
            #Split the data
            results = {}
            for category, subtypes in self.group_mapping.items():
                  print(f"\n>>> PROCESSING FLOW: {category}")
                  valid_subtypes = [c for c in subtypes if c in y_train.columns]
                  if not valid_subtypes:
                        print(f"Skipping {category}: columns not found in dataset")
                        continue
                  
                  #Level 1: Gatekeeper
                  y_train_gate = y_train[valid_subtypes].max(axis=1)
                  y_test_gate  = y_test[valid_subtypes].max(axis=1)

                  print(f"Training Gatekeeper for {category}...")
                  
                  if self.algo == 'xgb':
                        gate_model = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
                  else:
                        gate_model = RandomForestClassifier(class_weight='balanced', random_state=42)
                  
                  #Evaluate 
                  gate_model.fit(x_train, y_train_gate)
                  gate_pred = gate_model.predict(x_test)
                  print(f"Gatekeeper Accuracy: {accuracy_score(y_test_gate, gate_pred):.4f}")
                  
                  #Level 2: Specialist 
                  mask_train = y_train_gate == 1
                  x_spec_train = x_train[mask_train] # Select rows where gatekeeper predicts 1
                  y_spec_train = y_train.loc[mask_train, valid_subtypes] # Corresponding subtypes
                  
                  spec_model = None
                  if len(x_spec_train) > 5:
                        
                        if self.algo == 'xgb':
                              base = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
                        else:
                              base = RandomForestClassifier(class_weight='balanced', random_state=42)
                        spec_model = MultiOutputClassifier(base)
                        spec_model.fit(x_spec_train, y_spec_train)
                  else:
                        print(f"Warning: No positive training examples for {category}.")
                  
                  #Evaluate Specialist
                  #conditional prediction on gatekeeper positive
                  final_pred = pd.DataFrame(0, index=y_test.index, columns=valid_subtypes)
                  pos_indices = np.where(gate_pred == 1)[0]

                  if len(pos_indices) > 0 and spec_model is not None:
                        spec_pred = spec_model.predict(x_test.iloc[pos_indices])
                        final_pred.iloc[pos_indices] = spec_pred
                        
                        x_test_spec = x_test.iloc[pos_indices] # Subset of test data relevant to specialist
                        # Iterate through each sub-category column and its corresponding trained estimator
                        for idx, sub_col in enumerate(valid_subtypes):
                              estimator = spec_model.estimators_[idx] # Extract the specific model for this sub-category
                              print(f"Visualizing SHAP for Specialist Subtype {sub_col}...")
                              self.run_shap(estimator, 
                                            x_spec_train, 
                                            x_test_spec, 
                                            output_filename=f"shap_{category}_{sub_col}.csv")
                              print(f"Visualizing LIME for Specialist Subtype {sub_col}...")
                              sub_class_names = [f"No_{sub_col}", sub_col]
                              self.run_lime(estimator, 
                                            x_spec_train, 
                                            x_test_spec, 
                                            class_names= sub_class_names,
                                            output_filename=f"lime_{category}_{sub_col}.csv")
                              ablation_df = self.run_ablation(estimator, 
                                                              x_spec_train, 
                                                              x_test_spec, 
                                                              output_filename=f"ablation_{category}_{sub_col}.csv",
                                                              n_samples=15)
                              self.run_cumulative_ablation(estimator, 
                                                           x_spec_train,
                                                           x_test_spec,
                                                           ablation_df, 
                                                           output_filename=f"cum_ablation_{category}_{sub_col}.csv",
                                                           n_samples=15)
                  else:
                        print(f"Warning: No positive predictions from Gatekeeper for {category}, skipping Specialist evaluation.")
                  results[category] = (gate_model, spec_model)
            return results

class MultiLabelStrategy(BaseStrategy):
      def __init__(self, algo='xgb'):
            self.algo = algo
            
      def execute(self, x_train, x_test, y_train, y_test):
            # Validation: Ensure y is a multi-column DataFrame
            if not isinstance(y_train, pd.DataFrame) or y_train.shape[1] < 2:
                  raise ValueError("MultiLabelStrategy requires a multi-column DataFrame as target y.")
            
            print(f"Training 'Flat' Multi-Label Model on targets: {y_train.columns.tolist()}")
            
            # Define Base Model
            if self.algo == 'xgb':
                  base = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
            else:
                  base = RandomForestClassifier(class_weight='balanced', random_state=42)
            
            # Wrap in MultiOutputClassifier (Fits one model per target column)
            clf = MultiOutputClassifier(base)
            clf.fit(x_train, y_train)
            
            # global accuracy (subset accuracy: requires all labels for a row to be correct)
            global_acc = clf.score(x_test, y_test)
            print(f"Global Subset Accuracy: {global_acc:.4f}")
            # MultiOutputClassifier stores individual models in clf.estimators_
            for i, col_name in enumerate(y_train.columns):
                  estimator = clf.estimators_[i]
                  
                  print(f"\n>>> Explaining Target: {col_name}")
                  
                  # Calculate individual accuracy for this specific target
                  y_test_col = y_test.iloc[:, i]
                  y_pred_col = estimator.predict(x_test)
                  acc = accuracy_score(y_test_col, y_pred_col)
                  print(f"Accuracy for {col_name}: {acc:.4f}")
                  
                  # We pass the specific estimator for this column, not the whole wrapper
                  self.run_shap(estimator, x_train, x_test, 
                                output_filename=f"shap_flat_{col_name}.csv")
                  class_names = [f"No_{col_name}", str(col_name)]
                  self.run_lime(estimator, x_train, x_test, class_names, 
                                output_filename=f"lime_flat_{col_name}.csv")
                  ablation_df = self.run_ablation(estimator, x_train, x_test,
                                                  output_filename=f"ablation_flat_{col_name}.csv")
                  self.run_cumulative_ablation(estimator, x_train, x_test,
                                               ablation_df, output_filename=f"cum_ablation_flat_{col_name}.csv")   
                  
            return clf
               