import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
import joblib


df = pd.read_csv("feature_dataset.csv") 


# Feature Engineering
'''
Various feature combinations (e.g., unique_ratio, verb_ratio, entropy, bigram_repeat_ratio) were tested in preliminary experiments
The best-performing combination ('unique_ratio', 'verb_ratio', 'entropy') was selected,
and polynomial interaction features were applied to capture non-linear relationships
'''
selected_features = ['unique_ratio', 'verb_ratio', 'entropy']

poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(df[selected_features])
poly_feature_names = poly.get_feature_names_out(selected_features)

df_poly = pd.concat([
    df[['title', 'paragraph_text', 'ppl', 'generated']].reset_index(drop=True),
    pd.DataFrame(X_poly, columns=poly_feature_names)
], axis=1)

feature_cols = poly.get_feature_names_out(selected_features)
X = df_poly[list(feature_cols) + ['ppl']]
y = df_poly['generated']

# Train/Validation/Test Split
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.15, random_state=42, stratify=y_trainval
)

# Model Training
# Final training performed with the best hyperparameters found via RandomizedSearch
model = CatBoostClassifier(
    random_strength=3,
    learning_rate=0.01,
    l2_leaf_reg=7,
    iterations=1000,
    depth=6,
    bagging_temperature=0,
    verbose=100,
    task_type="CPU"
)

model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)

# Evaluation
val_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

print(f"Validation ROC-AUC: {val_auc:.4f}")
print(f"Test ROC-AUC: {test_auc:.4f}")

# Model Save
joblib.dump(model, "catboost_final_model.pkl")
