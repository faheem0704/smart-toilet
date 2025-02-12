import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load the dataset
data = pd.read_csv(r"C:\Users\Elite\Desktop\uni\5th sem\ML\smartoiletproject\data\urinalysis_tests.csv")  

# Preprocessing
data['Age'] = data['Age'].apply(lambda x: x if x >= 1 else x / 100)
data['Gender'] = data['Gender'].map({'male': 0, 'female': 1})
data['Diagnosis'] = data['Diagnosis'].map({'NEGATIVE': 0, 'POSITIVE': 1})

numeric_columns = data.select_dtypes(include=[np.number]).columns
categorical_columns = data.select_dtypes(exclude=[np.number]).columns

print("\nNaN Values Before Imputation:\n", data.isna().sum())

imputer_numeric = SimpleImputer(strategy='median')
imputer_categorical = SimpleImputer(strategy='most_frequent')


numeric_columns = numeric_columns[numeric_columns != 'Gender']

data[numeric_columns] = imputer_numeric.fit_transform(data[numeric_columns])
data[categorical_columns] = imputer_categorical.fit_transform(data[categorical_columns])

print("\nNaN Values After Imputation:\n", data.isna().sum())


for col in categorical_columns:
    label_encoder = LabelEncoder()
    data[col] = label_encoder.fit_transform(data[col])


scaler = StandardScaler()
scaled_features = ['Glucose', 'pH', 'Specific Gravity', 'Protein']
data[scaled_features] = scaler.fit_transform(data[scaled_features])


X_classification = data.drop('Diagnosis', axis=1)
y_classification = data['Diagnosis']


X_regression = data.drop(['Protein', 'Specific Gravity'], axis=1)
y_protein = data['Protein']
y_specific_gravity = data['Specific Gravity']


X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_classification, y_classification, test_size=0.2, random_state=42)


X_train_reg, X_test_reg, y_train_protein, y_test_protein = train_test_split(
    X_regression, y_protein, test_size=0.2, random_state=42)
X_train_gravity, X_test_gravity, y_train_gravity, y_test_gravity = train_test_split(
    X_regression, y_specific_gravity, test_size=0.2, random_state=42)

print("\nNaN Values in Training Data:\n", X_train_clf.isna().sum())


feature_names_clf = X_train_clf.columns
feature_names_reg = X_train_reg.columns
joblib.dump(feature_names_clf, "classification_feature_names.joblib")
joblib.dump(feature_names_reg, "regression_feature_names.joblib")


pipeline_clf = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
])

pipeline_reg = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
])

X_train_clf = pipeline_clf.fit_transform(X_train_clf)
X_test_clf = pipeline_clf.transform(X_test_clf)

X_train_reg = pipeline_reg.fit_transform(X_train_reg)
X_test_reg = pipeline_reg.transform(X_test_reg)
X_train_gravity = pipeline_reg.fit_transform(X_train_gravity)
X_test_gravity = pipeline_reg.transform(X_test_gravity)


classification_models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
}

print("\n--- Classification Model Results ---\n")
for model_name, model in classification_models.items():
    model.fit(X_train_clf, y_train_clf)
    y_pred_clf = model.predict(X_test_clf)
    print(f"{model_name}:\n")
    print(classification_report(y_test_clf, y_pred_clf))
    print("Confusion Matrix:\n", confusion_matrix(y_test_clf, y_pred_clf))
    print("-" * 40)
    joblib.dump(model, f"{model_name}_classification_model.joblib")


regression_models = {
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42)
}

print("\n--- Regression Model Results ---\n")


print("\nProtein Level Prediction:\n")
for model_name, model in regression_models.items():
    model.fit(X_train_reg, y_train_protein)
    y_pred_protein = model.predict(X_test_reg)
    print(f"{model_name}: MAE = {mean_absolute_error(y_test_protein, y_pred_protein):.3f}, R2 = {r2_score(y_test_protein, y_pred_protein):.3f}")
    joblib.dump(model, f"{model_name}_protein_regression_model.joblib")


print("\nSpecific Gravity Prediction:\n")
for model_name, model in regression_models.items():
    model.fit(X_train_gravity, y_train_gravity)
    y_pred_gravity = model.predict(X_test_gravity)
    print(f"{model_name}: MAE = {mean_absolute_error(y_test_gravity, y_pred_gravity):.3f}, R2 = {r2_score(y_test_gravity, y_pred_gravity):.3f}")
    joblib.dump(model, f"{model_name}_gravity_regression_model.joblib")


best_clf = RandomForestClassifier(random_state=42)
best_clf.fit(X_train_clf, y_train_clf)
y_pred_clf_rf = best_clf.predict(X_test_clf)

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test_clf, y_pred_clf_rf), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Random Forest')
plt.show()


plt.scatter(y_test_gravity, y_pred_gravity, alpha=0.5)
plt.xlabel('Actual Specific Gravity')
plt.ylabel('Predicted Specific Gravity')
plt.title('Actual vs Predicted Specific Gravity')
plt.show()


joblib.dump(scaler, "scaler.joblib")
joblib.dump(pipeline_clf, "classification_pipeline.joblib")
joblib.dump(pipeline_reg, "regression_pipeline.joblib")

print("\nModels and Pipelines saved for deployment.")
