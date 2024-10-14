import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('drug_dataset.csv')

# Create target variable based on the median of LN_IC50
data['TARGET'] = data['LN_IC50'].apply(lambda x: 1 if x < data['LN_IC50'].median() else 0)

# Prepare features and target
X = data[['GENE', 'PUTATIVE_TARGET', 'LN_IC50']]  
y = data['TARGET']

# Convert categorical variables to dummy/indicator variables
X = pd.get_dummies(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")  # Displays accuracy as a fraction

# Accept user input at runtime
gene_input = input("Enter GENE: ")
putative_target_input = input("Enter PUTATIVE TARGET: ")
ln_ic50_input = float(input("Enter LN_IC50: "))
tissue_input = input("Enter TISSUE: ")

# Prepare the input data for prediction
user_input = pd.DataFrame({
    'GENE': [gene_input],
    'PUTATIVE_TARGET': [putative_target_input],
    'LN_IC50': [ln_ic50_input]
})

# Convert categorical variables to dummy/indicator variables
user_input = pd.get_dummies(user_input)
user_input = user_input.reindex(columns=X.columns, fill_value=0)

# Standardize the user input using the same scaler
user_input_scaled = scaler.transform(user_input)

# Predict the response using the trained model
user_pred = rf_model.predict(user_input_scaled)

# Display the result
response = 'Sensitive' if user_pred[0] == 1 else 'Resistive'
print(f"\nPrediction for GENE: {gene_input}, PUTATIVE_TARGET: {putative_target_input}, LN_IC50: {ln_ic50_input}")
print(f"Predicted Response: {response}")

# If the response is sensitive, suggest the best drug for the given tissue
if response == 'Sensitive':
    sensitive_drugs = data[data['TARGET'] == 1]
    if sensitive_drugs[sensitive_drugs['TISSUE'] == tissue_input].empty:
        print(f"No effective drugs found for tissue '{tissue_input}'.")
    else:
        common_drug = sensitive_drugs[sensitive_drugs['TISSUE'] == tissue_input]['DRUG_NAME'].value_counts().idxmax()
        print(f"The suggested effective drug for tissue '{tissue_input}' is: {common_drug}")
else:
    print("No specific drug suggestion as the response is Resistive.")

# Plot feature importance
plt.figure(figsize=(10, 6))
feature_importances = rf_model.feature_importances_
features = X.columns

# Create a DataFrame for feature importance
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)


ln_ic50_values = np.linspace(data['LN_IC50'].min(), data['LN_IC50'].max(), num=100)
sensitivity = []
resistivity = []

for ln_ic50 in ln_ic50_values:
    # Prepare the input data for prediction
    user_input = pd.DataFrame({
        'GENE': [gene_input],
        'PUTATIVE_TARGET': [putative_target_input],
        'LN_IC50': [ln_ic50]
    })

    # Convert categorical variables to dummy/indicator variables
    user_input = pd.get_dummies(user_input)
    user_input = user_input.reindex(columns=X.columns, fill_value=0)

    # Standardize the user input using the same scaler
    user_input_scaled = scaler.transform(user_input)

    # Predict the response using the trained model
    user_pred = rf_model.predict(user_input_scaled)

    # Collect the responses
    if user_pred[0] == 1:
        sensitivity.append(1)
        resistivity.append(0)
    else:
        sensitivity.append(0)
        resistivity.append(1)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(ln_ic50_values, sensitivity, label='Sensitivity', color='green')
plt.plot(ln_ic50_values, resistivity, label='Resistivity', color='red')
plt.title(f'Sensitivity and Resistivity for {gene_input} and {putative_target_input}')
plt.xlabel('LN_IC50 Values')
plt.ylabel('Response')
plt.yticks([0, 1], ['Resistive', 'Sensitive'])
plt.axhline(0.5, color='grey', linestyle='--', label='Threshold')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Plotting
sns.barplot(data=importance_df, x='Importance', y='Feature', hue=None)
plt.title('Feature Importance for Drug Response Prediction')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()