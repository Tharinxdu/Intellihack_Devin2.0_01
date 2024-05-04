import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Step 1: Data Exploration and Preprocessing
# Load the dataset
data_path = './mnt/data/Crop_Dataset.csv'
data = pd.read_csv(data_path)

# Inspect the data structure
print(data.info())
print(data.describe())

# Handle missing values if any
data = data.dropna()

# Extract features and labels
# Ensure to drop columns that shouldn't be features and specify the correct target column
X = data.drop(['Label', 'Label_Encoded'], axis=1)
y = data['Label_Encoded']

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Model Training
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train a RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 3: Model Evaluation
# Make predictions and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))

# Step 4: Joblib Model Creation and Prediction
# Save the model to a joblib file
joblib.dump(model, './mnt/data/crop_recommendation_model.joblib')

# Load the model and make predictions on new data
loaded_model = joblib.load('./mnt/data/crop_recommendation_model.joblib')
new_predictions = loaded_model.predict(X_test)
new_accuracy = accuracy_score(y_test, new_predictions)
print(f'New Accuracy from loaded model: {new_accuracy:.2f}')
