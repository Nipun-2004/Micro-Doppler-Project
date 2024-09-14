import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import joblib

# Load data
df = pd.read_csv('radar_signals.csv')
X = df[['Frequency']].values
y = df['Label'].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Initialize and fit the scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the model
model = SVC(kernel='linear', random_state=0)
model.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))

# Save the trained model and scaler
joblib.dump(model, 'radar_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
