import joblib
import pandas as pd

# Load the model and scaler
model = joblib.load('radar_model.pkl')
scaler = joblib.load('scaler.pkl')

# Function to make predictions
def classify_signals(frequencies):
    # Convert frequencies to DataFrame
    data = pd.DataFrame({'Frequency': frequencies})
    
    # Preprocess the data (scale it)
    data_scaled = scaler.transform(data)
    
    # Make predictions
    predictions = model.predict(data_scaled)
    return predictions

# Example usage
frequencies = [12.5, 139.2, 15.0, 29.5]
predictions = classify_signals(frequencies)
print("Predictions:", predictions)
