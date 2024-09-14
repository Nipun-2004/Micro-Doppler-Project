from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Function to generate random frequencies
def generate_frequencies(min_freq, max_freq, num_samples):
    return np.random.uniform(min_freq, max_freq, num_samples)

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle form submission for ranges
@app.route('/set_ranges', methods=['POST'])
def set_ranges():
    try:
        bird_min_freq = float(request.form['bird_min_freq'])
        bird_max_freq = float(request.form['bird_max_freq'])
        drone_min_freq = float(request.form['drone_min_freq'])
        drone_max_freq = float(request.form['drone_max_freq'])
        
        bird_samples = 10000 # Define number of samples
        drone_samples = 10000  # Define number of samples

        bird_frequencies = generate_frequencies(bird_min_freq, bird_max_freq, bird_samples)
        drone_frequencies = generate_frequencies(drone_min_freq, drone_max_freq, drone_samples)

        # Create DataFrame
        data = {
            'Frequency': np.concatenate([bird_frequencies, drone_frequencies]),
            'Label': ['Bird'] * bird_samples + ['Drone'] * drone_samples
        }

        df = pd.DataFrame(data)
        df.to_csv('radar_signals.csv', index=False)

        # Train model
        X = df[['Frequency']]
        y = df['Label'].apply(lambda x: 1 if x == 'Drone' else 0)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = RandomForestClassifier()
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        joblib.dump(model, 'radar_model.pkl')
        joblib.dump(scaler, 'scaler.pkl')

        return render_template('set_ranges.html', accuracy=accuracy,
                               bird_min_freq=bird_min_freq, bird_max_freq=bird_max_freq,
                               drone_min_freq=drone_min_freq, drone_max_freq=drone_max_freq,
                               bird_samples=bird_samples, drone_samples=drone_samples)
    except ValueError as e:
        return f"Error: {e}"

# Route to handle frequency classification
@app.route('/classify', methods=['POST'])
def classify():
    try:
        frequency = float(request.form['frequency'])
        
        # Load the model and scaler
        model = joblib.load('radar_model.pkl')
        scaler = joblib.load('scaler.pkl')

        # Load the ranges
        bird_min_freq = float(request.form['bird_min_freq'])
        bird_max_freq = float(request.form['bird_max_freq'])
        drone_min_freq = float(request.form['drone_min_freq'])
        drone_max_freq = float(request.form['drone_max_freq'])

        # Check if the frequency is within the valid range
        if bird_min_freq <= frequency <= bird_max_freq or drone_min_freq <= frequency <= drone_max_freq:
            # Prepare the data for prediction
            data = np.array([[frequency]])
            scaled_data = scaler.transform(data)
            prediction = model.predict(scaled_data)
            
            # Map prediction to label
            label = 'Drone' if prediction[0] == 1 else 'Bird'
            
            return render_template('result.html', frequency=frequency, label=label)
        else:
            return render_template('out_of_range.html', frequency=frequency)
    except (ValueError, KeyError) as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True , host='0.0.0.0', port=8000)
