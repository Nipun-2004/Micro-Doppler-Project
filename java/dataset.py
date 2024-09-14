import pandas as pd
import numpy as np
import random 

# Function to generate random frequencies within a given range
def generate_frequencies(min_freq, max_freq, num_samples):
    return np.random.uniform(min_freq, max_freq, num_samples)

# Prompt user for input ranges and sample sizes
print("Enter frequency range for Birds:")
bird_min_freq = float(input("Minimum frequency: "))
bird_max_freq = float(input("Maximum frequency: "))
bird_samples = int(input("Number of samples: "))

print("Enter frequency range for Drones:")
drone_min_freq = float(input("Minimum frequency: "))
drone_max_freq = float(input("Maximum frequency: "))
drone_samples = int(input("Number of samples: "))

# Generate data for Birds and Drones
bird_frequencies = generate_frequencies(bird_min_freq, bird_max_freq, bird_samples)
drone_frequencies = generate_frequencies(drone_min_freq, drone_max_freq, drone_samples)

# Create DataFrame
data = {
    'Frequency': np.concatenate([bird_frequencies, drone_frequencies]),
    'Label': ['Bird'] * bird_samples + ['Drone'] * drone_samples
}

df = pd.DataFrame(data)

# Save to CSV
df.to_csv('radar_signals.csv', index=False)

print("Database created and saved as 'radar_signals.csv'.")
