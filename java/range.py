import pandas as pd

# Load the dataset
df = pd.read_csv('radar_signals.csv')

# Check the range of frequencies
min_freq = df['Frequency'].min()
max_freq = df['Frequency'].max()

print(f"Min Frequency: {min_freq}")
print(f"Max Frequency: {max_freq}")
