import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv('radar_signals.csv')

# Features and Labels
X = df[['Frequency']].values
y = df['Label'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Preprocessing
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save preprocessed data (Optional)
pd.DataFrame(X_train_scaled, columns=['Frequency']).to_csv('X_train_scaled.csv', index=False)
pd.DataFrame(X_test_scaled, columns=['Frequency']).to_csv('X_test_scaled.csv', index=False)
pd.DataFrame(y_train, columns=['Label']).to_csv('y_train.csv', index=False)
pd.DataFrame(y_test, columns=['Label']).to_csv('y_test.csv', index=False)
