
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# Create a dummy scaler
scaler = StandardScaler()

# Fit the scaler on dummy data
dummy_data = np.array([[0, 0, 0, 0, 0, 0, 0]])
scaler.fit(dummy_data)

# Save the scaler to a file
joblib.dump(scaler, "scaler.pkl")
print("scaler.pkl has been created and saved.")