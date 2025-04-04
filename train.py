import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Generate sample crypto price data
data = {
    "Previous_Price": [45000, 46000, 47000, 48000, 49000, 50000, 51000, 52000, 53000, 54000],
    "Next_Price": [46000, 47000, 48000, 49000, 50000, 51000, 52000, 53000, 54000, 55000]
}
df = pd.DataFrame(data)

# Features & Target
X = df[["Previous_Price"]]
y = df["Next_Price"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Save Model
with open("crypto_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as crypto_model.pkl")
