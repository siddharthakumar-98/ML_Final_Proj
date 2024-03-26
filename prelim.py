# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load your dataset
file_path = 'PTSH_snowdat.csv'  # Replace with the path to your dataset
data = pd.read_csv(file_path)

# Selecting the relevant features and the target variable
features = ['AirTC_Avg', 'SWE']
target = 'DBTCDT'  # Assuming 'DBTCDT' is the column for snow depth

# Preparing the data: dropping rows with missing values in the selected features or the target
data_clean = data.dropna(subset=features + [target])

# Splitting the dataset into features (X) and target (y)
X = data_clean[features]
y = data_clean[target]

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Creating and training the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions on the test set
y_pred = model.predict(X_test)

# Evaluating the model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse}")
print(f"R^2: {r2}")
