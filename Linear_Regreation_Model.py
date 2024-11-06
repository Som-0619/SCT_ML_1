import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Load data
train_data = pd.read_csv('/content/train.csv')
test_data = pd.read_csv('/content/test.csv')
sample_submission = pd.read_csv('/content/sample_submission.csv')

# Feature selection based on the columns available in train_data
X = train_data[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath']]
y = train_data['SalePrice']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on validation data
y_pred = model.predict(X_val)

# Calculate MSE
mse = mean_squared_error(y_val, y_pred)
print(f'Mean Squared Error: {mse}')

# **Visualizations**

# 1. Scatter Plot with Regression Line (GrLivArea vs. SalePrice)
plt.figure(figsize=(10, 6))
plt.scatter(X_val['GrLivArea'], y_val, color='blue', label='Actual Sale Price')
plt.scatter(X_val['GrLivArea'], y_pred, color='red', label='Predicted Sale Price')
plt.plot(X_val['GrLivArea'], y_pred, color='orange', linewidth=2)
plt.xlabel('GrLivArea')
plt.ylabel('SalePrice')
plt.title('GrLivArea vs SalePrice (Actual vs Predicted)')
plt.legend()
plt.show()

# 2. Residual Plot
residuals = y_val - y_pred
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.title('Distribution of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()

# 3. Predicted vs Actual Plot
plt.figure(figsize=(10, 6))
plt.scatter(y_val, y_pred, color='purple')
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'k--', lw=2)
plt.xlabel('Actual Sale Price')
plt.ylabel('Predicted Sale Price')
plt.title('Predicted vs Actual Sale Price')
plt.show()

# Prepare the test data
X_test = test_data[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath']]

# Predict on test data
test_predictions = model.predict(X_test)

# Prepare the submission file
submission = sample_submission.copy()
submission['SalePrice'] = test_predictions
submission.to_csv('house_price_predictions.csv', index=False)
