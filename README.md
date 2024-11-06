# House Price Prediction using Linear Regression
This project uses linear regression to predict house prices based on several key features. The model is trained on a dataset containing various house characteristics, allowing it to estimate the sale price of a house in the test dataset. The project also includes visualizations to help understand the model's performance and data distribution.
Project Overview
This repository demonstrates the application of linear regression to predict house prices based on features like ground living area, number of bedrooms, and number of bathrooms. The project includes data preprocessing, model training, evaluation, and visualization to gain insights into the model's predictions.

# Dataset
The dataset used includes:

train.csv: Training dataset with house features and sale prices.
test.csv: Test dataset with house features but without sale prices.
sample_submission.csv: A template file for formatting predictions in the expected output format.

# Selected Features
The following features are selected for training the model:

GrLivArea: Above-ground living area (square feet)
BedroomAbvGr: Number of bedrooms above ground
FullBath: Number of full bathrooms
HalfBath: Number of half bathrooms
Modeling Approach
Data Loading: Load training, test, and sample submission data.
Feature Selection: Select relevant features for predicting the sale price.
Data Splitting: Split the data into training and validation sets to evaluate model performance.
Model Training: Train a linear regression model on the training set.
Prediction and Evaluation: Use the model to predict house prices on the validation set and calculate the Mean Squared Error (MSE).
Test Prediction: Predict house prices for the test dataset and save predictions for submission.
Evaluation
The model's performance is evaluated using Mean Squared Error (MSE), which measures the average squared difference between actual and predicted sale prices. Lower MSE values indicate a better fit.

# Visualizations
The project includes the following visualizations to understand model performance:

Scatter Plot with Regression Line: Plots actual vs. predicted sale prices based on GrLivArea.
Residual Plot: Shows the distribution of residuals (errors) to assess the fit of the model.
Predicted vs Actual Plot: Compares predicted values with actual values, helping to visualize prediction accuracy.
Results
Mean Squared Error (MSE): The MSE on the validation set gives a numerical estimate of model performance.
Test Predictions: Predictions on the test data are saved in a CSV file ready for submission.
