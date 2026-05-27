# Implementation of Multivariate Linear Regression
## Name:PRASANNA P
## Reg no:212225230214
## Aim
To write a python program to implement multivariate linear regression and predict the output.
## Equipment’s required:
1.	Hardware – PCs
2.	Anaconda – Python 3.7 Installation / Moodle-Code Runner
## Algorithm:
### Step1
import pandas as pd.

### Step2
Read the csv file.

### Step3
Get the value of X and y variables

### Step4
Create the linear regression model and fit.

### Step5
Predict the CO2 emission of a car where the weight is 2300kg, and the volume is 1300cm cube.


## Program:
```
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
# Load dataset
data = fetch_california_housing()
# Feature matrix and target
X = data.data
y = data.target
# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=1)
# Create model
reg = linear_model.LinearRegression()
# Train model
reg.fit(X_train, y_train)
# Coefficients
print("Coefficients:", reg.coef_)
# Accuracy score
print("Variance score:", reg.score(X_test, y_test))
# Plot style
plt.style.use('fivethirtyeight')
# Training residuals
plt.scatter(reg.predict(X_train),
            reg.predict(X_train)-y_train,
            color='green', s=10, label='Train data')
# Testing residuals
plt.scatter(reg.predict(X_test),
            reg.predict(X_test)-y_test,
            color='blue', s=10, label='Test data')
# Zero residual line
plt.hlines(y=0, xmin=0, xmax=6, linewidth=2)
# Legend and title
plt.legend(loc='upper right')
plt.title("Residual Errors")
plt.show()






```
## Output:
<img width="1243" height="672" alt="image" src="https://github.com/user-attachments/assets/85be5393-cc1b-4447-a41d-7f103ca4269a" />


## Result
Thus the multivariate linear regression is implemented and predicted the output using python program.
