#Boston House Prices

#Requaired libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
import numpy as np

# Load Dataset
boston = load_boston()
print(boston.DESCR)
x = boston.data
y = boston.target

# Split Data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=42)

# LinearRegression
reg = LinearRegression()
reg.fit(x_train, y_train)

# Predict: test
y_predict = reg.predict(x_test)

n = np.arange(152)

# Plot
plt.plot(n, y_predict, label='y_predict')
plt.plot(n, y_test, label='y_test')
plt.legend()

# Accuracy
score = reg.score(x_test, y_test)
print(f"score: {score}")

# MSE :: Mean Square Error :: toevaluating Model
mse = mean_squared_error(y_test, y_predict)
print(f"MSE: {mse}")