#Boston House Prices

#Requaired libraries
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
import numpy as np

# Load Dataset
boston = load_boston()
#print(boston.DESCR)
x = boston.data
y = boston.target


# LinearRegression => Cross Validation
reg = LinearRegression()
cv_scores = cross_val_score(reg, x, y, cv=5)
cv = cv_scores
print(f"CV Scores: {cv}")

#plot
n = np.arange(5)
plt.plot(n, cv)
plt.title("Cross Validation")