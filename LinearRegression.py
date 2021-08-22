#Required libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#example data
x = np.arange(1,10)
y = np.array([28,25,26,31,32,29,30,35,36])

#Columnarize data
x = x.reshape(-1,1)
y = y.reshape(-1,1)

#LinearRegression
reg = LinearRegression()
reg.fit(x,y)
yhat = reg.predict(x)

#plot
plt.scatter(x,y)
plt.plot(x,yhat)

#plot details
plt.title("Linear Regresion")

