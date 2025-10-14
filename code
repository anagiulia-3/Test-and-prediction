import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

consumers_tendencies = pd.read_csv("consumers_tendencies.csv")

consumers_tendencies.head()

X = consumers_tendencies.iloc[:,0].values  # variavel independent
y = consumers_tendencies.iloc[:,1].values # variavel dependente

#  Treina o modelo para o linear
lr = LinearRegression()
lr.fit(X.reshape(-1, 1), y)

# treina o modelo para o polinomial 
pr = PolynomialFeatures(degree = 4) 
X_poly = pr.fit_transform(X.reshape(-1, 1))
lr_2 = LinearRegression()
lr_2.fit(X_poly, y)

# Predict results
y_pred_lr = lr.predict(X.reshape(-1, 1))           # Linear Regression
y_pred_poly = lr_2.predict(X_poly)   # Polynomial Regression

#  linear regression
plt.scatter(X, y, color = 'lightcoral')
plt.plot(X, lr.predict(X.reshape(-1, 1)), color = 'firebrick')
plt.title('Regressão linear')
plt.xlabel('anos')
plt.ylabel('percentual de buyers')
plt.legend(['X/y_pred_lr', 'X/y'], title = 'Regressão linear', loc='best', facecolor='white')
plt.box(False)
plt.show()
 
 #polinomial 
 # Visualize real data with polynomial regression
# Values to predict
X_pred = np.array([[13], [14], [15]])
X_pred_poly = pr.transform(X_pred)
y_pred_points = lr_2.predict(X_pred_poly)

# Polynomial regression plot
X_grid = np.arange(min(X), max(X), 0.1).reshape(-1, 1)
plt.scatter(X, y, color='lightcoral')
plt.plot(X_grid, lr_2.predict(pr.transform(X_grid)), color='firebrick')
plt.scatter(X_pred, y_pred_points, color='blue', marker='o', s=100, label='Predictions')
plt.title('Polynomial Regression')
plt.xlabel('Months')
plt.ylabel('Average Basket Value')
plt.legend(['Polynomial Fit', 'Actual Data', 'Predictions'], loc='best', facecolor='white')
plt.box(False)
plt.show()
