import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Load dataset
dataset = pd.read_csv('Position_Salaries.csv')

# Extracting independent variable (Position level)
x = dataset.iloc[:, 1:-1].values

# Extracting dependent variable (Salary)
y = dataset.iloc[:, -1].values

# Reshaping the dependent variable
y = y.reshape(len(y), 1)

# Feature scaling for independent and dependent variables
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)

# Fitting SVR model to the data with radial basis function (rbf) kernel
regressor = SVR(kernel='rbf')
regressor.fit(x, y)

# Predicting a new result for position level 6.5
y_predicted = sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]])).reshape(-1, 1))

# Creating a grid of points to plot the SVR results more smoothly
X_grid = np.arange(min(sc_x.inverse_transform(x)), max(sc_x.inverse_transform(x)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))

# Plotting the actual salary data points
plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color='red')

# Plotting the SVR prediction line
plt.plot(X_grid, sc_y.inverse_transform(regressor.predict(sc_x.transform(X_grid)).reshape(-1, 1)), color='blue')

# Adding titles and labels
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')

# Display the plot
plt.show()
