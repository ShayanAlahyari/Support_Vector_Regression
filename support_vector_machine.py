import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Load dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values  # Extracting independent variable (Position level)
y = dataset.iloc[:, -1].values    # Extracting dependent variable (Salary)
