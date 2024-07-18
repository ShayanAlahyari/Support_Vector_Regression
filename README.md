# Support_Vector_Regression

This repository contains a Python script that demonstrates the use of Support Vector Regression (SVR) on a dataset of position levels and salaries.

## Overview

The script performs the following steps:
1. Loads the dataset.
2. Preprocesses the data (feature scaling).
3. Fits an SVR model to the data.
4. Predicts a salary for a specific position level.
5. Visualizes the SVR results.

## Dataset

The dataset used in this example is `Position_Salaries.csv`, which contains the following columns:
- `Position`: Job position.
- `Level`: Position level (independent variable).
- `Salary`: Corresponding salary (dependent variable).

## Requirements

The script requires the following Python libraries:
- numpy
- pandas
- matplotlib
- scikit-learn

You can install these libraries using pip:
```bash
pip install numpy pandas matplotlib scikit-learn
