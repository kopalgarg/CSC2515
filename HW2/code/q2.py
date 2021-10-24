from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import arange

# 2 a. Load the dataset
def load_data():
    boston = datasets.load_boston()
    X = boston.data
    y = boston.target
    features = boston.feature_names
    return X, y, features

# 2 b. Describe and summarize the data in terms of # data points, dimensions, target, etc.
def describe_data(X, y, features):
  print("Number of data points:", y.shape)
  print("Dimensions of the dataset", X.shape)

  # Plot the distribution of the target variable
  sns.displot(y, bins=30, kde=True);
  print("y min: {}, max: {}, mean: {}, stdev: {}".format(min(y), max(y),
                                                       np.mean(y), np.std(y)))
  
  f, ax = plt.subplots(figsize=(10, 8))
  # Plot the correlations matrix to display the linear relations between pairs of features
  merged = pd.DataFrame(X,columns=features)
  merged['MEDV'] = y
  corr = merged.corr()

  sns.heatmap(corr, 
              cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)

# 2.c
def visualize(X, y, features):
    plt.figure(figsize=(20, 5))
    feature_count = X.shape[1]

    # i: index
    for i in range(feature_count):
        plt.subplot(3, 5, i + 1)
        plt.scatter(X[:, i], y, marker='o')
        plt.title(features[i])
        plt.xlabel(features[i])
        plt.ylabel('MEDEV')

    plt.tight_layout()
    plt.legend()
    plt.show()

def test_train_split(X, y):
  indices = np.random.permutation(X.shape[0])
  ratio = int(0.7*X.shape[0])
  train_idx, test_idx = indices[:ratio], indices[ratio:]
  X_train, X_test = X[train_idx,:], X[test_idx,:]
  y_train, y_test = y[train_idx], y[test_idx]
  return X_train, X_test, y_train, y_test

# 2 d
def fit_regression(X, y):
  # X with the bias term
  X_b = np.vstack([np.ones(X.shape[0]), X.T]).T 
  # l2_loss_no_regularizer
  # Lec 3 Slide 21: w^* = (X^T X)^-1 X^T y
  w = np.linalg.solve(np.dot(np.transpose(X_b), X_b), np.dot(np.transpose(X_b), y))
  return w

def compute_metrics(y_hat, y, X):
  MSE = sum((y_hat - y)**2)/X.shape[0]
  RMSE = np.power(MSE, .5)
  MAE = sum(np.abs(y_hat - y))/X.shape[0]
  return MSE, RMSE, MAE

def main():
    ### Q1a: Load the data
    X, y, features = load_data()
    print("Features: {}".format(features))
    ### Q1b: Describe data
    describe_data(X,y,features)

    ### Q1c: Visualize the features
    visualize(X, y, features)

    # Split data into train and test
    X_train, X_test, y_train, y_test = test_train_split(X,y)

    ### Q1d: Fit regression model
    w_train = fit_regression(X_train, y_train)

    ### Q1e: Tabulate each feature with weight
    w_f = pd.DataFrame(w_train[1:], features)
    w_f_abs = pd.DataFrame(abs(w_train[1:]), features)
    print(w_f)
    # sort in descending order - absolute weights
    print("sorted:")
    print(w_f_abs.sort_values(by=[0],  ascending=False).head())

    ### Q1f: Test performance, Q1g: Compute fitted values, MSE, etc.
    X_test_b = np.vstack([np.ones(X_test.shape[0]), X_test.T]).T # X with the bias term
    y_hat_test = np.dot(X_test_b, w_train)
    MSE, RMSE, MAE = compute_metrics(y_hat_test, y_test, X_test_b)
    print('Test set performance: MSE: {0}, RMSE: {1}, MAE: {2}'.format(MSE, RMSE, MAE))

if __name__ == "__main__":
    main()

