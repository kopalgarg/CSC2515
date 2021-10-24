# Q1 e
import pdb;
import numpy as np
import pandas as pd
from scipy import arange
import matplotlib.pyplot as plt

# One possible way of plotting
#lambda
l = np.linspace(0,10,100)

# given conditions
mu = 1
squared_sigma = 9
n = 10

# Q1.e
bias_term = np.power( ((mu*l)/(1+l) ) ,2)
variance_term = squared_sigma / (n*(np.power(1+l,2)))
expected_squared_error =  bias_term + variance_term

plt.figure(figsize=(20, 5))
plt.plot(bias_term, label = 'bias')
plt.plot(variance_term, label='variance')
plt.plot(expected_squared_error, label = 'expected squared error')
plt.legend()
plt.show()

# An alternative way that results in the same plot

#lambda
l = np.arange(0,10,.1)
error,bias,variance = [],[],[]
min_error = 1e6

for i in l:
  # given conditions
  mu = 1
  SqSigma = 9
  n = 10

  # Q1.d
  bias_term = np.power( ((mu*i)/(1+i) ) ,2)
  variance_term = SqSigma / (n*(np.power(1+i,2)))
  error.append(bias_term + variance_term)
  variance.append(variance_term)
  bias.append(bias_term)

plt.plot(bias, label = 'bias')
plt.plot(variance, label='variance')
plt.plot(error, label = 'expected squared error')
plt.legend()
#plt.show()