# Q1 e
import pdb;
import numpy as np
import pandas as pd
from scipy import arange
import matplotlib.pyplot as plt

#lambda
l = np.arange(0,100,0.1)
error,bias,variance = [],[],[]
min_error = 1e6
min_lam = 0

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
  if (bias_term+variance_term < min_error):
    min_error = bias_term + variance_term
    min_lam = i


plt.plot(l, bias, label = 'bias')
plt.plot(l, variance, label='variance')
plt.plot(l, error, label = 'expected squared error')
plt.xlabel("λ")
plt.legend()
plt.show()
print("min error {} occurs at lambda value {}".format(min_error, min_lam))