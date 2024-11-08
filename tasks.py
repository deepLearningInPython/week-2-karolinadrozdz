import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.optimize import minimize  # Python version of R's optim() function
from sklearn import datasets
 
# Carry out the exercises in your own copy of the notebook that you can find at
#    https://www.kaggle.com/code/datasniffer/perceptrons-mlp-s-and-gradient-descent.
# Then copy and paste code asked for below in between the dashed lines.
# Do not import additional packages.
 
# Task 1:
# Instructions:
# In the notebook, you wrote a function that implements an MLP with 2 hidden layers.
# The function should accept a vector of weights and a matrix X that stores input feature
# vectors in its **columns**.
# The name of the function should be my_mlp.
 
# Copy and paste the code for that function here:
# -----------------------------------------------
def my_mlp(w, X, sigma=np.tanh):
    W1 = (w[0:4*6]).reshape(4,6)
    W2 = (w[4*6:7*4+6*4]).reshape(7,4)
    W3 = (w[7*4+6*4:]).reshape(1,7)
   
    a1 = sigma(W1 @ X)
    a2 = sigma(W2 @ a1)
    f  = sigma(W3 @ a2)
    return f [0, 0, :, :]
# -----------------------------------------------
 
# Task 2:
# Instructions:
# In the notebook, you wrote a function that implements a loss function for training
# the MLP implemented by my_mlp of Task 1.
# The function should accept a vector of weights, a matrix X that stores input feature
# vectors in its **columns**, and a vector y that stores the target labels (-1 or +1).
# The name of the function should be MSE_func.
 
# Copy and paste the code for that function here:
# -----------------------------------------------
def MSE_func(w, X, y): 
    f = my_mlp(w, X)
    MSE = np.sum((y-f)**2)
    return MSE
# -----------------------------------------------
 
# Task 3:
# Instructions:
# In the notebook, you wrote a function that returns the gradient vector for the least
# squares (simple) linear regression loss function.
# The function should accept a vector beta that contains the intercept (β₀) and the slope (β₁),
# a vector x that stores values of the independent variable, and a vector y that stores
# the values of the dependent variable and should return an np.array() that has the derivative values
# as its components.
# The name of the function should be dR.
 
# Copy and paste the code for that function here:
# -----------------------------------------------
def dR(beta, x, y):
    dbeta_0 = 2*np.mean((beta[0] + beta[1]*x - y))   # implement the above formula for dR/dβ₀
    dbeta_1 = 2*np.mean((beta[0] + beta[1]*x - y)*x) # implement the above formula for dR/dβ₁
    return np.array([dbeta_0, dbeta_1])
 
# -----------------------------------------------
