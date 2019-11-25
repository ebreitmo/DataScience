import matplotlib.pyplot as plt
import numpy as np

# -------------------- Function warmUpExercise ------------------------                                                                         
def warmUpExercise():
 print(np.identity(5))
# print(np.eye(5))

# -------------------- Function computeCost ----------------------------
def computeCost(X, y, theta):
 m=len(y)
 Y=y.reshape(m,1)
 J=0.5/m*sum(np.square((np.dot(X,theta)-Y)))
 return (J)

# -------------------- Function gradientDescent -------------------------
def gradientDescent(X, y, theta, alpha, iterations):
 m=len(y)
 Y=y.reshape(m,1)
# For debugging print J_history
 #J_history=np.zeros((iterations,1))
 for i in range(1,iterations):
  theta = theta - alpha/m * (np.dot((np.dot(X,theta)-Y).transpose(),X)).transpose()
  #J_history[i]=computeCost(X,y,theta) 
  #print("J_history:  ",J_history[i])
 return(theta)

