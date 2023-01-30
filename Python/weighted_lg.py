from preprocess_data import *
import cvxpy as cp
import numpy as np
from typing import TypeVar


PandasSeries = TypeVar('pandas.core.frame.Series')
PandasDF = TypeVar('pandas.core.frame.DataFrame')
CVXvar = TypeVar('cp.Variable')
CVXexpression = TypeVar('cp.Expression')







class Logistics_Regression:

	def __init__(self, X_train, X_test, y_train, y_test):
		self.X_train = X_train
		self.y_train = y_train

	def sigmoid(self, z):
		#sigmoid function, z is linear transformation of the input
		return 1/(1 + cp.exp(-z))



	def objective(self, X, y, beta):
		#define cross entropy loss

		#y_hat = self.sigmoid(X.values @ beta)
		#loss = -cp.sum(y.values @ cp.log(y_hat) - (1-y.values) @ cp.log(1-y_hat))


		loss = cp.sum(cp.multiply(y.values, X.values @ beta) - cp.logistic(X.values @ beta))

		return loss


	def predict(self, X, beta_optimized):
		#make prediction with the optimized beta

		scores = self.sigmoid(X @ beta_optimized)
		scores[scores > 0] = 1
		scores[scores <= 0] = 0

		return scores


p = PreprocessData()
X_train, X_test, y_train, y_test = p.get_train_test_split()

lg = Logistics_Regression(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

beta = cp.Variable(X_train.shape[1])
problem = cp.Problem(cp.Maximize(lg.objective(X_train, y_train, beta)))
#problem.solve()

#y_hat = lg.sigmoid(X_train.values @ beta)


#-cp.sum(y_train.values @ cp.log(y_hat) - (1-y_train.values) @ cp.log(1-y_hat)) / X_train.shape[0]








