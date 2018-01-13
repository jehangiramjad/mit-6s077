from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm
from sklearn import metrics
from sklearn import linear_model
import statsmodels.api as sm

from pandas.tools.plotting import scatter_matrix

def drop_columns(dataframe, columns):

	for colname in dataframe.columns.values:
		if (colname in columns):
			del dataframe[colname]

	return dataframe

def subset_data(dataframe, columns):

	return dataframe[columns]


def introduce_power_terms(dataframe, columns, power=2):

	for colname in dataframe.columns.values:
		if (colname in columns):
			dataframe[colname+ "^%s" %power] = dataframe[colname]**power

	return dataframe


def introduce_log_terms(dataframe, columns):

	for colname in dataframe.columns.values:
		if (colname in columns):
			dataframe[colname+"_log"] = np.log(dataframe[colname])

	return dataframe


def introduce_interaction_terms(dataframe, column_pairs):

	for (col1, col2) in column_pairs:
		dataframe[col1+"-"+col2] = dataframe[col1] * dataframe[col2]

	return dataframe


def read_dataset_from_csv(filename, columns=None):

	df = pd.read_csv(filename)

	if (columns is not None):
		df = subset_data(df, columns)

	return df


def split_dataframe(df, train_proportion, validation_proportion, test_proportion=0.0):

	rand_nbrs = np.random.rand(len(df))

	train_msk =  rand_nbrs <= train_proportion
	validate_msk = ((rand_nbrs > train_proportion) & (rand_nbrs <= (train_proportion + validation_proportion)))
	test_msk = rand_nbrs > (train_proportion + validation_proportion)

	train_df = df[train_msk]
	val_df = df[validate_msk]
	test_df = df[test_msk]

	if (test_proportion > 0.0):
		return (train_df, val_df, test_df)
	else:
		return (train_df, val_df)

###################################################
# Set the dataframe up for regression

def get_independent_variable_matrix_X(dataframe, exclude_columns=[]):
	cols_included = []
	dim = np.shape(dataframe)
	nbr_cols = dim[1] - len(exclude_columns)
	X = np.zeros([dim[0], nbr_cols])

	columns = dataframe.columns.values
	index = 0
	for i in range(0, len(columns)):
		colname = columns[i]
		if (colname in exclude_columns):
			continue

		col_vector =  dataframe[colname]
		col_vector.values.reshape(len(col_vector), 1)

		try:
			X[:, index] = col_vector
			cols_included.append(colname)
		except:
			print("Not inserting column: (%s) because all rows don't have Floats" %colname)

		index += 1

	return (X, cols_included)


def get_dependent_variable_matrix_y(dataframe, target_column):
	dim = np.shape(dataframe)
	y = dataframe[target_column]
	y.values.reshape(len(y), 1)

	return y




###################################################
# Summary Statsitics


def correlation(a, b):
	return scipy.stats.pearsonr(a, b)[0]

def rmse(y, y_hat):
	return metrics.mean_squared_error(y, y_hat)

def r2(actual, pred):
	#actual.flatten()
	#pred.flatten()

	tss = np.sum((actual - np.mean(actual))**2)
	rss = np.sum((actual - pred)**2)
	return tss, rss, (tss-rss)/tss


### RSE = sqrt(RSS / (n-2))
def rse(rss, samples, dof):
	return np.sqrt(rss / (samples - dof))


## standard error in co-eff
def std_errs(X, y, RSS):

	# first calculate the rse
	dof = np.shape(X)[1]
	sigma = rse(RSS, len(y), dof)

	X_concat = np.hstack((np.ones((len(y), 1)), X))

	A = X_concat.transpose().dot(X_concat)
	Ainv = np.linalg.inv(A)
	Err_Cov = sigma**2 * Ainv
	return np.array(map(lambda i: Err_Cov[i,i]**0.5, range(Err_Cov.shape[0])))

# construct confidence intervals
def confidence_intervals(coeffs, standard_errors, nbr_std_devs):

	(ci_lower, ci_upper) = (coeffs - nbr_std_devs*standard_errors, 
			coeffs + nbr_std_devs*standard_errors)

	ci_array = []
	for i in range(0, len(ci_lower)):
		ci_array.append((ci_lower[i], ci_upper[i]))

	return ci_array

## p value
def pvalue(coeffs, standard_errors):
	t_statistic_array = coeffs / standard_errors

	pval_array = []
	for t in t_statistic_array:
		pval_array.append(2.0 *(1.0 - norm.cdf(np.abs(t))))

	return pval_array


# F test score
def F_score(tss, rss, p, n):

	NUM = (tss-rss)/p
	DENOM = rss / (n-p-1.)

	return NUM / DENOM



###################################################
# Plots

def correlation_plot(dataframe):
	corr_matrix = dataframe.corr()
	f = plt.figure(figsize=(10,10))
	plt.matshow(corr_matrix, vmin=-1, vmax=1, cmap='RdBu_r', fignum=f.number)
	plt.title('Correlation Plot')
	plt.colorbar()
	plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns)
	plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
	plt.show()


def scatter_plot_dataframe(dataframe):
	colors = ['red', 'green']
	scatter_matrix(dataframe, figsize=[20,20],marker='x', c='red')
	plt.suptitle('Scatter Plot for the DataFrame')
	plt.show()

# Statsmodels Residual Plots
def plot_sm_residuals(sm_result):
	
	fig = plt.figure(figsize=(10,10))
	plt.subplot(311)
	plt.plot(sm_result.resid)
	plt.ylabel('Residuals')

	plt.subplot(312)
	plt.hist(sm_result.resid, bins=20, normed=True)
	plt.xlabel('Residuals')
	plt.ylabel('Freq')

	plt.subplot(313)
	plt.scatter(sm_result.fittedvalues, sm_result.resid)
	plt.xlabel('Fitted Values')
	plt.ylabel('Residuals')

	fig.suptitle('Residuals Plots')

	sm.graphics.qqplot(sm_result.resid, dist=stats.norm, line='45', fit=True)
	plt.title('QQ Plot of Residuals')
	plt.show()


# Statsmodels Residual Plots
def plot_sm_residuals(sm_result):
	
	fig = plt.figure(figsize=(10,10))
	plt.subplot(311)
	plt.plot(sm_result.resid)
	plt.ylabel('Residuals')

	plt.subplot(312)
	plt.hist(sm_result.resid, bins=20, normed=True)
	plt.xlabel('Residuals')
	plt.ylabel('Freq')

	plt.subplot(313)
	plt.scatter(sm_result.fittedvalues, sm_result.resid)
	plt.xlabel('Fitted Values')
	plt.ylabel('Residuals')

	fig.suptitle('Residuals Plots')

	sm.graphics.qqplot(sm_result.resid, dist=stats.norm, line='45', fit=True)
	plt.title('QQ Plot of Residuals')
	plt.show()


# Sklearn Residual Plots
def plot_residuals(y, y_hat):
		
	residuals = y - y_hat
	fig = plt.figure(figsize=(10,10))
	
	# plt.subplot(221)
	# plt.scatter(y, residuals)
	# plt.ylabel('Residuals')

	plt.subplot(311)
	stats.probplot(residuals, dist="norm", plot=plt)
	#plt.title('QQ Plot of Residuals')

	plt.subplot(312)
	plt.hist(residuals, bins=20, normed=True)
	plt.xlabel('Residuals')
	plt.ylabel('Freq')

	plt.subplot(313)
	plt.scatter(y_hat, residuals)
	plt.xlabel('Fitted Values')
	plt.ylabel('Residuals')

	fig.suptitle('Residuals Plots')

	plt.show()



# Sklearn Regression Prediction Plot
def plot_predictions(y, y_hat, x_var=None):
	# assumed regression_model is coming from sklearn.linear_model class
	# X and y are the original data

	if (x_var is not None):
		plt.scatter(x_var, y,  color='black', label='Original')
		plt.plot(x_var, y_hat, color='blue', linewidth=3, label='Predictions')
		plt.xlabel('x')
		plt.ylabel('y and $\hat(y)$')
		plt.title('Original vs Predictions')
		legend = plt.legend(loc='upper right', shadow=True)

	else:
		plt.plot(y,  color='black', label="Original ($y$)")
		plt.plot(y_hat, color='blue', linewidth=1, label="Predictions ($\hat{y}$)")
		plt.xlabel('Samples')
		plt.ylabel('y and $\hat(y)$')
		plt.title('Original vs Predictions')
		legend = plt.legend(loc='lower right', shadow=True)
	
	plt.show()



###########################################################
# Regression Helpers

def all_coefficients(regression_model):
	# assumed regression_model is coming from sklearn.linear_model class
	# Intercept followed by all other variables

	intercept = np.array(regression_model.intercept_)
	return np.array(list(intercept.flatten()) + list(regression_model.coef_.flatten()))


def prediction_stats(model, X, y, plot=False):
	(samples, nbr_cols) = np.shape(X)

	coeffs = all_coefficients(model)
	#print("coefficients: ", coeffs)

	# predictions
	predictions = model.predict(X)

	# RMSE
	rmse_val = np.sqrt(rmse(y, predictions))

	# Prediction and Plot
	if (plot == True):
		if (nbr_cols == 1):
			plot_predictions(y, predictions, x_var= X.flatten())
		else:
			plot_predictions(y, predictions)
		
		plot_residuals(y, predictions)

	# Get Regression fit Stats (R^2)
	TSS, RSS, R2 = r2(y, predictions)

	#print("TSS, RSS, R^2: ", TSS, RSS, R2)

	# Standard Errors estimates
	se = std_errs(X, y, RSS)
	#print("Std. Errors: ", std_errs)

	# Confidence Intervals
	ce = confidence_intervals(coeffs, se, 2)
	#print("C.I's (95%): " , conf_intervals)

	# p-values
	pvals = pvalue(coeffs, se)
	#print("p-values: ", pvals)

	return {"coeffs":coeffs, "rmse": rmse_val, "tss": TSS, "rss":RSS, "r2": R2, "std_errors": se, "ci": ce, "pval": pvals}


def multiple_linear_regression(X, y, fit_intercept=True):

	##### use Scikit Learn for Regression
	model = linear_model.LinearRegression(fit_intercept = fit_intercept)
	model.fit(X, y)
	
	return model



def ridge_regression(X, y, alpha=0.0, fit_intercept=True):

	##### use Scikit Learn for Regression
	model = linear_model.Ridge(alpha=alpha, fit_intercept = fit_intercept)
	model.fit(X, y)
	
	return model


def lasso_regression(X, y, alpha=1.0, fit_intercept=True):

	##### use Scikit Learn for Regression
	model = linear_model.Lasso(alpha=alpha, fit_intercept = fit_intercept)
	model.fit(X, y)
	
	return model


