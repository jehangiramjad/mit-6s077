## Hypothesis Testing Utility functions

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
from datetime import datetime as dt

########################################################
#Helper functions

# returns a datetime.datetime object from numpy's datetime64
def convert_to_datetime_from_numpy_datetime64(np_datetime64):
	if (type(np_datetime64) is not np.datetime64):
		raise Exception('np_datetime64 needs to be of type \'numpy.datetime64\'')
	
	return dt.utcfromtimestamp(np_datetime64.tolist()/1e9)

########################################################
# Dataset Handling 

def read_dataset_from_csv(filename, columns=None):

	df = pd.read_csv(filename)

	if (columns is not None):
		df = subset_data(df, columns)

	return df

# returns the dataframe
def read_column_as_datetime(df, column_name, date_format_to_read):

	if (column_name not in df.columns):
		raise Exception('Column Name: (%s), not found in the dataframe.')

	df[column_name] = pd.to_datetime(df[column_name], format=date_format_to_read)
	return df

