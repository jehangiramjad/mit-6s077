import numpy as np
import tensorflow

def get_data_extract():
	
	from tensorflow.examples.tutorials.mnist import input_data
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

	X = mnist.train.images
	Y = mnist.train.labels
	X_val = mnist.validation.images
	Y_val = mnist.validation.labels
	X_test = mnist.test.images
	Y_test = mnist.test.labels

	prop = 0.3
	train_max_points = np.shape(X)[0]
	val_max_points = np.shape(X_val)[0]
	test_max_points = np.shape(X_test)[0]

	np.random.seed(1031) # this is to ensure the same random subsets
	random_rows_train = np.random.choice(range(0, train_max_points), int(prop * train_max_points), replace=False)
	random_rows_val = np.random.choice(range(0, val_max_points), int(prop * val_max_points), replace=False)
	random_rows_test = np.random.choice(range(0, test_max_points), int(prop * test_max_points), replace=False)

	X = X[random_rows_train, :]
	Y = Y[random_rows_train]

	X_val = X_val[random_rows_val, :]
	Y_val = Y_val[random_rows_val]

	X_test = X_test[random_rows_test, :]
	Y_test = Y_test[random_rows_test]

	return (X, Y, X_val, Y_val, X_test, Y_test)