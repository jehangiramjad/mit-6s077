##############################################################################
# Utils for Problem Set 0
#
##############################################################################
import numpy as np
import pandas as pd

def fib_n(n):
	if (n > 10):
		return None

	if(n <= 1):
		return n
	else:
		return fib_n(n-1) + fib_n(n-2)
