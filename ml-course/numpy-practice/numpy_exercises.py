# -*- coding: utf-8 -*-

"""The Complete Machine Learning Course, Numpy Practice Module
"""

# Base Import
import numpy as np
import pandas as pd

# Create a 1-dimensional NumPy array using np.array()
array_1d = np.array([1, 2, 3])

# Create a 2-dimensional NumPy array using np.array()
array_2d = np.array([[1, 2, 3], [4, 5, 6]])

# Create a 3-dimensional Numpy array using np.array()
array_3d = np.array(
    [[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[10, 11, 12], [13, 14, 15], [16, 17, 18]]]
)

# Attributes of 1-dimensional array (shape, number of dimensions, data type,
# size and type)

print(array_1d.shape, array_1d.ndim, array_1d.dtype, array_1d.size, type(array_1d))

# Attributes of 2-dimensional array
print(array_2d.shape, array_2d.ndim, array_2d.dtype, array_2d.size, type(array_2d))

# Attributes of 3-dimensional array
print(array_3d.shape, array_3d.ndim, array_3d.dtype, array_3d.size, type(array_3d))


# Import pandas and create a DataFrame out of one
# of the arrays you've created
df = pd.DataFrame(array_2d)

# Create an array of shape (10, 2) with only ones
array_ = np.ones([10, 2])
array_.shape

# Create an array of shape (7, 2, 3) of only zeros
array_ = np.zeros((7, 2, 3))

# Create an array within a range of 0 and 100 with step 3
array_ = np.arange(start=0, stop=100, step=3)

# Create a random array with numbers between 0 and 10 of size (7, 2)
array_ = np.random.randint(0, 10, size=(7, 2))

# Create a random array of floats between 0 & 1 of shape (3, 5)
array_ = np.random.random(size=(3, 5))

# Set the random seed to 42
np.random.seed(42)

# Create a random array of numbers between 0 & 10 of size (4, 6)
array_ = np.random.randint(0, 10, size=(4, 6))

# Create an array of random numbers between 1 & 10 of size (3, 7)
# and save it to a variable
array_ = np.random.randint(0, 10, size=(3, 7))

# Find the unique numbers in the array you just created
np.unique(array_)

# Find the 0'th index of the latest array you created
array_[0]

# Get the first 2 rows of latest array you created
array_[:2]

# Get the first 2 values of the first 2 rows of the latest array
array_[:2, :2]

# Create a random array of numbers between 0 & 10 and an array of ones
# both of size (3, 5), save them both to variables
numbers_ = np.random.randint(0, 10, size=(3, 5))
ones_ = np.ones((3, 5))

# Add the two arrays together
added = numbers_ + ones_

# Create another array of ones of shape (5, 3)
another_ones_ = np.ones((5, 3))

# Try adding the array of ones and the other most recent array together
added_again = numbers_ + another_ones_  # Does not work because arrays have diff shapes

# Create another array of ones of shape (3, 5)
ones_again_ = np.ones((3, 5))

# Subtract the new array of ones from the other most recent array
numbers_ - ones_again_

# Multiply the ones array with the latest array
numbers_ * ones_again_

# Take the latest array to the power of 2 using '**'
numbers_ ** 2

# Do the same thing with np.square()
np.square(numbers_)

# Find the mean of the latest array using np.mean()
np.mean(numbers_)
# Find the maximum of the latest array using np.max()
np.max(numbers_)
# Find the minimum of the latest array using np.min()
np.min(numbers_)
# Find the standard deviation of the latest array
np.std(numbers_)
# Find the variance of the latest array
np.var(numbers_)
# Reshape the latest array to (3, 5, 1)
np.reshape(a=numbers_, newshape=(3, 5, 1))
# Transpose the latest array
np.transpose(a=numbers_)

# Create two arrays of random integers between 0 to 10
# one of size (3, 3) the other of size (3, 2)
array_one = np.random.randint(0, 10, (3, 3))
array_two = np.random.randint(0, 10, (3, 2))

# Perform a dot product on the two newest arrays you created
np.dot(array_one, array_two)

# Take the latest two arrays, perform a transpose on one of them and then perform
# a dot product on them both
np.dot(np.transpose(array_one), array_two)


# Create two arrays of random integers between 0 & 10 of the same shape
# and save them to variables
array_one = np.random.randint(0, 10, (3, 3))
array_two = np.random.randint(0, 10, (3, 3))


# Compare the two arrays with '>'
array_one > array_two

# Compare the two arrays with '>='
array_one >= array_two

# Find which elements of the first array are greater than 7
array_one > 7

# Which parts of each array are equal? (try using '==')
array_one == array_two

# Sort one of the arrays you just created in ascending order
np.sort(array_one)

# Sort the indexes of one of the arrays you just created
np.argsort(array_one)

# Find the index with the maximum value in one of the arrays you've created
np.argmax(array_one)

# Find the index with the minimum value in one of the arrays you've created
np.argmin(array_one)

# Find the indexes with the maximum values down the verticial axis
# of one of the arrays you created
np.argmax(array_one, axis=1)
# Find the indexes with the minimum values across the horizontal axis
# of one of the arrays you created
np.argmax(array_one, axis=0)

# Create an array of normally distributed random numbers
ndist_array = np.random.randn(3, 5)

# Create an array with 10 evenly spaced numbers between 1 and 100
evenly_spaced_array = np.linspace(1, 100, 10)
