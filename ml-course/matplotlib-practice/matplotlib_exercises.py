# Imports
import matplotlib.pyplot as plt
import numpy as np

# Create a simple plot using plt.plot()
plt.plot()

# Plot a single Python list
plt.plot([1, 2, 3, 4, 5])

# Create two lists, one called X, one called y, each with 5 numbers in them
x = [1, 2, 3, 4, 5]
y = [22, 23, 14, 12, 10]

# Plot X & y (the lists you've created)
plt.plot(x, y)

# Create a plot using plt.subplots()
plt.subplots()

# Create a plot using plt.subplots() and then add X & y on the axes
figure, ax = plt.subplots()
ax.plot(x, y)
plt.show()


# Prepare data (create two lists of 5 numbers, X & y)
X = [34, 77, 21, 54, 9]
y = [9, 45, 89, 66, 4]

# Setup figure and axes using plt.subplots()
fig, ax = plt.subplots()

# Add data (X, y) to axes
ax.plot(X, y)

# Customize plot by adding a title, xlabel and ylabel
ax.set(title="Sample simple plot", xlabel="x-axis", ylabel="y-axis")

# Save the plot to file using fig.savefig()
fig.savefig("../images/simple-plot.png")

# Create an array of 100 evenly spaced numbers between 0 and 100 using NumPy and save it to variable X
x = np.linspace(0, 10, 100)

# Create a plot using plt.subplots() and plot X versus X^2 (X squared)
fig, ax = plt.subplots()
ax.plot(x, x**2)
plt.show()

# Create a scatter plot of X versus the exponential of X (np.exp(X))
plt.scatter(x, np.exp(x))

# Create a scatter plot of X versus np.sin(X)
plt.scatter(x, np.sin(x))

# Create a Python dictionary of 3 of your favourite foods with
# The keys of the dictionary should be the food name and the values their price
foods = {
    "Pizza Pericolosa": 15,
    "Red Curry": 11,
    "Gyudon": 9
}

# Create a bar graph where the x-axis is the keys of the dictionary
# and the y-axis is the values of the dictionary
fig, ax = plt.subplots()
ax.bar(foods.keys(), foods.values())

# Add a title, xlabel and ylabel to the plot
ax.set(title="Favourite Foods Prices", xlabel="Food Items", ylabel="prices")

# Make the same plot as above, except this time make the bars go horizontal


# Create a random NumPy array of 1000 normally distributed numbers using NumPy and save it to X
x = np.random.randn(1000)

# Create a histogram plot of X
fig, ax = plt.subplots()
ax.hist(x)

# Create a NumPy array of 1000 random numbers and save it to X
x = np.random.randint(0, 1000, size=1000)

# Create a histogram plot of X
fig, ax = plt.subplots()
ax.hist(x)


# Create an empty subplot with 2 rows and 2 columns (4 subplots total)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
# Create the same plot as above with 2 rows and 2 columns and figsize of (10, 5)


# Plot X versus X/2 on the top left axes
ax1.plot(x, x/2)

# Plot a scatter plot of 10 random numbers on each axis on the top right subplot
ax2.scatter(np.random.random(10), np.random.random(10))

# Plot a bar graph of the favourite food keys and values on the bottom left subplot
ax3.bar(foods.keys(), foods.values())

# Plot a histogram of 1000 random normally distributed numbers on the bottom right subplot
ax4.hist(np.random.randn(1000))