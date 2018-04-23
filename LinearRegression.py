# https://mp.weixin.qq.com/s/UltBigoduH76vs_pmLUOVQ
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
np.random.seed(123)

# generate data set
x = 2 * np.random.rand(500, 1)
y = 5 + 3 * x + np.random.rand(500, 1)
fig = plt.figure(figsize=(8, 6))
plt.scatter(x, y)
plt.title("Dataset")
plt.xlabel("First Feature")
plt.ylabel("Second Feature")
plt.show()

# split the data into training and testing set
x_train, x_test, y_train, y_test = train_test_split(x, y)
print(f'Shape x_train:{x_train.shape}')
print(f'Shape x_test:{x_test.shape}')
print(f'Shape y_train:{y_train.shape}')
print(f'Shape x_test:{x_test.shape}')

class LinearRegression:

    def __init__(self):
        pass

    def train_gradient_descent(self, x, y, learning_rate=0.01, n_iters=100):
        # step 0: Initials the parameters
        n_samples, n_features = x.shape;
        self.weights = np.zeros(shape=(n_features, 1))
        self.bias = 0
        costs = []

        for i in range(n_iters):
            # step 1: Compute a linear combination of the input features and weights
            y_predict = np.dot(x, self.weights) + self.bias

            # step 2: Compute the cost over training set
            cost = (1 / n_samples) * np.sum((y_predict - y)**2)
            costs.append(cost)

            if i % 100 == 0 :
                print(f"Cost at iteration {i}: {cost}")

            # step 3: Compute the gradients
            dJ_dw = (2 / n_samples) * np.dot(x.T, (y_predict - y))
            dJ_db = (2 / n_samples) * np.sum((y_predict - y))

            # step 4: Update the parameters
            self.weights = self.weights - learning_rate * dJ_dw
            self.bias = self.bias - learning_rate * dJ_db

        return self.weights, self.bias, costs

    def train_normal_equation(self, x, y):
        self.weights = np.dot(np.dot(np.linalg.inv(np.dot(x.T, x)), x.T), y)
        self.bias = 0
        return self.weights, self.bias

    def predict(self, x):
        return np.dot(x, self.weights) + self.bias



# training
regressor = LinearRegression()
n_iters = 600
w_trained, b_trained, costs = regressor.train_gradient_descent(x_train, y_train, learning_rate=0.05, n_iters=600)
fig = plt.figure(figsize=(8, 6))
plt.plot(np.arange(n_iters), costs)
plt.title("Development of cost during training")
plt.xlabel("Number of iterations")
plt.ylabel("Cost")
plt.show()


# test
n_sample, _ = x_train.shape
n_samples_test, _ = x_test.shape

y_p_train = regressor.predict(x_train)
y_p_test = regressor.predict(x_test)

error_train = (1 / n_sample) * np.sum((y_p_train - y_train) ** 2)
error_test = (1 / n_samples_test) * np.sum((y_p_test - y_test) ** 2)

print(f"Error on training set: {np.round(error_train, 4)}")
print(f"Error on test set: {np.round(error_test, 4)}")



###################################################
# train with normal equation
# To compute the parameters using the normal equation, we add a bias value of 1 to each input example
x_b_train = np.c_[np.ones((n_sample)), x_train]
x_b_test = np.c_[np.ones((n_samples_test)), x_test]

reg_normal = LinearRegression()
w_trained = reg_normal.train_normal_equation(x_b_train, y_train)

# test with normal equation
y_p_train = reg_normal.predict(x_b_train)
y_p_test = reg_normal.predict(x_b_test)

error_train = (1 / n_sample) * np.sum((y_p_train - y_train) ** 2)
error_test = (1 / n_samples_test) * np.sum((y_p_test - y_test) ** 2)

print(f"Error on training set by Normal Equation: {np.round(error_train, 4)}")
print(f"Error on test set by Normal Equation: {np.round(error_test, 4)}")
#######################################################


# plot the test predictions
fig = plt.figure(figsize=(8, 6))
plt.scatter(x_train, y_train)
plt.scatter(x_test, y_p_test)
plt.xlabel("First Feature")
plt.ylabel("Second Feature")
plt.show()
