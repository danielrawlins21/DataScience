import numpy as np

# Initialize the weights and bias (theta and theta0) to zero
"""theta = np.zeros(X.shape[1])  # Assuming X is your feature matrix
theta0 = 0.0"""

# Perceptron algorithm
def perceptron_train(X, y, theta, theta0, learning_rate, num_epochs):
    for epoch in range(num_epochs):
        breakpoint()
        for i in range(X.shape[0]):
            xi = X[i, :]
            yi = y[i]
            
            # Calculate the prediction
            prediction = np.dot(xi, theta) + theta0
            
            # Update weights and bias if the prediction is incorrect
            if yi * prediction <= 0:
                theta += learning_rate * yi * xi
                theta0 += learning_rate * yi
    
    return theta, theta0

# Example usage
# Generate some sample data
X = np.array([[-4, 2], [-2, 1], [-1, -1], [2, 2], [1,-2]])
y = np.array([1, 1, -1, -1,-1])

theta = np.array([0,0])  # Assuming X is your feature matrix
theta0 = 0

# Set the learning rate and number of epochs
learning_rate = 1
num_epochs = 100

# Train the perceptron
theta, theta0 = perceptron_train(X, y, theta, theta0, learning_rate, num_epochs)
print(theta,"\n")
print(theta0,"\n")
# Now, you can use the learned weights (theta) and bias (theta0) for prediction
