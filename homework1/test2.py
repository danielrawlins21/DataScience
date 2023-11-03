from sklearn.svm import SVC
import numpy as np

# Define your dataset
X = np.array([[0, 0], [2, 0], [3, 0], [0, 2], [2, 2], [5, 1], [5, 2], [2, 4], [4, 4], [5, 5]])
y = np.array([-1, -1, -1, -1, -1, 1, 1, 1, 1, 1])

# Create an SVM model with a linear kernel and C parameter to control regularization
model = SVC(kernel='linear', C=1)

# Fit the model to the data
model.fit(X, y)

# Get the parameters
theta0 = model.intercept_[0]
theta1 = model.coef_[0, 0]
theta2 = model.coef_[0, 1]
print(model.support_vectors_)
print("Theta0:", theta0)
print("Theta1:", theta1)
print("Theta2:", theta2)

margin = 2 / np.linalg.norm([theta1, theta2])
print("Margin:", margin)

hinge_losses = [max(0, 1 - y[i] * (theta0 + theta1 * X[i, 0] + theta2 * X[i, 1])) for i in range(len(X))]

# Compute the sum of hinge losses
sum_of_hinge_losses = np.sum(hinge_losses)

print("Sum of Hinge Losses:", sum_of_hinge_losses)