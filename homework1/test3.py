import numpy as np

# Define the feature map function
def feature_map(x):
    return np.array([x[0]**2, np.sqrt(2) * x[0] * x[1], x[1]**2])

# Data
labels = [-1, -1, -1, -1, -1, 1, 1, 1, 1, 1]
points = [(0, 0), (2, 0), (1, 1), (0, 2), (3, 3), (4, 1), (5, 2), (1, 4), (4, 4), (5, 5)]
mistakes = [1, 65, 11, 31, 72, 30, 0, 21, 4, 15]

# Initialize θ and θ0
theta = np.zeros(3)  # Assuming 3 features in the feature map
theta0 = 0

# Calculate θ and θ0 based on the support vectors
for i in range(len(labels)):
    if mistakes[i] > 0:
        alpha_i = mistakes[i]
        y_i = labels[i]
        phi_x_i = feature_map(points[i])
        theta += alpha_i * y_i * phi_x_i
        theta0 += alpha_i * y_i

# Print the results
print("θ:", theta)
print("θ0:", theta0)
