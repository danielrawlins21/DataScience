import numpy as np

# Given data
labels = np.array([-1, -1, -1, -1, -1, 1, 1, 1, 1, 1])
x_coordinates = np.array([0, 2, 3, 0, 2, 5, 5, 2, 4, 5])
y_coordinates = np.array([0, 0, 0, 2, 2, 1, 2, 4, 4, 5])
mistakes = np.array([1, 9, 10, 5, 9, 11, 0, 3, 1, 1])

# Calculate θ₀
theta_0 = -np.sum(labels * mistakes)

# Calculate θ₁
theta_1 = -np.sum(labels * x_coordinates * mistakes)

# Calculate θ₂
theta_2 = -np.sum(labels * y_coordinates * mistakes)
print("θ₀:", theta_0)
print("θ₁:", theta_1)
print("θ₂:", theta_2)
