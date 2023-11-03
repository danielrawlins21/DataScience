"""import numpy as np

# Given data
labels = np.array([-1, -1, -1, -1, -1, 1, 1, 1, 1, 1])
coordinates = np.array([[0, 0],
                        [2, 0],
                        [3, 0],
                        [0, 2],
                        [2, 2],
                        [5, 1],
                        [5, 2],
                        [2, 4],
                        [4, 4],
                        [5, 5]])
mistakes = np.array([1, 9, 10, 5, 9, 11, 0, 3, 1, 1])

# Calculate θ₀
theta_0 = -np.sum(labels * mistakes)

# Calculate θ₁ and θ₂ using the combined coordinates matrix
theta_1 = -np.sum(labels * coordinates[:, 0] * mistakes)
theta_2 = -np.sum(labels * coordinates[:, 1] * mistakes)

print("θ₀:", theta_0)
print("θ₁:", theta_1)
print("θ₂:", theta_2)
"""
import numpy as np

# Given data
x_coordinates = np.array([0, 2, 3, 0, 2, 5, 5, 2, 4, 5])
y_coordinates = np.array([0, 0, 0, 2, 2, 1, 2, 4, 4, 5])
mistakes = np.array([1, 9, 10, 5, 9, 11, 0, 3, 1, 1])

# Identify support vectors (mistakes == 1)
support_vectors = [(x, y) for x, y, m in zip(x_coordinates, y_coordinates, mistakes) if m == 1]
print(support_vectors)
# Calculate the normalized vector perpendicular to the decision boundary
theta_1, theta_2 = np.mean(support_vectors, axis=0)

# Calculate θ₀
theta_0 = 1 - np.dot([theta_1, theta_2], np.mean(support_vectors, axis=0))

#print("θ₀:", theta_0)
#print("θ₁:", theta_1)
#print("θ₂:", theta_2)

