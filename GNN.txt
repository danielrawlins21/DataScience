import numpy as np

def gaussian_pdf(x, mean, variance):
    return 1 / np.sqrt(2 * np.pi * variance) * np.exp(-0.5 * ((x - mean) ** 2) / variance)

def calculate_posterior_probability(x, mean, variance, prior_probability):
    likelihood_cluster_1 = gaussian_pdf(x, mean[0], variance)
    likelihood_cluster_2 = gaussian_pdf(x, mean[1], variance)
    
    posterior_cluster_1 = (likelihood_cluster_1 * prior_probability[0]) / \
                          (likelihood_cluster_1 * prior_probability[0] + likelihood_cluster_2 * prior_probability[1])
    
    return posterior_cluster_1

# Given parameters
mean = [-3, 2]
variance = 4
prior_probability = [0.5, 0.5]

# Data points
data_points = [0.2, -0.9, -1, 1.2, 1.8]

# Calculate posterior probabilities for each data point
posterior_probabilities = [calculate_posterior_probability(x, mean, variance, prior_probability) for x in data_points]

# Display the results
for i, x in enumerate(data_points, 1):
    print(f'Posterior probability for x^{i} = {x}: {posterior_probabilities[i-1]:.8f}')
