import numpy as np

# Define the MDP parameters
num_states = 5
num_actions = 3
gamma = 0.5
iterations = 100

# Initialize the value function
V = np.zeros(num_states)

# Define the transition probabilities
P = np.zeros((num_states, num_actions, num_states))
R = np.zeros((num_states, num_actions, num_states))

# Update the transition probabilities and rewards based on the given information
# ... (update P and R matrices based on the transition probabilities you have)

# Value iteration algorithm
for t in range(iterations):
    V_temp = np.copy(V)
    for s in range(num_states):
        # Compute the Q-value for each action
        Q_values = np.zeros(num_actions)
        for a in range(num_actions):
            Q_values[a] = np.sum(P[s, a, :] * (R[s, a, :] + gamma * V_temp[:]))
        
        # Update the value function
        V[s] = np.max(Q_values)

# Display the final value function V*100 as an array
V_100 = V
print("V*100:", V_100)
