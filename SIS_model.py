import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

# Define the model
class model():
    def __init__(self, A, beta=0.01, delta=0.05):
        self.adjacency = A
        self.beta = beta
        self.delta = delta
        self.Num_node = A.shape[0]
        self.report = []

    def get_report(self):
        report_dict = {'current_probs': self.current_probs, 'current_states': self.current_states, 'delta_nodes': self.delta_nodes, 'next_probs': self.next_probs}
        self.report.append( report_dict )
        
    def init_allinfected(self):
        self.current_probs = np.ones(self.Num_node)
        self.sample_infection_prob()

    def calc_prob_not_get_infected(self):
        self.delta_nodes = np.ones(self.Num_node)
        A = self.adjacency
        A_probs = np.tile(self.current_probs.reshape(-1, 1), [1, self.Num_node])
        A_probs = A_probs * A * self.beta
        self.delta_nodes = (1 - A_probs).prod(axis=0, where= (A==1))
        
    def calc_next_probs(self):
        self.next_probs = np.ones(self.Num_node)
        for node in range(self.Num_node):
            prob_healthy = self.delta_nodes[node] * (1 - self.current_probs[node]) + self.delta_nodes[node] * (self.current_probs[node]) * self.delta + (1 - self.delta_nodes[node]) * (self.current_probs[node]) * self.delta * 0.5
            self.next_probs[node] = 1 - prob_healthy

    def sample_infection_prob(self):
        self.current_states = np.random.rand(1, self.Num_node) < self.current_probs

    def update_state(self):
        self.get_report()
        self.current_probs = self.next_probs
        self.sample_infection_prob()




# -------------------------------------------------------- Load data
text = []
with open('social_network.txt', 'r') as file:
    lines = file.readlines()
    for line in lines:
        line_text = line.rstrip()
        text.append( [int(str) for str in line_text.split(',')] )
edgelist = np.array(text)
N = edgelist.max() + 1

# create adjacency matrix
A = np.zeros((N, N))
for ind in range(edgelist.shape[0]):
    edge = edgelist[ind,:]
    A[edge[0], edge[1]] = 1
    A[edge[1], edge[0]] = 1

# Extract the largest eigenvalue
v = np.linalg.eigvalsh(A)
# first large eigenvalue
v1 = sorted(v, reverse=True)[0]
print(f'The biggest eigenvalue of the adjacency matrix is {v1:.2f}')


# -------------------------------------------------------- Initiate and run models
model1 = model(A, beta=0.01, delta=0.05)
model1.init_allinfected()

model2 = model(A, beta=0.01, delta=0.40)
model2.init_allinfected()

for iter in range(100):
    model1.calc_prob_not_get_infected()
    model1.calc_next_probs()
    model1.update_state()

    model2.calc_prob_not_get_infected()
    model2.calc_next_probs()
    model2.update_state()


# extract infection rate over time
rate1 = []
rate2 = []
for iter in range(len(model1.report)):
    states = model1.report[iter]['current_states']
    rate1.append( ((states == 1).sum(), (states == 0).sum()) )
    states = model2.report[iter]['current_states']
    rate2.append( ((states == 1).sum(), (states == 0).sum()) )


# -------------------------------------------------------- plot results
fig, ax = plt.subplots(2, 1, figsize=(15, 12))
ax = ax.ravel()
_ = ax[0].plot([num[0] for num in rate1], color='r', marker='.', label='Infected')
_ = ax[0].plot([num[1] for num in rate1], color='g', marker='.', label='Healthy')
ax[0].set_ylim([0, model1.Num_node])
ax[0].legend()
ax[0].set_xlabel('iteration')
ax[0].set_ylabel('Number of nodes')
ax[0].set_title(f'Beta = {model1.beta} and Delta = {model1.delta}')

_ = ax[1].plot([num[0] for num in rate2], color='r', marker='.', label='Infected')
_ = ax[1].plot([num[1] for num in rate2], color='g', marker='.', label='Healthy')
ax[1].set_ylim([0, model1.Num_node])
ax[1].legend()
ax[1].set_xlabel('iteration')
ax[1].set_ylabel('Number of nodes')
ax[1].set_title(f'Beta = {model2.beta} and Delta = {model2.delta}')

plt.show()

