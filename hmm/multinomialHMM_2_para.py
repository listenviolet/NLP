import numpy as np 
from hmmlearn import hmm 

# Known the observations 
# solve: the para of the model
states = ['box 1', 'box 2', 'box 3']
n_states = len(states)

observations = ['red', 'white']
n_observations = len(observations)

model2 = hmm.MultinomialHMM(n_components = n_states, n_iter = 20, tol = 0.01)
# tol (float, optional) – Convergence threshold. 
# EM will stop if the gain in log-likelihood is below this value.

# the observations:
X2 = np.array([[0,1,0,1],[0,0,0,1],[1,0,1,1]]) 

model2.fit(X2)
print(model2.startprob_)
print(model2.transmat_)
print(model2.emissionprob_)
print(model2.score(X2))  # P(O | lambda)

print()

model2.fit(X2)
print(model2.startprob_)
print(model2.transmat_)
print(model2.emissionprob_)
print(model2.score(X2))  # P(O | lambda)

print()

model2.fit(X2)
print(model2.startprob_)
print(model2.transmat_)
print(model2.emissionprob_)
print(model2.score(X2))  # P(O | lambda)

print()