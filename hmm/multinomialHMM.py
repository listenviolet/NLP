import numpy as np 
from hmmlearn import hmm 

''' Create the model
'''
states = ['box 1', 'box 2', 'box 3']
n_states = len(states)

observations = ['red', 'white']
n_observations = len(observations)

start_probability = np.array([0.2, 0.4, 0.4])

transition_probability = np.array([
	[0.5, 0.2, 0.3],
  	[0.3, 0.5, 0.2],
  	[0.2, 0.3, 0.5]
])

emission_probability = np.array([
	[0.5, 0.5],
  	[0.4, 0.6],
  	[0.7, 0.3]
])

model = hmm.MultinomialHMM(n_components = n_states)
model.startprob_ = start_probability
model.transmat_ = transition_probability
model.emissionprob_ = emission_probability

''' 3. Viterbi
'''
seen = np.array([[0, 1, 0]]).T  # red, white, red
logprob, box = model.decode(seen, algorithm = 'viterbi') # seen here has to be an 2-D array
print("The ball picked: ", ", ".join(map(lambda x: observations[x], [0, 1, 0]))) # seen here has to be and 1-D array
print("The hidden box: ", ", ".join(map(lambda x: states[x], box)))

# logprob: Log probability of the produced state sequence.
# output:
# The ball picked:  red, white, red
# The hidden box:  box 3, box 3, box 3


''' 1. P(O|lambda)
'''
print(model.score(seen))
# -2.038545309915233
# model.score(seen) = log_e (prob)
# prob = 0.13022
