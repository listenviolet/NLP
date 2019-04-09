# GaussianHMM和GMMHMM是连续观测状态的HMM模型
# GaussianHMM类假设观测状态符合高斯分布，
# 而GMMHMM类则假设观测状态符合混合高斯分布。
# 一般情况下我们使用GaussianHMM即高斯分布的观测状态即可。

# 我们的观测状态是二维的，而隐藏状态有4个。
# 因此我们的“means”参数是4×2的矩阵，
# 而“covars”参数是4×2×2的张量。

import numpy as np 
from hmmlearn import hmm 

startprob = np.array([0.6, 0.3, 0.1, 0.0])

# The transition matrix, note that there are no transitions possible
# between component 1 and 3
transmat = np.array([[0.7, 0.2, 0.0, 0.1],
                     [0.3, 0.5, 0.2, 0.0],
                     [0.0, 0.3, 0.5, 0.2],
                     [0.2, 0.0, 0.2, 0.6]])

# The means of each component
means = np.array([[0.0,  0.0],
                  [0.0, 11.0],
                  [9.0, 10.0],
                  [11.0, -1.0]])

# The covariance of each component
covars = .5 *np.tile(np.identity(2), (4, 1, 1))
# print(covars)
# [[[0.5 0. ]
#   [0.  0.5]]

#  [[0.5 0. ]
#   [0.  0.5]]

#  [[0.5 0. ]
#   [0.  0.5]]

#  [[0.5 0. ]
#   [0.  0.5]]]


# Build an HMM instance and set parameters
model3 = hmm.GaussianHMM(n_components = 4, covariance_type = 'full')
# ”full” — each state uses a full (i.e. unrestricted) covariance matrix.
# 取值为"full"意味所有的μ,Σ都需要指定。
# 取值为“spherical”则Σ的非对角线元素为0，对角线元素相同。
# 取值为“diag”则Σ的非对角线元素为0，对角线元素可以不同，
# "tied"指所有的隐藏状态对应的观测状态分布使用相同的协方差矩阵Σ

# Instead of fitting it from the data, we directly set the estimated
# parameters, the means and covariance of the components
model3.startprob_ = startprob
model3.transmat_ = transmat
model3.means_ = means
model3.covars_ = covars

''' 1. P(O|lambda)
'''

# 由于观测状态是二维的，我们用的三维观测序列， 所以这里的 输入是一个3×2的矩阵
seen = np.array([[1.1,2.0],[-1,2.0],[3,7]])
logprob, state = model3.decode(seen, algorithm="viterbi")
print(state)
# [0 0 1]

print(model3.score(seen))
# -41.121128137687