# M adjacency matrix
# d damping factor (default value 0.85)
# eps quadratic error for v (default value 1.0e-8)
# Return v, a vector of ranks such that v_i is the i-th rank from [0, 1]

import numpy as np

def Pagerank(Mat, eps=1.0e-8, d=0.85):
    N = M.shape[1]
    v = np.random.rand(N, 1)
    v = v / np.linalg.norm(v, 1)
    last_v = np.ones((N, 1), dtype=np.float32) * 100
    M_hat = (d * M) + (((1 - d) / N) * np.ones((N, N), dtype=np.float32))
    
    while np.linalg.norm(v - last_v, 2) > eps:
        last_v = v
        v = np.matmul(M_hat, v)
    return v


# Testing
M = np.array([[0, 0, 0, 0, 1], [0.5, 0, 0, 0, 0], [0.5, 0, 0, 0, 0], [0, 1, 0.5, 0, 0], [0, 0, 0.5, 1, 0]])
v = Pagerank(Mat=M, eps=0.001, d=0.85)
print(v)
