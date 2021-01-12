import numpy as np

def filter_func(x, d, r):
    return x > d - r and d < d + r

def distance(p1, p2):
    #return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)**0.5
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2) ** 0.5

#b = xyz.shape[0]
#n = xyz.shape[1]
#m = new_xyz.shape[1]


#xyz_np = xyz.cpu().numpy()
#new_xyz_np = new_xyz.cpu().numpy()
#d_np = batch_distances
#inds_np = inds.cpu().numpy()

b = 2
n = 9
m = 2
radius = 1
nsample = 2

xyz_np = np.array([[[-1.5256, -0.7502, -0.6540],
         [-1.6095, -0.1002, -0.6092],
         [-0.9798, -1.6091, -0.7121],
         [ 0.3037, -0.7773, -0.2515],
         [-0.2223,  1.6871,  0.2284],
         [ 0.4676, -0.6970, -1.1608],
         [ 0.6995,  0.1991,  0.8657],
         [ 0.2444, -0.6629,  0.8073],
         [ 1.1017, -0.1759, -2.2456]],

        [[-1.4465,  0.0612, -0.6177],
         [-0.7981, -0.1316,  1.8793],
         [-0.0721,  0.1578, -0.7735],
         [ 0.1991,  0.0457,  1.1651],
         [ 2.0154,  0.2152, -0.5242],
         [-1.8034, -1.3083,  0.4533],
         [ 1.1422, -3.3312, -0.7479],
         [ 1.1173,  0.2981,  0.1099],
         [-0.6463,  0.4285,  1.4761]]], dtype=float)
         
new_xyz_np = np.array([[[ 0.3037, -0.7773, -0.2515],
         [-1.6095, -0.1002, -0.6092]],

         [[ 0.1991,  0.0457,  1.1651],
         [-1.4465,  0.0612, -0.6177]]], dtype=float)    
d_np = np.array([[1.44304748, 1.47029484, 1.55111209, 0.67711474, 2.10896685, 1.05045989,
  1.63205747, 1.33456112, 2.23330393],
  [1.73048126, 1.8026257,  1.1810547,  1.02564653, 2.27958346, 2.00054397,
  3.32090884, 1.35253274, 1.58555361]], dtype=float)
inds_np = np.array([[1,3], [0,3]])

for b_idx in range(b):
    for m_idx in range(m):
        center = new_xyz_np[b_idx, m_idx]
        # find the distance of center point
        d = d_np[b_idx, inds_np[b_idx, m_idx]]

        # find candidate points whos are between d-r and d+r
        j = 0            
        #print("d: " + d)
        #print("d_np[b_idx,i]: " + d_np[b_idx, 0])
        #for i in range(n):
        #    if filter_func(d_np[b_idx, i], d, radius):
        #        j += 1
        #candidates = np.empty(j, dtype=int)
        candidates = np.array([])

        j = 0
        for i in range(n):   
            if i == inds_np[b_idx, m_idx]: continue 
            if filter_func(d_np[b_idx, i], d, radius):
                #candidates[j] = i
                #Check if candidate point is really within radius distance
                if distance(xyz_np[b_idx, i], center) < radius:                    
                    candidates = np.append(candidates, [i], axis = 0)
                    j += 1   
                    if j == nsample: break

        while len(candidates) < nsample:            
            candidates = np.append(candidates, [candidates[len(candidates)-1]], axis = 0)
    
        if m_idx == 0:
            solution_set = np.expand_dims(candidates, axis = 0)
        else:
            solution_set = np.append(solution_set, np.expand_dims(candidates, axis = 0), axis = 0)
    
    if b_idx == 0:
        b_solution_set = np.expand_dims(solution_set, axis = 0)
    else:
        b_solution_set = np.append(b_solution_set, np.expand_dims(solution_set, axis = 0), axis = 0)

print(b_solution_set)