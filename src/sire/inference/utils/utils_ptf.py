import numpy as np

def normalize(point):
    norm = np.linalg.norm(point)
    if norm < 1e-12:
        return norm
    return point/norm

def rodrigues_rotation(v, axis, angle):
    '''
    Rotation of an vector in space given an axis and an angle of rotation.
    based on the formula by Olinde Rodrigues.
    v_rot = v*cos(angle) + (axis x v)*sin(angle) + axis(dot(axis, v))(1 - cos(angle))
    '''
    axis = normalize(axis)
    return v*np.cos(angle) + (np.cross(axis, v))*np.sin(angle) + axis*(np.dot(axis, v))*(1-np.cos(angle))

def parallel_transport_frame(filtered_points):

    num_p = len(filtered_points)

    T = normalize(filtered_points[1:] - filtered_points[:-1])
    T = np.vstack([T, T[-1]])

    N, B = np.zeros_like(T), np.zeros_like(T)

    candidate_N = np.array([0,0,1], dtype = float)
    if abs(np.dot(candidate_N, T[0])) > 0.9: # Means it is almost parallel to T_0:
        candidate_N = np.array([0,1,0], dtype = float)

    # Initializing N and B
    N0 = candidate_N - np.dot(candidate_N, T[0]) * T[0]
    N0 = normalize(N0)
    B0 = np.cross(T[0], N0)

    N[0], B[0] = N0, B0

    for i in range(1, num_p):

        v, w = T[i-1], T[i]
        a = np.cross(v, w) # Rotation axis.

        if np.linalg.norm(a) < 1e-6: # We don't need to rotate.
            N[i] = N[i-1]
            B[i] = B[i-1]
            continue
        
        axis = a/np.linalg.norm(a)
        angle = np.arccos(np.clip(np.dot(v,w), -1.0, 1.0))

        N[i] = rodrigues_rotation(N[i-1], axis, angle)
        B[i] = rodrigues_rotation(B[i-1], axis, angle)

    return T, N, B