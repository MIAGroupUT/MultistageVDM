import numpy as np

from typing import Any, List
from dataclasses import asdict
from scipy.signal import savgol_filter

# For larger vessels, RMF is also possible.

class ParallelTransportFrame:
    def __init__(self, centers,
                 filter_fn = None,
                 output_filter_fn = None):      # This is the function for the filter.
        self.centers = np.asarray(centers, dtype=float)
        self.filter_fn = filter_fn                  # To filter the original centers.
        self.output_filter_fn = output_filter_fn    # To filter the output of the frame.

    def __call__(self):
        centers = self.centers.copy()

        if self.filter_fn is not None:
            centers = self.filter_fn(centers)

        T, N, B = self._compute_frame(centers, self.output_filter_fn)

        return T, N, B, centers
    
    @staticmethod
    def _normalize(v):
        norm = np.linalg.norm(v)
        if norm < 1e-12:
            return v
        return v/norm
    
    @classmethod
    def _rodrigues_rotation(cls, v, axis, angle):
        '''
        Rotation of an vector in space given an axis and an angle of rotation.
        based on the formula by Olinde Rodrigues.
        v_rot = v*cos(angle) + (axis x v)*sin(angle) + axis(dot(axis, v))(1 - cos(angle))
        '''
        axis = cls._normalize(axis)    
        return v*np.cos(angle) + (np.cross(axis, v))*np.sin(angle) + axis*(np.dot(axis, v))*(1-np.cos(angle))

    @classmethod
    def _compute_frame(cls, centers, filter_fn):
        # Applying Parallel Transport Frame
        num_p = len(centers)

        T = centers[1:] - centers[:-1]
        T = np.array([cls._normalize(t) for t in T])
        T = np.vstack([T, T[-1]])

        N, B = np.zeros_like(T), np.zeros_like(T)

        candidate_N = np.array([0,0,1], dtype = float)
        if abs(np.dot(candidate_N, T[0])) > 0.9: # Means it is almost parallel to T_0:
            candidate_N = np.array([0,1,0], dtype = float)

        # Initializing N and B
        N0 = candidate_N - np.dot(candidate_N, T[0]) * T[0]
        N0 = cls._normalize(N0)
        B0 = np.cross(T[0], N0)

        N[0], B[0] = N0, B0

        for i in range(1, num_p):

            v, w = T[i-1], T[i]
            a = np.cross(v, w) # Rotation axis.

            if np.linalg.norm(a) < 1e-6: # We don't need to rotate.
                N[i] = N[i-1]
                B[i] = B[i-1]
                continue
            
            axis = cls._normalize(a)
            angle = np.arccos(np.clip(np.dot(v,w), -1.0, 1.0))

            N[i] = cls._rodrigues_rotation(N[i-1], axis, angle)
            B[i] = cls._rodrigues_rotation(B[i-1], axis, angle)

        if filter_fn is None:
            return T, N, B

        else: # Possible smoothing of the output.
            return filter_fn(T), filter_fn(N), filter_fn(B)

    
    '''
    Definition of some filter pre-sets.
    '''
    @staticmethod
    def savgol(window_length: int = 21,
               polyorder: int = 3,
               axis: int = 0):
        
        return lambda x: savgol_filter(x, # Recieves the x.
                                       window_length = window_length,
                                       polyorder = polyorder,
                                       axis = axis)
    
    # @staticmethod
    # def moving_average(window_length: int = 21):

    #     return lambda x, y, z 