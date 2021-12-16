import numpy as np
import scipy.special


class CooperationTask:
    def __init__(self):
        self._cooperation_relation = None

    def get_cooperation_relation(self, shape):
        if self._cooperation_relation is None:
            indices = np.arange(shape)
            self._cooperation_relation = np.array([(current, neighbor)
                                                   for current in indices for neighbor in indices[current + 1:]])
        return self._cooperation_relation
