import numpy as np

# Define class for LP problem
class FoodTruckLP:
    def __init__(self):
        # Objective for minimization: minimize -4x - 5y  (equivalent to maximizing 4x + 5y)
        # TODO: Replace None with the correct values
        self.c = np.array([None, None], dtype=float)
        
        # Constraints:
        # 2x + y <= 10
        # x + 2y <= 12
        # TODO: Replace None with the correct values
        self.A_ub = np.array([
            [None, None],
            [None, None]
        ], dtype=float)

        # TODO: Replace None with the correct values
        self.b_ub = np.array([None, None], dtype=float)
        
        # Bounds: 0 <= x <= 6, 0 <= y <= 6
        # TODO: Replace None with the correct values
        self.bounds = [(None, None), (None, None)]



