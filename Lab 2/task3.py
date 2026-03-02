import numpy as np

# Define class for LP problem
class FoodTruckLP:
    def __init__(self):
        # Objective for minimization: minimize -4x - 5y  (equivalent to maximizing 4x + 5y)
        self.c = np.array([-4.0, -5.0], dtype=float)
        
        # Constraints:
        # 2x + y <= 10
        # x + 2y <= 12
        self.A_ub = np.array([
            [2.0, 1.0],
            [1.0, 2.0]
        ], dtype=float)

        self.b_ub = np.array([10.0, 12.0], dtype=float)
        
        # Bounds: 0 <= x <= 6, 0 <= y <= 6
        self.bounds = [(0.0, 6.0), (0.0, 6.0)]



