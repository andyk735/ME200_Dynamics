"""
Created on Thursday Oct 8 2020
@author: Andrew Kim
Dynamics Workshop 4: Problem 13-6
"""

import matplotlib.pyplot as plt
from numpy import sqrt, sin, cos
import math

# INPUT PARAMETER
g = 32.2                        # gravity, ft/s^2
v = 8                           # velocity of crane, ft/s
l = 30                          # length of crane, ft

# CALCULATIONS
y_max = (v**2) / (2*g)
x_max = sqrt((60*y_max) - (y_max**2))

# PRINT RESULTS
print(y_max)
print("The maximum horizontal distance of the bucket:", x_max, " ft")   