"""
Created on Monday Oct 12 2020
@author: Andrew Kim
Dynamics Workshop 4: Problem 13-29
"""

import matplotlib.pyplot as plt
from numpy import sqrt, sin, cos, pi
import math

# INPUT PARAMETER
g = 32.2                        # gravity, ft/s^2
w = 7.5                         # weight of collar, lb
dist_slide = 1.5                # distance the collar slides before hitting the spring, ft
dist_spring = 5/12              # max distance the spring compresses, ft
theta = pi/6                    # angle of inclined rod, degrees
k = 60                          # spring constant, lb/ft

# CALCULATIONS
u_k = ((w*(dist_slide + dist_spring)*sin(theta)) - (0.5*k*(dist_spring)**2)) / (w*cos(theta)*(dist_slide+dist_spring))
v_max = sqrt(2*g*dist_slide*(sin(theta) - u_k*cos(theta)))
back_up = (0.5*k*(dist_spring)**2) / (w*(sin(theta) + u_k*cos(theta)))

# PRINT RESULTS
print("The coefficient of kinetic fricton between the collar and the rod:", u_k)       
print("The maximum speed of the collar:", v_max, " ft/s")   
print("The maximum distance along the rod that the collar reaches when it rebounds back up", back_up, " ft") 


