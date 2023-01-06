#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 21:22:41 2020

@author: andyk

Program to simulate Todhunter 1
"""

import matplotlib.pyplot as plt
from numpy import sqrt, linspace, ones


# INPUT PARAMETERS
g = 9.81    # m/s^2, gravity
M = 10.0      # kg, larger mass
m = 5.0       # kg, smaller mass
H = 2.5     # m, initial height

# DERIVED PARAMETERS
t_star = sqrt((M + m)*H / (M - m)/g)        # sec, characteristic time
v_star = sqrt((M - m)*g*H / (M + m))        # m/s, characteristic speed
a_star = v_star/t_star                      # m/s^2, characteristic acceleration

# SOLUTION
datapoints = 150                            # number of points to plot
t_hat = linspace(0, sqrt(2), datapoints)    # dimensionless time
a_hat = -1 * ones(len(t_hat))               # dimensionless acceleration
v_hat = -1 * (t_hat)                              # dimensionless velocity
y_hat = -1 * (t_hat**2)/2 + 1

# PLOT RESULTS
fig = plt.figure(1)
plt.suptitle('Todhunter 1')

plt.subplot(3,1,1)
plt.plot(t_hat * t_star, a_hat * a_star)        
plt.ylabel('Acceleration (m/s^2)')
plt.xlabel('Time (sec)')

plt.subplot(3,1,2)
plt.plot(t_hat * t_star, v_hat * v_star)
plt.ylabel('Velocity (m/s)')
plt.xlabel('Time (sec)')

plt.subplot(3,1,3)
plt.plot(t_hat * t_star, y_hat * H)
plt.ylabel('Height (m)')
plt.xlabel('Time (sec)')

fig.subplots_adjust(hspace=1.5)
plt.savefig('tod1.png')
plt.show()

