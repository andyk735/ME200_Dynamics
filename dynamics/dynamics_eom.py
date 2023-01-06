"""
Created on Tue Sep 29 2020
@author: andyk
Program to find the equations of motion for the pendulum/cart system
"""
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np

# Parameters


k = 0.01 #M * g / l
A = 0.0

def eom(x, t):                      #g(x,t) = xdot, returns xdot
    
    x2, theta, x2dot, omega = x     # we are assuming someone will input a vector with theses values in this order
    #Fe = 0                         # external force on cart (= -k*x3 + A*np.cos(omegac*t))

    xdot = np.zeros(4)              # we are returning xdot(initialize vector with 4 zeros) 
    xdot[0] = x2dot
    xdot[1] = omega

    # Mass matrix
    M_mat = np.array({
        [M + m*np.sin(theta)**2,    0],
        [m*np.cos(theta),         m*l]
    })

    # Force vector
    f_vec = np.array([
    ])

    # Solve for accelerations q = [x2ddot, omegadot]
    xdot[2:] = np.linalg.solve(M_mat, f_vec).flatten()          # you get the acceleration a vector (we assign this to the remaining entries in xdot vector)
    return xdot

# ICs (in order to integrate to find x(t) for all time, we need initial conditions)
x0 = [1, np.pi/6, 0, 0]                                         # x2, theta, x2dot, omega
t = np.linspace(0, 6, 1000)                                     # 1000 points between 0 and 6
solution = odeint(eom, x0, t, rtol=1e-10)                       # solve integral of g(x,t) = xdot from x0 to t (idk abt that last part)

# Plot
plt.figure(1)
plt.plot(t, sol[:, 0], label = 'Displacement (m)')
plt.plot(t, sol[:, 1], label = 'Angle (rad)')
plt.xlabel('Time (s)')
plt.ylabel('Displacement (m), Angle (rad)')
plt.legend()