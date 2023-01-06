"""
ME200 - Workshop 3
Inverted Pendulum on a Springy Cart
@authors: Aaron Schmitz, Andrew Kim, Zachary Potoskie
"""

from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

# Close all plots
plt.close('all')

# Parameters
M = 1                       # mass of cart, kg
m = 0.25                    # mass of pendulum, kg
g = 10                      # gravity, m/s^2

ell =  0.2                  # length of pendulum, m
so = 0.1                    # original distance of cart from wall, m

b = 0.2                     # damping coefficient
k = 2                       # spring coefficient

# Equation of Motion
def eom(x, t):
    """
    Time: t
    Out: dxdt
    State vector: x = [q theta qdot omega]
    """
    q, theta, qdot, omega = x
    xdot = np.zeros(4)
    xdot[0] = qdot
    xdot[1] = omega
    # Mass matrix
    Mmat = np.array([
        [M + m*np.sin(theta)**2, 0    ],
        [m*np.cos(theta)       , m*ell],
        ])
    # Forces
    fvec = np.array([
        [-1*k*(q-so) -b*qdot - m*g*np.cos(theta)*np.sin(theta) + m*ell*omega**2*np.sin(theta)],
        [m*g*np.sin(theta)]
        ])
    # Solve for accelerations
    xdot[2:] = np.linalg.solve(Mmat, fvec).flatten()
    return xdot

# ICs
x0_1 = [so, np.pi/6, 0, 0]          # simulation 1 (q, theta, qdot, omega)
x0_2 = [so, np.pi-np.pi/6, 0, 0]    # simulation 2 (q, theta, qdot, omega)
t = np.linspace(0,20, 1000)
sol_1 = odeint(eom, x0_1, t)        # solution to sim1
sol_2 = odeint(eom, x0_2, t)        # solution to sim2


# Plot
fig = plt.figure(1)
plt.suptitle('Inverted Pendulum on a Springy Cart')                             # main title

plt.subplot(2,2,1)
plt.title('Simulation 1 (Distance, Angle vs Time)')                             # sub title 
plt.plot(t, sol_1[:, 0],label='Displacement [m]')
plt.plot(t, sol_1[:, 1],label='Angle [rad]')
plt.xlabel('Time [s]')
plt.ylabel('Displacement [m], Angle [rad]')
plt.legend(loc = 'upper right')

plt.subplot(2,2,2)
plt.title('Simulation 1 (Velocity, Angular Velocity vs Time)')                  # sub title 
plt.plot(t, sol_1[:, 2],label='Velocity [m/s]')
plt.plot(t, sol_1[:, 3],label='Angular Velocity [rad/s]')
plt.xlabel('Time [s]')
plt.ylabel('Velocity [m/s], Angular Velocity [rad/s]')
plt.legend(loc = 'upper right')

plt.subplot(2,2,3)
plt.title('Simulation 2 (Distance, Angle vs Time)')                             # sub title 
plt.plot(t, sol_2[:, 0],label='Displacement [m]')
plt.plot(t, sol_2[:, 1],label='Angle [rad]')
plt.xlabel('Time [s]')
plt.ylabel('Displacement [m], Angle [rad]')
plt.legend(loc = 'upper right')

plt.subplot(2,2,4)
plt.title('Simulation 2 (Velocity, Angular Velocity vs Time)')                  # sub title 
plt.plot(t, sol_2[:, 2],label='Velocity [m/s]')
plt.plot(t, sol_2[:, 3],label='Angular Velocity [rad/s]')
plt.xlabel('Time [s]')
plt.ylabel('Velocity [m/s], Angular Velocity [rad/s]')
plt.legend(loc = 'upper right')

fig.subplots_adjust(wspace=0.4, hspace = 0.4)                                   # adjust spacing
plt.show()                                                                      # show plots
