from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

# Close all plots
plt.close('all')


# Parameters
M = 100
m = 1
g = 10
ell =  1

k = 0 # M * g / ell # matched resonant freq
#k = M * g / ell # matched resonant freq
A = 0.0
omegap = np.sqrt(g/ell)
omegac = np.sqrt(k/M)

# Equation of Motion
def eom(x, t):
    """
    Time: t
    Out: dxdt
    State vector: x = [theta x2 omega x2dot]
    """
    x2, theta, x2dot, omega = x
    Fe = - k*x2 + A*np.cos(omegac*t)
    xdot = np.zeros(4)
    xdot[0] = x2dot
    xdot[1] = omega
    # Mass matrixc
    Mmat = np.array([
        [M + m*np.sin(theta)**2, 0    ],
        [m*np.cos(theta)       , m*ell],
        ])
    # Forces
    fvec = np.array([
        [Fe + m*g*np.cos(theta)*np.sin(theta) + m*ell*omega**2*np.sin(theta)],
        [-m*g*np.sin(theta)]
        ])
    # Solve for accelerations q = [x2ddot, omegadot] in Mmat q = fvec
    xdot[2:] = np.linalg.solve(Mmat, fvec).flatten()
    return xdot

# ICs
x0 = [1, np.pi/6, 0, 0] # x2, theta, x2dot, omega
t1 = np.linspace(0, 10, 1000)
sol = odeint(eom, x0, t1)


# Plot
plt.figure(1)
plt.plot(t1, sol[:, 0],label='Displacement [m]')
plt.plot(t1, sol[:, 1],label='Angle [rad]')
plt.xlabel('Time [s]')
plt.ylabel('Displacement [m], Angle [rad]')
plt.legend()
plt.ylim(-1.25, 1.25)
plt.show()
#plt.savefig('images/pos_vel.pdf')
