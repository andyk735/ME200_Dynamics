from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

# Close all plots
plt.close('all')


# Parameters
m1 = 1                  # kg
m2 = 1                  # kg
g = 9.8                 # m/s^2
ell = 1                 # m
ell_1 = 0.25 * ell      # m

# Equation of Motion
def eom(x, t):
    """
    Time: t
    Out: dxdt
    State vector: x = [ell_2 theta v_2 omega]
    """
    ell_2, theta, v_2, omega = x
    xdot = np.zeros(4)
    xdot[0] = v_2
    xdot[1] = omega
    # Mass matrix
    Mmat = np.array([
        [m1+m2   , 0        ],
        [0       , 1        ]
        ])
    # Force matrix
    fvec = np.array([
        [(m2*ell_2*(omega**2)) + m2*g*np.sin(theta) - m1*g  ],
        [(g*np.cos(theta) - 2*v_2*omega) / ell_2            ]
        ])
    # Solve for accelerations q = [ell_2ddot, omegadot] in Mmat q = fvec
    xdot[2:] = np.linalg.solve(Mmat, fvec).flatten()
    return xdot

# ICs
x0 = [ell - ell_1, 0, 0, 0]          # simulation 1 (ell_2, theta, v_2, omega)
t = np.linspace(0, 0.5, 500)
sol1 = odeint(eom, x0, t)


# Plot q, theta (t)
fig = plt.figure(1)
plt.suptitle('ME 200 Take-Home Midterm')                             # main title

plt.subplot(3,3,1)
plt.title('Lengths vs Time (Simulation 1: m1 < m2)')
plt.plot(t, ell - sol1[:, 0],label='l_1 [m]')
plt.plot(t, sol1[:, 0],label='l_2 [m]')
plt.xlabel('Time [s]')
plt.ylabel('l1, l2 [m]')
plt.legend(loc = 'upper left')
plt.ylim(0, 1.2)

plt.subplot(3,3,2)
plt.title('Theta vs Time (Simulation 1: m1 < m2)') 
plt.plot(t, sol1[:, 1], label='Theta [rad]')
plt.xlabel('Time [s]')
plt.ylabel('Theta [rad]')
plt.legend(loc = 'upper left')

plt.subplot(3,3,3)
plt.title('Velocities vs Time (Simulation 1: m1 < m2)') 
plt.plot(t, sol1[:, 2], label='v_2 [m/s]')
plt.plot(t, sol1[:, 3], label='omega [rad/s]')
plt.xlabel('Time [s]')
plt.ylabel('v_2 [m/s], omega [rad/s]')
plt.legend(loc = 'upper left')

fig.subplots_adjust(wspace=0.4, hspace = 0.4) 
plt.show()