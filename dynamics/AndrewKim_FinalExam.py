'''
ME200 - Take Home Final Exam
"Tumbling Cell Phone"
Andrew Kim
'''

from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

# Close all plots
plt.close('all')

# Parameters
h = 144         # mm
w = 71.4        # mm
d = 8.1         # mm
m = 188         # g

# Mass Moments of Inertia
J1 = (1/12)*m*(h**2+w**2)
J2 = (1/12)*m*(h**2+d**2)
J3 = (1/12)*m*(d**2+w**2)

# Equation of Motion
def eom(x, t):
    """
    Time: t
    Out: dxdt
    State vector: x = [omega1 omega2 omega3]
    """
    omega1, omega2, omega3 = x
    xdot = np.zeros(3)
    # no initializations necessary because we are solving for all 3 terms
    # LHS Mass Moment of Inertia matrix
    Mmat = np.array([
        [J1     , 0     , 0     ],
        [0      , J2    , 0     ],
        [0      , 0     , J3     ]
        ])
    # RHS matrix
    fvec = np.array([
        [   omega2*omega3*(J2-J3)   ],
        [   omega1*omega3*(J3-J1)   ],
        [   omega1*omega2*(J1-J2)   ]
        ])
    # Solve for accelerations q = [ell_2ddot, omegadot] in Mmat q = fvec
    xdot[0:] = np.linalg.solve(Mmat, fvec).flatten()
    return xdot

# ICs
t = np.linspace(0, 30, 1000)            # run simulation for 30 seconds

x1 = [1, 0.01, 0.01]                   
sol1 = odeint(eom, x1, t)               # solution 1
x2 = [0.01, 1, 0.01]
sol2 = odeint(eom, x2, t)               # solution 2
x3 = [0.01, 0.01, 1]
sol3 = odeint(eom, x3, t)               # solution 3

# Plot q, theta (t)
fig = plt.figure(figsize=(30, 8))
plt.suptitle('ME 200 Take-Home Final Exam (Andrew Kim)')                            # main title

plt.subplot(1,3,1)                                                                  # simulation 1
plt.title('Scenario 1: Spinning around 1-axis')
plt.plot(t, sol1[:, 0],label='Omega 1 [rad/s]')
plt.plot(t, sol1[:, 1],label='Omega 2 [rad/s]')
plt.plot(t, sol1[:, 2],label='Omega 3 [rad/s]')
plt.xlabel('Time [s]')
plt.ylabel('Angular Velocity [rad/s]')
plt.legend(loc = 'upper right')
#plt.ylim(0, 1.2)

plt.subplot(1,3,2)                                                                  # simulation 2
plt.title('Scenario 2: Spinning around 2-axis')
plt.plot(t, sol2[:, 0],label='Omega 1 [rad/s]')
plt.plot(t, sol2[:, 1],label='Omega 2 [rad/s]')
plt.plot(t, sol2[:, 2],label='Omega 3 [rad/s]')
plt.xlabel('Time [s]')
plt.ylabel('Angular Velocity [rad/s]')
plt.legend(loc = 'upper right')

plt.subplot(1,3,3)                                                                  # simulation 3
plt.title('Scenario 3: Spinning around 3-axis')
plt.plot(t, sol3[:, 0],label='Omega 1 [rad/s]')
plt.plot(t, sol3[:, 1],label='Omega 2 [rad/s]')
plt.plot(t, sol3[:, 2],label='Omega 3 [rad/s]')
plt.xlabel('Time [s]')
plt.ylabel('Angular Velocity [rad/s]')
plt.legend(loc = 'upper right')

fig.subplots_adjust(wspace=0.4, hspace = 0) 
plt.show()                                                                          # show plots
