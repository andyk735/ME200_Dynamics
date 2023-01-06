'''
ME200 - Workshop 5
Plendulum
Andrew Kim
'''

from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

# Close all plots
plt.close('all')

# Parameters
g = 9.8                 # gravity, m/s^2
p = 700                 # density, m^3/kg
h = 0.58                # height of plate, m
w = 0.197               # width of plate, m
t = 0.007               # thickness of plate, m

# Equation of Motion
def eom(u, t):
    # u is a vector such that theta = u[0] and thetadot = u[1]
    return [u[1], (3*g/2)*((w*np.cos(u[0]) + h*np.sin(u[0]))/(w**2 + h**2))]   #return thetadot, theta2dot

# ICs
u0 = [0, 0]                                 # theta, thetadot
time = np.linspace(0, 10, 1000)             # run simulation for 0.5 seconds
uSol = odeint(eom, u0, time)                # integrate [thetadot, theta2dot] to get [theta, thetadot]
thetaSol = uSol[:,0]                        # solution for theta

# Plot q, theta (t)
fig = plt.figure(1)
plt.suptitle('ME 200 Workshop 5 (Andrew Kim)')      # main title

plt.subplot(1,1,1)                                  # simulation 1
plt.title('Theta and Omega')
plt.plot(time, thetaSol,label='Theta')
plt.plot(time, uSol[:,1],label='Omega')
plt.xlabel('Time [s]')
plt.ylabel('Angle [rad], Angular Velocity [rad/s]')
plt.legend(loc = 'upper left')

#plt.savefig('wksp5.png') 
plt.show()                                          # show plots