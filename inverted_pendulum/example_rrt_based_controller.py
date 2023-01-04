import time

import matplotlib.colors
import matplotlib.pyplot as plt
import matplotlib.patches as patch
from dynamics_library import PendulumCartContinuous, SampleAndHold, PendulumCartBroken
from dynamical_rrt import DynamicalRRT
from zonotope_lib import Zonotope, Box, plot_zonotope
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import pickle
from matplotlib.patches import Rectangle
import matplotlib as mpl
from copy import copy
mpl.rcParams['hatch.linewidth'] = 2.0

input_min = -10.0
input_max = -input_min
input_space = Box(np.array([[input_min, input_max]]))

theta_min = -1.5*np.pi
theta_max = -theta_min
theta_dot_min = -15.0
theta_dot_max = 15.0
state_space = Box(np.array([[theta_min, theta_max], [theta_dot_min, theta_dot_max]]))

root_state = np.array([[0], [0]])

dynamics = PendulumCartBroken()
discretization_step_s = 0.001
sample_time_s = 0.01
dynamics_discrete = SampleAndHold(dynamics, sample_time=sample_time_s, discretization_step=discretization_step_s)

x_0 = np.array([[-np.pi],[0.0]])

# '''Generate and Save tree'''
# target_set = Box(np.array([[1.0, 1.001], [1.0, 1.001]]))
# rrt = DynamicalRRT(dynamics, state_space, input_space, root_state, backward=True, target_region=target_set)
# t0 = time.time()
# rrt.planning(animation=False, max_iter=12000)
# print('Success? ', rrt.success)
# print("Building time: ", time.time()-t0, " seconds.")

'''Load Unrewired Tree'''
with open('generated_trees/tree_15000_unrewired_input10_doublefriction.pickle', 'rb') as f:
    rrt = pickle.load(f)
assert(isinstance(rrt, DynamicalRRT))

'''Cost before rewiring'''
costs = np.zeros((rrt.nnodes,))
for n in range(rrt.nnodes):
    costs[n] = rrt.get_cost(n)
print('Cost before rewiring: ', rrt.get_avg_cost())
states = state_space.sample(250)

fig, ax = plt.subplots()
fig, ax = rrt.draw_rand_trajectories(states, fig1=fig, ax1=ax)
high_fric_patch = Rectangle((0.5, theta_dot_min), 1.5, theta_dot_max-theta_dot_min, fill=True, facecolor='dimgray', hatch='//')
low_fric_patch = Rectangle((-2.0, theta_dot_min), 1.5, theta_dot_max-theta_dot_min, fill=True, facecolor='lightgray', hatch='//')
ax.add_patch(high_fric_patch)
ax.add_patch(low_fric_patch)
ax.set_ylim([-15, 15])
ax.set_xlabel(r'$\theta$')
ax.set_ylabel(r'$\dot{\theta}$')
ax.set_title('Unrewired tree')
ax.legend(handles=[low_fric_patch, high_fric_patch], labels=['2x friction zone', '4x friction zone'], loc='upper right')
'''Scenario'''
x = copy(x_0)
u = rrt.get_control(x)
ax.plot(x[0], x[1], 'r*')

T = 100
x_hist = np.zeros((2,T))
for t in range(T):
    u = rrt.get_control(x)
    x_hist[:, t:t + 1] = x
    for i in range(len(u)):
        x = dynamics_discrete(x, u[i])

ax.plot(x_hist[0, :], x_hist[1, :], 'r--', linewidth=3)

plt.savefig('tree_unrewired.svg')
plt.savefig('tree_unrewired.png')
plt.show()


'''Load Rewired Tree'''
with open('generated_trees/tree_15000_rewired_input10_doublefriction.pickle', 'rb') as f:
    rrt = pickle.load(f)
assert(isinstance(rrt, DynamicalRRT))

'''Cost after rewiring'''
costs = np.zeros((rrt.nnodes,))
for n in range(rrt.nnodes):
    costs[n] = rrt.get_cost(n)
print('Cost after rewiring: ', rrt.get_avg_cost())

fig, ax = plt.subplots()
fig, ax = rrt.draw_rand_trajectories(states, fig1=fig, ax1=ax)
high_fric_patch = Rectangle((0.5, theta_dot_min), 1.5, theta_dot_max-theta_dot_min, fill=True, facecolor='dimgray', hatch='//')
low_fric_patch = Rectangle((-2.0, theta_dot_min), 1.5, theta_dot_max-theta_dot_min, fill=True, facecolor='lightgray', hatch='//')
ax.add_patch(high_fric_patch)
ax.add_patch(low_fric_patch)
ax.set_ylim([-15, 15])
ax.set_xlabel(r'$\theta$')
ax.set_ylabel(r'$\dot{\theta}$')
ax.set_title('Rewired tree')
ax.legend(handles=[low_fric_patch, high_fric_patch], labels=['2x friction zone', '4x friction zone'], loc='upper right')
'''Scenario'''
x = copy(x_0)
u = rrt.get_control(x)
ax.plot(x[0], x[1], 'r*')

T = 100
x_hist = np.zeros((2,T))
for t in range(T):
    u = rrt.get_control(x)
    x_hist[:, t:t + 1] = x
    for i in range(len(u)):
        x = dynamics_discrete(x, u[i])

ax.plot(x_hist[0, :], x_hist[1, :], 'r--', linewidth=3)


plt.savefig('tree_rewired.png')
plt.savefig('tree_rewired.svg')
plt.show()