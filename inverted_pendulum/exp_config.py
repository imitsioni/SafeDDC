"""
This file is used to set up parameters for the experiments with pendulum_mpc.py.
"""

import numpy as np

FRST_DID = 0  # First dataset ID

# ---- Pendulum Simulation Params
INP_MIN = -10.0  # Input space minimum
TH_MIN = -1.5 * np.pi  # Theta min (pendulum's position)
TH_D_MIN = -10.0  # Theta dot min (pendulum's velocity)
SIM_STPS = 150  # Simulation steps
TRLS = 10  # Number of MPC trials
FIX_INTIAL_ST = False  # Set the flag to True will use hardcoded values as initial states for the simulation
# Hardcoded initial states for comparisons
INIT_STATES = np.array([[-1.99515118, 3.5697508, -1.23121458, 4.70902953, 1.73190895, 1.71519266,
                         -1.90229281, -2.40907181, -3.7114882, -1.39460226],
                        [0.13083611, 0.44138971, -0.03106526, -0.12031505, -0.10615971, 0.12072505,
                         0.47966152, -0.10425984, -0.22563822, -0.28744314]])


# ---- Dynamics Model Params
BLOCK_SIZE = 1
HRZN = 10  # Predictive horizon
DO_MLTSTP = True  # Set the flag to True to use multi-step predictions from the network
OUTP_SIZ = 2  # Output size for the network

DNM_MDL = 'rrtc+random/converged_trials/relative_net/'  # Name of the dynamics model to be used

# ---- MPC Params
FSB_INP_MIN = -10  # Minimum of MPC's feasible inputs
N_FSB_INPS = 10  # Number of feasible inputs

# ---- OOD Detection Params
AE = "trained_models/vae_lr0.001_wd_0.0001_bet_1_hor_10_epo_254517766290206988366.pt"  # Encoder weights
OOD = "trained_models/gpr_flat_multistep_bl5.pt"  # OOD detector to use
PLG = "trained_models/poly_gpr_flat_multistep_th4_bl5_vae.joblib"  # Polygon approximation of GPR levels

USE_FLOW = False  # Setting it to False will use GPR for safety estimate instead of a Normalizing Flow
USE_PLG = True  # Setting it to True wil use the multipolygon approximation for GPR
SFT_STPS = 30