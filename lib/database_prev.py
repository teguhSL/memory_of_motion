from crocoddyl import ShootingProblem, SolverDDP
from crocoddyl.fddp import SolverFDDP
from crocoddyl import CallbackDDPLogger, CallbackDDPVerbose, CallbackSolverDisplay
from crocoddyl import loadTalos, loadTalosLegs, loadHyQ, m2a
#from utils import printPhaseInfo, setWhiteBackground, displayPhaseMotion, displayContactPlacements
from crocoddyl.utils import *
#from utils import runningModel, impactModel
import numpy as np
import pickle
from pinocchio.libpinocchio_pywrap import SE3
from locomote import ContactSequenceHumanoid

def subsample(X,N):
    '''Subsample in N iterations the trajectory X. The output is a 
    trajectory similar to X with N points. '''
    nx  = X.shape[0]
    idx = np.arange(float(N))/(N-1)*(nx-1)
    hx  = []
    for i in idx:
        i0 = int(np.floor(i))
        i1 = int(np.ceil(i))
        di = i%1
        x  = X[i0,:]*(1-di) + X[i1,:]*di
        hx.append(x)
    return np.vstack(hx)

def extract_contact_sequence(filename):
    cs = ContactSequenceHumanoid(0)
    cs.loadFromXML(filename + '/contact_sequence_trajectory.xml', 'contact_sequence')
    raw_phases = cs.contact_phases

    rfs = []
    lfs = []
    timesteps = []
    num_phases = len(raw_phases)
    for cur_phase in raw_phases:
        rf = cur_phase.RF_patch
        lf = cur_phase.LF_patch
        rfs.append(rf)
        lfs.append(lf)
        timesteps.append(cur_phase.time_trajectory[-1] - cur_phase.time_trajectory[0])

    timesteps = np.array(timesteps)/0.005
    timesteps = timesteps.astype(int)
    return rfs, lfs, timesteps

#read the trajectory
def extract_traj(filename, timesteps, sub_fac = 0.0375/0.005,  offset_time = 100):
    f = open(filename ,'rb')
    traj = []
    for line in f.readlines():
        traj.append(np.array([float(l) for l in line.split()])[1:]) #start from 1, because column 0 is the timestep

    #starting from time 0.5s due to the dataset error
    traj = traj[offset_time:]
    
    #split the trajectory according to the contact phases
    trajs = []
    i = 0
    for timestep in timesteps:
        cur_traj = np.vstack(traj[i:i+timestep])
        n = cur_traj.shape[0]
        cur_traj = list(subsample(cur_traj, np.round(n/sub_fac)))
        i+= timestep
        trajs.append(cur_traj)
        
    f.close()
    return trajs

def store_for_crocoddyl(timesteps, lfs, rfs, trajs, vel_trajs, delta_t):
    q_init = trajs[0][0].copy()
    v_init = vel_trajs[0][0].copy()
    q = np.zeros(39)
    v = np.zeros(38)
    u = np.zeros(32)

    phases = []
    for i in range(len(timesteps)):
        phase = dict()
        phase['support_contacts'] = []
        phase['swing_contacts'] = []
        rf = rfs[i]
        lf = lfs[i]
        timestep = timesteps[i]

        #contact phase
        if rf.active:
            phase['support_contacts'] += ['right_sole_link']
        else:
            phase['swing_contacts'] += ['right_sole_link']
            trans =  rfs[i+1].placement.translation
            trans[2] -= 0.107
            rot =  rfs[i+1].placement.rotation
            phase['right_sole_link'] = SE3(rot,trans)

        if lf.active:
            phase['support_contacts'] += ['left_sole_link']
        else:
            phase['swing_contacts'] += ['left_sole_link']
            trans =  lfs[i+1].placement.translation
            trans[2] -= 0.107
            rot =  lfs[i+1].placement.rotation
            phase['left_sole_link'] = SE3(rot,trans)

        #trajectory info
        phase['ts'] = timestep*[delta_t]
        phase['qs'] = trajs[i][:]
        phase['vs'] = vel_trajs[i][:]
        phase['us'] = timestep*[u]

        #if there is no swing contact, delete the field
        if len(phase['swing_contacts']) < 1:
            phase.pop('swing_contacts')

        #insert initial state 
        if i == 0: 
            phase['qs'].insert(0,q_init)
            phase['vs'].insert(0,v_init)

        phases.append(phase)

    return phases