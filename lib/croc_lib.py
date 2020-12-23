import crocoddyl
from crocoddyl import ShootingProblem, SolverDDP
from crocoddyl import SolverFDDP
from crocoddyl import CallbackLogger, CallbackVerbose, CallbackDisplay
from crocoddyl import loadTalos, loadTalosLegs, loadHyQ, m2a
from utils import printPhaseInfo, setWhiteBackground, displayPhaseMotion, displayContactPlacements
from utils import runningModel, impactModel
import numpy as np
import pickle
from pinocchio.libpinocchio_pywrap import SE3
from locomote import ContactSequenceHumanoid
from database import *
from memmo_utils import *



def examine_phases(phases,left_foot, right_foot,viz,  foot_marker,is_pause=True):
    for i,phase in enumerate(phases):
        print 'Phase number ' + str(i)
        qs = phase['qs']
        vs = phase['vs']
        ts = phase['ts']
        us = phase['us']
        print 'The lengths of qs,vs,us,ts are {},{},{},{}'.format(len(qs),len(vs),len(us),len(ts))
        Tl,Tr = calc_foot_pose(qs[0],left_foot, right_foot)
        print 'The initial foot poses are:\n {} \n{}'.format(Tl,Tr)
        foot_poses = [Tl,Tr]
        Tl,Tr = calc_foot_pose(qs[-1],left_foot, right_foot)
        print 'The final foot poses are:\n {} \n{}'.format(Tl,Tr)
        
        
        #obtain the swing foot destination
        try:
            swing_link = phase['swing_contacts'][0]
            print '{} is moving to'.format(swing_link)
            swing_goal = phase[swing_link]
            print(SE3toMat(swing_goal))
            foot_poses += [SE3toMat(swing_goal)]
            foot_marker.publish(foot_poses)
            print foot_poses
            viz.vis_traj(np.array(qs))
            if is_pause: raw_input()
        except:
            print 'Double support phase!'
            viz.vis_traj(np.array(qs))
            if is_pause: raw_input()
            continue

            
def compare_phases_with_result(phases, xs,left_foot, right_foot,viz, is_pause=True):
    cur_index = 0
    for i,phase in enumerate(phases):
        print 'Phase number ' + str(i)
        interval = len(phase['ts'])
        print interval
        qs = xs[cur_index:cur_index+interval,0:39]
        cur_index += interval
        
        Tl,Tr = calc_foot_pose(qs[0],left_foot, right_foot)
        print 'The initial foot poses are:\n Left: \n {}  \n Right: \n{}'.format(Tl,Tr)
        Tl,Tr = calc_foot_pose(qs[-1],left_foot, right_foot)
        print 'The final foot poses are:\n Left: \n {}  \n Right: \n{}'.format(Tl,Tr)
        #obtain the swing foot destination
        try:
            swing_link = phase['swing_contacts'][0]
            print '{} is moving to'.format(swing_link)
            swing_goal = phase[swing_link]
            print(SE3toMat(swing_goal))
            foot_marker.publish([swing_goal])
            viz.vis_traj(np.array(qs))
            if is_pause: raw_input()
        except:
            print 'Double support phase!'
            viz.vis_traj(np.array(qs))
            if is_pause: raw_input()
            continue


    

def store_for_crocoddyl(traj, vel_traj, u_traj, T_lefts, T_rights, data_phases, delta_t, q_init = None, v_init = None):
    if q_init is None: q_init = traj[0].copy()
    if v_init is None: v_init = vel_traj[0].copy()
    q = np.zeros(39)
    v = np.zeros(38)
    u = np.zeros(32)

    phases = []
    for phase_index, data_phase in enumerate(data_phases):
        phase = dict()
        phase['support_contacts'] = []
        phase['swing_contacts'] = []

        #add contacts
        #r_active = contact_rights[data_phase[0]]
        #l_active = contact_lefts[data_phase[0]]

        #contact phase
        if T_rights[phase_index] is not None:
            phase['support_contacts'] += ['right_sole_link']
        else:
            phase['swing_contacts'] += ['right_sole_link']
            pose = SE3()
            pose.translation = T_rights[phase_index + 1][:3,3:]  #- np.array([[0],[0],[0.103]])
            pose.rotation = T_rights[phase_index + 1][:3,:3]
            phase['right_sole_link'] = pose

        if T_lefts[phase_index] is not None:
            phase['support_contacts'] += ['left_sole_link']
        else:
            phase['swing_contacts'] += ['left_sole_link']
            pose = SE3()
            pose.translation = T_lefts[phase_index + 1][:3,3:]  #- np.array([[0],[0],[0.103]])
            pose.rotation = T_lefts[phase_index + 1][:3,:3]
            phase['left_sole_link'] = pose

        phases.append(phase)

        #if there is no swing contact, delete the field
        if len(phase['swing_contacts']) < 1:
            phase.pop('swing_contacts')         

        #trajectory info
        timestep = len(data_phase)
        phase['ts'] = timestep*[delta_t]
        phase['qs'] = list(traj[data_phase])#timestep*[q]#
        phase['vs'] = list(vel_traj[data_phase])#timestep*[v]#l
        phase['us'] = list(u_traj[data_phase])#timestep*[u]#

        #insert initial state 
        if phase_index == 0: 
            phase['qs'].insert(0,q_init)
            phase['vs'].insert(0,v_init)

    return phases

def convert_to_croc_format(x_input, traj, key = 'left', vel_traj = None, u_traj = None, data_phases = None, sub_fac = 40, q_init = None, v_init = None, filename = 'data.txt'):
    if q_init is None: q_init = traj[0].copy()
    if v_init is None: v_init = vel_traj[0].copy()
        
    T_lefts, T_rights = calc_foot_T(x_input,move=key)
    delta_t = 0.001*sub_fac
    
    #subsample the trajectories
    T = traj.shape[0]
    intervals = np.concatenate([np.arange(0,T,sub_fac),[T-1]])
    T_new = len(intervals)
    traj = traj[intervals]
    
    if vel_traj is None:
        vel_traj = np.zeros((traj.shape[0], 38))
    else:
        vel_traj = vel_traj[intervals]

    if u_traj is None:
        u_traj = np.zeros((traj.shape[0], 32))
    else:
        u_traj = u_traj[intervals]
        
    if data_phases is None:
        data_phases = []
        data_phases.append(np.arange(0,38))
        data_phases.append(np.arange(38,73))
        data_phases.append(np.arange(73,110))
    else:
        data_phases = subsample_phases(data_phases, sub_fac=sub_fac)
    
    #store in crocoddyl format
    phases = store_for_crocoddyl(traj, vel_traj, u_traj, T_lefts, T_rights, data_phases, delta_t, q_init = q_init, v_init = v_init)
    pickle.dump(phases,open(filename,'wb'))
    return traj,data_phases

def define_croc_problem(ROBOT, FILENAME, num_phases = 4, is_warmstart=True):
    ENABLE_ARMATURE = False
    ENABLE_DISPLAY = False

    if not ENABLE_ARMATURE: ROBOT.model.armature[6:] = 1. # this disable any armature

    
    # Loading the memory of motion
    memmofile = open(FILENAME, 'rb')
    memmo = pickle.load(memmofile)
    memmofile.close()
    memmo = memmo[:num_phases]
    
    # Create the initial state
    q0 = memmo[0]['qs'][0].copy()
    # q0[0:2] = XY_INIT TODO: discuss if this is needed
    v0 = memmo[0]['vs'][0].copy()
    x0 = m2a(np.concatenate([q0, v0]))
    ROBOT.model.defaultState = x0.copy()
    
    #if there is no warmstart, set the states and controls to be all zeros
    if is_warmstart is False:
        for i in range(len(memmo)):
            for j in range(len(memmo[i]['qs'])):
                memmo[i]['qs'][j] *= 0
                memmo[i]['vs'][j] *= 0
            for j in range(len(memmo[i]['us'])):
                memmo[i]['us'][j] *= 0
    
    #copy back the initial state         
    memmo[0]['qs'][0] = q0.copy()
    memmo[0]['vs'][0] = v0.copy()
    
    #reference posture
    qref = np.array(ROBOT.model.referenceConfigurations['half_sitting']).flatten()
    vref = np.zeros(len(v0))
    xref = np.concatenate([qref,vref])
    
    xs = []
    us = []
    ts = []
    memmoModels = []
    print('Building the set of action models and warmstart from Memmo')
    if ENABLE_DISPLAY:
        setWhiteBackground(ROBOT)
    for i, phase in enumerate(memmo):
        # Printing the phase info and display contact plan
        printPhaseInfo(phase, i)
        if ENABLE_DISPLAY:
            displayContactPlacements(ROBOT, phase, i)

        # Extracting the warm-start and contact sequences
        qs = phase['qs']
        vs = phase['vs']
        for q, v in zip(qs, vs):
            xs.append(np.concatenate([q, v]))
        for u in phase['us']:
            us.append(u)
        tp = phase['ts']
        ts += tp
        assert(len(xs) == len(us) + 1), "qs and us should have the same dimension"

        contactIds = \
            [ROBOT.model.getFrameId(f) for f in phase['support_contacts']]
        if phase.has_key('swing_contacts'):
            for j, t in enumerate(tp[:-1]):
                if i == 0:
                    xp = xs[j + 1]
                else:
                    xp = xs[j]
                #if is_warmstart is False:
                xp = xref
                memmoModels += \
                    [runningModel(rmodel=ROBOT.model,
                                  contactIds=contactIds,
                                  integrationStep=t,
                                  defaultState=xp)]
            contactPlacements = {c: phase[c] for c in phase['swing_contacts']}
            contactIds += [ROBOT.model.getFrameId(c) for c in phase['swing_contacts']]
            memmoModels += [impactModel(rmodel=ROBOT.model,
                                        contactIds=contactIds,
                                        contactPlacements=contactPlacements,
                                        defaultState=xp)]
            us[-1] = np.zeros(0)
        else:
            for j, t in enumerate(tp):
                xp = xs[j]
                #if is_warmstart is False:
                xp = xref
                memmoModels += \
                    [runningModel(rmodel=ROBOT.model,
                                  contactIds=contactIds,
                                  integrationStep=t,
                                  defaultState=xp)]
    termModel = runningModel(rmodel=ROBOT.model,
                             contactIds=contactIds,
                             integrationStep=0.,
                             defaultState=xp)

    # Creating the shooting problem
    problem = ShootingProblem(x0, memmoModels, termModel)
    
    return problem,xs,us,ts

def solve_problem(ROBOT,problem, xs, us, TYPE_OF_SOLVER = 'DDP',STOP_THRESHOLD=1e-6, ENABLE_ITER_DISPLAY = False, maxiter = 50, recalc_u = True):
    # Selecting the desired optimal control solver
    if TYPE_OF_SOLVER == 'FDDP':
        solver = SolverFDDP(problem)
    elif TYPE_OF_SOLVER == 'DDP':
        solver = SolverDDP(problem)
    else:
        print("Warning: wrong type of solver ... selecting DDP one.")
        solver = SolverDDP(problem)

    # Setting the stopping threshold
    solver.th_stop = STOP_THRESHOLD

    # Enabling the desired set of callbacks
    solver.callback = [CallbackLogger(), CallbackVerbose()]
    if ENABLE_ITER_DISPLAY:
        solver.callback += [CallbackDisplay(ROBOT, 4, 1)]

    # Solving the optimal control problem
    print
    print("*** SOLVE ***")
    from crocoddyl import IntegratedActionModelEuler
    if recalc_u:
        us0 = [ m.differential.quasiStatic(d.differential, xs[k][:m.nq]) \
                if isinstance(m,IntegratedActionModelEuler) else np.zeros(0) \
                for k, (m,d) in enumerate(zip(solver.problem.runningModels, solver.problem.runningDatas)) ]
        solver.solve(maxiter=maxiter,
                     init_xs=xs,
                     init_us=us0
                    )
    else:
        solver.solve(maxiter=maxiter,
                     init_xs=xs,
                     init_us=us
                    )
        
    return solver


def runningModel(rmodel, contactIds, integrationStep=1e-2, defaultState=None, defaultCtrl=None):
    """ Creating the action model for floating-base systems.

    :param rmodel: Pinocchio robot model.
    :param contactIds: list of frame ids that should be in contact.
    :param integrationStep: duration of the integration step in seconds.
    :param defaultState: state uses for regularization
    :param defaultCtrl: control uses for regularization
    """
    actModel = crocoddyl.ActuationModelFreeFloating(rmodel)
    State = crocoddyl.StatePinocchio(rmodel)

    # Creating a 6D multi-contact model, and then including the supporting foot
    contactModel = crocoddyl.ContactModelMultiple(rmodel)
    for cid in contactIds:
        contact = crocoddyl.ContactModel6D(rmodel, cid, ref=None, gains=[0., 50.])
        contactModel.addContact('contact%d' % cid, contact)

    # Creating the cost model for a contact phase
    costModel = crocoddyl.CostModelSum(rmodel, actModel.nu)
    stateWeights = np.array([1.] * 6 + [1.] * (rmodel.nv - 6) + [10] * rmodel.nv)
    if defaultState is None:
        defaultState = rmodel.defaultState
    stateReg = crocoddyl.CostModelState(
        rmodel, State, defaultState, actModel.nu,
        activation=crocoddyl.ActivationModelWeightedQuad(stateWeights))
    if defaultCtrl is None:
        defaultCtrl = np.zeros(actModel.nu)
    ctrlReg = crocoddyl.CostModelControl(rmodel, actModel.nu, defaultCtrl)
    costModel.addCost('stateReg', stateReg, 1e-1)
    costModel.addCost('ctrlReg', ctrlReg, 1e-4)

    # Creating the action model for the KKT dynamics with simpletic Euler
    # integration scheme
    dmodel = crocoddyl.DifferentialActionModelFloatingInContact(
        rmodel, actModel, contactModel, costModel)
    model = crocoddyl.IntegratedActionModelEuler(dmodel)
    model.timeStep = integrationStep
    return model


def impactModel(rmodel, contactIds, contactPlacements, defaultState=None):
    """ Creating the impact action model for floating-base systems.

    :param rmodel: Pinocchio robot model.
    :param contactIds: list of frame ids that should be in contact.
    :param contactPlacements: dict of key frame ids and SE3 values of contact
    placement references. This value should typically be provided for effector
    landing.
    """
    State = crocoddyl.StatePinocchio(rmodel)

    # Creating a 6D multi-contact model, and then including the supporting foot
    impulses = {"impulse%d" % cid:
                crocoddyl.ImpulseModel6D(rmodel, cid) for cid in contactIds}
    impulseModel = crocoddyl.ImpulseModelMultiple(rmodel, impulses)

    # Creating the cost model for a contact phase
    costModel = crocoddyl.CostModelSum(rmodel, 0)
    stateWeights = np.array([1.] * 6 + [1.] * (rmodel.nv - 6) + [10] * rmodel.nv)
    if defaultState is None:
        defaultState = rmodel.defaultState
    stateReg = crocoddyl.CostModelState(
        rmodel, State, defaultState, 0,
        activation=crocoddyl.ActivationModelWeightedQuad(stateWeights))
    costModel.addCost('stateReg', stateReg, 1e-1)

    for cp, ref in contactPlacements.items():
        fid = rmodel.getFrameId(cp)
        contactWeights = np.array([1.] * 6)
        contactPlacementCost = \
            crocoddyl.CostModelFramePlacement(
                rmodel, fid, ref, nu=0,
                activation=crocoddyl.ActivationModelWeightedQuad(contactWeights))
        costModel.addCost("contactPlacement%d" % fid, contactPlacementCost, 1e5)

    # Creating the action model for the KKT dynamics with simpletic Euler
    # integration scheme
    model = crocoddyl.ActionModelImpact(rmodel, impulseModel, costModel)
    return model



def setWhiteBackground(robot):
    if not hasattr(robot, 'viewer'):
        # Spawn robot model
        robot.initDisplay(loadModel=True)
        # Set white background and floor
        window_id = robot.viewer.gui.getWindowID('python-pinocchio')
        robot.viewer.gui.setBackgroundColor1(window_id, [1., 1., 1., 1.])
        robot.viewer.gui.setBackgroundColor2(window_id, [1., 1., 1., 1.])
        robot.viewer.gui.addFloor('hpp-gui/floor')
        robot.viewer.gui.setScale('hpp-gui/floor', [0.5, 0.5, 0.5])
        robot.viewer.gui.setColor('hpp-gui/floor', [0.7, 0.7, 0.7, 1.])
        robot.viewer.gui.setLightingMode('hpp-gui/floor', 'OFF')


def displayPhaseMotion(robot, qs, ts):
    if len(qs) == len(ts) + 1:
        for k, q in enumerate(qs[1:]):
            dt = ts[k]
            robot.display(np.matrix(q).T)
            time.sleep(dt)
    else:
        for k, q in enumerate(qs):
            dt = ts[k]
            robot.display(np.matrix(q).T)
            time.sleep(dt)


def displayContactPlacements(robot, phase, index):
    if not hasattr(robot, 'viewer'):
        # Spawn robot model
        robot.initDisplay(loadModel=True)
    idx = 0
    if phase.has_key('swing_contacts'):
        for c in phase['swing_contacts']:
            body_id = robot.model.getBodyId(c)
            parent_id = robot.model.frames[body_id].parent
            M = robot.model.frames[body_id].placement
            mesh = robot.visual_model.geometryObjects[parent_id - 1]
            meshName = robot.getViewerNodeName(mesh, pinocchio.GeometryType.VISUAL) + '_loc%d%d' % (index, idx)
            robot.viewer.gui.addMesh(meshName, mesh.meshPath)
            # robot.viewer.gui.setColor(meshName, [0., 0., 1., 0.5])
            robot.viewer.gui.setHighlight(meshName, 2)
            contactPose = pinocchio.se3ToXYZQUATtuple(M.actInv(phase[c]))
            robot.viewer.gui.applyConfiguration(meshName, contactPose)
            idx += 1

            


def printPhaseInfo(phase, index):
    print('phase %d:' % index)
    print('  nodes: %d' % len(phase['qs']))
    print('  contacts:')
    print('    suppport: %s' % phase['support_contacts'])
    if phase.has_key('swing_contacts'):
        print('    swing: %s' % phase['swing_contacts'])
    else:
        print('    swing: []')



def recordSolutionToFile(robot, filename, solver):
    MEMMO = []
    for i, d in enumerate(solver):
        start = 0
        end = 0
        previousContactNames = []
        previousSwingNames = []
        for k, m in enumerate(d.models()):
            contacts = m.differential.contact.contacts
            costs = m.differential.costs.costs
            contactNames = [robot.model.frames[int(c.split('_')[1])].name for c in contacts]
            swingNames = []
            for c in costs:
                if len(c.split('_')) == 2 and c.split('_')[0] == 'footTrack':
                    swingNames += [robot.model.frames[int(c.split('_')[1])].name]
            if contactNames != previousContactNames and k != 0 or k is len(d.models()) - 1:
                start = end
                end = k
                if start == 0 and i == 0:
                    qs = [x[:robot.nq] for x in d.xs[start:end + 1]]
                    vs = [x[robot.nq:] for x in d.xs[start:end + 1]]
                else:
                    qs = [x[:robot.nq] for x in d.xs[start + 1:end + 1]]
                    vs = [x[robot.nq:] for x in d.xs[start + 1:end + 1]]
                ts = [t.timeStep for t in d.models()[start:end]]
                us = [u for u in d.us[start:end]]
                phase = {}
                phase['support_contacts'] = previousContactNames
                if len(previousSwingNames) != 0:
                    phase['swing_contacts'] = previousSwingNames
                phase['qs'] = qs
                phase['vs'] = vs
                phase['us'] = us
                phase['ts'] = ts
                qT = crocoddyl.a2m(qs[-1])
                rdata = robot.model.createData()
                pinocchio.forwardKinematics(robot.model, rdata, qT)
                pinocchio.updateFramePlacements(robot.model, rdata)
                for s in previousSwingNames:
                    M = rdata.oMf[robot.model.getFrameId(s)]
                    phase[s] = M
                MEMMO += [phase]
            previousContactNames = contactNames
            previousSwingNames = swingNames

    # Recording the data
    output = open(filename, 'wb')
    pickle.dump(MEMMO, output)
    output.close()
