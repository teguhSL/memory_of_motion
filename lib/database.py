import numpy as np
from pinocchio import SE3
from memmo_utils import *
from mlp.utils.status import Status
import mlp.utils.wholebody_result as wb_res


def vec2mat(q):
    #convert a flattened SE3 from the dataset (foot trajectories) to a 4x4 matrix 
    q = np.array(q).flatten()
    p = q[0:3]
    R = q[3:].reshape(3,3).T
    return affines.compose(p, R, np.ones(3))

def rectify_quat(q):
    #change the dataset quaternion format (x,y,z,w) to transform3d format (w,x,y,z)
    return np.concatenate([q[-1:], q[0:-1]])

def derectify_quat(q):
    #change  transform3d format (w,x,y,z) to the dataset quaternion format (x,y,z,w) 
    return np.concatenate([q[1:], q[0:1]])

def SE3toMat(T):
    mat = np.eye(4)
    mat[:3,:3] = np.array(T.rotation)
    mat[:3,3] = np.array(T.translation).flatten()
    return mat

def PosetoMat(pose, is_3D = False):
    #convert the foot pose (x,y,(z), angle) to homogeneous matrix
    if is_3D:
        p = np.array([pose[0],pose[1],pose[2]])
        R = axangles.axangle2mat(np.array([0.,0.,1.]), pose[3])    
    else:
        p = np.array([pose[0],pose[1],0.])
        R = axangles.axangle2mat(np.array([0.,0.,1.]), pose[2])
    T = affines.compose(p,R, np.ones(3))
    return T

def MattoPose(T, pose_type = '2D'):
    if pose_type == '3D':
        pose = np.zeros(4)
        pose[0:3] = T[:3,3]
        pose[3] = axangles.mat2axangle(T[:3,:3])[1]
    elif pose_type == '2D':
        pose = np.zeros(3)
        pose[0:2] = T[:2,3]
        pose[2] = axangles.mat2axangle(T[:3,:3])[1]
    elif pose_type == '7D':
        pose = np.zeros(7)
        pose[0:3] = T[:3,3]
        pose[3:7] = derectify_quat(quaternions.mat2quat(T[:3,:3]))
    return pose

            
def normalize(x):
    return x/np.linalg.norm(x)

def calc_foot_pose(q,left_foot, right_foot, pose_type='Mat'):
    q_left = q[7:13]
    q_right = q[13:19]
    Tl = np.array(left_foot.forward(q_left))
    Tr = np.array(right_foot.forward(q_right))

    T = affines.compose(T=q[0:3], R=quaternions.quat2mat(rectify_quat(q[3:7])),Z=np.ones(3))
    Tl = np.dot(T,Tl)
    Tr = np.dot(T,Tr)
    if pose_type == 'Mat':
        return Tl, Tr
    else:
        return MattoPose(Tl,pose_type), MattoPose(Tr,pose_type)

def calc_root_pose(q):
    return affines.compose(q[0:3], quaternions.quat2mat(rectify_quat(q[3:7])), np.ones(3))


def calc_foot_T(x, move = 'left'):
    #calculate the footstep transformation matrix based on the input x
    #x is defined as: [left_foot, right_foot, 'foot_to_move']
    x = x.reshape(-1,3)
    Ts = []
    for x_i in x:
        T = PosetoMat(x_i)
        Ts.append(T)
    
    if move == 'left':
        T_lefts = [Ts[0], None, Ts[2]]
        T_rights = [Ts[1]]*3
    else:
        T_rights = [Ts[1], None, Ts[2]]
        T_lefts = [Ts[0]]*3
        
    return T_lefts, T_rights




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

def subsample_phases(data_phases, sub_fac = 40):
    new_phases = []
    for i,data_phase in enumerate(data_phases):
        low = int(np.ceil((data_phase[0])/(sub_fac*1.)))
        up = int(np.floor((data_phase[-1]-1e-5)/(sub_fac*1.)))
        if i == len(data_phases)-1:
            up+=1
        new_phases.append(np.arange(low,up+1))
    return new_phases

def extract_from_dataset(directory, filename, is_subsample = True, sub_fact = 40, version='v1.0' ):
    stat_raw = Status(directory +'/res/infos.log')
    stat = [stat_raw.gen_cs_success,stat_raw.centroidal_success,stat_raw.wholebody_success, stat_raw.wholebody_reach_goal, stat_raw.motion_valid ]
    if False in  stat:
        print 'The status contains False!'
        print stat
        return None
    
    try:
        res = wb_res.loadFromNPZ(directory + '/res/' + filename + '.npz')
        print 'Obtain a valid trajectory file!'
    except:
        print 'There is no trajectory in this file!'
        print filename
        return None
    
    #extract the data
    traj =  np.array(res.q_t).T
    vel_traj = np.array(res.dq_t).T
    u_traj = np.array(res.tau_t).T
    data_phases = np.array(res.phases_intervals)
    print traj.shape
    T_lefts, T_rights = extract_foot_contacts(res,version)

    #subsample
    delta_t = 0.001
    T = traj.shape[0]
    if is_subsample:
        delta_t *= sub_fact
        intervals = np.concatenate([np.arange(0,T,sub_fact),[T-1]])
        traj = traj[intervals]
        vel_traj = vel_traj[intervals]
        u_traj = u_traj[intervals]
        data_phases = subsample_phases(data_phases, sub_fact)
    
    return traj, vel_traj, u_traj, T_lefts, T_rights, data_phases, delta_t, stat, res
    


def transform_SE3s(Ts, M, mult_from = 'left'):
    new_Ts = []
    for T in Ts:
        if T is None:
            new_Ts.append(None)
            continue
            
        if mult_from == 'left':
            new_Ts.append(np.dot(M,T))
        else:
            new_Ts.append(np.dot(T,M))
    return new_Ts

def extract_foot_contacts(res, version='v1.0'):
    #extract foot contact sequence from the database file
    if version == 'v1.0':
        left_key = 'leg_left_6_joint'#'leg_left_sole_fix_joint'#
        right_key = 'leg_right_6_joint'#'leg_right_sole_fix_joint'#
    elif version == 'v1.2':
        left_key = 'leg_left_sole_fix_joint'#
        right_key = 'leg_right_sole_fix_joint'#
    else:
        print version + " is not defined!"
        return None 
        
    phases = res.phases_intervals
    contact_lefts = np.array(res.contact_activity[left_key]).flatten().astype('bool')
    contact_rights = np.array(res.contact_activity[right_key]).flatten().astype('bool')

    T_lefts = []
    T_rights = []

    for i,phase in enumerate(phases):
        if contact_lefts[phase[0]] == True:
            T_left = vec2mat(res.effector_trajectories[left_key][:,phase[0]])
            #T_left[2,3] -= 0.107 + 0.004 #move from left_key to left_sole_link 
        else:
            T_left = None
        if contact_rights[phase[0]] == True:
            T_right = vec2mat(res.effector_trajectories[right_key][:,phase[0]])
            #T_right[2,3] -= 0.107 + 0.004 #move from right_key to right_sole_link
        else:
            T_right = None
        T_lefts.append(T_left)
        T_rights.append(T_right)
    return T_lefts, T_rights



def extract_root_poses(res):
    #extract foot contact sequence from the database file
    phases = res.phases_intervals
    traj = np.array(res.q_t).T
    Twrs = []
    for phase in phases[0::2]:   
        q = traj[phase[0],0:7]
        Twr = affines.compose(q[0:3], quaternions.quat2mat(rectify_quat(q[3:])), np.ones(3))
        Twrs.append(Twr)
    return Twrs

def construct_foot_poses(T_lefts, T_rights, is_3D = False):
    #transform SE3 foot poses (left and right) to [x,y,(z),theta] for visualization and regression
    #if 3D, include the z position. else, only 2D
    n = len(T_lefts)
    poses = []
    for i in range(n):
        #if there is only one foot (the other is None), skip
        if T_lefts[i] is None or T_rights[i] is None: continue
            
        angle_left = axangles.mat2axangle(T_lefts[i][:3,:3])[1]
        angle_right = axangles.mat2axangle(T_rights[i][:3,:3])[1]
        if is_3D:
            left_pose = np.concatenate([T_lefts[i][:3,3], [angle_left]])
            right_pose = np.concatenate([T_rights[i][:3,3], [angle_right]])
        else:
            left_pose = np.concatenate([T_lefts[i][:2,3], [angle_left]])
            right_pose = np.concatenate([T_rights[i][:2,3], [angle_right]])
            
        poses.append(left_pose)
        poses.append(right_pose)
    return np.array(poses)

def transform_traj(traj, T):
    #transform the trajectories by multiplying the root pose by T
    for i,q in enumerate(traj):
        q = q[0:7]
        Twr = affines.compose(q[0:3], quaternions.quat2mat(rectify_quat(q[3:])), np.ones(3))
        Twr_new = np.dot(T, Twr)
        p,R, _, _ = affines.decompose44(Twr_new)
        q_new = np.concatenate([p, derectify_quat(quaternions.mat2quat(R))])
        traj[i,0:7] = q_new
    return traj


    
def project_traj(traj, foot_pose, l_stat = True):
    #rectify trajectories based on the stationary foot
    print 'Yes'
    return
    
'''

for j in range(len(traj)):
    q = traj[j]
    Twr = affines.compose(q[0:3], quaternions.quat2mat(rectify_quat(q[3:7])), np.ones(3))
    if l_stat:
        Trl = left_foot.forward(q[7:13])
        Twl = np.dot(Twr, Trl)
        xyz_true = np.concatenate([x[0:2], [0]])
        xyz_pred = Twl[:3,3].flatten()
        delta_xyz = np.array(xyz_true-xyz_pred)
        traj[j][0:3] += delta_xyz.flatten()
    else:
        Trri = right_foot.forward(q[13:19])
        Twri = np.dot(Twr, Trri)
        xyz_true = np.concatenate([x[3:5], [0]])
        xyz_pred = Twri[:3,3].flatten()
        delta_xyz = np.array(xyz_true-xyz_pred)
        traj[j][0:3] += delta_xyz.flatten()
''' 

def determine_which_foot(foot_poses):
    #determine which foot that moves
    pose_init_left = foot_poses[0]
    pose_init_right = foot_poses[1]
    pose_goal_left = foot_poses[2]
    pose_goal_right = foot_poses[3]
    
    dist_left = np.linalg.norm(pose_goal_left - pose_init_left)
    dist_right = np.linalg.norm(pose_goal_right - pose_init_right)
    
    
    if np.allclose(pose_init_left, pose_goal_left, rtol=1e-2,atol=1e-2):
        return 'right'
    elif np.allclose(pose_init_right, pose_goal_right, rtol=1e-2,atol=1e-2):
        return 'left'
    elif dist_left < dist_right: 
        return 'left'
    elif dist_left > dist_right: 
        return 'right'
    #else:
    #    return 'Cannot determine which foot!'
    
