
import crocoddyl
import pinocchio
import numpy as np
import time
import pickle

import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import roslib
import tf
from geometry_msgs.msg import Quaternion, Pose, Point, Vector3
from std_msgs.msg import Header, ColorRGBA
from visualization_msgs.msg import Marker,MarkerArray
from transforms3d import axangles
from transforms3d import affines, quaternions
from pinocchio import SE3
from database import *


import numpy.matlib
import scipy.stats

class RvizMarker():
    def __init__(self,name = 'marker', queue_size = 5, marker_type = Marker.CUBE, n = 1, max_n = 200, duration = 1000, color = ColorRGBA(0.0, 1.0, 0.0, 0.8),scale = Vector3(0.25, 0.13, 0.01)):
        self.n = n
        self.max_n = max_n
        self.duration = duration
        self.marker_type = marker_type
        self.marker_publisher = rospy.Publisher(name,   MarkerArray, queue_size = queue_size)
        #rospy.init_node('marker_publisher')
        self.markers = []
        self.marker_array = MarkerArray()
        self.color = color
        self.scale = scale
        for i in range(self.max_n):
            marker = Marker(
                type=self.marker_type,
                id=i,
                lifetime=rospy.Duration(self.duration),
                pose=Pose(Point(10., 10., 10.), Quaternion(0, 0, 0, 1)),
                scale=scale,
                header=Header(frame_id='world'),
                color=color)
            self.markers.append(marker)
            
    def add_marker(self, color = ColorRGBA(0.0, 1.0, 0.0, 0.8)):
        marker = Marker(
        type=self.marker_type,
        id=self.n,
        lifetime=rospy.Duration(self.duration),
        pose=Pose(Point(0., 0., 0.), Quaternion(0, 0, 0, 1)),
        scale=self.scale,
        header=Header(frame_id='world'),
        color=color)
        self.n += 1
        self.markers.append(marker)
            
    def publish(self,pos, quats):
        for i,marker in enumerate(self.markers):
            marker.pose.position = Point(pos[i][0], pos[i][1], pos[i][2]) 
            marker.pose.orientation = Quaternion(quats[i][1], quats[i][2], quats[i][3], quats[i][0]) #convert transform3d format to rviz/dataset format

            #self.marker_publisher.publish(marker)
        self.marker_array.markers = self.markers
        self.marker_publisher.publish(self.marker_array)
    def publish(self, poses):
        n = len(poses)
        if n > self.n:
            for i in range(n-self.n):
                #self.add_marker()
                self.markers[self.n].action = self.markers[0].ADD
                self.n +=1
        else:
            for i in range(self.n-n):
                self.markers[self.n-1].action = self.markers[0].DELETE
                self.n -= 1
                #self.marker_publisher.publish(self.markers[-1])
                #del self.markers[-1]
            self.n = n
                
        for i,marker in enumerate(self.markers[:self.n]):
            pose = poses[i] #pose = x, y, z, theta
            if isinstance(pose,SE3):
                marker.pose.position = Point(pose.translation[0,0], pose.translation[1,0], pose.translation[2,0] ) #2D cases
                marker.pose.orientation = Quaternion(*derectify_quat(quaternions.mat2quat(pose.rotation)))
            elif pose.shape == (4,):
                marker.pose.position = Point(pose[0], pose[1], pose[2])
                marker.pose.orientation = Quaternion(0,0,np.sin(pose[3]/2), np.cos(pose[3]/2)) #rviz/dataset format: x,y,z,w
            elif pose.shape == (3,):
                marker.pose.position = Point(pose[0], pose[1], 0.) #2D cases
                marker.pose.orientation = Quaternion(0,0,np.sin(pose[2]/2), np.cos(pose[2]/2)) 
            elif pose.shape == (4,4):
                marker.pose.position = Point(*pose[:3,3]) #2D cases
                marker.pose.orientation = Quaternion(*derectify_quat(quaternions.mat2quat(pose[:3,:3])))
                
                
            #self.marker_publisher.publish(marker)
            self.marker_array.markers = self.markers
            self.marker_publisher.publish(self.marker_array)
            
    def set_color(self,colors):
        for i,color in enumerate(colors):
            self.markers[i].color = color

class Visual():
    def __init__(self,rate = 26):
        self.pub = rospy.Publisher('joint_states', JointState, queue_size=10)
        rospy.init_node('joint_state_publisher')
        self.rate = rospy.Rate(rate) # 10hz

        self.hello_str = JointState()
        self.hello_str.header = Header()
        self.hello_str.header.stamp = rospy.Time.now()
        self.hello_str.name = ['leg_left_1_joint','leg_left_2_joint','leg_left_3_joint','leg_left_4_joint','leg_left_5_joint',
        'leg_left_6_joint','leg_right_1_joint','leg_right_2_joint','leg_right_3_joint',
        'leg_right_4_joint','leg_right_5_joint','leg_right_6_joint','torso_1_joint',
        'torso_2_joint','arm_left_1_joint','arm_left_2_joint','arm_left_3_joint',
        'arm_left_4_joint','arm_left_5_joint','arm_left_6_joint','arm_left_7_joint',
        'gripper_left_joint','arm_right_1_joint','arm_right_2_joint','arm_right_3_joint',
        'arm_right_4_joint','arm_right_5_joint','arm_right_6_joint','arm_right_7_joint',
        'gripper_right_joint','head_1_joint','head_2_joint']
        
        self.hello_str.velocity = []
        self.hello_str.effort = []
        self.br = tf.TransformBroadcaster()
        
        self.br.sendTransform([0.,0.,0.], [0,0,0,1],
                 rospy.Time.now(),
                 "init",
                 "world"
                 )

    def vis_traj(self, traj):
        for traj_i in traj:
            xyz = traj_i[:3].tolist()
            quat = traj_i[3:7]
            quat = quat/np.linalg.norm(quat)
            quat = quat.tolist()
            self.hello_str.header.stamp = rospy.Time.now()
            self.hello_str.position = traj_i[7:].tolist()

            self.br.sendTransform(xyz, quat, 
                         rospy.Time.now(),
                         "base_link",
                         "world"
                         )
            self.pub.publish(self.hello_str)
            self.rate.sleep()
            
    def set_dof(self, q):
        xyz = q[:3].tolist()
        quat = q[3:7]
        quat = quat/np.linalg.norm(quat)
        quat = quat.tolist()
        self.hello_str.header.stamp = rospy.Time.now()
        self.hello_str.position = q[7:].tolist()

        self.br.sendTransform(xyz, quat, 
                     rospy.Time.now(),
                     "base_link",
                     "world"
                     )
        self.pub.publish(self.hello_str)
        self.rate.sleep()
        
    def set_rate(self, freq):
        self.rate = rospy.Rate(freq)


        
        
def define_RBF(dof=39, nbStates=60, offset=200, width=60, T=4000, coeff = 250):
    tList = np.arange(T)

    Mu = np.linspace(tList[0]-offset, tList[-1]+offset, nbStates)
    Sigma  = np.reshape(np.matlib.repmat(width, 1, nbStates),[1, 1, nbStates])
    Sigma.shape
    Phi = np.zeros((T, nbStates))
    for i in range(nbStates):
        Phi[:,i] = coeff*scipy.stats.norm(Mu[i], Sigma[0,0,i]).pdf(tList)
    print Phi
    return Phi


'''
import gtk.gdk
def save_screenshot(x,y,w,h,file_name):
    window = gtk.gdk.get_default_root_window()
    sz = window.get_size()
    print "The size of the window is %d x %d" % sz
    pb = gtk.gdk.Pixbuf(gtk.gdk.COLORSPACE_RGB,False,8,sz[0],sz[1])
    pb = pb.get_from_drawable(window,window.get_colormap(),0,0,0,0,sz[0],sz[1])
    pb = pb.subpixbuf(int(x),int(y),int(w),int(h)) 
    if (pb != None):
        pb.save(file_name,"png")
        print "Screenshot saved."
    else:
        print "Unable to get the screenshot."'''
        
import pyscreenshot as ImageGrab
def save_screenshot(x,y,w,h,file_name, to_show='False'):
    # part of the screen
    im=ImageGrab.grab(bbox=(x,y,w,h))
    if to_show:
        im.show()
    # save to file
    im.save(file_name)
    return im
