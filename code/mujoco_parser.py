import os,time,cv2,glfw,mujoco_py
import numpy as np
import matplotlib.pyplot as plt
from screeninfo import get_monitors # get monitor size
from util import r2w,trim_scale,quat2r

# MuJoCo Parser class
class MuJoCoParserClass(object):
    def __init__(self,
                 name     = 'Robot',
                 xml_path = ''
                ):
        """
            Initialize MuJoCo parser
        """
        self.name        = name
        self.xml_path    = xml_path
        self.cwd         = os.getcwd()
        # Simulation
        self.tick         = 0
        self.VIEWER_EXIST = False
        # Parse the xml file
        self._parse_xml()
        # Reset
        self.reset()
        
    def _parse_xml(self):
        """
            Parse an xml file
        """
        # Basic MuJoCo model and sim
        self.full_xml_path = os.path.abspath(os.path.join(self.cwd,self.xml_path))
        self.model         = mujoco_py.load_model_from_path(self.full_xml_path)
        self.sim           = mujoco_py.MjSim(self.model)
        # Parse model information
        self.dt              = self.sim.model.opt.timestep 
        self.HZ              = int(1/self.dt)
        self.n_body          = self.model.nbody
        self.body_names      = list(self.sim.model.body_names)
        self.n_joint         = self.model.njnt
        self.joint_idxs      = np.arange(0,self.n_joint,1)
        self.joint_names     = [self.sim.model.joint_id2name(x) for x in range(self.n_joint)]
        self.joint_types     = self.sim.model.jnt_type # 0:free, 1:ball, 2:slide, 3:hinge
        self.joint_range     = self.sim.model.jnt_range
        self.actuator_names  = list(self.sim.model.actuator_names)
        self.n_actuator      = len(self.actuator_names)
        self.torque_range    = self.sim.model.actuator_ctrlrange
        self.rev_joint_idxs  = np.where(self.joint_types==3)[0].astype(np.int32) # revolute joint indices
        self.rev_joint_names = [self.joint_names[x] for x in self.rev_joint_idxs]
        self.n_rev_joint     = len(self.rev_joint_idxs)
        self.rev_qvel_idxs   = [self.sim.model.get_joint_qvel_addr(x) for x in self.rev_joint_names]
        self.geom_names      = list(self.sim.model.geom_names)
        self.n_geom          = len(self.geom_names)
        
    def print_env_info(self):
        """
            Print env info
        """
        print ("[%s] Instantiated from [%s]"%(self.name,self.full_xml_path))
        print ("- Simulation timestep is [%.4f]sec and frequency is [%d]HZ"%(self.dt,self.HZ))
        print ("- [%s] has [%d] bodies"%(self.name,self.n_body))
        for b_idx in range(self.n_body):
            body_name  = self.body_names[b_idx]
            print (" [%02d] body name:[%s]"%(b_idx,body_name))
        print ("- [%s] has [%d] joints"%(self.name,self.n_joint))
        for j_idx in range(self.n_joint):
            joint_name = self.joint_names[j_idx]
            joint_type = self.joint_types[j_idx]
            if joint_type == 0:
                joint_type_str = 'free'
            elif joint_type == 1:
                joint_type_str = 'ball'
            elif joint_type == 2:
                joint_type_str = 'prismatic'
            elif joint_type == 3:
                joint_type_str = 'revolute'
            else:
                joint_type_str = 'unknown'
            print (" [%02d] name:[%s] type:[%s] joint range:[%.2f to %.2f]"%
                (j_idx,joint_name,joint_type_str,self.joint_range[j_idx,0],self.joint_range[j_idx,1]))
        print ("- [%s] has [%d] revolute joints"%(self.name,self.n_rev_joint))
        for j_idx in range(self.n_rev_joint):
            rev_joint_idx  = self.rev_joint_idxs[j_idx]
            rev_joint_name = self.rev_joint_names[j_idx]
            print (" [%02d] joint index:[%d] and name:[%s]"%(j_idx,rev_joint_idx,rev_joint_name))
        print  ("- [%s] has [%d] actuators"%(self.name,self.n_actuator))
        for a_idx in range(self.n_actuator):
            actuator_name = self.actuator_names[a_idx]
            print (" [%02d] actuator name:[%s] torque range:[%.2f to %.2f]"%
            (a_idx,actuator_name,self.torque_range[a_idx,0],self.torque_range[a_idx,1]))
        print  ("- [%s] has [%d] geometries"%(self.name,self.n_geom))
        for g_idx in range(self.n_geom):
            geom_name = self.geom_names[g_idx]
            print (" [%02d] geometry name:[%s]"%(g_idx,geom_name))
            
    def plot_scene(self,
                   figsize       = (12,8),
                   render_w      = 1200,
                   render_h      = 800,
                   title_str     = None,
                   title_fs      = 10,
                   cam_azimuth   = None,
                   cam_distance  = None,
                   cam_elevation = None,
                   cam_lookat    = None,
                   RETURN_IMG    = False,
                   N_TRY         = 5
                   ):
        """
            Plot scene
        """
        self.init_viewer()
        for _ in range(N_TRY): # render multiple times to properly apply plot configurations
            for r_idx in range(len(self.sim.render_contexts)):
                if cam_azimuth is not None:
                    self.sim.render_contexts[r_idx].cam.azimuth   = cam_azimuth
                if cam_distance is not None:
                    self.sim.render_contexts[r_idx].cam.distance  = cam_distance
                if cam_elevation is not None:
                    self.sim.render_contexts[r_idx].cam.elevation = cam_elevation
                if cam_lookat is not None:
                    self.sim.render_contexts[r_idx].cam.lookat[0] = cam_lookat[0]
                    self.sim.render_contexts[r_idx].cam.lookat[1] = cam_lookat[1]
                    self.sim.render_contexts[r_idx].cam.lookat[2] = cam_lookat[2]
            img = self.sim.render(width=render_w,height=render_h)
        img = cv2.flip(cv2.rotate(img,cv2.ROTATE_180),1) # 0:up<->down, 1:left<->right
        if RETURN_IMG: # return RGB image
            return img
        else: # plot image
            plt.figure(figsize=figsize)
            plt.imshow(img)
            if title_str is not None:
                plt.title(title_str,fontsize=title_fs)
            plt.axis('off')
            plt.show()
            
    def reset(self,RESET_GLFW=False):
        """
             Reset simulation
        """
        self.tick = 0
        self.sim.reset()
            
    def init_viewer(self,
                    window_width  = None,
                    window_height = None,
                    cam_azimuth   = None,
                    cam_distance  = None,
                    cam_elevation = None,
                    cam_lookat    = None
                    ):
        """
            Initialize viewer
        """
        if not self.VIEWER_EXIST:
            self.VIEWER_EXIST = True
            self.viewer = mujoco_py.MjViewer(self.sim) # this will make a new window
        # Set viewer
        if (window_width is not None) and (window_height is not None):
            self.set_viewer(
                window_width=window_width,window_height=window_height,
                cam_azimuth=cam_azimuth,cam_distance=cam_distance,
                cam_elevation=cam_elevation,cam_lookat=cam_lookat)

    def set_viewer(self,
                   window_width  = 0.5,
                   window_height = 0.5,
                   cam_azimuth   = None,
                   cam_distance  = None,
                   cam_elevation = None,
                   cam_lookat    = None
                   ):
        """
            Set viewer
        """
        if self.VIEWER_EXIST:
            self.window = self.viewer.window
            width  = int(window_width*get_monitors()[0].width)
            height = int(window_height*get_monitors()[0].height)
            glfw.set_window_size(window=self.window,width=width,height=height)
            # Viewer setting
            if cam_azimuth is not None:
                self.viewer.cam.azimuth = cam_azimuth
            if cam_distance is not None:
                self.viewer.cam.distance = cam_distance
            if cam_elevation is not None:
                self.viewer.cam.elevation = cam_elevation
            if cam_lookat is not None:
                self.viewer.cam.lookat[0] = cam_lookat[0]
                self.viewer.cam.lookat[1] = cam_lookat[1]
                self.viewer.cam.lookat[2] = cam_lookat[2]

    def print_viewer_info(self):
        """
            Print current viewer information
        """
        print ("azimuth:[%.2f] distance:[%.2f] elevation:[%.2f] lookat:[%.2f,%.2f,%.2f]"%(
            self.viewer.cam.azimuth,self.viewer.cam.distance,self.viewer.cam.elevation,
            self.viewer.cam.lookat[0],self.viewer.cam.lookat[1],self.viewer.cam.lookat[2]))

    def get_viewer_info(self):
        """
            Get viewer information
        """
        cam_azimuth = self.viewer.cam.azimuth
        cam_distance = self.viewer.cam.distance
        cam_elevation = self.viewer.cam.elevation
        cam_lookat = self.viewer.cam.lookat
        viewer_info = {
            'cam_azimuth':cam_azimuth,'cam_distance':cam_distance,
            'cam_elevation':cam_elevation,'cam_lookat':cam_lookat
        }
        return viewer_info
            
    def terminate_viewer(self):
        """
            Terminate viewer
        """
        if self.VIEWER_EXIST:
            self.VIEWER_EXIST = False 
            self.viewer.render() # render once before terminate
            time.sleep(1.0)
            glfw.terminate() # terminate
            time.sleep(1.0) 
            glfw.init() # initialize once

    def step(self,ctrl=None,ctrl_idxs=None):
        """
            Step simulation
        """
        # Increase tick
        self.tick = self.tick + 1
        # Control
        if ctrl is not None:
            if ctrl_idxs is None:
                self.sim.data.ctrl[:] = ctrl
            else:
                self.sim.data.ctrl[ctrl_idxs] = ctrl
        # Forward dynamics
        self.sim.step()

    def forward(self,q_pos=None,q_pos_idxs=None,INCREASE_TICK=True):
        """
            Forward kinemaatics
        """
        # Increase tick
        if INCREASE_TICK:
            self.tick = self.tick + 1
        # Forward kinematicaaqs
        if q_pos is not None:
            if q_pos_idxs is None:
                self.sim.data.qpos[:] = q_pos
            else:
                self.sim.data.qpos[q_pos_idxs] = q_pos
        self.sim.forward()
        
    def render(self,RENDER_ALWAYS=False):
        """
            Render simulation
        """
        if RENDER_ALWAYS:
            self.viewer._render_every_frame = True
        else:
            self.viewer._render_every_frame = False
        self.viewer.render()
        
    def get_sim_time(self):
        """
            Get simulation time [sec]
        """
        return self.sim.get_state().time

    def get_q_pos(self,q_pos_idxs=None):
        """
            Get current revolute joint position
        """
        self.sim_state = self.sim.get_state()
        if q_pos_idxs is None:
            q_pos = self.sim_state.qpos[:]
        else:
            q_pos = self.sim_state.qpos[q_pos_idxs]
        return q_pos
        
    def apply_xfrc(self,body_name,xfrc):
        """
            Apply external force (6D) to body
        """
        self.sim.data.xfrc_applied[self.body_name2idx(body_name),:] = xfrc

    def body_name2idx(self,body_name='panda_eef'):
        """
            Body name to index
        """
        return self.sim.model.body_name2id(body_name)

    def body_idx2name(self,body_idx=0):
        """
            Body index to name
        """
        return self.sim.model.body_id2name(body_idx)

    def add_marker(self,pos,type=2,radius=0.02,color=[0.0,1.0,0.0,1.0],label=''):
        """
            Add a maker to renderer
        """
        self.viewer.add_marker(
            pos   = pos,
            type  = type, # mjtGeom: 2:sphere, 3:capsule, 6:box, 9:arrow
            size  = radius*np.ones(3),
            mat   = np.eye(3).flatten(),
            rgba  = color,
            label = label)

    def get_p_body(self,body_name):
        """
            Get body position
        """
        self.sim_state = self.sim.get_state()
        p = np.array(self.sim.data.body_xpos[self.body_name2idx(body_name)])
        return p

    def get_R_body(self,body_name):
        """
            Get body rotation
        """
        self.sim_state = self.sim.get_state()
        R = np.array(self.sim.data.body_xmat[self.body_name2idx(body_name)].reshape([3, 3]))
        return R

    def get_J_body(self,body_name):
        """
            Get body Jacobian
        """
        J_p    = np.array(self.sim.data.get_body_jacp(body_name).reshape((3, -1))[:,self.rev_qvel_idxs])
        J_R    = np.array(self.sim.data.get_body_jacr(body_name).reshape((3, -1))[:,self.rev_qvel_idxs])
        J_full = np.array(np.vstack([J_p,J_R]))
        return J_p,J_R,J_full

    def one_step_ik(self,body_name,p_trgt=None,R_trgt=None,stepsize=5.0*np.pi/180.0,eps=1e-6):
        """
            One-step inverse kinematics
        """
        J_p,J_R,J_full = self.get_J_body(body_name=body_name)
        p_curr = self.get_p_body(body_name=body_name)
        R_curr = self.get_R_body(body_name=body_name)
        if (p_trgt is not None) and (R_trgt is not None): # both p and R targets are given
            p_err = (p_trgt-p_curr)
            R_err = np.linalg.solve(R_curr,R_trgt)
            w_err = R_curr @ r2w(R_err)
            J,err = J_full,np.concatenate((p_err,w_err))
        elif (p_trgt is not None) and (R_trgt is None): # only p target is given
            p_err = (p_trgt-p_curr)
            J,err = J_p,p_err
        elif (p_trgt is None) and (R_trgt is not None): # only R target is given
            R_err = np.linalg.solve(R_curr,R_trgt)
            w_err = R_curr @ r2w(R_err)
            J,err = J_R,w_err
        else:
            raise Exception('At least one IK target is required!')
        # Compute dq using least-square
        dq = np.linalg.solve(a=(J.T@J)+eps*np.eye(J.shape[1]),b=J.T@err)
        # Trim scale 
        dq = trim_scale(x=dq,th=stepsize)
        return dq,err

    def backup_sim_data(self,joint_idxs=None):
        """
            Backup sim data (qpos, qvel, qacc)
        """
        if joint_idxs is None:
            joint_idxs = self.rev_joint_idxs
        self.qpos_bu = self.sim.data.qpos[joint_idxs]
        self.qvel_bu = self.sim.data.qvel[joint_idxs]
        self.qacc_bu = self.sim.data.qacc[joint_idxs]

    def restore_sim_data(self,joint_idxs=None):
        """
            Restore sim data (qpos, qvel, qacc)
        """
        if joint_idxs is None:
            joint_idxs = self.rev_joint_idxs
        self.sim.data.qpos[joint_idxs] = self.qpos_bu
        self.sim.data.qvel[joint_idxs] = self.qvel_bu
        self.sim.data.qacc[joint_idxs] = self.qacc_bu

    def solve_inverse_dynamics(self,qvel,qacc,joint_idxs=None):
        """
            Solve inverse dynamics to get torque from qvel and qacc
        """
        if joint_idxs is None:
            joint_idxs = self.rev_joint_idxs
        # Backup
        self.backup_sim_data(joint_idxs=joint_idxs)
        # Compute torque
        self.sim.data.qpos[joint_idxs] = self.get_q_pos(q_pos_idxs=joint_idxs)
        self.sim.data.qvel[joint_idxs] = qvel
        self.sim.data.qacc[joint_idxs] = qacc
        mujoco_py.functions.mj_inverse(self.sim.model,self.sim.data)
        torque = self.sim.data.qfrc_inverse[joint_idxs].copy()
        # Restore
        self.restore_sim_data(joint_idxs=joint_idxs)
        return torque

def get_env_obj_names(env,prefix='obj_'):
    """
        Accumulate object names by assuming that the prefix is 'obj_'
    """
    obj_names = [x for x in env.joint_names if x[:len(prefix)]==prefix]
    return obj_names

def set_env_obj(
    env,
    obj_name  = 'obj_box_01',
    obj_pos   = [1.0,0.0,0.75],
    obj_quat  = [0,0,0,1],
    obj_color = None
    ):
    """
        Set a single object in an environment
    """
    # Get address
    qpos_addr = env.sim.model.get_joint_qpos_addr(obj_name)
    # Set position
    env.sim.data.qpos[qpos_addr[0]]   = obj_pos[0] # x
    env.sim.data.qpos[qpos_addr[0]+1] = obj_pos[1] # y
    env.sim.data.qpos[qpos_addr[0]+2] = obj_pos[2] # z
    # Set rotation
    env.sim.data.qpos[qpos_addr[0]+3:qpos_addr[1]] = obj_quat # quaternion
    # Color
    if obj_color is not None:
        idx = env.sim.model.geom_name2id(obj_name)
        env.sim.model.geom_rgba[idx,:] = obj_color

def set_env_objs(
    env,
    obj_names,
    obj_poses,
    obj_colors=None):
    """
        Set multiple objects
    """
    for o_idx,obj_name in enumerate(obj_names):
        obj_pos = obj_poses[o_idx,:]
        if obj_colors is not None:
            obj_color = obj_colors[o_idx,:]
        else:
            obj_color = None
        set_env_obj(env,obj_name=obj_name,obj_pos=obj_pos,obj_color=obj_color)


def get_env_obj_poses(env,obj_names):
    """
        Get object poses 
    """
    n_obj     = len(obj_names)
    obj_ps = np.zeros(shape=(n_obj,3))
    obj_Rs = np.zeros(shape=(n_obj,3,3))
    for o_idx,obj_name in enumerate(obj_names):
        qpos_addr = env.sim.model.get_joint_qpos_addr(obj_name)
        # Get position
        x = env.sim.data.qpos[qpos_addr[0]]
        y = env.sim.data.qpos[qpos_addr[0]+1]
        z = env.sim.data.qpos[qpos_addr[0]+2]
        # Set rotation (upstraight)
        quat = env.sim.data.qpos[qpos_addr[0]+3:qpos_addr[1]]
        R = quat2r(quat)
        # Append
        obj_ps[o_idx,:] = np.array([x,y,z])
        obj_Rs[o_idx,:,:] = R
    return obj_ps,obj_Rs