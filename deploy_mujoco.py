import sys
import os
import time
import torch
import mujoco
import mujoco.viewer
import numpy as np
import faulthandler
from omegaconf import OmegaConf

from residual_policy import OnnxResidualPolicyWrapper, OnnxBasePolicyWrapper  
from scipy.spatial.transform import Rotation as R
from isaac_utils import  rotations as _rot
import onnxruntime as ort


BASE_ONNX="policy/base_policy_trt.onnx"
RESIDUAL_ONNX="policy/residual_policy.onnx"
FK_QUAT_ONNX="policy/fk_trt.onnx"

HW_DOF = 29
VISUAL = True

# Reindex DOF order from BeyondMimic motionlib to URDF definition
PERM = np.array([0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28])

# Reindex DOF order from URDF to BeyondMimic motionlib definition
INV_PERM = np.empty_like(PERM)
INV_PERM[PERM] = np.arange(len(PERM))


# Joint position bias for pd control and joint related observation computation(urdf order)
PD_BIAS_JOINT_POS = np.array([
            -0.3120,  0.0000, 0.0000, 0.6690, -0.3630,  0.0000,
            -0.3120,  0.0000, 0.0000, 0.6690, -0.3630,  0.0000,  
            0.0000,  0.0000,  0.0000, 
            0.2000, 0.2000,  0.0000,  0.6000,  0.0000,  0.0000,  0.0000, 
            0.2000, -0.2000, 0.0000,  0.6000,  0.0000,  0.0000,  0.0000
])

# Action scale for robot control(urdf order)
ACTION_SCALE=torch.tensor([[ 
            0.3506614663788243, 0.3506614663788243, 0.5475464652142303, 0.3506614663788243, 0.30700411974635705, 0.30700411974635705,
            0.3506614663788243, 0.3506614663788243, 0.5475464652142303, 0.3506614663788243, 0.30700411974635705, 0.30700411974635705,
            0.5475464652142303, 0.30700411974635705, 0.30700411974635705,
            0.43857731392336724, 0.43857731392336724, 0.43857731392336724, 0.43857731392336724, 0.43857731392336724, 0.07450087032950714, 0.07450087032950714,
            0.43857731392336724, 0.43857731392336724, 0.43857731392336724, 0.43857731392336724, 0.43857731392336724, 0.07450087032950714, 0.07450087032950714        
]])

# Pd control kp parameters(urdf order)
P_GAINS=np.array([
            99.0984, 99.0984, 40.1792, 99.0984, 28.5012, 28.5012, 
            99.0984, 99.0984, 40.1792, 99.0984, 28.5012, 28.5012, 
            40.1792, 28.5012, 28.5012, 
            14.2506, 14.2506, 14.2506, 14.2506, 14.2506, 16.7783, 16.7783, 
            14.2506, 14.2506, 14.2506, 14.2506, 14.2506, 16.7783, 16.7783


])

# Pd control kd parameters(urdf order)
D_GAINS=np.array([
            6.3088, 6.3088, 2.5579, 6.3088, 1.8144, 1.8144, 
            6.3088, 6.3088, 2.5579, 6.3088, 1.8144, 1.8144, 
            2.5579, 1.8144, 1.8144, 
            0.9072, 0.9072, 0.9072, 0.9072, 0.9072, 1.0681, 1.0681, 
            0.9072, 0.9072, 0.9072, 0.9072, 0.9072, 1.0681, 1.0681
])

# Unitree actuator torque-speed envelope per joint (urdf order)
# The model and parameters are referenced from https://github.com/unitreerobotics/unitree_rl_lab/blob/main/source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree_actuators.py
X1_7520_22, X2_7520_22, Y1_7520_22, Y2_7520_22 = 14.5, 22.7, 111.0, 131.0
X1_7520_14, X2_7520_14, Y1_7520_14, Y2_7520_14 = 22.63, 35.52, 71.0, 83.3
X1_5020_16, X2_5020_16, Y1_5020_16, Y2_5020_16 = 30.86, 40.13, 24.8, 31.9
X1_4010_25, X2_4010_25, Y1_4010_25, Y2_4010_25 = 15.3, 24.76, 4.8, 8.6

X1_list = [
    X1_7520_22, X1_7520_22, X1_7520_14, X1_7520_22, X1_5020_16, X1_5020_16,
    X1_7520_14, X1_5020_16, X1_5020_16,
    X1_7520_22, X1_7520_22, X1_7520_14, X1_7520_22, X1_5020_16, X1_5020_16,
    X1_5020_16, X1_5020_16, X1_5020_16, X1_5020_16, X1_5020_16, X1_4010_25, X1_4010_25,
    X1_5020_16, X1_5020_16, X1_5020_16, X1_5020_16, X1_5020_16, X1_4010_25, X1_4010_25,
]
X2_list = [
    X2_7520_22, X2_7520_22, X2_7520_14, X2_7520_22, X2_5020_16, X2_5020_16,
    X2_7520_22, X2_7520_22, X2_7520_14, X2_7520_22, X2_5020_16, X2_5020_16,
    X2_7520_14, X2_5020_16, X2_5020_16,
    X2_5020_16, X2_5020_16, X2_5020_16, X2_5020_16, X2_5020_16, X2_4010_25, X2_4010_25,
    X2_5020_16, X2_5020_16, X2_5020_16, X2_5020_16, X2_5020_16, X2_4010_25, X2_4010_25,
]
Y1_list = [
    Y1_7520_22, Y1_7520_22, Y1_7520_14, Y1_7520_22, Y1_5020_16, Y1_5020_16,
    Y1_7520_22, Y1_7520_22, Y1_7520_14, Y1_7520_22, Y1_5020_16, Y1_5020_16,
    Y1_7520_14, Y1_5020_16, Y1_5020_16,
    Y1_5020_16, Y1_5020_16, Y1_5020_16, Y1_5020_16, Y1_5020_16, Y1_4010_25, Y1_4010_25,
    Y1_5020_16, Y1_5020_16, Y1_5020_16, Y1_5020_16, Y1_5020_16, Y1_4010_25, Y1_4010_25,
]
Y2_list = [
    Y2_7520_22, Y2_7520_22, Y2_7520_14, Y2_7520_22, Y2_5020_16, Y2_5020_16,
    Y2_7520_22, Y2_7520_22, Y2_7520_14, Y2_7520_22, Y2_5020_16, Y2_5020_16,
    Y2_7520_14, Y2_5020_16, Y2_5020_16,
    Y2_5020_16, Y2_5020_16, Y2_5020_16, Y2_5020_16, Y2_5020_16, Y2_4010_25, Y2_4010_25,
    Y2_5020_16, Y2_5020_16, Y2_5020_16, Y2_5020_16, Y2_5020_16, Y2_4010_25, Y2_4010_25,
]

# Joint friction model parameters(urdf order)
# The model and parameters are referenced from https://github.com/unitreerobotics/unitree_rl_lab/blob/main/source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree_actuators.py
Va_7520_22, Fs_7520_22, Fd_7520_22 = 0.01, 2.4, 0.24
Va_7520_14, Fs_7520_14, Fd_7520_14 = 0.01, 1.6, 0.16
Va_5020_16, Fs_5020_16, Fd_5020_16 = 0.01, 0.6, 0.06
Va_4010_25, Fs_4010_25, Fd_4010_25 = 0.01, 0.6, 0.06
Va_list = [
    Va_7520_22, Va_7520_22, Va_7520_14, Va_7520_22, Va_5020_16, Va_5020_16,
    Va_7520_22, Va_7520_22, Va_7520_14, Va_7520_22, Va_5020_16, Va_5020_16,
    Va_7520_14, Va_5020_16, Va_5020_16,
    Va_5020_16, Va_5020_16, Va_5020_16, Va_5020_16, Va_5020_16, Va_4010_25, Va_4010_25,
    Va_5020_16, Va_5020_16, Va_5020_16, Va_5020_16, Va_5020_16, Va_4010_25, Va_4010_25,
]
Fs_list = [
    Fs_7520_22, Fs_7520_22, Fs_7520_14, Fs_7520_22, Fs_5020_16, Fs_5020_16,
    Fs_7520_22, Fs_7520_22, Fs_7520_14, Fs_7520_22, Fs_5020_16, Fs_5020_16,
    Fs_7520_14, Fs_5020_16, Fs_5020_16,
    Fs_5020_16, Fs_5020_16, Fs_5020_16, Fs_5020_16, Fs_5020_16, Fs_4010_25, Fs_4010_25,
    Fs_5020_16, Fs_5020_16, Fs_5020_16, Fs_5020_16, Fs_5020_16, Fs_4010_25, Fs_4010_25,
]
Fd_list = [
    Fd_7520_22, Fd_7520_22, Fd_7520_14, Fd_7520_22, Fd_5020_16, Fd_5020_16,
    Fd_7520_22, Fd_7520_22, Fd_7520_14, Fd_7520_22, Fd_5020_16, Fd_5020_16,
    Fd_7520_14, Fd_5020_16, Fd_5020_16,
    Fd_5020_16, Fd_5020_16, Fd_5020_16, Fd_5020_16, Fd_5020_16, Fd_4010_25, Fd_4010_25,
    Fd_5020_16, Fd_5020_16, Fd_5020_16, Fd_5020_16, Fd_5020_16, Fd_4010_25, Fd_4010_25,
]

def _make_fk_onnx_session(onnx_path: str):
    """Create an ONNXRuntime session for FK with optional TensorRT EP.

    Uses the same env knobs as base policy:
    - TENSORRT_EP / ONNX_TRT / ORT_TRT
    - ORT_TRT_FP16 / ORT_TRT_INT8 / ORT_TRT_CACHE / ORT_TRT_CACHE_PATH / ORT_TRT_WORKSPACE
    """
    if ort is None:
        raise ImportError("onnxruntime is not installed, cannot load FK ONNX model")
    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(f"FK ONNX file does not exist: {onnx_path}")

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    want_trt = os.environ.get("TENSORRT_EP", "0") not in ("0", "false", "False") or \
               os.environ.get("ONNX_TRT", "0") not in ("0", "false", "False") or \
               os.environ.get("ORT_TRT", "0") not in ("0", "false", "False")
    avail = set(getattr(ort, "get_available_providers", lambda: [])())
    providers: list = []
    if torch.cuda.is_available():
        if want_trt and "TensorrtExecutionProvider" in avail:
            trt_opts = {
                "trt_fp16_enable": os.environ.get("ORT_TRT_FP16", "1") not in ("0", "false", "False"),
                "trt_int8_enable": os.environ.get("ORT_TRT_INT8", "0") not in ("0", "false", "False"),
                "trt_engine_cache_enable": os.environ.get("ORT_TRT_CACHE", "1") not in ("0", "false", "False"),
                "trt_engine_cache_path": os.environ.get("ORT_TRT_CACHE_PATH", ".ort_trt_cache"),
            }
            ws = os.environ.get("ORT_TRT_WORKSPACE", None)
            if ws is not None:
                try:
                    trt_opts["trt_max_workspace_size"] = int(ws)
                except Exception:
                    pass
            providers.append(("TensorrtExecutionProvider", trt_opts))
        if "CUDAExecutionProvider" in avail:
            providers.append(("CUDAExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"}))
    providers.append("CPUExecutionProvider")

    debug = os.environ.get("ONNX_DEBUG", "0") not in ("0", "false", "False")
    try:
        sess = ort.InferenceSession(onnx_path, sess_options=sess_opts, providers=providers)
    except Exception as e:
        if debug:
            print(f"[FK ONNX] Initial session with providers={providers} failed: {e}, falling back to CPUExecutionProvider")
        sess = ort.InferenceSession(onnx_path, sess_options=sess_opts, providers=["CPUExecutionProvider"])

    if debug:
        try:
            print(f"[FK ONNX] Requested providers: {providers}")
            print(f"[FK ONNX] Session providers:  {sess.get_providers()}")
        except Exception:
            pass
    return sess

def normalize_angle(angle):
    """normalize angles to [-pi, pi]"""

    pi=3.14159265358979323846

    while angle > pi:
        angle -= 2 *pi
    while angle < -pi:
        angle += 2 *pi
    return angle


def add_visual_capsule(scene, point1, point2, radius, rgba):
    """Adds one capsule to an mjvScene."""
    if scene.ngeom >= scene.maxgeom:
        return
    scene.ngeom += 1  # increment ngeom
    # initialise a new capsule, add it to the scene using mjv_makeConnector
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom-1],
                        mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                        np.zeros(3), np.zeros(9), rgba.astype(np.float32))
    mujoco.mjv_makeConnector(scene.geoms[scene.ngeom-1],
                            mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
                            point1[0], point1[1], point1[2],
                            point2[0], point2[1], point2[2])
    
def get_euler_xyz(q):
    qx, qy, qz, qw = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = q[:, qw] * q[:, qw] - q[:, qx] * \
                q[:, qx] - q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz]
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    pitch=torch.asin(sinp)
    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = q[:, qw] * q[:, qw] + q[:, qx] * \
                q[:, qx] - q[:, qy] * q[:, qy] - q[:, qz] * q[:, qz]
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack((roll, pitch, yaw), dim=-1)

def rot_subtract_frame_transforms(t01: torch.Tensor, q01: torch.Tensor, t02: torch.Tensor , q02: torch.Tensor ,w_last=True) :
    q10 = _rot.quat_conjugate(q01,w_last)
    if q02 is not None:
        q12 = _rot.quat_mul(q10, q02,w_last)
    else:
        q12 = q10
    return None, q12
class G1():
    def __init__(self):
        self.decimation=5
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.num_envs = 1 
        self.num_observations = 161
        self.num_actions = 29
        self.obs_context_len=15
    
      
        self.p_gains = P_GAINS
        self.p_gains_tensor = torch.tensor(self.p_gains, dtype=torch.float32, device=self.device, requires_grad=False)
        self.d_gains = D_GAINS
        self.d_gains_tensor = torch.tensor(self.d_gains, dtype=torch.float32, device=self.device, requires_grad=False)
            
        self.default_dof_pos_np = PD_BIAS_JOINT_POS

        self.X1 = torch.tensor(X1_list, dtype=torch.float32, device=self.device)
        self.X2 = torch.tensor(X2_list, dtype=torch.float32, device=self.device)
        self.Y1 = torch.tensor(Y1_list, dtype=torch.float32, device=self.device)
        self.Y2 = torch.tensor(Y2_list, dtype=torch.float32, device=self.device)
      
        # dq=0 neighborhood threshold for envelope selection
        self.v_eps = 1e-2
        self.action_scale=ACTION_SCALE.to(self.device)
        
        default_dof_pos = torch.tensor(self.default_dof_pos_np, dtype=torch.float, device=self.device, requires_grad=False)
        self.default_dof_pos = default_dof_pos.unsqueeze(0)

        print(f"default_dof_pos.shape: {self.default_dof_pos.shape}")

        # prepare osbervations buffer
        self.obs_tensor = torch.zeros(1, self.num_observations*self.obs_context_len, dtype=torch.float, device=self.device, requires_grad=False)

        self.motion_times=torch.zeros((1),dtype=torch.float32, device=self.device)
        
    def init_mujoco_viewer(self, robot_xml):
        self.mj_model = mujoco.MjModel.from_xml_path(robot_xml)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_model.opt.timestep =  0.02/self.decimation#0.005
        self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)

        for _ in range(33):
            add_visual_capsule(self.viewer.user_scn, np.zeros(3), np.array([0.001, 0, 0]), 0.05, np.array([0, 1, 0, 1]))
        self.viewer.user_scn.geoms[32].pos = [0,0,0]

        for _ in range(33):
            add_visual_capsule(self.viewer.user_scn, np.zeros(3), np.array([0.001, 0, 0]), 0.05, np.array([0, 0, 1, 1]))
        self.viewer.user_scn.geoms[65].pos = [0,0,0]


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd
def pd_control_friction(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands with motor friction model"""
    tq=(target_q - q) * kp + (target_dq - dq) * kd
    applied_effort =tq- 1*(
            np.array(Fs_list) * np.tanh(dq / np.array(Va_list)) + np.array(Fd_list) * dq
        )
    return  applied_effort
def load_npz(path: str, device="cpu"):
    """Load npz created by csv_to_npz.py"""
    data = np.load(path)
    required = ["joint_pos", "body_pos_w", "body_quat_w"]
    for k in required:
        if k not in data:
            raise KeyError(f"Missing key '{k}' in {path}")
    return {k: torch.from_numpy(data[k]).float().to(device).numpy() for k in data.keys()}
def get_npz_motion(file_path,device='cpu'):
    data=load_npz(file_path)
    re=[]
    for i in range(len(data['joint_pos'])):
        sub_dict={}
        perm = PERM
        sub_dict['rg_pos_t']=torch.tensor(data['body_pos_w'][i][None,:30]).to(device)
        sub_dict['rg_rot_t']=torch.tensor(data['body_quat_w'][i][:,[1,2,3,0]][None,:30]).to(device)
        sub_dict['dof_pos']=torch.tensor(data['joint_pos'][i][None][:,perm]).to(device)
        sub_dict['dof_vel']=torch.tensor(data['joint_vel'][i][None][:,perm]).to(device)
        re.append(sub_dict)
    return re
class DeployNode():

    def __init__(self):
        self.quat_wxyz_r=np.zeros(4)
        self.quat_wxyz_r[0]=1
        
        self.quat_xyzw_r=np.zeros(4)
        self.quat_xyzw_r[3]=1
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.motor_pub_freq = 50
        self.dt = 1/self.motor_pub_freq
        self.joint_pos = torch.zeros(HW_DOF, device=self.device)
        self.joint_vel = torch.zeros(HW_DOF, device=self.device)
        # motion
        self.motion_ids = torch.arange(1).to(self.device)
        self.motion_start_times = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=False)
        self.motion_len = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=False)
        self.config = OmegaConf.load("configs/config.yaml")
        # init policy
        self.init_policy()
 
        clip_mode="envelope"
        self.clip_mode = clip_mode

        print(f"[DeployNode] Clip mode: {self.clip_mode}")
        self.prev_action = torch.zeros(self.env.num_actions, device=self.device)
        self.prev_prev_action = torch.zeros(self.env.num_actions, device=self.device)
        self.prev_base_action = torch.zeros(self.env.num_actions, device=self.device)
        self.start_policy = True

        self.obs_dict={}
        self.history_data: torch.Tensor = torch.zeros(15, 90, device=self.device, dtype=torch.float32)
        self.obs_dict['real_obs']=torch.zeros(1,90).to(self.device)
        self.obs_dict['command_obs']=torch.zeros(1,64).to(self.device)
        self.obs_dict['residual_obs']=torch.zeros(1,183).to(self.device)
        self.obs_dict['real_historical_obs_raw']=self.history_data[None]

          
        self.motion_length_buf = torch.zeros(1, device=self.device, dtype=torch.long)
        self.motion_res_buf = []

        self.stidx=0
        self.motion_res_buf_bm=get_npz_motion(self.config["motion"]['motion_file_npz'], self.device)
        self._ref_motion_length = len(self.motion_res_buf_bm)
        self.motion_res_buf=self.motion_res_buf_bm[self.stidx:]  
        self.bm_anchor_id=9
        self.cur_gt=None
        self.init_motion_rot = self.motion_res_buf[0]["rg_rot_t"][0][0]
        self.init_motion_rot = R.from_quat([self.init_motion_rot[0].item(), self.init_motion_rot[1].item(), self.init_motion_rot[2].item(), self.init_motion_rot[3].item()]).as_euler('xyz', degrees=False)
        
        self.init_motion_yaw=0
        self.initial_yaw=0

        self.env.init_mujoco_viewer(robot_xml=self.config["xml_path"])
        # Initialize robot pose to reference motion first frame
        motion_res_cur = self.motion_res_buf[0]
        # Root position and rotation (xyzw) at first frame
        root_pos = motion_res_cur["rg_pos_t"][0][0].detach().cpu().numpy()
        root_rot_xyzw = motion_res_cur["rg_rot_t"][0][0].detach().cpu().numpy()
        # Convert xyzw -> wxyz for MuJoCo
        self.init_motion_rot_quat=torch.tensor(root_rot_xyzw)
        root_rot_wxyz = np.array([root_rot_xyzw[3], root_rot_xyzw[0], root_rot_xyzw[1], root_rot_xyzw[2]], dtype=np.float32)
        # Joint DOF positions
        ref_dof_pos = motion_res_cur["dof_pos"][0].detach().cpu().numpy()
        # Apply to MuJoCo qpos
        self.rootpos_offset=root_pos[:]
        self.rootpos_offset[2]=0
        self.env.mj_data.qpos[:3] =  root_pos 
        self.env.mj_data.qpos[:3] -=  self.rootpos_offset
        self.env.mj_data.qpos[2]=0.79
        self.env.mj_data.qpos[3:7] = root_rot_wxyz

        self.env.mj_data.qpos[7:] = ref_dof_pos
        mujoco.mj_forward(self.env.mj_model, self.env.mj_data)

        # Sync internal joint state
        self.joint_pos = torch.from_numpy(self.env.mj_data.qpos[7:].copy()).to(self.device)

        tau = pd_control_friction(self.angles, 
                            self.env.mj_data.qpos[7:], 
                            self.env.p_gains, 
                            np.zeros(self.env.num_actions), 
                            self.env.mj_data.qvel[6:], 
                            self.env.d_gains)
        self.env.mj_data.ctrl[:] = tau
        mujoco.mj_step(self.env.mj_model, self.env.mj_data)
            
        self.env.viewer.sync()
    
        self.episode_length_buf = torch.zeros(1, device=self.device, dtype=torch.long)
        self.motion_times=torch.zeros(1,device=self.device)
        time.sleep(1)
        self.update_initial_yaw=True

    def lowlevel_state_mujoco(self):
        """
        Update robot state.
        """
        if  self.start_policy:
            # imu data
            quat = self.env.mj_data.qpos[3:7]
            obs_ang_vel = torch.from_numpy(self.env.mj_data.qvel[3:6]).to(self.device)
            self.obs_ang_vel = (obs_ang_vel + (torch.rand_like(obs_ang_vel)* 2. - 1.) * self.config.noise_scales.base_ang_vel)* self.config.obs_scales.base_ang_vel
            quat_xyzw = torch.tensor([
                quat[1],
                quat[2],
                quat[3],
                quat[0],
            ], device= self.device, dtype= torch.float32).unsqueeze(0)
            obs_base_euler_xyz = get_euler_xyz(quat_xyzw)[:, :].squeeze(0)
            self.obs_base_euler_xyz = (obs_base_euler_xyz + (torch.rand_like(obs_base_euler_xyz)* 2. - 1.) * self.config.noise_scales.base_euler_xyz)* self.config.obs_scales.base_euler_xyz

            rpy = R.from_quat(quat_xyzw.cpu().numpy()) 
            self.roll, self.pitch, self.yaw = rpy.as_euler('xyz', degrees=False)[0]
            self.yaw = normalize_angle(self.yaw - self.initial_yaw)
            rpy = R.from_euler('xyz', [self.roll, self.pitch, self.yaw], degrees=False)
          
            self.quat_xyzw_r[:]=rpy.as_quat()[:]
            self.quat_wxyz_r[:]=_rot.xyzw_to_wxyz(self.quat_xyzw_r)[:]
            # motor data
            self.joint_pos = torch.from_numpy(self.env.mj_data.qpos[7:]).to(self.device)  
            self.joint_vel = torch.from_numpy(self.env.mj_data.qvel[6:]).to(self.device)
            obs_joint_vel = self.joint_vel
            self.obs_joint_vel = (obs_joint_vel + (torch.rand_like(obs_joint_vel)* 2. - 1.) * self.config.noise_scales.dof_vel)* self.config.obs_scales.dof_vel

    def init_policy(self):
        faulthandler.enable()

        # prepare environment
        self.env = G1()
        # Optional FK ONNX (e.g. for arm anchor)
        self.fk_quat_session = None
        fk_quat_path = FK_QUAT_ONNX
        if fk_quat_path is not None and os.path.isfile(fk_quat_path):
            print(f"Loading FK quat ONNX from: {fk_quat_path}")
            self.fk_quat_session = _make_fk_onnx_session(fk_quat_path)

        # Base policy priority: BASE_ONNX -> FM_RESULTS_DIR -> error
        base_onnx = BASE_ONNX
        if base_onnx is not None and os.path.isfile(base_onnx):
            print(f"Loading ONNX base policy from: {base_onnx}")
            self.base_policy = OnnxBasePolicyWrapper(base_onnx, device=str(self.env.device))
            self._base_mode = 'onnx'

            if os.environ.get("ONNX_DEBUG", "0") not in ("0", "false", "False"):
                prov = getattr(self.base_policy, 'session').get_providers()
                print(f"[ONNX] Runtime session providers: {prov}")

            # Optional residual ONNX (separate, summed at inference)
            residual_onnx = RESIDUAL_ONNX
            
            residual_gain = float(os.environ.get("RESIDUAL_GAIN", "1.0"))
            use_tanh = os.environ.get("RESIDUAL_TANH", "0") not in ("0", "false", "False")
            self.residual_gain = residual_gain
            self.residual_use_tanh = use_tanh
            self.residual_policy = OnnxResidualPolicyWrapper(residual_onnx, device=str(self.env.device))
            # Maintain current step base action (env order) for residual input
            self._last_base_action_env = torch.zeros(1, self.env.num_actions, device=self.env.device)
            print(f"Loaded ONNX base + Residual(ONNX) policy: base={base_onnx}, residual={residual_onnx}")
                
        

        self.angles = np.zeros(HW_DOF, dtype=np.float32)
        self.angles_last = np.zeros(HW_DOF, dtype=np.float32)
    
    
    def compute_observations(self):
        """ Computes observations"""
        motion_res_cur=self.motion_res_buf[self.episode_length_buf+1]

        ref_body_pos_extend = motion_res_cur["rg_pos_t"][0]
        ref_body_rot_extend=motion_res_cur["rg_rot_t"][0]
        self.ref_joint_pos = motion_res_cur["dof_pos"][0]
        self.ref_joint_vel = motion_res_cur["dof_vel"][0]

        # reference motion
        ref_joint_angles = self.ref_joint_pos
        ref_joint_angles = (ref_joint_angles + (torch.rand_like(ref_joint_angles)* 2. - 1.) * self.config.noise_scales.ref_joint_angles)* self.config.obs_scales.ref_joint_angles
        ref_joint_velocities = self.ref_joint_vel
        ref_joint_velocities = (ref_joint_velocities + (torch.rand_like(ref_joint_velocities)* 2. - 1.) * self.config.noise_scales.ref_joint_velocities)* self.config.obs_scales.ref_joint_velocities

        dof_pos = self.joint_pos
        dof_pos = (dof_pos + (torch.rand_like(dof_pos)* 2. - 1.) * self.config.noise_scales.dof_pos)* self.config.obs_scales.dof_pos
        dof_vel=self.joint_vel

        base_ang_vel_local=self.obs_ang_vel
        prev_actions=self.prev_action
        prev_base_actions=self.prev_base_action

        dof_pos_relative=dof_pos-self.env.default_dof_pos[0]
        
        real_obs = torch.cat([
                dof_pos_relative,  # Use relative joint positions
                dof_vel,
                base_ang_vel_local,
                prev_base_actions,
            ], dim=-1)

        self.history_data[:-1]= self.history_data[1:].clone()
        self.history_data[-1]= real_obs.squeeze(0).clone()
        dof_command = torch.cat([
            self.ref_joint_pos,
            self.ref_joint_vel,
        ], dim=-1)

        # --- FK timing: optional ONNX FK (same anchor, subset of joints) ---
        if getattr(self, "fk_quat_session", None) is not None:

            ja = dof_pos[[12, 13, 14]][None].detach().cpu().numpy().astype(np.float32)
            base_pos = np.array([[0.0, 0.0, 0.79]], dtype=np.float32)
            base_quat = np.asarray(self.quat_wxyz_r, dtype=np.float32)[None, :]
            pos_np, rot_np = self.fk_quat_session.run(
                    None,
                    {
                        "joint_angles": ja,
                        "base_pos": base_pos,
                        "base_quat": base_quat,
                    },
            )

            # rot_np is assumed wxyz from FK model; convert to xyzw for w_last=True
            cur_gt_anchor = torch.from_numpy(pos_np).to(self.device)
            rot_wxyz = torch.from_numpy(rot_np).to(self.device)
            cur_gr_anchor = rot_wxyz[:, [1, 2, 3, 0]]  # xyzw

          
        self.cur_gr_anchor = cur_gr_anchor.clone()

        _, motion_anchor_ori_b_quat = rot_subtract_frame_transforms(
                cur_gt_anchor,
                cur_gr_anchor,
                ref_body_pos_extend[[self.bm_anchor_id]],
                ref_body_rot_extend[[self.bm_anchor_id]],
            )

        tmp_matrix = _rot.my_quaternion_to_matrix(motion_anchor_ori_b_quat, w_last=True)  # [1, 3, 3]
        anchor_ori_6d = tmp_matrix[..., :2].reshape(-1, 6)  # [1, 6]
        anchor_ori_6d = anchor_ori_6d.squeeze(0)  # [6]
        command_obs = torch.cat([dof_command, anchor_ori_6d], dim=-1)
        real_obs = real_obs.view(1, -1)
        
        inv_perm=INV_PERM
        if self.residual_policy is not None:
           
            B=1

            jpos =  dof_pos_relative[inv_perm]
            jvel = dof_vel[inv_perm]
            base = base_ang_vel_local 
            act  = prev_actions[inv_perm]
            real_reordered = torch.cat([base, jpos, jvel, act], dim=-1)

            cmd_pos = self.ref_joint_pos[inv_perm]
            cmd_vel = self.ref_joint_vel[inv_perm]
            anchor6d = anchor_ori_6d
            cmd_reordered = torch.cat([cmd_pos, cmd_vel, anchor6d], dim=-1)
            a_base_env = self._last_base_action_env.clone()
            a_base_proto = a_base_env[0, inv_perm]
            concat_obs = torch.cat([cmd_reordered, real_reordered, a_base_proto], dim=-1)
    
            self.obs_dict['residual_obs'][:]= concat_obs[None,:]

        self.obs_dict['real_obs'][:]=real_obs[:]
        self.obs_dict['command_obs'][:]=command_obs[None,:]
        self.obs_dict['real_historical_obs_raw'][:]= self.history_data[None,].clone()

        # Optional external initial noise for FM ONNX export (keeps randomness while avoiding RandomNormalLike in ONNX)
        try:
            B = 1
            act_dim = int(getattr(self.env, "num_actions", real_obs.shape[-1]))
            init_noise = torch.randn(B, act_dim, device=self.device, dtype=torch.float32)
            self.obs_dict['initial_noise'] = init_noise
        except Exception:
            pass


    @torch.no_grad()
    def main_loop(self):
        # warm-up
        try:
            _ = self.base_policy(torch.rand_like(self.env.obs_tensor).to(self.env.device))
        except Exception:
            pass
        
        try:
            _ = self.residual_policy(torch.rand_like(self.env.obs_tensor).to(self.env.device))
        except Exception:
            pass
        
        while True:
            loop_start_time = time.monotonic()
            if self.start_policy:        
                self.lowlevel_state_mujoco()
                if self.start_policy:
                    use_quat_yaw_diff=True
                    if self.update_initial_yaw:
                        if use_quat_yaw_diff==False:
                            self.initial_yaw = self.yaw.copy()-self.init_motion_yaw 
                            self.initial_torso_yaw = self.torso_yaw.copy()-self.init_motion_torso_yaw
                        
                        else:
                            diff_root_rot=((R.from_quat(self.init_motion_rot_quat.cpu().numpy()))*(R.from_quat(self.quat_xyzw_r).inv())).as_euler('xyz', degrees=False)
                            self.initial_yaw=-diff_root_rot[2]
                        self.update_initial_yaw=False    
                self.compute_observations()
                self.episode_length_buf += 1
        
                a_base = self.base_policy(self.obs_dict) 
                self._last_base_action_env = a_base.detach().clone()
                self.obs_dict['residual_obs'][:,-29:]=self._last_base_action_env[:,INV_PERM]
                    
                
                a_res = self.residual_policy(self.obs_dict['residual_obs'].detach())
                perm = PERM 
                if perm is not None:
                        a_res = a_res[:, perm]
                raw_actions = a_base + float(getattr(self, "residual_gain", 0.0)) * a_res
                      
                if torch.any(torch.isnan(raw_actions)):
                    assert False

                self.prev_prev_action=self.prev_action.clone()
                self.prev_action = raw_actions.squeeze(0)
                self.prev_base_action = a_base.squeeze(0)
                whole_body_action = raw_actions.squeeze(0)
                
                actions_scaled = whole_body_action * self.env.action_scale + self.env.default_dof_pos
                if self.clip_mode == "envelope":
                    # Envelope-based PD back-solve limits
                    eps = 1e-6
                    abs_dq = self.joint_vel.abs()
                    over = torch.clamp(abs_dq - self.env.X1, min=0.0)
                    # positive torque limit (>=0), dq=0 uses Y2 to match UnitreeActuator
                    base_pos = torch.where(
                        abs_dq <= self.env.v_eps,
                        self.env.Y2,
                        torch.where(self.joint_vel >= 0.0, self.env.Y1, self.env.Y2),
                    )
                    slope_pos = base_pos / torch.clamp(self.env.X2 - self.env.X1, min=eps)
                    tau_high = torch.clamp(base_pos - slope_pos * over, min=0.0)
                    # negative torque limit (<=0), dq=0 uses -Y2
                    base_neg = torch.where(
                        abs_dq <= self.env.v_eps,
                        -self.env.Y2,
                        torch.where(self.joint_vel >= 0.0, -self.env.Y2, -self.env.Y1),
                    )
                    slope_neg = (-base_neg) / torch.clamp(self.env.X2 - self.env.X1, min=eps)
                    tau_low = torch.clamp(base_neg + slope_neg * over, max=0.0)
                    # back-solve for position bounds with dq_des=0
                    p_limits_low = tau_low + self.env.d_gains_tensor * self.joint_vel
                    p_limits_high = tau_high + self.env.d_gains_tensor * self.joint_vel
                    actions_low = (p_limits_low / self.env.p_gains_tensor) + self.joint_pos
                    actions_high = (p_limits_high / self.env.p_gains_tensor) + self.joint_pos

                    _as = actions_scaled.view(-1)
                    _al = actions_low.view(-1)
                    _ah = actions_high.view(-1)
                    clipped_mask = (_as != torch.clip(_as, _al, _ah))
                    if clipped_mask.any(): 
                        clipped_pos_mask = _as > _ah
                        clipped_neg_mask = _as < _al
                        try:
                            pos_idx = torch.nonzero(clipped_pos_mask).squeeze(-1).tolist()
                        except Exception:
                            pos_idx = []
                        try:
                            neg_idx = torch.nonzero(clipped_neg_mask).squeeze(-1).tolist()
                        except Exception:
                            neg_idx = []
                        print("\n[Envelope] Clipping detected:")
                        print(f"  +ve indices (>high): {pos_idx}")
                        if len(pos_idx) > 0:
                            print("    desired:", _as[clipped_pos_mask].detach().cpu().numpy())
                            print("    high   :", _ah[clipped_pos_mask].detach().cpu().numpy())
                            print("    dq     :", self.joint_vel.view(-1)[clipped_pos_mask].detach().cpu().numpy())
                            print("    over   :", (_as - _ah)[clipped_pos_mask].detach().cpu().numpy())
                        print(f"  -ve indices (<low):  {neg_idx}")
                        if len(neg_idx) > 0:
                            print("    desired:", _as[clipped_neg_mask].detach().cpu().numpy())
                            print("    low    :", _al[clipped_neg_mask].detach().cpu().numpy())
                            print("    dq     :", self.joint_vel.view(-1)[clipped_neg_mask].detach().cpu().numpy())
                            print("    over   :", (_al - _as)[clipped_neg_mask].detach().cpu().numpy())
                    angles = torch.clip(actions_scaled, actions_low, actions_high).view(-1)
                
                self.angles = angles.cpu().numpy()
                
                motion_res_cur=self.motion_res_buf[self.episode_length_buf]
                ref_body_pos_extend = motion_res_cur["rg_pos_t"][0]

                if VISUAL:
                    for i in range(ref_body_pos_extend.shape[0]):
                        self.env.viewer.user_scn.geoms[i].pos = ref_body_pos_extend[i].cpu()-self.rootpos_offset
                    if self.cur_gt is not None:
                        for i in range(ref_body_pos_extend.shape[0]):
                            self.env.viewer.user_scn.geoms[i+ref_body_pos_extend.shape[0]].pos = self.cur_gt[i].cpu()

                action_delay_decimation = np.random.randint(self.config["action_depaly_decimation"][0], self.config["action_depaly_decimation"][1]+1)
                for i in range(self.env.decimation):
                    if i == action_delay_decimation:
                        self.angles_last = self.angles.copy()
                    self.env.viewer.sync()
                    tau = pd_control_friction(self.angles_last, 
                                    self.env.mj_data.qpos[7:], 
                                    self.env.p_gains, 
                                    np.zeros(self.env.num_actions), 
                                    self.env.mj_data.qvel[6:], 
                                    self.env.d_gains)
                    self.env.mj_data.ctrl[:] = tau
                    mujoco.mj_step(self.env.mj_model, self.env.mj_data)
                if self.episode_length_buf+1 >= self._ref_motion_length:
                    self.env.viewer.close()
                    break
                            
                sys.stdout.flush()                


if __name__ == "__main__":
    dp_node = DeployNode()
    dp_node.main_loop()
