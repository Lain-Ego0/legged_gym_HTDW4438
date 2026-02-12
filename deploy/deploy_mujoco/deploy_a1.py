import numpy as np
import mujoco, mujoco_viewer
import onnxruntime as ort
import os, time, re

try:
    import glfw
except ImportError:
    raise ImportError("缺少 glfw 库，请运行: pip install glfw")

# ===================== 1. 配置 (Configuration) =====================
class Cfg:
    # 路径
    ROOT = os.path.dirname(os.path.abspath(__file__))
    ROBOT_DIR = os.path.abspath(os.path.join(ROOT, "../../resources/robots/a1"))
    XML = os.path.join(ROBOT_DIR, "xml/scene.xml")
    MESHES = os.path.join(ROBOT_DIR, "meshes")
    POLICY = os.path.abspath(os.path.join(ROOT, "../../onnx/policy_1500.onnx"))

    # 仿真参数
    dt = 0.005      # 200Hz
    decimation = 4  # 50Hz Policy
    
    # 物理参数 (Kp, Kd, Default Pos)
    kps = np.array([60.0] * 12)
    kds = np.array([2.0] * 12)
    tau_limit = 20.0
    default_dof_pos = np.array([0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 
                                0.1, 1.0, -1.5, -0.1, 1.0, -1.5])
    
    # 归一化与控制灵敏度
    obs_scales = [0.25, 2.0, 2.0, 0.25, 1.0, 0.05] # ang_vel, lin_vel_x, lin_vel_y, ang_vel_yaw, dof_pos, dof_vel
    clip_obs = 5.0
    vel_scales = [0.05, 0.05, 0.1] # x, y, yaw step
    vel_decay = 0.95

# ===================== 2. 工具函数 (Utils) =====================
def load_model(xml_path, meshes_dir):
    if not os.path.exists(xml_path): raise FileNotFoundError(xml_path)
    xml_dir = os.path.dirname(xml_path)
    assets = {}
    # 加载 XML 依赖和 Mesh
    for f in os.listdir(xml_dir):
        if f.endswith('.xml') and f != os.path.basename(xml_path):
            with open(os.path.join(xml_dir, f), 'rb') as file: assets[f] = file.read()
    for f in os.listdir(meshes_dir):
        if f.endswith('.stl'):
            with open(os.path.join(meshes_dir, f), 'rb') as file: assets[f] = file.read()
    # 正则修复路径
    with open(xml_path, 'r') as f: content = f.read()
    content = re.sub(r'file="[^"]*?([^\/"]+\.stl)"', r'file="\1"', content)
    return mujoco.MjModel.from_xml_string(content, assets=assets)

def get_controller_input(window, cmd):
    """检测按键并更新速度指令 (Shift+左右=平移)"""
    # 读取状态
    keys = {
        'up': glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS,
        'down': glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS,
        'left': glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS,
        'right': glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS,
        'shift': (glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or 
                  glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS),
        'enter': glfw.get_key(window, glfw.KEY_ENTER) == glfw.PRESS
    }

    # 更新逻辑
    if keys['up']:    cmd[0] += Cfg.vel_scales[0]
    if keys['down']:  cmd[0] -= Cfg.vel_scales[0]
    
    if keys['shift']: # 平移模式
        if keys['left']:  cmd[1] += Cfg.vel_scales[1]
        if keys['right']: cmd[1] -= Cfg.vel_scales[1]
        cmd[2] *= Cfg.vel_decay
    else:             # 旋转模式
        if keys['left']:  cmd[2] += Cfg.vel_scales[2]
        if keys['right']: cmd[2] -= Cfg.vel_scales[2]
        cmd[1] *= Cfg.vel_decay

    if keys['enter']: cmd[:] = 0.0
    
    cmd[:] = np.clip(cmd * Cfg.vel_decay, -1.0, 1.0)
    if np.linalg.norm(cmd) < 0.01: cmd[:] = 0.0
    return cmd

def quat_rotate_inverse(q, v):
    q_w, q_vec = q[-1], q[:3]
    a = v * (2.0 * q_w ** 2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c

# ===================== 3. 主循环 (Main) =====================
if __name__ == '__main__':
    # 1. 初始化
    print(f"🚀 Sim Starting... Policy: {os.path.basename(Cfg.POLICY)}")
    ort_sess = ort.InferenceSession(Cfg.POLICY, providers=['CPUExecutionProvider'])
    input_name = ort_sess.get_inputs()[0].name
    
    model = load_model(Cfg.XML, Cfg.MESHES)
    model.opt.timestep = Cfg.dt
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    
    viewer = mujoco_viewer.MujocoViewer(model, data)
    
    # 2. 状态变量
    cmd_vel = np.zeros(3)
    last_action = np.zeros(12, dtype=np.float32)
    target_q = Cfg.default_dof_pos.copy()
    
    print("🎮 Control: [↑/↓] Move | [←/→] Turn | [Shift + ←/→] Strafe | [Enter] Stop")

    # 3. 循环
    while viewer.is_alive:
        step_start = time.time()

        # Input
        cmd_vel = get_controller_input(viewer.window, cmd_vel)

        # Policy (50Hz)
        if data.time % (Cfg.dt * Cfg.decimation) < Cfg.dt:
            # Get State
            q = data.qpos[-12:].astype(np.double)
            dq = data.qvel[-12:].astype(np.double)
            quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double)
            omega = data.sensor('angular-velocity').data.astype(np.double)
            proj_g = quat_rotate_inverse(quat, np.array([0., 0., -1.]))

            # Build Obs [1, 45]
            obs = np.concatenate([
                omega * Cfg.obs_scales[0],
                proj_g,
                cmd_vel * Cfg.obs_scales[1:4],
                (q - Cfg.default_dof_pos) * Cfg.obs_scales[4],
                dq * Cfg.obs_scales[5],
                last_action
            ]).astype(np.float32).reshape(1, -1)
            
            # Inference
            obs = np.clip(obs, -Cfg.clip_obs, Cfg.clip_obs)
            action = ort_sess.run(None, {input_name: obs})[0][0]
            action = np.clip(action, -10, 10)
            last_action = action.copy()
            
            # Action Scaling
            scaled = action.copy()
            scaled[[0, 3, 6, 9]] *= 0.5 
            scaled *= 0.25              
            target_q = scaled + Cfg.default_dof_pos

        # PD Control (200Hz)
        tau = Cfg.kps * (target_q - data.qpos[-12:]) + Cfg.kds * (0 - data.qvel[-12:])
        data.ctrl = np.clip(tau, -Cfg.tau_limit, Cfg.tau_limit)

        mujoco.mj_step(model, data)
        viewer.render()
        
        # Sync time
        while time.time() - step_start < Cfg.dt: pass

    viewer.close()