import numpy as np
import mujoco, mujoco_viewer
import onnxruntime as ort
import os
import time
import re

# 检查 GLFW 依赖 (必须)
try:
    import glfw
except ImportError:
    raise ImportError("缺少 glfw 库，请运行: pip install glfw")

# ===================== 1. 全局配置类 (Configuration) =====================
class Cfg:
    # --- 路径配置 ---
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
    ROBOT_ROOT = os.path.join(PROJECT_ROOT, "resources/robots/a1")
    XML_PATH = os.path.join(ROBOT_ROOT, "xml/scene.xml")
    MESHES_DIR = os.path.join(ROBOT_ROOT, "meshes")
    POLICY_PATH = os.path.join(PROJECT_ROOT, "onnx/policy_1500.onnx")

    # --- 物理与控制参数 ---
    sim_duration = 60.0
    dt = 0.005      # 200Hz 物理频率
    decimation = 4  # 50Hz 策略频率
    
    # 默认关节角度 (与训练时保持一致)
    default_dof_pos = np.array([0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 
                                0.1, 1.0, -1.5, -0.1, 1.0, -1.5], dtype=np.double)
    
    # PD 控制参数 (较硬的参数以保证站立稳定)
    kps = np.array([60.0] * 12, dtype=np.double)
    kds = np.array([2.0] * 12, dtype=np.double)
    tau_limit = 20.0
    
    # 观测值归一化参数
    class ObsScales:
        ang_vel = 0.25
        lin_vel = 2.0
        dof_pos = 1.0
        dof_vel = 0.05
    clip_obs = 5.0

    # --- 操控灵敏度 ---
    vel_lin_step = 0.05  # 线速度增量
    vel_ang_step = 0.1   # 角速度增量
    vel_decay = 0.95     # 速度自然衰减

# ===================== 2. 核心工具函数 (Utils) =====================
def load_model_robust(xml_path, meshes_dir):
    """加载模型，自动处理 mesh 路径和 xml 依赖"""
    if not os.path.exists(xml_path): raise FileNotFoundError(xml_path)
    xml_dir = os.path.dirname(xml_path)
    
    assets = {}
    # 加载同级 XML 依赖
    for filename in os.listdir(xml_dir):
        if filename.endswith('.xml') and filename != os.path.basename(xml_path):
            with open(os.path.join(xml_dir, filename), 'rb') as f:
                assets[filename] = f.read()
    # 加载 Mesh 资源
    for mf in os.listdir(meshes_dir):
        if mf.endswith('.stl'):
            with open(os.path.join(meshes_dir, mf), 'rb') as f:
                assets[mf] = f.read()
    
    # 读取主 XML 并正则修复路径
    with open(xml_path, 'r') as f: xml_content = f.read()
    xml_content = re.sub(r'file="[^"]*?([^\/"]+\.stl)"', r'file="\1"', xml_content)
    
    return mujoco.MjModel.from_xml_string(xml_content, assets=assets)

def get_obs(data):
    """提取机器人状态"""
    q = data.qpos.astype(np.double)[-12:]
    dq = data.qvel.astype(np.double)[-12:]
    quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double)
    omega = data.sensor('angular-velocity').data.astype(np.double)
    return q, dq, quat, omega

def quat_rotate_inverse(q, v):
    """四元数反向旋转"""
    q_w, q_vec = q[-1], q[:3]
    a = v * (2.0 * q_w ** 2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c

def poll_keyboard(window, current_cmd):
    """主动轮询键盘状态，返回更新后的速度指令"""
    cmd = current_cmd.copy()
    
    # 获取按键状态 (PRESS=1)
    up    = glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS
    down  = glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS
    left  = glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS
    right = glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS
    enter = glfw.get_key(window, glfw.KEY_ENTER) == glfw.PRESS

    # 更新指令
    if up:    cmd[0] += Cfg.vel_lin_step
    if down:  cmd[0] -= Cfg.vel_lin_step
    if left:  cmd[2] += Cfg.vel_ang_step  # 左转
    if right: cmd[2] -= Cfg.vel_ang_step  # 右转
    if enter: cmd[:] = 0.0                # 急停

    # 衰减与限幅
    cmd[0:2] = np.clip(cmd[0:2] * Cfg.vel_decay, -1.0, 1.0)
    cmd[2]   = np.clip(cmd[2] * Cfg.vel_decay,   -1.0, 1.0)
    if np.linalg.norm(cmd) < 0.01: cmd[:] = 0.0
    
    return cmd

# ===================== 3. 主程序 (Main) =====================
if __name__ == '__main__':
    print(f"✅ 加载策略: {os.path.basename(Cfg.POLICY_PATH)}")
    
    # 1. 初始化推理引擎
    policy = ort.InferenceSession(Cfg.POLICY_PATH, providers=['CPUExecutionProvider'])
    input_name = policy.get_inputs()[0].name
    output_name = policy.get_outputs()[0].name

    # 2. 初始化环境
    model = load_model_robust(Cfg.XML_PATH, Cfg.MESHES_DIR)
    model.opt.timestep = Cfg.dt
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data) 

    # 3. 初始化查看器
    viewer = mujoco_viewer.MujocoViewer(model, data)
    window_handle = viewer.window  # 获取 GLFW 窗口句柄

    # 运行时变量
    cmd_vel = np.zeros(3)
    last_action = np.zeros(12, dtype=np.float32)
    target_q = Cfg.default_dof_pos.copy()

    print("\n" + "="*50)
    print("🤖 Unitree A1 仿真就绪 (Stable Polling Ver.)")
    print("🎮 控制: [↑/↓] 前后 | [←/→] 转向 | [Enter] 急停")
    print("ℹ️  提示: 请确保点击黑色仿真窗口以获取焦点")
    print("="*50 + "\n")

    # 4. 仿真主循环
    try:
        start_time = time.time()
        steps = int(Cfg.sim_duration / Cfg.dt)
        
        for i in range(steps):
            if not viewer.is_alive: break

            # --- 控制输入 (Polling) ---
            cmd_vel = poll_keyboard(window_handle, cmd_vel)

            # --- 策略推理 (50Hz) ---
            if i % Cfg.decimation == 0:
                q, dq, quat, omega = get_obs(data)
                proj_gravity = quat_rotate_inverse(quat, np.array([0., 0., -1.]))
                
                # 构造观测向量
                obs_list = [
                    omega * Cfg.ObsScales.ang_vel,
                    proj_gravity,
                    cmd_vel * [Cfg.ObsScales.lin_vel, Cfg.ObsScales.lin_vel, Cfg.ObsScales.ang_vel],
                    (q - Cfg.default_dof_pos) * Cfg.ObsScales.dof_pos,
                    dq * Cfg.ObsScales.dof_vel,
                    last_action
                ]
                obs = np.concatenate(obs_list).astype(np.float32).reshape(1, -1)
                obs = np.clip(obs, -Cfg.clip_obs, Cfg.clip_obs)

                # 推理
                raw_action = policy.run([output_name], {input_name: obs})[0][0]
                raw_action = np.clip(raw_action, -10, 10)
                last_action = raw_action.copy()

                # 动作缩放与映射
                scaled_action = raw_action.copy()
                scaled_action[[0, 3, 6, 9]] *= 0.5 
                scaled_action *= 0.25              
                target_q = scaled_action + Cfg.default_dof_pos

            # --- 底层控制 (200Hz) ---
            tau = Cfg.kps * (target_q - data.qpos[-12:]) + Cfg.kds * (0 - data.qvel[-12:])
            tau = np.clip(tau, -Cfg.tau_limit, Cfg.tau_limit)
            data.ctrl = tau

            # --- 物理步进与渲染 ---
            mujoco.mj_step(model, data)
            viewer.render()
            
            # 实时同步 (简单的 Sleep)
            time.sleep(Cfg.dt)

    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()
        print("仿真结束。")