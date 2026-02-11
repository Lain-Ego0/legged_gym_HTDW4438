# HTDW4438_Isaacgym 
**基于Leggedgym，用于高擎小型四足的训练仓库**

- 飞书安装文档[🎯 欢迎来到IsaacGym](https://wcn9j5638vrr.feishu.cn/wiki/Aq7mwJ0Zmi0e0TkquqDcNSHYnnf)

## 相关参考链接
- 项目官网：https://leggedrobotics.github.io/legged_gym/
- 核心论文：https://arxiv.org/abs/2109.11978

## 一、完整安装步骤
整体安装思路为：
1. 安装支持 CUDA 的 NVIDIA 驱动
2. 创建 Python 3.8 的conda环境。
3. 安装CUDA
4. 安装与 CUDA 版本匹配的 PyTorch。
5. 安装Isaacgym
6. 下载isacgym官方环境包
### 分步安装命令与操作
1.  安装支持 CUDA 的 NVIDIA 驱动
   
    ```bash
    # 卸载所有NVIDIA相关包
    sudo apt-get remove --purge '^nvidia-.*'
    sudo apt-get autoremove
    sudo apt-get autoclean

    # 清理驱动模块残留
    sudo rm -rf /lib/modules/$(uname -r)/kernel/drivers/video/nvidia.ko
    sudo update-initramfs -u  # 更新启动镜像

    # 确保驱动编译和加载所需的工具齐全：
    sudo apt install build-essential dkms linux-headers-$(uname -r)

    # 添加官方显卡驱动源
    sudo add-apt-repository ppa:graphics-drivers/ppa
    sudo apt update

    # 安装535版本（相对比较稳定）
    sudo apt install nvidia-driver-535
    ```

2.  创建 Python 3.8 的conda环境
   
    ```bash
    # 安装miniconda 
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    chmod +x Miniconda3-latest-Linux-x86_64.sh
    ./Miniconda3-latest-Linux-x86_64.sh

    # 创建并验证虚拟环境
    conda create -n leggedgym python=3.8.10
    # 进入环境
    conda activate leggedgym

    ```

3.  安装CUDA
    1. 首先进入CUDA官网：https://developer.nvidia.com/cuda-toolkit-archive
    2. 我们以12.1为例：
    ```bash
    wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
    sudo sh cuda_12.1.0_530.30.02_linux.run
    ```
4.  安装与 CUDA 版本匹配的 PyTorch
   
    ```bash
    # 安装环境内cuda
    conda activate leggedgym
    conda install pytorch torchvision pytorch-cuda=12.1.0 -c pytorch -c nvidia

    # 安装pytorch
    sudo apt install python3-pip
    pip install torchvision -i https://pypi.tuna.tsinghua.edu.cn/simple
    pip install torch -i https://pypi.tuna.tsinghua.edu.cn/simple
    pip install pyquaternion -i https://pypi.tuna.tsinghua.edu.cn/simple
    pip install pyyaml -i https://pypi.tuna.tsinghua.edu.cn/simple
    pip install pexpect -i https://pypi.tuna.tsinghua.edu.cn/simple
    pip install matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple
    pip install einops -i https://pypi.tuna.tsinghua.edu.cn/simple
    pip install tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple
    pip install packaging -i https://pypi.tuna.tsinghua.edu.cn/simple
    pip install h5py -i https://pypi.tuna.tsinghua.edu.cn/simple
    pip install ipython -i https://pypi.tuna.tsinghua.edu.cn/simple
    pip install getkey -i https://pypi.tuna.tsinghua.edu.cn/simple
    pip install wandb -i https://pypi.tuna.tsinghua.edu.cn/simple
    pip install chardet -i https://pypi.tuna.tsinghua.edu.cn/simple
    pip install matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple
    pip install numpy==1.23.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
    pip install h5py_cache -i https://pypi.tuna.tsinghua.edu.cn/simple
    pip install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple
    pip install tensorboard -i https://pypi.tuna.tsinghua.edu.cn/simple
    pip install onnxruntime
    pip install mujoco-python-viewer

    ```
5.  安装Isaacgym
    从官网 https://developer.nvidia.com/isaac-gym 下载 Isaac Gym Preview 4，下载解压即可

6.  下载isacgym官方包环境
   
    ```bash
    # 克隆仓库
    git clone https://github.com/isaac-sim/IsaacGymEnvs.git
    conda activate leggedgym 
    pip install -e ./IsaacGymEnvs
    ```

7.  安装rsl_rl（PPO算法实现，必须切换v1.0.2版本）
    ```bash
    # 克隆仓库
    git clone https://github.com/leggedrobotics/rsl_rl
    # 切换指定版本并安装
    cd rsl_rl && git checkout v1.0.2 && pip install -e .
    ```

8.  安装legged_gym本体
    ```bash
    # 克隆仓库
    git clone https://github.com/leggedrobotics/legged_gym
    # 执行可编辑模式安装
    cd legged_gym && pip install -e .
    ```

## 二、代码核心结构
1.  每个环境由两部分定义：
    - 环境文件：`legged_robot.py`，实现环境核心逻辑
    - 配置文件：`legged_robot_config.py`，包含两类配置类：环境参数类`LeggedRobotCfg`、训练参数类`LeggedRobotCfgPPo`
2.  环境与配置类均支持继承机制，便于快速扩展新任务，一般建议在各个机器人子类中设置。
3.  配置文件`cfg`中，所有非零的奖励缩放系数，都会自动对应同名奖励函数，最终总奖励为所有激活奖励的加权和
4.  任务必须通过`task_registry.register(name, EnvClass, EnvConfig, TrainConfig)`完成注册，注册入口为`envs/__init__.py`，也支持仓库外外部注册。

## 三、核心使用方法
### 1. 训练策略
#### 基础训练命令
```bash
# 基础示例
python legged_gym/scripts/train.py --task=anymal_c_flat

# 用户自定义任务训练
python legged_gym/scripts/train.py --task=htdw_4438 --headless
```

#### 关键命令行参数与附加命令
| 参数 | 功能说明 | 示例命令 |
| :--- | :--- | :--- |
| `--sim_device=cpu`/`--rl_device=cpu` | 指定仿真/强化学习运算设备（CPU模式） | `python legged_gym/scripts/train.py --task=anymal_c_flat --sim_device=cpu --rl_device=cpu` |
| `--headless` | 无渲染无头模式运行，提升训练性能 | `python legged_gym/scripts/train.py --task=anymal_c_flat --headless` |
| `--resume` | 从检查点恢复训练 | `python legged_gym/scripts/train.py --task=anymal_c_flat --resume` |
| `--experiment_name` | 指定实验名称 | `python legged_gym/scripts/train.py --task=anymal_c_flat --experiment_name my_exp` |
| `--run_name` | 指定单次运行名称 | `python legged_gym/scripts/train.py --task=anymal_c_flat --run_name my_run` |
| `--load_run` | 指定恢复训练的运行记录名称 | `python legged_gym/scripts/train.py --task=anymal_c_flat --resume --load_run Jan27_17-56-48_h` |
| `--checkpoint` | 指定恢复训练的模型检查点迭代数 | `python legged_gym/scripts/train.py --task=anymal_c_flat --resume --load_run Jan27_17-56-48_h --checkpoint 1500` |
| `--num_envs` | 指定并行创建的环境数量 | `python legged_gym/scripts/train.py --task=anymal_c_flat --num_envs 4096` |
| `--seed` | 指定随机种子 | `python legged_gym/scripts/train.py --task=anymal_c_flat --seed 42` |
| `--max_iterations` | 指定训练最大迭代次数 | `python legged_gym/scripts/train.py --task=anymal_c_flat --max_iterations 3000` |

> 性能优化提示：训练启动后，按键盘`v`可关闭渲染提升性能，后续可再次按`v`恢复渲染查看进度。
> 模型保存路径：`issacgym_anymal/logs/<experiment_name>/<date_time>_<run_name>/model_<iteration>.pt`

### 2. 加载并回放训练好的策略
#### 基础回放命令
```bash
# 基础示例
python legged_gym/scripts/play.py --task=anymal_c_flat

# 用户自定义任务+指定运行记录+检查点回放
export PYTHONPATH=. && python legged_gym/scripts/play.py --task=htdw_4438 --load_run Jan27_17-56-48_h --checkpoint 1500
```

> 默认加载规则：默认加载实验文件夹下，最近一次运行的最新模型；可通过`load_run`和`checkpoint`参数指定特定模型。

### 3. Tensorboard 日志查看命令
```bash
# 查看当前目录下的训练日志
tensorboard --logdir .
```

## 四、新增自定义环境指南
基础环境`legged_robot`已实现粗糙地形运动核心任务，对应配置未指定机器人资产、无奖励缩放系数，可基于此快速扩展新环境：
1.  在`envs/`下新建文件夹，创建`<your_env>_config.py`配置文件，继承现有环境配置
2.  新增全新机器人时：
    - 将机器人URDF/MJCF等资产放入`resources/`目录
    - 在配置文件`cfg`中设置资产路径，定义刚体名称、默认关节位置、PD增益，指定训练配置与环境类名
    - 在训练配置`train_cfg`中设置`experiment_name`和`run_name`
3.  如需自定义环境逻辑，在`<your_env>.py`中继承现有环境类，重写对应方法、新增奖励函数
4.  在`isaacgym_anymal/envs/__init__.py`中完成新环境注册
5.  按需调优`cfg`和`cfg_train`中其他参数，无需某奖励时将其缩放系数设为0即可，禁止修改其他已有环境的参数。

## 五、故障排查
### 常见报错解决方案
若出现报错：`ImportError: libpython3.8m.so.1.0: cannot open shared object file: No such file or directory`
1.  执行系统安装命令：
    ```bash
    sudo apt install libpython3.8
    ```
2.  若仍未解决，执行Python库路径导出命令（conda环境需替换为对应环境lib路径）：
    ```bash
    # 通用系统Python
    export LD_LIBRARY_PATH=/path/to/libpython/directory
    # Conda虚拟环境
    export LD_LIBRARY_PATH=/path/to/conda/envs/your_env/lib
    ```

### 已知问题说明
GPU仿真下，使用三角网格地形时，`net_contact_force_tensor`上报的接触力数据不可靠。
- 解决方案：在机器人足部/末端执行器添加力传感器，参考代码如下：
```python
sensor_pose = gymapi.Transform()
for name in feet_names:
    sensor_options = gymapi.ForceSensorProperties()
    sensor_options.enable_forward_dynamics_forces = False # 例如重力
    sensor_options.enable_constraint_solver_forces = True # 例如接触力
    sensor_options.use_world_frame = True # 世界坐标系下返回力数据，便于获取垂直分量
    index = self.gym.find_asset_rigid_body_index(robot_asset, name)
    self.gym.create_asset_force_sensor(robot_asset, index, sensor_pose, sensor_options)

# 传感器张量读取
sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
self.gym.refresh_force_sensor_tensor(self.sim)
force_sensor_readings = gymtorch.wrap_tensor(sensor_tensor)
self.sensor_forces = force_sensor_readings.view(self.num_envs, 4, 6)[..., :3]

self.gym.refresh_force_sensor_tensor(self.sim)
contact = self.sensor_forces[:, :, 2] > 1.
```

## 六、全量命令汇总表
| 命令类别 | 完整可执行命令 |
| :--- | :--- |
| 日志查看 | `tensorboard --logdir .` |
| 策略回放 | `export PYTHONPATH=. && python legged_gym/scripts/play.py --task=htdw_4438 --load_run Jan27_17-56-48_h --checkpoint 1500` |
| 策略训练 | `python legged_gym/scripts/train.py --task=htdw_4438 --headless` |
| Pytorch安装 | `pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html` |
| Isaac Gym安装 | `cd isaacgym/python && pip install -e .` |
| Isaac Gym安装验证 | `cd examples && python 1080_balls_of_solitude.py` |
| rsl_rl克隆 | `git clone https://github.com/leggedrobotics/rsl_rl` |
| rsl_rl安装 | `cd rsl_rl && git checkout v1.0.2 && pip install -e .` |
| legged_gym安装 | `cd legged_gym && pip install -e .` |
| 基础训练示例 | `python legged_gym/scripts/train.py --task=anymal_c_flat` |
| 基础回放示例 | `python legged_gym/scripts/play.py --task=anymal_c_flat` |
| 系统依赖修复 | `sudo apt install libpython3.8` |
| 库路径环境变量配置 | `export LD_LIBRARY_PATH=/path/to/libpython/directory` |
| Conda库路径环境变量配置 | `export LD_LIBRARY_PATH=/path/to/conda/envs/your_env/lib` |