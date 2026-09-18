# examples_urdf_all_train

这个目录是一个从零开始的极简闭环 demo：

```text
手写 URDF
  -> Isaac Sim 转 USD/USDA
  -> 检查 USD 里的 link、joint、axis、limit、drive
  -> 手动修改 USD drive
  -> Isaac Lab 加载为 articulation
  -> SpaceMouse 遥操作
  -> HDF5 录制
  -> 行为克隆 BC 训练
```

目标不是做一个真实机械臂，而是把每个环节拆开看清楚。

## 目录结构

```text
examples/examples_urdf_all_train/
  README.md
  00_convert_urdf_to_usd.py
  01_inspect_usd_joints.py
  02_patch_usd_drives.py
  03_teleop_spacemouse_simple_arm.py
  04_record_spacemouse_simple_arm.py
  05_train_bc_simple_arm.py
  assets/simple_urdf_arm/simple_6dof_arm.urdf
```

机器人 asset 注册在：

```text
isaaclab_arena/embodiments/simple_urdf_arm/simple_urdf_arm.py
```

注册名：

```text
simple_urdf_arm_ik
```

## 0. 为什么是 6 关节？

这个 demo 用的是一个很小的 6-DOF 串联机械臂。原因是 SpaceMouse 输出的是 6D 末端增量：

```text
dx, dy, dz, droll, dpitch, dyaw
```

如果只写 2 关节，机械臂本身自由度太少，没法自然演示 6D IK 遥操作。所以这里采用：

```text
base_link
  -- joint_1 -->
link_1
  -- joint_2 -->
link_2
  -- joint_3 -->
link_3
  -- joint_4 -->
link_4
  -- joint_5 -->
link_5
  -- joint_6 -->
link_6
```

但每个 link 都是 box/cylinder，没有 mesh 文件，尽量保持可读。

## 1. 看 URDF

先打开：

```text
assets/simple_urdf_arm/simple_6dof_arm.urdf
```

重点看两类元素。

### link

```xml
<link name="link_2">
  <inertial>...</inertial>
  <visual>...</visual>
  <collision>...</collision>
</link>
```

含义：

```text
inertial   质量、质心、惯量，给物理用
visual     给人看的形状
collision  给碰撞检测用的形状
```

### joint

```xml
<joint name="joint_2" type="revolute">
  <parent link="link_1"/>
  <child link="link_2"/>
  <origin xyz="0 0 0.24" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>
  <limit lower="-1.5708" upper="1.5708" effort="20" velocity="2.0"/>
  <dynamics damping="0.0" friction="0.0"/>
</joint>
```

含义：

```text
joint_2 把 link_2 接到 link_1 上
origin 是 joint_2 在 link_1 坐标系里的安装位置
axis="0 1 0" 表示绕 Y 轴旋转
lower/upper 是角度范围，单位 rad
effort 是 URDF 里的最大力矩提示
velocity 是 URDF 里的最大速度提示，单位 rad/s
```

## 2. URDF 转 USD

进入 Docker 容器后运行：

```bash
cd /workspaces/isaaclab_arena

/isaac-sim/python.sh examples/examples_urdf_all_train/00_convert_urdf_to_usd.py
```

预期输出：

```text
[INFO] imported USD: .../assets/simple_urdf_arm/usd/simple_6dof_arm/simple_6dof_arm.usda
[INFO] defaultPrim: /simple_6dof_arm
[INFO] drive joint_1: stiffness=80, damping=8
...
[INFO] saved Isaac Lab friendly USD: ...
```

生成结果大概在：

```text
assets/simple_urdf_arm/usd/simple_6dof_arm/simple_6dof_arm.usda
assets/simple_urdf_arm/usd/simple_6dof_arm/payloads/...
```

转换脚本做了三件事：

```text
1. 调 Isaac Sim URDFImporter，把 URDF 导入成 USD。
2. 给 joint_1 到 joint_6 写入 USD drive stiffness/damping。
3. 把 PhysicsArticulationRootAPI 放到 defaultPrim 上，让 Isaac Lab 更容易识别 /Robot 是 articulation。
```

## 3. 检查 USD 关节是否正确

运行：

```bash
/isaac-sim/python.sh examples/examples_urdf_all_train/01_inspect_usd_joints.py
```

你应该重点看：

```text
[INFO] defaultPrim APIs
[LINKS]
[JOINTS]
```

一条正确的 revolute joint 应该能看到：

```text
/simple_6dof_arm/Physics/joint_2 (PhysicsRevoluteJoint)
  body0: [/simple_6dof_arm/Geometry/base_link/link_1]
  body1: [/simple_6dof_arm/Geometry/base_link/link_1/link_2]
  physics:axis: Y
  physics:lowerLimit: -90
  physics:upperLimit: 90
  drive:angular:stiffness: 80
  drive:angular:damping: 8
```

检查逻辑：

```text
body0/body1 是否接对了父子 link
axis 是否和 URDF 的 axis 对应
lower/upper 是否和 URDF 限位对应
每个 joint 是否都有 drive stiffness/damping
defaultPrim 是否有 PhysicsArticulationRootAPI
```

注意：USD 里的 revolute limit 很多时候显示为角度 degree，而 URDF 里是 rad。比如：

```text
URDF:  1.5708 rad
USD:   90 degree
```

## 4. 手动修改 USD 驱动器

如果你想练习不重新转换 URDF，直接改 USD drive：

```bash
/isaac-sim/python.sh examples/examples_urdf_all_train/02_patch_usd_drives.py \
  --stiffness 120 \
  --damping 12
```

再检查：

```bash
/isaac-sim/python.sh examples/examples_urdf_all_train/01_inspect_usd_joints.py
```

你会看到：

```text
drive:angular:stiffness: 120
drive:angular:damping: 12
```

这一步说明：

```text
URDF 主要描述结构和限制；
USD 可以承载更具体的仿真 drive 参数；
Isaac Lab 的 ArticulationCfg/ImplicitActuatorCfg 也能在加载后覆盖或补充驱动参数。
```

## 5. Isaac Lab 里驱动器怎么写

本 demo 的 Isaac Lab asset 在：

```text
isaaclab_arena/embodiments/simple_urdf_arm/simple_urdf_arm.py
```

核心配置：

```python
actuators={
    "arm": ImplicitActuatorCfg(
        joint_names_expr=["joint_[1-6]"],
        effort_limit=40.0,
        velocity_limit=2.5,
        stiffness=120.0,
        damping=12.0,
    ),
}
```

这里的含义：

```text
joint_names_expr=["joint_[1-6]"]
  只驱动 joint_1 到 joint_6。

stiffness=120
  关节偏离目标时，控制器多积极拉回目标。

damping=12
  关节运动时的刹车/减震。

effort_limit=40
  实际输出力矩最多 40。

velocity_limit=2.5
  关节转速最多 2.5 rad/s。
```

这个 demo 里 USD drive 和 Isaac Lab actuator 都写了，是为了教学对照。实际项目里通常以 Isaac Lab 的 actuator 为主，因为训练/控制配置更集中。

## 6. SpaceMouse 遥操作

先确保 USD 已生成，然后运行：

```bash
/isaac-sim/python.sh examples/examples_urdf_all_train/03_teleop_spacemouse_simple_arm.py \
  --pos_sensitivity 0.10 \
  --rot_sensitivity 0.25
```

启动后会打印：

```text
[INFO] Simple URDF arm USD: ...
[INFO] joints: [...]
[INFO] bodies: [...]
```

重点检查：

```text
joints 里有 joint_1 到 joint_6
bodies 里有 link_6
```

如果 FrameTransformer 报：

```text
No matching rigid-body prims were found
```

通常是：

```text
FrameTransformer 的 prim_path 不是真正 rigid body
或者 USD 的 articulation root 层级不适合 Isaac Lab
```

## 7. 录制 HDF5

运行：

```bash
/isaac-sim/python.sh examples/examples_urdf_all_train/04_record_spacemouse_simple_arm.py \
  --dataset_file /tmp/simple_arm_demos.hdf5 \
  --num_demos 3 \
  --steps_per_demo 200
```

这个录制 demo 不做复杂任务成功判定。每段达到 `steps_per_demo` 后，就把当前 episode 标为 success 并导出。

这样做是为了学习数据格式：

```text
data/demo_0/actions
data/demo_0/obs/actions
data/demo_0/obs/joint_pos
data/demo_0/obs/joint_vel
data/demo_0/obs/eef_pos
data/demo_0/obs/eef_quat
data/demo_0/states/...
```

## 8. 训练最小 BC

训练脚本不启动 Isaac Sim，只需要普通 Python + torch + h5py：

```bash
python3 examples/examples_urdf_all_train/05_train_bc_simple_arm.py \
  --dataset_file /tmp/simple_arm_demos.hdf5 \
  --epochs 100 \
  --output /tmp/simple_arm_bc.pt
```

它做的是：

```text
读取 HDF5
把 obs/actions 拼成监督学习数据
训练 MLP：obs -> action
保存 checkpoint
```

这就是最小行为克隆：

```text
示范数据 (obs, action)
  -> 监督学习
  -> 策略学会模仿你的动作
```

## 9. 录制、训练、部署环境要一致吗？

对行为克隆 BC 来说，最稳的是：

```text
录制环境 = 部署/评估环境
```

因为 BC 学的是：

```text
看到某种 obs -> 模仿人当时输出的 action
```

如果录制时是 A 世界，部署时变成 B 世界，模型可能就不会了。

比如录制时：

```text
机械臂在 (0.0, 0.0, 0.0)
方块在   (0.45, 0.0, 0.04)
```

部署/评估时如果改成：

```text
机械臂在 (1.0, 0.0, 0.0)
方块还在 (0.45, 0.0, 0.04)
```

那机器人和方块的相对关系完全变了，BC 策略很可能失效。

但要注意：`05_train_bc_simple_arm.py` 是离线训练脚本，它不启动 Isaac Sim，也不创建环境。它只做：

```text
读取 HDF5 里的 obs/actions
训练 MLP
保存 checkpoint
```

所以严格说：

```text
训练脚本本身没有“机械臂摆在哪里”这个概念。
```

真正需要保持一致的是：

```text
录制时的环境
未来部署/评估这个 BC policy 时的环境
```

可以不一样吗？可以，但有条件。

### 可以轻微不一样

比如：

```text
方块位置小范围随机
光照变化
摄像机变化
初始关节角度小范围随机
```

前提是训练数据里也覆盖了这些变化。这通常叫 domain randomization 或数据增强。

### 不建议完全不一样

比如：

```text
录制用 simple_urdf_arm_ik，部署用 Franka 或 ABB
录制 action 是 6D IK，部署 action 换成关节位置
录制 obs 有 eef_pos/eef_quat，部署 obs 变了
录制任务是移动末端，部署任务变成抓取放置
录制时机械臂和物体相对位置固定，部署时相对位置大变
```

这种就不是同一个学习问题了。

如果你希望策略适应不同机械臂摆放位置，录制阶段就要覆盖这些情况：

```text
robot 初始位置 A
robot 初始位置 B
robot 初始位置 C
cube 相对位置 A
cube 相对位置 B
```

或者把 observation 设计成相对坐标，让策略看到的是：

```text
物体相对末端的位置
目标相对末端的位置
```

而不是强依赖世界坐标。

一句话：

```text
训练脚本只吃 HDF5；
录制和部署/评估最好一致；
如果要不一致，就要让训练数据覆盖这些变化，或者把 obs 设计成相对关系。
```

## 10. 推荐学习顺序

按这个顺序最稳：

```text
1. 读 simple_6dof_arm.urdf
2. 跑 00_convert_urdf_to_usd.py
3. 跑 01_inspect_usd_joints.py
4. 跑 02_patch_usd_drives.py 改 stiffness/damping
5. 再跑 01_inspect_usd_joints.py 确认 USD 被改了
6. 跑 03_teleop_spacemouse_simple_arm.py 看 Isaac Lab 是否能遥操作
7. 跑 04_record_spacemouse_simple_arm.py 录 HDF5
8. 跑 05_train_bc_simple_arm.py 训练 BC
```

## 11. 常见问题

### 找不到 USD

先运行：

```bash
/isaac-sim/python.sh examples/examples_urdf_all_train/00_convert_urdf_to_usd.py
```

### 找不到 retargeter

报错类似：

```text
component spacemouse__simple_urdf_arm_ik not found
```

说明 `isaaclab_arena/assets/retargeter_library.py` 里的 `SimpleURDFArmSpaceMouseRetargeter` 没被注册或代码没更新到容器。

### joint 数量不对

看：

```bash
/isaac-sim/python.sh examples/examples_urdf_all_train/01_inspect_usd_joints.py
```

再看 Isaac Lab 启动打印的：

```text
[INFO] joints: [...]
```

两边都应该能看到 `joint_1` 到 `joint_6`。

### 机械臂不动、软、抖

先看四个参数：

```text
effort_limit   不够 -> 推不动
velocity_limit 太低 -> 只能慢慢动
stiffness      太低 -> 软、跟不上
damping        太低 -> 抖、冲过头
damping        太高 -> 钝、慢
```
