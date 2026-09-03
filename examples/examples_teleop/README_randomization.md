# Isaac Lab reset / event randomization 学习笔记

这份文档配合下面这个脚本看：

```text
examples/examples_teleop/example01_teleop_spacemouse_record_randomized.py
```

它基于：

```text
examples/examples_teleop/example01_teleop_spacemouse_record.py
```

新增的重点是：

```text
每次 episode reset 时，随机改变方块初始位置和朝向。
```

## 1. 随机化到底有什么用

如果每次录制都是同一个场景：

```text
方块永远在 x=0.30, y=0.00
目标区永远在同一个位置
机械臂永远从同一个角度开始
灯光、相机、颜色都不变
```

那训练出来的策略很容易只学会“这一种摆法”。

随机化的目的就是让数据更像真实世界：

```text
第 1 条 demo：方块偏左
第 2 条 demo：方块偏右
第 3 条 demo：方块转了一个角度
第 4 条 demo：机械臂初始姿态略有不同
```

这样模型更不容易死记硬背。

一句话：

```text
随机化 = 主动制造变化，让策略学会规律，而不是记住固定答案。
```

## 2. reset randomization 和 Replicator randomization 的区别

先分清两类随机化。

### Isaac Lab reset / event randomization

更偏“仿真状态随机化”：

```text
物体位置
物体姿态
物体速度
机器人初始关节角度
门/抽屉/按钮初始状态
质量
摩擦
关节阻尼
```

这些会影响机器人实际怎么运动、怎么接触、怎么完成任务。

### Replicator randomization

更偏“视觉数据随机化”：

```text
颜色
材质
灯光
相机位置
背景
纹理
视觉标注输出
```

这些主要影响相机图像和视觉数据。

当然边界不是绝对的。比如颜色/材质 Isaac Lab 也能改，Replicator 也能改物体 pose。但实践里可以先记：

```text
会改变物理任务本身的，优先放 Isaac Lab reset/event。
主要为了生成视觉图像变化的，优先考虑 Replicator。
```

## 3. 本例怎么随机方块位置

原始脚本里方块是固定位置：

```python
cube.set_initial_pose(Pose(position_xyz=(0.3, 0.0, 0.04)))
```

新增脚本里改成了 `PoseRange`：

```python
cube_pose_range = PoseRange(
    position_xyz_min=(0.22, -0.18, 0.04),
    position_xyz_max=(0.42, 0.18, 0.04),
    rpy_min=(0.0, 0.0, -3.14159),
    rpy_max=(0.0, 0.0, 3.14159),
)
cube.set_initial_pose(cube_pose_range)
```

可以把它理解成：

```text
x 从 0.22 到 0.42 随机
y 从 -0.18 到 0.18 随机
z 固定 0.04
yaw 从 -pi 到 pi 随机
```

每次执行：

```python
env.reset()
```

Arena 会给这个方块生成一个 reset event，然后 Isaac Lab 在 reset 阶段重新采样一次。

## 4. 运行这个随机化录制脚本

容器里运行：

```bash
cd /workspaces/isaaclab_arena
/isaac-sim/python.sh examples/examples_teleop/example01_teleop_spacemouse_record_randomized.py \
  --dataset_file /tmp/franka_spacemouse_randomized_demos.hdf5 \
  --num_demos 5 \
  --randomize_cube_pose
```

调窄随机范围：

```bash
/isaac-sim/python.sh examples/examples_teleop/example01_teleop_spacemouse_record_randomized.py \
  --dataset_file /tmp/franka_spacemouse_randomized_small_range.hdf5 \
  --num_demos 5 \
  --cube_x_min 0.26 --cube_x_max 0.34 \
  --cube_y_min -0.08 --cube_y_max 0.08
```

关闭随机化，退回固定方块：

```bash
/isaac-sim/python.sh examples/examples_teleop/example01_teleop_spacemouse_record_randomized.py \
  --dataset_file /tmp/franka_spacemouse_fixed_demos.hdf5 \
  --num_demos 5 \
  --no-randomize_cube_pose
```

## 5. 随机化范围怎么选

不要一上来随机很大。

比如抓方块任务，推荐从小范围开始：

```text
x: 0.26 到 0.34
y: -0.08 到 0.08
yaw: -0.5 到 0.5
```

确认你能稳定遥操作成功后，再扩大：

```text
x: 0.22 到 0.42
y: -0.18 到 0.18
yaw: -3.14 到 3.14
```

如果范围太大，会出现：

```text
方块离机械臂太远
方块靠近桌边
目标不合理
人录制很累
成功率太低
训练数据质量下降
```

调参顺序建议：

```text
先随机 x/y
再随机 yaw
再随机机器人初始关节
再随机目标区
最后再考虑颜色、灯光、相机
```

## 6. 常见随机化种类

### 物体初始位置

用途：

```text
让策略学会从不同位置抓取物体。
```

例子：

```python
cube.set_initial_pose(
    PoseRange(
        position_xyz_min=(0.25, -0.10, 0.04),
        position_xyz_max=(0.35, 0.10, 0.04),
    )
)
```

### 物体初始朝向

用途：

```text
让策略适应物体旋转。
```

例子：

```python
cube.set_initial_pose(
    PoseRange(
        position_xyz_min=(0.30, 0.00, 0.04),
        position_xyz_max=(0.30, 0.00, 0.04),
        rpy_min=(0.0, 0.0, -1.57),
        rpy_max=(0.0, 0.0, 1.57),
    )
)
```

### 机器人初始关节

用途：

```text
让机械臂不要每次都从完全相同姿态开始。
```

Franka embodiment 里已经有类似配置：

```python
randomize_franka_joint_state = EventTerm(
    func=franka_stack_events.randomize_joint_by_gaussian_offset,
    mode="reset",
    params={
        "mean": 0.0,
        "std": 0.02,
        "asset_cfg": SceneEntityCfg("robot"),
    },
)
```

直觉：

```text
std=0.02：每个关节在默认角度附近小幅抖动
std 越大：初始姿态变化越大，但也越可能进入不好操作的位置
```

### 目标位置

用途：

```text
让“放到哪里”也变化，而不是永远放到同一个目标区。
```

本例的 `GoalPoseTask` 目前使用固定目标区范围：

```python
TARGET_X_RANGE = (0.4, 0.6)
TARGET_Y_RANGE = (-0.15, 0.15)
TARGET_Z_RANGE = (0.02, 0.3)
```

如果要做“每条 demo 一个不同目标区”，需要把目标区本身也做成可 reset 的资产或任务状态。这个比随机方块初始位置更复杂，因为成功判定和绿色可视化区域也要同步更新。

### 质量、摩擦、接触参数

用途：

```text
让策略不只适应一个完美物理世界。
```

例子：

```text
方块稍微重一点/轻一点
桌面稍微滑一点/涩一点
夹爪摩擦稍微变化
```

这类属于 domain randomization。好处是训练出来更鲁棒，坏处是范围太大时任务会变难，甚至录制数据风格不一致。

### 颜色、材质、灯光、相机

用途：

```text
让视觉模型不要只记住固定颜色和固定光照。
```

如果只是遥操作动作数据，不启用相机，这些对 HDF5 里的动作学习影响不大。

如果你训练的是视觉策略，比如输入 RGB 图像，那这些非常重要。

这类通常 Replicator 更顺手：

```text
Replicator 随机颜色/灯光/相机
Isaac Lab 负责机器人控制和任务 reset
```

## 7. 录制和训练时要不要随机一致

要。

如果录制时：

```text
方块只在 x=0.25 到 0.35
y=-0.10 到 0.10
```

训练或部署时突然变成：

```text
方块在 x=0.10 到 0.70
y=-0.40 到 0.40
```

模型很可能不会做。

所以要记住：

```text
训练时看到的变化范围，应该覆盖部署时会遇到的变化范围。
```

但也不要反过来过度随机：

```text
录制时范围太大 -> 人录制困难 -> 成功 demo 少 -> 数据质量差
```

比较稳的流程是：

```text
小范围录 10 条
确认训练能学会
扩大范围录 30 条
再训练
继续扩大范围
```

## 8. HDF5 里会记录随机化结果吗

通常会。

录制 HDF5 里会保存：

```text
initial_state
states
obs
actions
```

如果 reset 时方块位置被随机了，这个随机后的状态会体现在：

```text
initial_state/...  episode 初始状态
states/...         每一步仿真状态
obs/...            如果观测项包含物体 pose，也会反映出来
```

也就是说，BC 训练时不能只看动作，还要看输入观测里有没有包含对应随机变量。

如果你随机了方块位置，但训练观测里没有方块位置、也没有相机图像，策略就不知道方块在哪里。

一句话：

```text
随机了什么，策略输入里最好能看到什么。
```

## 9. 最小心智模型

把 reset randomization 想成开局洗牌：

```text
episode 开始
    |
    v
reset events 执行
    |
    +-- 随机方块位置
    +-- 随机方块朝向
    +-- 随机机器人初始关节
    +-- 随机物理参数
    |
    v
人开始遥操作
    |
    v
录制 obs/actions/states
```

重点是：

```text
reset randomization 改的是“这一局开始时世界长什么样”。
Replicator randomization 更偏“相机看到的图像长什么样”。
```

