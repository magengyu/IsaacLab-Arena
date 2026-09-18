# IsaacLab-Arena 自定义环境实践（fork 版）

> 从零定义自己的环境，从最小可视化到完整 RL 训练。六个级别难度递增，每个都可跑。
> 本目录是 [`docs/IsaacLab-Arena-自定义环境实践.md`](../../../../docs/IsaacLab-Arena-自定义环境实践.md) 适配 **`references/IsaacLab-Arena_magengyu`** 后的版本，所有 `.py` 脚本抽离为独立文件放在本目录，通过 `--external_environment_class_path` 加载。

---

## 一、前置：理解三板斧

定义一个 Arena 环境 = 写 3 样东西：

```
1. 环境类 (@register_environment)
   └─ build() 方法：组装 Scene + Embodiment + Task → IsaacLabArenaEnvironment

2. 任务类 (@register_task, 继承 TaskBase)
   └─ 定义终止条件、奖励函数、观察空间、指标

3. 资产类 (@register_asset, 可选)
   └─ 定义新的物体/场景/机器人
```

**与 fork 原版 `examples/external_env/` 的区别**：本目录用**当前推荐的 `ArenaEnvironmentFactory + build(cfg)` 模式**；fork 的 `examples/external_env/my_env.py` 用的是已弃用的 `ExampleEnvironmentBase + get_env()` 兼容层（正在迁移，见 `isaaclab_arena_environments/example_environment_base.py` 的 TODO）。

---

## 二、目录结构（全部外挂式）

本目录 9 个脚本都通过 `--external_environment_class_path` 加载，**无需修改 fork 的 `isaaclab_arena_environments/` 包**：

| 文件 | 环境名 | 类名 | 说明 |
|------|--------|------|------|
| `level0a_minimal.py` | `level0a_minimal` | `Level0aMinimal` | 地面 + 机器人，秒开 |
| `level0b_simple.py` | `level0b_simple` | `Level0bSimple` | 桌子 + 机器人 + 方块 |
| `level0c_kitchen.py` | `level0c_kitchen` | `Level0cKitchen` | 厨房背景 |
| `level0d_hello_world.py` | `level0_hello_world` | `Level0HelloWorld` | 背景/机器人/物体全可换 |
| `level1_goal_pose.py` | `level1_goal_pose` | `Level1GoalPose` | 加任务：方块到目标 |
| `level2_rl_reach.py` | `level2_rl_reach` | `Level2RLReach` | 加奖励/观察，可 RL |
| `level3_custom_object.py` | `level3_custom_object` | `Level3CustomObject` | 自定义资产 + 关系放置 |
| `level4_pick_and_place.py` | `level4_pick_and_place` | `Level4PickAndPlace` | 顺序复合任务链 |
| `level5_sequential_rl.py` | `level5_sequential_rl` | `Level5SequentialRL` | 顺序复合 RL 环境 |
| `precache_assets.sh` | — | — | 资产预热脚本（headless 下载所有 USD，见「三、运行方式」） |

> **内置式说明**：原版文档里还有一个「内置式」做法——把脚本放进 `isaaclab_arena_environments/my_envs/`，靠 `@register_environment` + `__init__.py` 自动加载，运行时不带 `--external_environment_class_path`。本目录统一走外挂式（不改 fork 包）；若你要内置，把脚本拷进 `isaaclab_arena_environments/` 并在 `cli.py` 里 `import` 即可（原版踩坑见 `docs/IsaacLab-Arena-自定义环境实践.md`）。

---

## 三、运行方式

所有命令都在**容器内**执行（前置：已构建 `isaaclab_arena_magengyu:latest`，见[部署实战记录](../../../../docs/IsaacLab-Arena_magengyu-部署实战记录.md)）。

通用启动（交互式，需要显示器）：

```bash
cd references/IsaacLab-Arena_magengyu
./docker/run_docker.sh -n isaaclab_arena_magengyu
# 容器内：
/isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py \
  --policy_type zero_action --num_steps 200 --viz kit \
  --external_environment_class_path examples.examples_custom_env.<文件模块>:<类名> \
  <环境名>
```

> ⚠️ **资产下载**：用到 `kitchen` 背景 / YCB 物体 / GR1 等 USD 的级别，首次运行会从 AWS S3（us-west-2）+ Lightwheel 阻塞下载，国内网络可能很慢。`level0a/0b/1/2/4/5` 用 `table`/`ground_plane`/`dex_cube` 等本地/已缓存资产，启动快。

### 提前下载资产（precache，推荐首次先跑）

本目录自带 `precache_assets.sh`，一次性 headless 跑通 fork 全部 19 个内置环境（每个 5 步），把 USD 资产预热缓存到宿主机 `~/.cache`，之后 GUI 运行秒开、不再卡下载。

```bash
# 容器内（在仓库根目录）
./examples/examples_custom_env/precache_assets.sh
```

- 耗时约 20-40 分钟；资产缓存到 `~/.cache/ov/`（S3 USD）+ `~/.cache/lightwheel_sdk/`（Lightwheel）。
- **缓存与原版 `references/IsaacLab-Arena` 共享**：原版已下载的资产（当前 `~/.cache/ov/` 约 45GB）会直接复用，无需重复下载。
- 失败的环境会跳过并留日志 `/tmp/precache_<env>.log`，不影响其它环境。

---

## 四、Level 0 · 场景构建（4 个递进）

### 0a 最简：地面 + 机器人

```bash
/isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py \
  --policy_type zero_action --num_steps 200 --viz kit \
  --external_environment_class_path examples.examples_custom_env.level0a_minimal:Level0aMinimal \
  level0a_minimal
```

### 0b 简单：桌子 + 方块

```bash
/isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py \
  --policy_type zero_action --num_steps 2000 --viz kit \
  --external_environment_class_path examples.examples_custom_env.level0b_simple:Level0bSimple \
  level0b_simple
```

### 0c 厨房

```bash
/isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py \
  --policy_type zero_action --num_steps 200 --viz kit \
  --external_environment_class_path examples.examples_custom_env.level0c_kitchen:Level0cKitchen \
  level0c_kitchen
```

### 0d 全可配置（背景/机器人/物体都能换）

```bash
/isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py \
  --policy_type zero_action --num_steps 200 --viz kit \
  --external_environment_class_path examples.examples_custom_env.level0d_hello_world:Level0HelloWorld \
  level0_hello_world \
  --background packing_table --robot franka_ik --object mustard_bottle
```

**四个版本对比**：

| | 0a | 0b | 0c | 0d |
|---|---|---|---|---|
| 加载 | 秒开 | 几秒 | 首次慢 | 首次慢 |
| 场景 | 地面 | 桌子 | 厨房 | 厨房（可换） |
| 可配置字段 | robot | robot/object | background/robot/object | background/robot/object |

**学到**：`build()` 就是同一个模式 —— 「拿资产 → 设位置 → 组装 Scene → 返回环境」。复杂度只取决于你拿多少东西、从哪拿。

---

## 五、Level 1 · 加任务：方块到目标位置

在 Level 0 基础上加 `TaskBase` 子类，定义终止条件（方块到目标即成功）。核心是 `TerminationTermCfg(func=你的函数, params=参数)`，`time_out=True` 由框架处理。

```bash
/isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py \
  --policy_type zero_action --num_steps 500 --viz kit \
  --external_environment_class_path examples.examples_custom_env.level1_goal_pose:Level1GoalPose \
  level1_goal_pose

# 换目标位置（_legacy_argparse_cfg_type 字段用 --key value 格式）
/isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py \
  --policy_type zero_action --num_steps 500 --viz kit \
  --external_environment_class_path examples.examples_custom_env.level1_goal_pose:Level1GoalPose \
  level1_goal_pose --target_x 0.5 --target_y 0.2 --target_z 0.5
```

> Level 1 只有终止条件，**没有**奖励/观察，不能 RL 训练（会因 `get_rewards_cfg()` 返回 `None` 报错）。升级到 Level 2 才可训练。

---

## 六、Level 2 · RL 环境：加奖励 + 观察

在 Level 1 基础上，Task 额外实现 `get_rewards_cfg()` + `get_observation_cfg()`。关键点：

- RL 任务 = 非 RL 任务 + `get_rewards_cfg()` + `get_observation_cfg()`
- 奖励函数接收 `(env, **params)` → 返回 per-env 标量
- 环境类加两行 `rl_framework_entry_point` + `rl_policy_cfg`
- **RL 训练必须用关节位置控制 `franka_joint_pos`，不能用 IK**

**训练**：

```bash
/isaac-sim/python.sh \
  submodules/IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py \
  --external_callback isaaclab_arena.environments.isaaclab_interop.environment_registration_callback \
  --task level2_rl_reach \
  --num_envs 256 --max_iterations 500
```

**评估**：

```bash
/isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py \
  --policy_type rsl_rl \
  --checkpoint_path logs/rsl_rl/generic_experiment/<时间戳>/model_500.pt \
  --num_steps 500 --viz kit \
  --external_environment_class_path examples.examples_custom_env.level2_rl_reach:Level2RLReach \
  level2_rl_reach
```

---

## 七、Level 3 · 自定义物体 + 关系放置

`@register_asset` + `LibraryObject` 注册新资产；`On()` / `NextTo()` / `IsAnchor()` 关系让 Arena 自动算摆放位置。

```bash
/isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py \
  --policy_type zero_action --num_steps 500 --viz kit \
  --external_environment_class_path examples.examples_custom_env.level3_custom_object:Level3CustomObject \
  level3_custom_object
```

---

## 八、Level 4 · 顺序复合任务链：先抓后放

`SequentialTaskBase` 组合两个子任务（`LiftObjectTask` 举起 → `GoalPoseTask` 放置），自动维护状态机。

```bash
/isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py \
  --policy_type zero_action --num_steps 50 \
  --external_environment_class_path examples.examples_custom_env.level4_pick_and_place:Level4PickAndPlace \
  level4_pick_and_place
```

> Level 4 用的是 IL 版 `LiftObjectTask`（无 RL 奖励/观察），**不能直接 RL 训练**。要做 RL 顺序任务见 Level 5。

---

## 九、Level 5 · 顺序复合 RL 环境（可训练）

`Level5SequentialRLTask` 继承 `SequentialTaskBase`，覆写 `get_rewards_cfg`/`get_observation_cfg`/`get_commands_cfg`，**委托给 `LiftObjectTaskRL`**（内置完整 RL 栈）。

**训练**：

```bash
/isaac-sim/python.sh \
  submodules/IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py \
  --external_callback isaaclab_arena.environments.isaaclab_interop.environment_registration_callback \
  --task level5_sequential_rl \
  --num_envs 256 --max_iterations 1000
```

启动后日志里可见 5 个奖励项被激活：`reaching_object` / `lifting_object` / `object_goal_tracking` / `object_goal_tracking_fine_grained` 等。

**评估**：

```bash
/isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py \
  --policy_type rsl_rl \
  --checkpoint_path logs/rsl_rl/generic_experiment/<时间戳>/model_1000.pt \
  --num_steps 500 --viz kit \
  --external_environment_class_path examples.examples_custom_env.level5_sequential_rl:Level5SequentialRL \
  level5_sequential_rl
```

---

## 十、六个级别总结

| 级别 | 核心技能 | 可 RL 训练 |
|------|---------|-----------|
| 0a-0d | `build()` 模式 + `@dataclass` 可配置化 | ❌ |
| 1 | 自定义终止条件 | ❌ |
| 2 | 奖励函数 + 观察空间 | ✅ |
| 3 | 注册新资产 + 关系放置 | ❌ |
| 4 | 顺序复合任务链 | ❌ |
| 5 | 顺序复合 RL 环境 | ✅ |

每个级别都遵循同一个公式：

```
@dataclass Cfg        → 定义"改什么参数"
@register_environment → 注册环境名
build(cfg)            → 组装返回 IsaacLabArenaEnvironment
TaskBase 子类         → 定义判据（终止/奖励/观察）
```

从 Level 2 起就能接入 RL 训练，打通「自定义环境 → 训练模型 → 评估模型」完整闭环。

---

## 十一、fork 注意事项

1. **资产下载**：`kitchen` / YCB / GR1 等 USD 从 AWS S3（us-west-2）+ Lightwheel 下载，首次慢（见[示例运行指南](../../../../docs/IsaacLab-Arena_magengyu-示例运行指南.md)的资产下载警告）。本目录的 `table`/`ground_plane`/`dex_cube` 是已缓存资产。
2. **`variation_recorder is None` 警告**：RL 训练（`train.py`）时每局会刷这条，是无害的——RL 路径不挂场景变化记录器，只有评估（`policy_runner.py`）才记录。可 `2>&1 | grep -v "variation_recorder is None"` 静音。
3. **`_legacy_argparse_cfg_type`**：fork 仍需要（argparse 兼容，alpha 阶段），每个环境类都要写。
4. **RL 关节控制**：RL 训练必须用 `franka_joint_pos`（关节位置控制），不能用 `franka_ik`（IK），否则训练不稳定。
