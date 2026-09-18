# GR1 Open Microwave Door Task — 完整模仿学习工作流（fork 版）

> 官方教程 [`static_manipulation`](https://isaac-sim.github.io/IsaacLab-Arena/release/0.2.0/pages/example_workflows/static_manipulation/index.html) 适配 **`references/IsaacLab-Arena_magengyu`** 后的操作指南。
> 覆盖完整流水线：**环境搭建 → 遥操作采集 → Mimic 数据生成 → GR00T N1.6 微调 → 闭环评估**。
> 官方源文件在 fork 内：`docs/pages/example_workflows/static_manipulation/`（5 个 step 的 `.rst`）。

---

## 〇、任务概览

| 项 | 值 |
|----|----|
| **Task ID** | `gr1_open_microwave` |
| **任务** | GR1T2 人形用上半身（双臂+双手）够到微波炉并开门 |
| **Embodiment** | Fourier GR1T2（54 DOF） |
| **场景** | 厨房 + 微波炉（铰接物体） |
| **Policy** | GR00T N1.6（视觉-语言基础模型，3B） |
| **数据集** | [nvidia/Arena-GR1-Manipulation-Task](https://huggingface.co/datasets/nvidia/Arena-GR1-Manipulation-Task) |
| **Checkpoint** | [nvidia/GN1x-Tuned-Arena-GR1-Manipulation](https://huggingface.co/nvidia/GN1x-Tuned-Arena-GR1-Manipulation) |
| **指标** | Success rate（开门成功率）、Door moved rate |

---

## 前置条件

### 1. 启动 Docker 容器

```bash
cd references/IsaacLab-Arena_magengyu
./docker/run_docker.sh -n isaaclab_arena_magengyu
```

> 容器内 `python` 已 alias 到 `/isaac-sim/python.sh`；本指南统一用 `/isaac-sim/python.sh`，`docker exec` 场景也能用。

### 2. 登录 Hugging Face + 建数据/模型目录

```bash
# 容器内
hf auth login

export DATASET_DIR=/datasets/isaaclab_arena/static_manipulation_tutorial
mkdir -p $DATASET_DIR
export MODELS_DIR=/models/isaaclab_arena/static_manipulation_tutorial
mkdir -p $MODELS_DIR
```

> 宿主机的 `~/datasets`、`~/models` 会挂载到容器的 `/datasets`、`/models`（`run_docker.sh` 默认行为）。

### 3. 硬件约束（先看清再动手）

| 步骤 | 需要什么 | 4090D(24G) 是否可行 |
|------|---------|---------------------|
| Step 2 遥操作采集 | Meta Quest 头显 + CloudXR | ⚠️ 需头显 |
| Step 3 Mimic 数据生成 | CPU 即可（`--device cpu`） | ✅ 可行 |
| Step 4 GR00T 微调 | 单卡 24G（低配）或 8×48G（高配） | ⚠️ 低配单卡 24G 可尝试，2-3 小时 |
| Step 5 闭环评估 | 单卡（推理） | ✅ 可行 |

> 说明：GR00T N1.6 官方给了**低硬件需求**档（1×24G 单卡），4090D 理论上能跑微调；但 3B 模型微调仍较贵，建议先在服务器试。所有步骤都提供**跳过选项**（直接下载上一步产出的数据/权重）。

---

## Step 1 · 环境搭建与验证

**目标**：确认 `gr1_open_microwave` 环境能加载，并能回放预录数据。

### 1.1 下载测试数据集

```bash
hf download \
    nvidia/Arena-GR1-Manipulation-Task \
    arena_gr1_manipulation_dataset_generated.hdf5 \
    --repo-type dataset \
    --revision arena_v0.2_lab_v3.0 \
    --local-dir $DATASET_DIR
```

### 1.2 回放验证

```bash
/isaac-sim/python.sh submodules/IsaacLab/scripts/tools/replay_demos.py \
  --viz kit \
  --device cpu \
  --enable_cameras \
  --dataset_file "${DATASET_DIR}/arena_gr1_manipulation_dataset_generated.hdf5" \
  --external_callback isaaclab_arena.environments.isaaclab_interop.environment_registration_callback \
  --task gr1_open_microwave \
  --embodiment gr1_pink
```

预期：GR1 机器人在厨房里回放「开门」动作。

> ⚠️ 首次会下载 kitchen/GR1 的 USD 资产（AWS S3 us-west-2），慢；可先跑 `examples/examples_custom_env/precache_assets.sh` 预热（见[示例运行指南](../../../../docs/IsaacLab-Arena_magengyu-示例运行指南.md)）。

---

## Step 2 · 遥操作采集（Quest XR）

**目标**：用 Quest 头显遥控 GR1 开门，录制示范数据。

### 2.1 开防火墙

```bash
sudo ufw allow 49100/tcp   # Signaling
sudo ufw allow 47998/udp   # Media stream
sudo ufw allow 48322/tcp   # Proxy
```

### 2.2 开始录制

```bash
export CLOUDXR_ENV=cloudxrjs  # Apple Vision Pro 用 "avp"
/isaac-sim/python.sh submodules/IsaacLab/scripts/tools/record_demos.py \
  --device cpu \
  --viz kit \
  --xr \
  --cloudxr_env $CLOUDXR_ENV \
  --dataset_file $DATASET_DIR/arena_gr1_manipulation_dataset_recorded.hdf5 \
  --num_demos 10 \
  --num_success_steps 2 \
  --external_callback isaaclab_arena.environments.isaaclab_interop.environment_registration_callback \
  --task gr1_open_microwave \
  --arena_teleop_device openxr
```

> 若 Ctrl-C 退出，需清理 CloudXR 残留进程：`pkill -KILL -f '[i]saacteleop.cloudxr.runtime'`，否则下次会报 `XR_ERROR_INSTANCE_LOST`。

### 2.3 连 Quest 并录制

1. Quest 浏览器开 `https://nvidia.github.io/IsaacTeleop/client` → 填主机 IP → 接受证书 → Connect
2. Isaac Sim **XR** 标签页启动 Session → Quest 点 Play
3. 双手控制机械手、手指控制夹爪，完成开门任务；成功后环境自动重置，重复 `num_demos`(10) 次

> 首次使用需在 Quest 开 **Hand and Body Tracking**。录制建议：动作慢而稳、手保持追踪范围内、光照充足。

---

## Step 3 · Mimic 数据生成

**目标**：把少量示范「标注 + 增广」成更多训练数据。

### 3.1 标注示范（切成 reach → open door 两段）

```bash
/isaac-sim/python.sh submodules/IsaacLab/scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
  --viz kit \
  --device cpu \
  --input_file $DATASET_DIR/arena_gr1_manipulation_dataset_recorded.hdf5 \
  --output_file $DATASET_DIR/arena_gr1_manipulation_dataset_annotated.hdf5 \
  --mimic \
  --external_callback isaaclab_arena.environments.isaaclab_interop.environment_registration_callback \
  --task gr1_open_microwave
```

> 跳过：`hf download nvidia/Arena-GR1-Manipulation-Task arena_gr1_manipulation_dataset_annotated.hdf5 --repo-type dataset --revision arena_v0.2_lab_v3.0 --local-dir $DATASET_DIR`

### 3.2 增广数据集

```bash
/isaac-sim/python.sh submodules/IsaacLab/scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
  --device cpu \
  --generation_num_trials 50 \
  --num_envs 10 \
  --input_file $DATASET_DIR/arena_gr1_manipulation_dataset_annotated.hdf5 \
  --output_file $DATASET_DIR/arena_gr1_manipulation_dataset_generated.hdf5 \
  --enable_cameras \
  --headless \
  --mimic \
  --external_callback isaaclab_arena.environments.isaaclab_interop.environment_registration_callback \
  --task gr1_open_microwave
```

> 30-60 分钟；想看过程去掉 `--headless` 加 `--viz kit`。跳过：下载 `arena_gr1_manipulation_dataset_generated.hdf5`。

### 3.3（可选）回放增广数据验证

```bash
/isaac-sim/python.sh submodules/IsaacLab/scripts/tools/replay_demos.py \
  --viz kit --device cpu --enable_cameras \
  --dataset_file $DATASET_DIR/arena_gr1_manipulation_dataset_generated.hdf5 \
  --external_callback isaaclab_arena.environments.isaaclab_interop.environment_registration_callback \
  --task gr1_open_microwave --embodiment gr1_pink
```

---

## Step 4 · GR00T N1.6 微调

**目标**：用生成的 LeRobot 数据微调 GR00T N1.6。**在 Arena 容器外、`submodules/Isaac-GR00T` 的原生 `uv` 环境跑**。

### 4.1 转 LeRobot 格式（容器内）

```bash
/isaac-sim/python.sh isaaclab_arena_gr00t/lerobot/convert_hdf5_to_lerobot.py \
  --yaml_file isaaclab_arena_gr00t/lerobot/config/gr1_manip_config.yaml
```

产出 `$DATASET_DIR/arena_gr1_manipulation_dataset_generated/lerobot`（parquet + MP4 + metadata）。

### 4.2 微调（低硬件档：单卡 24G，适配 4090D）

在**容器外**，先按 [GR00T 安装指南](https://github.com/NVIDIA/Isaac-GR00T#installation-guide) 配好 `submodules/Isaac-GR00T` 的 `uv` 环境：

```bash
cd references/IsaacLab-Arena_magengyu/submodules/Isaac-GR00T
CUDA_VISIBLE_DEVICES=0 uv run python gr00t/experiment/launch_finetune.py \
  --dataset-path ~/datasets/isaaclab_arena/static_manipulation_tutorial/arena_gr1_manipulation_dataset_generated/lerobot \
  --output-dir ~/models/isaaclab_arena/static_manipulation_tutorial \
  --modality-config-path ../../isaaclab_arena_gr00t/embodiments/gr1/gr1_arms_only_data_config.py \
  --global-batch-size 16 \
  --max-steps 30000 \
  --num-gpus 1 \
  --save-steps 5000 \
  --base-model-path nvidia/GR00T-N1.6-3B \
  --no-tune-llm \
  --tune-visual \
  --tune-projector \
  --tune-diffusion-model \
  --dataloader-num-workers 4 \
  --embodiment-tag GR1 \
  --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
  --save-total-limit 5
```

> 高配档：`--nproc_per_node=8 --global-batch-size 24 --num-gpus 8`（8×48G，4-8 小时）。

---

## Step 5 · 闭环评估

**目标**：微调后的 GR00T 策略在 Arena 里闭环跑，测成功率。

### 5.1 下载预训练权重（跳过 Step 4 时）

```bash
hf download --revision gn1_6 nvidia/GN1x-Tuned-Arena-GR1-Manipulation \
  --local-dir $MODELS_DIR/checkpoint-20000
```

### 5.2 启动 GR00T 策略服务（容器外）

```bash
cd references/IsaacLab-Arena_magengyu/submodules/Isaac-GR00T
uv run python gr00t/eval/run_gr00t_server.py \
  --modality-config-path ../../isaaclab_arena_gr00t/embodiments/gr1/gr1_arms_only_data_config.py \
  --model-path /models/isaaclab_arena/static_manipulation_tutorial/checkpoint-20000 \
  --embodiment-tag GR1 \
  --device cuda --host 127.0.0.1 --port 5555
```

### 5.3 单环境评估（容器内）

```bash
/isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py \
  --viz kit \
  --policy_type isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy.Gr00tRemoteClosedloopPolicy \
  --policy_config_yaml_path isaaclab_arena_gr00t/policy/config/gr1_manip_gr00t_closedloop_config.yaml \
  --remote_host 127.0.0.1 \
  --remote_port 5555 \
  --num_steps 2000 \
  --enable_cameras \
  gr1_open_microwave \
  --embodiment gr1_joint
```

预期指标：`success_rate > 0.8`、`revolute_joint_moved_rate > 0.9`、`num_episodes` 10-20。

### 5.4 并行评估（多环境）

```bash
/isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py \
  --viz kit \
  --policy_type isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy.Gr00tRemoteClosedloopPolicy \
  --policy_config_yaml_path isaaclab_arena_gr00t/policy/config/gr1_manip_gr00t_closedloop_config.yaml \
  --remote_host 127.0.0.1 --remote_port 5555 \
  --num_steps 2000 --num_envs 10 --enable_cameras \
  gr1_open_microwave --embodiment gr1_joint
```

> ⚠️ 注意 embodiment 区别：数据生成/遥操作用 `gr1_pink`（PINK IK 末端控制）；闭环推理用 `gr1_joint`（关节位置控制，GR00T 训练的就是关节位）。

---

## fork 注意事项

1. **资产下载**：kitchen + GR1 的 USD 首次要下（S3 us-west-2），先跑 `examples/examples_custom_env/precache_assets.sh` 预热。
2. **GR00T 子模块**：`submodules/Isaac-GR00T` 需 `git submodule update --init`（Step 4/5 用）；微调/服务都在容器外的 `uv` 环境跑，**不是** Arena 容器。
3. **遥操作需 Quest**：本机无头显的话，Step 2 只能跳过（下载预录数据）。
4. **两条训练路线**：GR00T N1.6（本文档，24G 低配单卡）或 LeRobot/SmolVLA（见 [lerobot+smolVLA配置流程.md](../../lerobot+smolVLA配置流程.md)）。
5. 与 fork 里 `examples/train_pick_and_place/`（G1 苹果装盘，GR00T N1.7）是同构流程，可对照。
