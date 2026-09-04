#!/bin/bash
# ═══════════════════════════════════════════════════════
# Isaac Lab-Arena (magengyu fork) 资源预热脚本
# 一次性 headless 跑所有环境，把 USD 资产缓存到本地
# 之后 GUI 运行就秒开了
#
# 用法: ./examples/examples_custom_env/precache_assets.sh
# 耗时: 约 20-40 分钟（取决于 AWS S3 / Lightwheel 速度）
#
# 说明: 资产缓存到宿主机 ~/.cache（ov/ + lightwheel_sdk/），
#       与 references/IsaacLab-Arena 共享同一份缓存，无需重复下载。
# ═══════════════════════════════════════════════════════
set -e

ENVS=(
  # 机械臂 (Franka)
  cube_goal_pose
  lift_object
  kitchen_pick_and_place
  press_button
  tabletop_sort_cubes
  gear_mesh
  peg_insert
  tabletop_place_upright
  pick_and_place_maple_table
  franka_put_and_close_door
  pick_and_place_airpod
  # 人形 (GR1 / G1)
  gr1_open_microwave
  gr1_turn_stand_mixer_knob
  gr1_table_multi_object_no_collision
  put_item_in_fridge_and_close_door
  galileo_pick_and_place
  galileo_g1_locomanip_pick_and_place
  droid_table_multi_object_placement
  # 灵巧手
  dexsuite_lift
)

TOTAL=${#ENVS[@]}
SUCCESS=0
FAIL=0

echo "═════════════════════════════════════════════════"
echo "  预热 Isaac Lab-Arena 所有环境 ($TOTAL 个)"
echo "  每个跑 5 步，仅下载资源，不做训练"
echo "═════════════════════════════════════════════════"
echo ""

for i in "${!ENVS[@]}"; do
  env="${ENVS[$i]}"
  idx=$((i + 1))
  echo "[$idx/$TOTAL] 预热: $env ..."

  if /isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py \
      --policy_type zero_action --num_steps 5 "$env" \
      > /tmp/precache_${env}.log 2>&1; then
    echo "  ✅ 成功"
    SUCCESS=$((SUCCESS + 1))
  else
    echo "  ⚠️  跳过（可能缺依赖或资产下载失败，日志见 /tmp/precache_${env}.log）"
    FAIL=$((FAIL + 1))
  fi
done

echo ""
echo "═════════════════════════════════════════════════"
echo "  完成！成功: $SUCCESS / 失败: $FAIL / 总计: $TOTAL"
echo "  所有资源已缓存到 ~/.cache，之后 GUI 运行秒开"
echo "═════════════════════════════════════════════════"
