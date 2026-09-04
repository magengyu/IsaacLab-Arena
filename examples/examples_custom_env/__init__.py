# examples_custom_env — 自定义 Isaac Lab-Arena 环境示例（fork 版）

# 用法：通过 --external_environment_class_path 加载，例如：
#   /isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py \
#     --policy_type zero_action --num_steps 200 --viz kit \
#     --external_environment_class_path examples.examples_custom_env.level0a_minimal:Level0aMinimal \
#     level0a_minimal
