# RL Framework NPU 集群启动手册

本文档面向第一次在当前 Huawei Ascend 集群上运行项目的合作者。目标是从登录跳板机开始，完成环境检查、C-MLFQ、Global Resource Planner（GRP，以及 Elastic Hybrid Pool 对照实验。

文档根据项目代码、现有启动脚本和服务器上的已验证运行目录整理，最后核对时间为 2026-08-18。

## 0. 前提

1. 当前集群通过跳板机上的 SSH、远端 `torchrun` 和 Ascend HCCL 启动；每个节点有 8 张 Ascend 910B3 NPU，每张 64 GiB HBM。
2. 推荐先跑 CPU 控制面验证和 GRP `PREFLIGHT_ONLY`，确认配置/节点/设备映射正确后，再启动真实训练。真实任务会占用节点，不要在不确认空闲的情况下运行。

## 1. 集群与目录信息

### 1.1 登录入口

- 公网入口/跳板机：`119.13.124.91`
- 跳板机主机名：通常为 `aura-5.novalocal`
- 跳板机内网地址：`192.168.0.85`
- 用户：`root`
- 内网节点：以下 20 个地址来自项目的节点探测和启动脚本；`192.168.0.85` 同时是当前跳板机节点。

```text
192.168.0.2    192.168.0.17   192.168.0.24   192.168.0.36   192.168.0.41
192.168.0.47   192.168.0.48   192.168.0.50   192.168.0.51   192.168.0.53
192.168.0.63   192.168.0.85   192.168.0.88   192.168.0.89   192.168.0.99
192.168.0.112  192.168.0.155  192.168.0.189  192.168.0.195  192.168.0.252
```

### 1.2 登录方式

在本地电脑上先登录跳板机：

```bash
ssh -o StrictHostKeyChecking=no root@119.13.124.91
```

登录跳板机后，再连接某个内网节点：

```bash
ssh -o StrictHostKeyChecking=no root@192.168.0.2
```

也可以在本地使用 ProxyJump：

```bash
ssh -J root@119.13.124.91 root@192.168.0.2
```

如果 ProxyJump 在本地网络策略下不可用，就采用“两段式”登录：先进入 `119.13.124.91`，再从跳板机执行第二条 `ssh`。不要把密码写进命令、脚本、Git、shell 历史或文档。

### 1.3 已验证软件基线

```text
OS: Huawei Cloud EulerOS 2.0 (aarch64)
Kernel: 5.10.0-136.12.0.86.r1526_92.hce2.aarch64
NPU: Ascend 910B3, 8 devices/node, 64 GiB/device
npu-smi / driver: 25.5.1
CANN toolkit: 8.5.2
Python training: 3.10.20
Python rollout: 3.11.13
torch training: 2.7.1 (+cpu runtime string)
torch-npu training: 2.7.1.post2
Megatron-Core: 0.14.0
Megatron Bridge: 0.2.0rc6
vLLM: 0.11.0
vLLM-Ascend: 0.11.0
```

CANN 环境必须先加载：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

已验证的训练和 rollout 环境是分开的，不要交叉替换：

```text
训练/Megatron-Core: /data/qianzhirong/envs/rl_mindspeed_260/bin/python
基础 Python 包目录:  /data/qianzhirong/envs/rl_framework_py310
rollout/vLLM:       /root/vllm_ascend_env/bin/python
```

关键公共目录：

```text
项目工作树:         /root/RL_Framework_npu
已验证运行源码:     /data/qianzhirong/runtime_sources/RL_Framework_npu_ehp_ab_8ec69d6
GRP Python 包根:     /data/qianzhirong/runtime_sources/grp_unrestricted_20260812
模型目录:           /data/qianzhirong/models
Qwen3-0.6B:         /data/qianzhirong/models/Qwen3-0.6B
Qwen3-14B:          /data/qianzhirong/models/Qwen3-14B
运行结果根目录:     /data/qianzhirong/runs
环境快照:           /data/qianzhirong/environment_snapshots/ascend_megatron_validated_20260805
```

环境快照是无密码、无 token 的基线记录，可用来核对依赖：

```bash
cd /data/qianzhirong/environment_snapshots/ascend_megatron_validated_20260805
grep -E '^(torch|torch_npu|megatron|transformers|tokenizers|datasets|accelerate|vllm)' \
  training.pip-freeze.txt rollout.pip-freeze.txt
```

## 2. 第一次登录后的环境检查

在跳板机执行：

```bash
set -e
source /usr/local/Ascend/ascend-toolkit/set_env.sh
echo "CANN_ENV=$ASCEND_HOME_PATH"
hostname
uname -m
npu-smi info | sed -n '1,80p'
/data/qianzhirong/envs/rl_mindspeed_260/bin/python -c \
  'import torch, torch_npu; print(torch.__version__); print(torch_npu.__version__); print(torch.npu.device_count())'
/root/vllm_ascend_env/bin/python -c \
  'import torch, torch_npu, vllm; print(torch.__version__); print(torch_npu.__version__); print(vllm.__version__)'
```

预期结果：`npu_count=8`，NPU 名称为 `910B3`，每张卡没有正在运行的进程。若出现 `libhccl.so: cannot open shared object file`，说明还没有执行 `source /usr/local/Ascend/ascend-toolkit/set_env.sh`，不要重新安装 torch。

检查一个内网节点：

```bash
ssh root@192.168.0.2 'source /usr/local/Ascend/ascend-toolkit/set_env.sh; \
  hostname; uname -m; npu-smi info | grep -E "910B3|No running processes" | tail -n 8'
```

批量操作使用项目提供的 `scripts/internal_ssh.sh`。它要求在跳板机当前 shell 中存在 `NODE_PASSWORD`，但不会把密码写进脚本：

```bash
cd /data/qianzhirong/runtime_sources/RL_Framework_npu_ehp_ab_8ec69d6
read -rsp 'Internal SSH password: ' NODE_PASSWORD; echo
export NODE_PASSWORD
export INTERNAL_SSH_TIMEOUT=20
./scripts/internal_ssh.sh 192.168.0.2 192.168.0.17 192.168.0.24 -- \
  'hostname; npu-smi info | grep -c -F "No running processes found in NPU"'
unset NODE_PASSWORD
```

返回值为 `8` 表示该节点的 8 张 NPU 都没有进程。正式运行前，必须对本次使用的所有节点都做这个检查。不要使用未经确认的 `pkill -f python` 或 `pkill -f vllm`，因为节点可能被其他实验占用。

## 3. 先做 CPU 控制面验证

这一步不启动模型、不占用 NPU，用于验证 C-MLFQ、GRP 和 Elastic Hybrid Pool 的控制逻辑。建议在跳板机执行：

```bash
cd /root/RL_Framework_npu
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH=/data/qianzhirong/runtime_sources/grp_unrestricted_20260812
export WANDB_MODE=disabled

/data/qianzhirong/envs/rl_framework_py310/bin/python \
  examples/test_r2e_cmlfq_flow.py

/data/qianzhirong/envs/rl_framework_py310/bin/python \
  examples/global_resource_planner_full_flow.py

CORE_BENCH_REQUESTS=100 \
EHP_BENCH_ITERATIONS=10 \
CORE_COMPONENT_REPORT=/tmp/r2e_core_component_report.json \
/data/qianzhirong/envs/rl_framework_py310/bin/python \
  examples/validate_r2e_core_components.py
```

应看到以下类型的成功标记：

```text
R2E_CMLFQ_FLOW_OK
FLOW_OK
R2E_CORE_COMPONENTS_OK output=/tmp/r2e_core_component_report.json
```

若只想检查单元测试：

```bash
cd /root/RL_Framework_npu
export PYTHONPATH=/data/qianzhirong/runtime_sources/grp_unrestricted_20260812
/data/qianzhirong/envs/rl_framework_py310/bin/python -m pytest -q \
  tests/test_cmlfq_scheduler.py \
  tests/test_preflight_planner.py \
  tests/test_global_resource_planner_simulators.py \
  tests/test_hetero_cmlfq_integration.py
```

## 4. C-MLFQ：配置、启动和检查

### 4.1 C-MLFQ/no-C-MLFQ 对照启动

两组实验共用同一套 5 节点/40 NPU 模型、数据、训练拓扑和 GRP 设置，唯一
变化是 rollout scheduler：C-MLFQ 使用 `scheduler_type: cmlfq`，控制组使用
`load_balance + round_robin`。在跳板机确认节点空闲后分别执行：

```bash
cd /root/RL_Framework_npu
read -rsp 'Internal SSH password: ' NODE_PASSWORD; echo
export NODE_PASSWORD INTERNAL_HOSTS='192.168.0.2 192.168.0.24 192.168.0.63 192.168.0.99'
export RUN_ROOT=/data/qianzhirong/runs/r2e_gym_qwen3_14b_5node40_cmlfq_ab
MODE=cmlfq ARM=with bash scripts/run_npu_ablation.sh
MODE=cmlfq ARM=without bash scripts/run_npu_ablation.sh
unset NODE_PASSWORD
```

### 4.1 C-MLFQ 做什么

C-MLFQ 不是一个独立的训练进程，而是 rollout 请求路由器：

1. 新请求根据输入长度、历史前缀树和当前负载，先分配到 short/medium/long bucket。
2. 工具调用返回后，调度器读取工具类型、成功/失败状态、payload 大小和剩余长度。
3. 如果新的因果状态表明更适合另一个 TP bucket，就产生 migration decision 并迁移请求。
4. 请求结束后把 trajectory 写回前缀树；达到 `cmlfq_rebuild_interval` 后重建或合并统计。
5. 多进程运行时，各进程通过共享负载目录交换 heartbeat 和队列状态。

项目中主要实现位于：

```text
infra/scheduling/cmlfq_scheduler.py
infra/scheduling/cmlfq_prefix_tree.py
infra/scheduling/cmlfq_tool_state.py
infra/scheduling/cmlfq_shared_state.py
infra/scheduling/cmlfq_offline_profile.py
infra/scheduling/cmlfq_migration.py
```

### 4.2 关键 YAML 开关

在配置中确认以下字段：

```yaml
heterogeneous_rollout:
  enabled: true
  scheduling:
    scheduler_type: cmlfq
    cmlfq_rebuild_interval: 5
    cmlfq_tree_persist_interval: 1
    cmlfq_tree_path: /data/qianzhirong/runs/<run>/cmlfq_tree.json
    cmlfq_shared_load_dir: /data/qianzhirong/runs/<run>/cmlfq_shared_load
    cmlfq_shared_load_ttl_s: 60
    cmlfq_shared_load_heartbeat_s: 10
    cmlfq_payload_small_threshold: 1000
    cmlfq_payload_large_threshold: 12000
    cmlfq_buckets:
      short:
        tp_degrees: [1]
        max_tokens: 4000
      medium:
        tp_degrees: [2]
        max_tokens: 12000
      long:
        tp_degrees: [4]
        max_tokens: 30000
```

当前项目中可直接参考：

```text
configs/r2e_gym_qwen3_14b_mcore_npu_5node40_100step.yaml
configs/r2e_gym_qwen3_14b_mcore_npu_6node48_grp_ab_equal_common.yaml
configs/r2e_gym_cmlfq_grp_qwen3_14b_npu_capability_curve.yaml
configs/r2e_gym_cmlfq_qwen3_14b_npu_pilot.yaml
```

### 4.3 启动 rollout 服务

C-MLFQ 只负责路由，必须先有 OpenAI-compatible vLLM endpoint。单个服务的启动脚本是 `scripts/run_vllm_ascend_server.sh`，使用 rollout 专用环境：

```bash
cd /root/RL_Framework_npu
source /usr/local/Ascend/ascend-toolkit/set_env.sh
VENV_DIR=/root/vllm_ascend_env \
ASCEND_DEVICES=0 \
PORT=8000 \
TP_SIZE=1 \
MAX_MODEL_LEN=4096 \
GPU_MEMORY_UTILIZATION=0.88 \
bash scripts/run_vllm_ascend_server.sh \
  /data/qianzhirong/models/Qwen3-0.6B
```

另开终端确认服务：

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/models
```

多实例、TP-1/2/4 的服务由 `scripts/start_r2e_rollout_manifest_npu.sh` 根据 `device_placement.json` 管理。不要手工复用同一张 NPU，也不要把训练环境的 Python 用来启动 vLLM。

### 4.4 如何确认 C-MLFQ 真正在工作

运行后重点看：

```bash
RUN_DIR=/data/qianzhirong/runs/<run>
find "$RUN_DIR" -maxdepth 3 -type f | sort | grep -E 'cmlfq|history|rollout|manifest'
grep -R -E '\[CMLFQ\]|C-MLFQ|migration|prefix-tree' "$RUN_DIR" | tail -n 100
```

应能找到：

- `cmlfq_tree*.json`：前缀树快照，可能按 rank 分片；
- `cmlfq_shared_load/`：跨进程负载 heartbeat；
- rollout manifest/placement：实例、主机、设备、TP、端口；
- 日志中的初始路由、tool-return 后的迁移和请求完成事件。

如果树文件不生成，优先检查 `cmlfq_tree_path` 的父目录权限和所有 rank 是否能访问同一个共享路径；如果一直只有 fallback 路由，检查是否实际产生了 tool return，以及 `cmlfq_buckets` 的 TP 资源是否和 rollout placement 一致。

## 5. EHP：弹性混合池

### 5.1 EHP 解决什么问题

Elastic Hybrid Pool（EHP）允许一个 rollout worker 在运行过程中暂时加入 training pool，训练压力下降后再回到 rollout pool。它不是额外增加 NPU，而是在 rollout 和 training 之间复用同一批物理资源：

```text
正常状态：     core training + core rollout
训练压力升高：  rollout worker -> hybrid joining -> hybrid training
训练压力下降：  hybrid training -> hybrid rollout -> rollout service
```

EHP 的关键约束是：核心 Megatron-Core 的 TP/PP/DP process group 保持稳定，借入的 worker 不直接改变核心训练拓扑，而是在 inter-replica gradient domain 中贡献梯度，避免在线修改 Megatron distributed optimizer 和核心 collective group。

项目实现位于：

```text
infra/elastic/hybrid_pool.py       # worker 状态机、join/release、梯度聚合
infra/elastic/runtime_executor.py # GRP 计划到运行时动作的执行器
scripts/elastic_hybrid_worker.py  # 外部 hybrid worker 进程
```

### 5.2 Worker 生命周期

| 状态 | 含义 |
| --- | --- |
| `core_training` | 固定的核心训练 worker，不被 EHP 借走 |
| `core_rollout` | 固定的 rollout worker |
| `hybrid_rollout` | 可被 planner 借给训练池的 rollout worker |
| `hybrid_joining` | 正在获取训练快照、进行 zero-gradient 对齐 |
| `hybrid_training` | 已经加入训练，并向目标 core replica 贡献梯度 |

一次 join 的顺序是：

1. GRP 根据队列压力和成本模型决定 `target_train_gpus`。
2. `RuntimeElasticExecutor` 选择一个 `core_rollout`/`hybrid_rollout` worker。
3. `ElasticHybridPool.join_training()` 异步启动加入流程。
4. worker 获取目标 core replica 的 snapshot/state version。
5. 在 `zero_sync_steps` 内发送 zero placeholder，保证梯度结构稳定。
6. worker 进入 `hybrid_training`，梯度被路由到 `target_core_id` 并聚合。
7. 训练压力降低时调用 `release_to_rollout()`，解除梯度绑定并恢复 rollout 角色。

### 5.3 EHP 与 cluster-swap

没有 spare NPU 时，`cluster_swap` 会先停止或迁移不再需要的 rollout instance，再把释放出的物理 NPU 分配给 hybrid training worker；计划反向变化时，再释放 training worker 并恢复 rollout instance。

```yaml
global_resource_planner:
  runtime_dynamic_reconfiguration_enabled: true
  runtime_online_replanning: true
  runtime_manage_rollout_processes: true
  runtime_reconfigure_training: true
  runtime_training_pool_plan_only: false
  runtime_cluster_swap_enabled: true
  runtime_rollout_reconfigure_strategy: cluster_swap
  runtime_training_pool_only: true
```

重要边界：

- `runtime_training_pool_plan_only: true` 只记录借入/释放计划，不会真的接入 worker。
- `runtime_reconfigure_training: false` 时，EHP 不会改变 training pool。
- `hybrid_worker_launch_enabled: false` 时，不会自动启动外部 hybrid worker。
- `runtime_manage_rollout_processes: false` 时，Libra 不会替你管理 vLLM 进程。

EHP 梯度通信有两种模式：

- `decouple_communication_domains=true`：elastic traffic 使用独立 hybrid process group，隔离核心训练 group。
- `decouple_communication_domains=false`：复用已经初始化的训练通信组，适合当前某些 NPU smoke，但没有通信域隔离。

当前 2 节点 EHP 配置使用 `decouple_communication_domains: false`，因为该 NPU launcher 没有初始化独立的 elastic CCL process group。不要只改成 `true`，必须先确认独立通信组和 peer barrier 已实现。

### 5.4 EHP 关键配置

以当前 2 节点配置为参考：

```yaml
global_resource_planner:
  elastic_hybrid_planning_enabled: true
  elastic_hybrid_borrow_train_rollout_ratio: 1.15
  elastic_hybrid_release_train_rollout_ratio: 0.90
  elastic_hybrid_max_rollout_pressure: 0.80
  elastic_hybrid_join_timeout_s: 180
  elastic_hybrid_signal_ttl_steps: 20
  hybrid_worker_launch_enabled: true
  hybrid_worker_mode: megatron_core
  hybrid_worker_python: /data/qianzhirong/envs/rl_mindspeed_260/bin/python
  hybrid_worker_task_dir: /data/qianzhirong/runs/<run>/elastic_training_tasks
  hybrid_worker_remote_control_enabled: true
  gradient_transport_backend: tcp
  gradient_server_host: 0.0.0.0
  gradient_server_public_host: 192.168.0.36
  gradient_server_port: 29852
```

借入/释放阈值不是“每一步都切换”；实际还会受到 `min_gain_ratio`、`reconfiguration_cost_s`、`runtime_replan_cooldown_steps` 和 drain 状态影响。

### 5.5 验证 EHP

先做不占用 NPU 的控制面验证：

```bash
cd /root/RL_Framework_npu
export PYTHONPATH=/data/qianzhirong/runtime_sources/grp_unrestricted_20260812
EHP_BENCH_ITERATIONS=100 \
EHP_TENSOR_ELEMENTS=262144 \
CORE_COMPONENT_REPORT=/data/qianzhirong/runs/ehp_control_plane_report.json \
/data/qianzhirong/envs/rl_framework_py310/bin/python \
  examples/validate_r2e_core_components.py
```

输出中的 `elastic_hybrid_pool` 应包含 join/release latency、gradient aggregation 和 `communication_domains`。这验证的是 EHP 状态机和本地梯度聚合，不代表多节点 external hybrid worker 已经连通。

相关测试：

```bash
cd /root/RL_Framework_npu
export PYTHONPATH=/data/qianzhirong/runtime_sources/grp_unrestricted_20260812
/data/qianzhirong/envs/rl_framework_py310/bin/python -m pytest -q \
  tests/test_elastic_hybrid_pool.py \
  tests/test_runtime_elastic_executor.py \
  tests/test_elastic_cross_process_worker.py \
  tests/test_hetero_cmlfq_integration.py
```

真实 EHP 2 节点 smoke 使用：

```text
configs/r2e_gym_qwen3_14b_mcore_npu_2node16_ehp_docker.yaml
```

该配置依赖 `vllm_launch_command_template`、`vllm_stop_command_template` 指向的服务器专用 runtime manager 和 Docker 容器。启动前检查：

```bash
test -x /data/qianzhirong/runtime_sources/RL_Framework_npu_4node32_ehp_ab_300step_20260812/scripts/runtime_manage_r2e_rollout_npu.sh
test -x /data/qianzhirong/runtime_sources/RL_Framework_npu_4node32_ehp_ab_300step_20260812/scripts/ehp_docker_exec.sh
docker ps --format '{{.Names}}' | grep -F ehp-test-chenkaiwen
```

如果依赖不完整，不要直接运行该 YAML；先使用普通 manifest/GRP 路径，或先补齐 runtime manager。

### 5.6 判断 EHP 是否实际生效

不能只看 `GRP candidate`，还要检查：

```bash
RUN_DIR=/data/qianzhirong/runs/<run>
grep -R -E 'join_training|release_to_rollout|hybrid_training|elastic_gradient_domain|cluster_swap_(begin|complete)' \
  "$RUN_DIR" --include='*.log' --include='*.json' | tail -n 200
find "$RUN_DIR" -type f | grep -E 'elastic_training_tasks|runtime_reconfiguration|gradient|cluster_swap'
```

有效证据包括 `join_training:<worker>-><core>`、`release_to_rollout:<worker>`、`elastic_gradient_domain`、`cluster_swap_complete`、hybrid worker ready marker，以及对应的 rollout stop/start 和 ACK 文件。只有 `plan_join_training` 而没有 `join_training`，表示仍是 plan-only。

当前 6 节点 GRP/no-GRP 对照配置默认关闭：

```yaml
global_resource_planner:
  runtime_reconfigure_training: false
  elastic_hybrid_planning_enabled: false
  hybrid_worker_launch_enabled: false
```

因此 6 节点脚本适合比较 GRP 资源规划和 rollout placement，不适合证明 EHP 已加入训练。要做 EHP 对照，应使用 2 节点 EHP/no-EHP 配置，或创建启用上述字段的实验配置。

## 6. GRP：先预检，再启动真实对照实验

### 6.1 GRP 做什么

Global Resource Planner 根据历史 batch 的输入/输出长度、训练和 rollout 成本模型、当前队列压力以及 reconfiguration cost，搜索训练/rollout 的资源分配。它会同时考虑：

- training TP/PP/DP/micro-batch；
- rollout TP bucket 列表，例如 `[1, 1, 2, 4]` 或 `[2, 2, 2, 2]`；
- 总 NPU 预算和设备级 placement；
- 是否启用 online replanning、cluster swap、EHP；
- 预估收益是否足以覆盖重配置成本。

主要实现位于：

```text
infra/cost_model/global_resource_planner.py
infra/cost_model/preflight_planner.py
infra/elastic/runtime_executor.py
scripts/plan_unrestricted_grp_device_placement.py
```

### 6.2 先做 6 节点 GRP 预检

`run_6node48_grp_vs_no_grp_equal.sh` 会使用以下 6 个节点，总计 48 个 NPU：

```text
192.168.0.112 192.168.0.252 192.168.0.189
192.168.0.47  192.168.0.88  192.168.0.89
```

在跳板机执行：

```bash
cd /data/qianzhirong/runtime_sources/RL_Framework_npu_ehp_ab_8ec69d6
read -rsp 'Internal SSH password: ' NODE_PASSWORD; echo
export NODE_PASSWORD
export PROJECT_DIR=/data/qianzhirong/runtime_sources/RL_Framework_npu_ehp_ab_8ec69d6
export RUNTIME_PROJECT_DIR=/data/qianzhirong/runtime_sources/RL_Framework_npu_ehp_ab_8ec69d6
export CONFIG_PYTHON=/data/qianzhirong/envs/rl_framework_py310/bin/python
export TRAIN_PYTHON=/data/qianzhirong/envs/rl_mindspeed_260/bin/python
export MODEL_PATH=/data/qianzhirong/models/Qwen3-14B
export AVAILABLE_HOSTS='192.168.0.112 192.168.0.252 192.168.0.189 192.168.0.47 192.168.0.88 192.168.0.89'
export RUN_ROOT=/data/qianzhirong/runs/r2e_6node48_grp_vs_no_grp_equal_20260812
export PREFLIGHT_ONLY=1
bash scripts/run_6node48_grp_vs_no_grp_equal.sh
unset NODE_PASSWORD
```

`PREFLIGHT_ONLY=1` 只做空闲检查、GRP/固定 placement 计算和配置验证，不启动 vLLM 或训练。检查输出目录：

```bash
RUN_ROOT=/data/qianzhirong/runs/r2e_6node48_grp_vs_no_grp_equal_20260812
find "$RUN_ROOT" -maxdepth 3 -type f | sort | tail -n 100
sed -n '1,220p' "$RUN_ROOT"/formal_6node48_grp_unrestricted_*/effective_config.yaml
cat "$RUN_ROOT"/formal_6node48_grp_unrestricted_*/device_placement.json
```

重点确认：

- `train_devices` 和 `rollout_instances` 都非空；
- 每个 rollout instance 的 `tp` 与 `gpus` 数量相等；
- 所有设备编号在同一 host 内不重复；
- `train_gpus + rollout_gpus = 48`；
- `master_addr` 是训练设备的第一个 host；
- `master_port` 没有被其他任务占用；
- C-MLFQ 的 TP bucket 能在 placement 中找到对应 rollout 实例。

GRP 预检产生的典型文件包括：

```text
effective_config.yaml
device_placement.json
grp_initial_placement.json
selected_hosts.txt
```

### 6.3 启动 GRP 与 no-GRP 对照实验

预检无误且 6 个节点确认空闲后，去掉 `PREFLIGHT_ONLY`：

```bash
cd /data/qianzhirong/runtime_sources/RL_Framework_npu_ehp_ab_8ec69d6
read -rsp 'Internal SSH password: ' NODE_PASSWORD; echo
export NODE_PASSWORD
export PROJECT_DIR=/data/qianzhirong/runtime_sources/RL_Framework_npu_ehp_ab_8ec69d6
export RUNTIME_PROJECT_DIR=/data/qianzhirong/runtime_sources/RL_Framework_npu_ehp_ab_8ec69d6
export CONFIG_PYTHON=/data/qianzhirong/envs/rl_framework_py310/bin/python
export TRAIN_PYTHON=/data/qianzhirong/envs/rl_mindspeed_260/bin/python
export MODEL_PATH=/data/qianzhirong/models/Qwen3-14B
export AVAILABLE_HOSTS='192.168.0.112 192.168.0.252 192.168.0.189 192.168.0.47 192.168.0.88 192.168.0.89'
export RUN_ROOT=/data/qianzhirong/runs/r2e_6node48_grp_vs_no_grp_equal_20260812
export MAX_MODEL_LEN=4096
export RUN_ONLY=both
bash scripts/run_6node48_grp_vs_no_grp_equal.sh
unset NODE_PASSWORD
```

该脚本会串行运行两个 arm：

```text
GRP:     configs/r2e_gym_qwen3_14b_mcore_npu_6node48_grp_ab_equal_grp.yaml
control: configs/r2e_gym_qwen3_14b_mcore_npu_6node48_grp_ab_equal_no_grp.yaml
```

也可以只启动单个 arm：

```bash
MODE=grp ARM=with bash scripts/run_npu_ablation.sh
MODE=grp ARM=without bash scripts/run_npu_ablation.sh
```

脚本会在每个 arm 前检查 6 个节点是否 8 卡全空闲，调用 `plan_unrestricted_grp_device_placement.py` 生成设备级 placement，启动 rollout manifest，然后用 SSH 在各训练设备上启动一个单进程/单 NPU 的 `torchrun` rank。它会自动在退出时停止本 arm 自己登记的 rollout 进程。

默认关键端口：

```text
GRP training master: 31240
fixed training master: 31260
HCCL base port: 52000
rollout HTTP ports: each host starts at 8000 and increments locally
```

若端口冲突，换一组未占用的 `MASTER_PORT`/`HCCL_IF_BASE_PORT`；同一实验的所有 rank 必须使用一致的 HCCL base port。

### 6.4 观察 GRP 是否真的应用了计划

运行期间另开一个跳板机终端：

```bash
RUN_ROOT=/data/qianzhirong/runs/r2e_6node48_grp_vs_no_grp_equal_20260812
find "$RUN_ROOT" -maxdepth 3 -type f | sort | tail -n 120
grep -R -E 'GRP|planner|candidate|reconfig|cluster_swap|plan' "$RUN_ROOT" \
  --include='*.log' --include='*.json' --include='*.yaml' | tail -n 150
```

完成后，脚本会打印类似：

```text
STATUS=complete GRP_RUN=<name> NO_GRP_RUN=<name> REPORT=<path>
```

报告通常位于：

```text
/data/qianzhirong/runs/r2e_6node48_grp_vs_no_grp_equal_20260812/grp_vs_no_grp_<timestamp>.md
```

也可以直接重新分析已有结果：

```bash
PYTHONPATH=/data/qianzhirong/runtime_sources/grp_unrestricted_20260812 \
/data/qianzhirong/envs/rl_framework_py310/bin/python \
  scripts/analyze_libra_experiment.py \
  --run 'no-GRP-equal=/data/qianzhirong/runs/r2e_6node48_grp_vs_no_grp_equal_20260812/<fixed_run>' \
  --run 'GRP=/data/qianzhirong/runs/r2e_6node48_grp_vs_no_grp_equal_20260812/<grp_run>' \
  --output /data/qianzhirong/runs/r2e_6node48_grp_vs_no_grp_equal_20260812/reanalysis.md
```

报告里至少检查：训练/rollout 实际设备数、候选计划、全局预测时间、实际 step 时间、rollout 吞吐、C-MLFQ 路由/迁移统计、重配置事件和错误日志。只看到 `candidate plan` 不代表计划已经应用；必须同时看到 placement/effective config 和 runtime reconfiguration 记录。

## 7. 2 节点 EHP/GRP smoke 的参考配置

### 7.1 EHP/no-EHP 对照启动

EHP 对照必须使用 2 节点/16 NPU 配置；EHP 组允许 runtime planner 管理
rollout/training 角色，no-EHP 组关闭 online planning、hybrid worker 和运行时
重配置，二者共享启动计划和工作负载：

```bash
cd /root/RL_Framework_npu
read -rsp 'Internal SSH password: ' NODE_PASSWORD; echo
export NODE_PASSWORD INTERNAL_HOSTS='192.168.0.36 192.168.0.41'
export NPROC_PER_NODE=8 SKIP_ROLLOUT_HEALTH=1
MODE=ehp ARM=with bash scripts/run_npu_ablation.sh
MODE=ehp ARM=without bash scripts/run_npu_ablation.sh
unset NODE_PASSWORD
```

目标环境必须已经提供配置中 `vllm_launch_command_template` 和
`vllm_stop_command_template` 引用的 runtime manager/Docker 脚本；缺少这些依赖
时先做配置预检，不要启动真实 EHP 任务。

如果 6 节点正式对照太重，可以先参考当前服务器上已经跑过的 2 节点 16 NPU 配置：

```text
configs/r2e_gym_qwen3_14b_mcore_npu_2node16_ehp_docker.yaml
```

该配置的关键设定是：

```text
train_gpus=8, rollout_gpus=8
train_tp_size=4, train_dp_size=2
rollout: 4 个 TP=2 实例
C-MLFQ scheduler_type=cmlfq
GRP enabled=true, initial_allocation_strategy=grp
runtime cluster_swap=true
model=/data/qianzhirong/models/Qwen3-14B
```

服务器上最近的参考运行目录：

```text
/data/qianzhirong/runs/ehp_2node16_docker_20260818/
/data/qianzhirong/runs/r2e_gym_qwen3_14b_2node16_ehp_docker/
```

其中 `rollout_placement.json` 记录过类似以下布局：一个节点的 8 张 NPU 分成四个 TP=2 服务，端口 `8000`–`8003`。如果沿用该配置，先确认对应的 EHP Docker 管理脚本在目标源码树中存在；没有 `runtime_manage_r2e_rollout_npu.sh` 等依赖时，不要直接执行 docker 配置，应先使用第 6 节的 6 节点 manifest 路径。

## 8. 训练环境变量模板

非 Slurm、SSH/HCCL 启动时，各 rank 至少需要以下变量。实际脚本会根据 placement 为每个 rank 填入 `RANK`、`WORLD_SIZE`、`MASTER_ADDR`、`MASTER_PORT` 和单卡 `ASCEND_RT_VISIBLE_DEVICES`：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH=/data/qianzhirong/runtime_sources/grp_unrestricted_20260812
export DEVICE_BACKEND=npu
export DIST_BACKEND=hccl
export GLOO_SOCKET_IFNAME=enp67s0f5
export HCCL_SOCKET_IFNAME=enp67s0f5
export HCCL_CONNECT_TIMEOUT=1800
export HCCL_EXEC_TIMEOUT=1800
export TORCH_DISTRIBUTED_TIMEOUT=3600
export MCORE_MOE_GROUPED_GEMM=0
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export OMP_NUM_THREADS=1
export R2E_GYM_INDEX=/data/qianzhirong/runtime_sources/RL_Framework_npu_ehp_ab_8ec69d6/data/r2e_gym_v1/index.jsonl
```

训练环境和 rollout 环境都必须加载 CANN，但 Python 解释器不同。已验证基线采用磁盘导出/重启式 rollout weight sync；不要在这套 CANN 8.5.2 + vLLM 0.11.0 环境里自行切到官方 HCCL in-place weight transfer。

官方 vLLM Ascend HCCL in-place 路径需要另一套未作为当前基线的组合（CANN 9.0、torch/torch-npu 2.10、vLLM 0.22.1、vLLM-Ascend 0.22.1rc1）。如果要切换，必须另建环境并重新做整套验证，不能覆盖当前两个环境。

## 9. 可直接使用的启动/检查脚本

下面优先列出适用于当前 Ascend NPU、SSH/HCCL 集群的脚本。带 Slurm、CUDA 或旧占位路径的脚本不要直接使用。

### 9.1 推荐脚本总表

| 脚本 | 用途 | 是否占用 NPU | 备注 |
| --- | --- | ---: | --- |
| `scripts/internal_ssh.sh` | 从跳板机向内网节点批量执行命令 | 否 | 所有批量启动脚本的底层工具，需要 `NODE_PASSWORD` |
| `scripts/probe_internal_npu_free.sh` | 检查指定节点是否 8 卡空闲 | 否 | 只读，适合启动前检查 |
| `scripts/run_global_resource_preflight.py` | 使用 synthetic/history 数据做 GRP 规划 | 否 | 不启动模型，不需要 NPU |
| `scripts/plan_unrestricted_grp_device_placement.py` | 生成 6 节点设备级 placement | 否 | 通常由 6 节点入口调用 |
| `scripts/run_multinode_mcore_npu_preflight.sh` | 多节点 Megatron-Core 模型/optimizer 初始化预检 | 是 | 默认 4 节点、每节点 8 NPU |
| `scripts/run_r2e_qwen3_14b_npu_pilot.sh` | 单节点 pilot：vLLM、eval、4 卡训练 | 是 | 默认训练 NPU 0–3，rollout NPU 4–7 |
| `scripts/run_multinode_r2e_mcore_npu.sh` | 4 节点训练 + 已启动 rollout 服务 | 是 | 要求本机 8000–8003 health 已通 |
| `scripts/run_6node48_grp_vs_no_grp_equal.sh` | 6 节点 GRP 与 no-GRP 串行对照 | 是 | 当前最完整的 48-NPU 入口 |
| `scripts/run_6node48_grp_vs_no_grp_when_idle.sh` | 等待 6 个节点全空闲后启动上一个脚本 | 是 | 适合在 tmux/screen 中等待 |
| `scripts/start_r2e_rollout_manifest_npu.sh` | 按 placement 启停单个节点上的 rollout instances | 是 | `start/stop PLACEMENT_JSON HOST` |
| `scripts/run_vllm_ascend_server.sh` | 启动单个 Ascend vLLM endpoint | 是 | 使用 `/root/vllm_ascend_env` |
| `scripts/sync_npu_worker_from_jump.sh` | 同步环境和源码到一个 worker | 否/写文件 | 不使用 `--delete`，但会覆盖目标同名文件 |

### 9.2 控制面和 GRP 预检

只做 GRP 规划、不启动训练：

```bash
cd /root/RL_Framework_npu
export PYTHONPATH=/data/qianzhirong/runtime_sources/grp_unrestricted_20260812
/data/qianzhirong/envs/rl_framework_py310/bin/python \
  scripts/run_global_resource_preflight.py \
  --config configs/r2e_gym_qwen3_14b_mcore_npu_2node16_ehp_docker.yaml \
  --output-config /tmp/r2e_grp_planned.yaml \
  --decision-json /tmp/r2e_grp_decision.json \
  --synthetic-requests 64 \
  --synthetic-input-len 2048 \
  --synthetic-output-len 4096
```

多节点 Megatron-Core 硬件 preflight：

```bash
cd /data/qianzhirong/runtime_sources/RL_Framework_npu_ehp_ab_8ec69d6
read -rsp 'Internal SSH password: ' NODE_PASSWORD; echo
export NODE_PASSWORD
export INTERNAL_HOSTS='192.168.0.2 192.168.0.24 192.168.0.63 192.168.0.99'
export MODEL_PATH=/data/qianzhirong/models/Qwen3-14B
export MAX_SEQ_LENGTH=512
export INITIALIZE_OPTIMIZER=0
bash scripts/run_multinode_mcore_npu_preflight.sh
unset NODE_PASSWORD
```

成功标记是 `MCORE_NPU_MODEL_INIT_OK`。需要额外验证 distributed optimizer 时，再把 `INITIALIZE_OPTIMIZER=1`，但内存压力更高。

### 9.3 单节点 pilot

这是最方便的真实端到端入口，会启动本机 rollout、等待 health、运行 baseline eval、启动 4 卡训练，再运行 post eval：

```bash
cd /root/RL_Framework_npu
export PROJECT_DIR=/root/RL_Framework_npu
export MODEL_PATH=/data/qianzhirong/models/Qwen3-14B
export CONFIG_PATH=/root/RL_Framework_npu/configs/r2e_gym_cmlfq_qwen3_14b_npu_pilot.yaml
export TRAIN_VENV_DIR=/data/qianzhirong/envs/rl_mindspeed_260
export VLLM_VENV_DIR=/root/vllm_ascend_env
export MAX_MODEL_LEN=4096
export R2E_EVAL_MAX_SAMPLES=4
export R2E_EVAL_CONCURRENCY=1
bash scripts/run_r2e_qwen3_14b_npu_pilot.sh
```

默认设备布局是 rollout 使用 NPU `4`、`5`、`6,7`，训练使用 `0,1,2,3`。本机必须完全空闲。

### 9.4 已有 rollout 服务时的 4 节点训练

该入口默认把 4 个节点作为训练节点，并要求跳板机本地 `127.0.0.1:8000`–`8003` 已经 health 正常：

```bash
cd /data/qianzhirong/runtime_sources/RL_Framework_npu_ehp_ab_8ec69d6
read -rsp 'Internal SSH password: ' NODE_PASSWORD; echo
export NODE_PASSWORD
export INTERNAL_HOSTS='192.168.0.2 192.168.0.24 192.168.0.63 192.168.0.99'
export MASTER_ADDR=192.168.0.2
export MASTER_PORT=29720
export CONFIG_PATH=configs/r2e_gym_qwen3_14b_mcore_npu_5node40_100step.yaml
export RUN_ROOT=/data/qianzhirong/runs/r2e_gym_qwen3_14b_5node40
bash scripts/run_multinode_r2e_mcore_npu.sh
unset NODE_PASSWORD
```

rollout health 未通时，该脚本会拒绝启动训练。

### 9.5 rollout manifest 的单节点管理

GRP placement 生成后，可按 host 手动启停该 host 上的 rollout instances：

```bash
PROJECT_DIR=/data/qianzhirong/runtime_sources/RL_Framework_npu_ehp_ab_8ec69d6
RUN_ROOT=/data/qianzhirong/runs/<run>
PLACEMENT=$RUN_ROOT/device_placement.json
HOST=192.168.0.112

bash "$PROJECT_DIR/scripts/start_r2e_rollout_manifest_npu.sh" \
  start "$PLACEMENT" "$HOST"

for port in 8000 8001 8002 8003; do
  curl -fsS "http://${HOST}:${port}/health" || true
done

bash "$PROJECT_DIR/scripts/start_r2e_rollout_manifest_npu.sh" \
  stop "$PLACEMENT" "$HOST"
```

它通过 PID registry 和进程命令行校验停止实例，比共享节点上执行全局 `pkill` 安全。

### 9.6 等待空闲后自动启动

```bash
cd /data/qianzhirong/runtime_sources/RL_Framework_npu_ehp_ab_8ec69d6
read -rsp 'Internal SSH password: ' NODE_PASSWORD; echo
export NODE_PASSWORD
export AVAILABLE_HOSTS='192.168.0.47 192.168.0.88 192.168.0.89 192.168.0.189 192.168.0.195 192.168.0.252'
export POLL_SECONDS=120
bash scripts/run_6node48_grp_vs_no_grp_when_idle.sh
unset NODE_PASSWORD
```

脚本会持续轮询，直到 6 个节点都返回 `idle_npus=8`，然后调用 6 节点 GRP/no-GRP 入口。建议放在 tmux/screen 中运行。

### 9.7 同步源码/环境到新 worker

目标节点空闲且确实需要同步时使用：

```bash
cd /root/RL_Framework_npu
read -rsp 'Internal SSH password: ' NODE_PASSWORD; echo
export NODE_PASSWORD
bash scripts/sync_npu_worker_from_jump.sh 192.168.0.112
unset NODE_PASSWORD
```

该脚本同步 `/root/rl_framework_py310` 和 `/root/RL_Framework_npu`，并建立 `/root/RL_Framework -> /root/RL_Framework_npu` 软链接。不会使用 `rsync --delete`，但会覆盖目标同名文件。

### 9.8 不建议直接使用的脚本

- `scripts/run_hetero_distributed.sh`、`scripts/run_hetero_distributed_4GPU.sh`：旧 GPU/CUDA/Slurm 风格，包含占位路径，不适用于当前 Ascend NPU 集群。
- `scripts/run_distributed.sh`、`scripts/run_slurm_distributed.sh`：通用/旧入口，除非按当前集群改写并验证，不要直接执行。
- `scripts/cleanup_zombie.sh`：调用 `pkill -9` 和 `nvidia-smi`，是 GPU 清理脚本；不要在共享 NPU 节点上运行。
- `scripts/validate_hccl_weight_transfer_env.py`：用于官方 vLLM Ascend HCCL in-place 新版本栈，不是当前 CANN 8.5.2 + vLLM 0.11.0 基线的启动脚本。
- `scripts/setup_npu_env.sh`：安装依赖用，不是启动脚本；当前服务器已有已验证环境，不要重复运行覆盖环境。

## 10. 停止任务与清理

优先在启动脚本所在终端按 `Ctrl-C`，脚本的 trap 会停止自己登记的 rank 和 rollout 实例。若终端断开，使用对应 run 的 placement 文件精确停止：

```bash
RUN_DIR=/data/qianzhirong/runs/<run>
PLACEMENT="$RUN_DIR/device_placement.json"
for host in 192.168.0.112 192.168.0.252 192.168.0.189 192.168.0.47 192.168.0.88 192.168.0.89; do
  PROJECT_DIR=/data/qianzhirong/runtime_sources/RL_Framework_npu_ehp_ab_8ec69d6 \
  RUN_ROOT="$RUN_DIR" \
  ./scripts/internal_ssh.sh "$host" -- \
    "cd /data/qianzhirong/runtime_sources/RL_Framework_npu_ehp_ab_8ec69d6 && \
     bash scripts/start_r2e_rollout_manifest_npu.sh stop '$PLACEMENT' '$host'"
done
```

执行批量 stop 前先重新设置当前 shell 的 `NODE_PASSWORD`。如果 placement 文件不存在，不要用全局 `pkill` 猜测清理目标，先根据 `ps`、`npu-smi info` 和 run 日志确定进程归属。

## 11. 常见问题排查

| 现象 | 原因/处理 |
| --- | --- |
| `libhccl.so` 找不到 | 先 `source /usr/local/Ascend/ascend-toolkit/set_env.sh`；确认 CANN 8.5.2 路径，不要盲目重装 torch。 |
| `npu_count` 不是 8 | 当前节点驱动或 CANN 环境异常；运行 `npu-smi info`，并和其他节点对比。 |
| `host is not fully idle` | 节点有其他任务；不要杀进程，换节点或等待。 |
| 内网 SSH 一直超时 | 确认命令是在跳板机执行，`NODE_PASSWORD` 已设置，内网地址属于 192 网段。 |
| `No module named RL_Framework` | `PYTHONPATH` 必须指向包含 `RL_Framework` 目录/软链接的包根；6 节点脚本默认使用 `/data/qianzhirong/runtime_sources/grp_unrestricted_20260812`。 |
| HCCL 初始化卡住 | 检查所有 rank 的 `MASTER_ADDR/PORT`、`HCCL_SOCKET_IFNAME=enp67s0f5`、`GLOO_SOCKET_IFNAME=enp67s0f5`、HCCL base port 是否一致且未冲突。 |
| vLLM health 不通 | 先看 `/data/qianzhirong/runs/<run>/rollout_logs/`；确认 rollout Python、`ASCEND_DEVICES`、端口和模型路径。 |
| C-MLFQ 没有 migration | 需要真实 tool-return；检查 bucket TP 和 rollout placement 是否匹配，检查 `cmlfq_shared_load` 是否可写。 |
| GRP 只输出 candidate 没有变更 | 可能只是 plan-only、收益低于 `min_gain_ratio` 或被 reconfiguration cost 拒绝；查看 effective config 和 planner/runtime 日志。 |
| `MASTER_PORT` 被占用 | 换端口，并确保所有 rank 的配置一致；GRP/fixed 两个 arm 不要共用端口。 |
| 训练环境能 import、远端 rank 不能 | 每个远端命令都必须 source CANN、设置 `PYTHONPATH`，并使用 `/data/qianzhirong/envs/rl_mindspeed_260/bin/python`。 |

## 12. 推荐执行顺序

```text
登录 119.13.124.91
  -> source CANN，检查本机 8 NPU
  -> 检查目标内网节点空闲
  -> CPU 跑 test_r2e_cmlfq_flow.py / global_resource_planner_full_flow.py
  -> 6 节点 PREFLIGHT_ONLY=1
  -> 检查 effective_config.yaml 和 device_placement.json
  -> 启动 rollout manifest，逐个 curl /health
  -> 启动 GRP + no-GRP 对照
  -> 观察 driver logs、C-MLFQ tree、planner/reconfiguration 记录
  -> 运行 analyze_libra_experiment.py 生成报告
  -> Ctrl-C 或按 placement 精确 stop
```

完成一次实验后，请把以下信息发回项目负责人：`RUN_ROOT`、GRP/fixed run 名称、使用的 6 个节点、实际 `device_placement.json`、最终报告路径、任何 `Traceback`/`HCCL`/`vLLM` 错误，以及是否生成了 C-MLFQ tree 和 planner reconfiguration 事件。
