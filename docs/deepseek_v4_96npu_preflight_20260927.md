# DeepSeek-V4 96 卡实验现场预检（2026-09-27）

最新状态：新增节点后已确认 12 台、96 卡空闲；正式性能实验未启动，训练后端兼容性预检失败。下方原始节点表为先前快照。
通过 root@119.13.124.91 跳板机检查，认证信息仅用于交互式进程，不写入本文件。
空闲定义为 npu-smi 报告该卡无运行进程；结果为检查时快照，不表示预约或独占。

| IP 后缀（192.168.0） | 无运行进程的 NPU 数 |
| --- | ---: |
| 50 | 8 |
| 99 | 8 |
| 189 | 8 |
| 41 | 8 |
| 2 | 0 |
| 88 | SSH connection refused |
| 51 | 0 |
| 155 | 5 |
| 48 | 8 |
| 89 | 0 |
| 195 | 8 |
| 47 | 8 |
| 63 | 8 |
| 252 | 0 |
| 36 | No route to host |
| 85 | 8 |
| 24 | 0 |
| 17 | 0 |
| 112 | 0 |
| 53 | 0 |

候选整机池：50、99、189、41、48、195、47、63、85，共 72 卡。
155 的 5 张零散空闲卡不计入 12 台整机的实验布局。未停止现有任务。

## 权重核验

跳板机完整模型路径：`/data_nv1/models/DeepSeek-V4-Flash-DSpark-BF16`。
根据 model.safetensors.index.json 枚举全部分片，检查文件存在、读取 safetensors
头部、核对 data_offsets 对应文件长度，并检查索引中每个 tensor 所属文件：

- 48/48 分片存在，无缺失。
- 分片长度错误 0，tensor 索引错误 0。
- 总文件大小 608454419616 字节。
- tensor dtype：BF16、F32、I64。

这是结构完整性检查，未逐字节计算校验和，也未在所有候选节点执行完整模型加载。
旧路径 `/data/qianzhirong/models/DeepSeek-V4-Flash-0731` 仅 13/48 分片，
不应当用作本轮模型路径。

## 实验与归因要求

在至少 12 台节点空闲且加载、反向、权重同步预检成功后启动 96 卡实验。
当前未确认完整 DeepSeek-V4 Libra RL 训练适配成功；已有 canary 日志中的
推理前向输出不等价于训练成功，也不等价于对照实验结果。

固定模型、数据、精度、样本数、随机种子、上下文预算和全局 batch，记录真实
训练更新、rollout、同步、迁移事件。分别开展匹配的 C-MLFQ 调度对照、EHP
开关对照、GRP 规划对照。其余机制保持一致，明确每项提升的条件；若采用逐步
叠加实验，结果应称为条件增量，不能声称是相互独立贡献。

主要吞吐提升 = (处理组有效完成轨迹/计时秒数 ÷ 对照组有效完成轨迹/计时秒数 - 1) × 100%。
同时报告步骤时间、token/s、延迟分位数、失败率、实际卡数和重配置耗时。
目前三项实际提升率均为未测，不是 0%。先前最小 80 卡估算未获实机验证。

## 新节点与启动前实测更新

选定节点：192.168.0.{50,189,41,99,48,195,63,85,138,217,131,47}。
本轮再次逐台查询 npu-smi，12 台均报告 8 张无运行进程的 NPU。
其中 10 台在 `/data_nv1/models/DeepSeek-V4-Flash-DSpark-BF16` 下有 48 个分片；
217、131 的该路径无分片，但有共享 `/data` 挂载。所有节点未预约或独占。
共享路径 `/data/l00619320/models/DeepSeek-V4-Flash-DSpark-BF16` 已在跳板机核验：
48/48 分片，文件长度错误 0、tensor 索引错误 0；未做完整文件校验和或分布式加载。

训练兼容性检查：

1. 使用 `/data/qianzhirong/envs/rl_mindspeed_260/bin/python` 对 BF16 模型执行
   `AutoConfig.from_pretrained(..., trust_remote_code=True, local_files_only=True)`，
   实际报错 `KeyError: 'deepseek_v4'`，随后 ValueError：Transformers 不识别此架构。
2. `Libra-chenkaiwen` 容器对共享 BF16 模型执行相同配置加载，返回 `DeepseekV4Config`。
   但 `importlib.util.find_spec('megatron.bridge')` 返回 None；Libra 的
   MegatronCoreTrainEngine.initialize 明确依赖该模块的 AutoBridge。
3. 容器中的替代包 mbridge 在导入时因缺少 transformer_engine 报错。
   mbridge 与 megatron.bridge 不是可以直接替换的同一接口。
4. 远端 Libra_DeepSeekV4_compat_20260919 训练入口明确限制 PP=1；已有其他项目
   DeepSeek-V4 的 train.sh 使用 PP=2 和专用 MindSpeed/mbridge 路径，不能直接
   当作 Libra 的 96 卡实验启动器。

因此本轮仅执行节点、模型路径、Python 配置加载与依赖预检，没有启动训练、
rollout 性能压测或消融组。需要先提供已验证的 Libra DeepSeek-V4 训练适配，
或开展该后端的集成开发，验证真实梯度更新、权重同步以及 EHP 的一致性后再计时。
