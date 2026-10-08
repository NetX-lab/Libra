# DeepSeek-V4-Flash：96 NPU 实验预检（2026-10-07）

状态：未启动性能实验。C-MLFQ、EHP、GRP 的提升率均未测量。
本文件记录本轮实际读取的现场状态；automodelwire 的历史性能是参考证据，不能作为 Libra 的结果。

## 现场连接与资源

本地可免密登录 root@119.13.124.91（192.168.0.85）。直接连接本地不可达的
192.168.0.50 超时；ProxyJump 和跳板机内 SSH 均对该节点返回认证失败。
从跳板机对用户提供的全部 20 台执行 BatchMode SSH / npu-smi 只读检查：

| 节点后缀 | 当前结果 |
| --- | --- |
| 85 | 8 张 910B3，各 65536 MiB；8 张均无运行进程，HBM 基础占用约 3433–3439 MiB |
| 47 | 8 张均有 rayRayWorkerP 进程，HBM 各约 59040–59044 MiB |
| 17 | 8 张均有 Ray 进程，HBM 各约 13233–32006 MiB |
| 88、36 | SSH connection refused |
| 50、99、189、41、2、51、155、48、89、195、63、252、24、112、53 | SSH Permission denied；无法判定空闲状态 |

没有停止其他任务、启动训练、重启容器或占用 NPU。空闲快照不等于资源预约。
需要可用的内网 SSH 认证方式，并重新确认至少 12 台整机可用于实验。

## 权重和最少卡数

重新读取跳板机 `/data_nv1/models/DeepSeek-V4-Flash-DSpark-BF16`：

- config：DeepseekV4ForCausalLM，43 层、256 专家、每 token 6 专家、BF16、DSpark block size 5。
- index 引用的 48 个分片全部可读，总文件大小 608454419616 字节（约 566.67 GiB）。
- 共 304180418494 个 tensor 元素，其中 BF16 304140349824、F32 37741630、I64 2327040；包括整数缓冲区，不能全部解释为可训练参数。
- 逐分片读取 safetensors 头部，检查索引 tensor 存在及 data_offsets 末端不超过文件大小，无错误；未执行完整内容校验和、分布式加载或数值正确性验证。

**容量下界**：单份 BF16 权重全驻 HBM 时，ceil(566.67/64)=9 张。这是假设理想均匀分片的纯容量下界，未包含设备基础占用、激活、KV cache、梯度、优化器、通信缓存及加载峰值，也不保证 9 卡是合法并行拓扑。单台 8 卡不能全驻这份权重。

**96 卡参考布局**：automodelwire/configs/train_e4_reduce.sh 使用 64 张训练卡和
32 张推理卡，训练 TP=4、PP=2、EP=32、ETP=1、CP=1；推理 TP=8、DP=4、跨组 EP。
TP8 的推理组不能被理解成独立装下完整权重的一份副本。
归档报告记录过这套 verl/MindSpeed 布局运行，但本轮尚未复现，更未验证 Libra。

**完整 Libra 最小卡数尚未确定**。需要固定精度、上下文、batch、CPU offload、参考策略存储方式和并行拓扑，验证端到端峰值显存后，按合法拓扑逐级缩减；不能将 9、80 或 96 宣称为已验证最小值。测试 EHP 还需要可用于完整训练副本的弹性资源，C-MLFQ 需要实际可调度的多个服务组，均须单独验证。

## 当前代码适配缺口

本地及远端 `/data/qianzhirong/runtime_sources/Libra_DeepSeekV4_compat_20260919`
的 MegatronCoreTrainEngine.initialize 均显式拒绝 PP != 1，并导入
`megatron.bridge.AutoBridge`。automodelwire 的路径是 MindSpeed-LLM DeepSeek4Model、
mbridge 与 PP=2；移除 PP 检查或修改模型名称不足以完成适配。

需要在专用后端中接通 DeepSeek4Model 的 DSA/mHC、PP 前后向调度、Libra GRPO loss、
分片加载和导出；再把 automodelwire 的 HCCL 同步及 MoE w2 格式修复接到 Libra 的
policy version 语义。EHP 还需检验完整分片副本的加入/退出、梯度归并与同步一致性。
参考 overlay/README.md、overlay/new-files/pretrain_deepseek4.py、overlay/verl.patch、
overlay/mbridge.patch，不能用 verl 的训练结果替代 Libra。

当前跳板机上的 Libra-chenkaiwen 容器为 Exited (137)，未启动它；本轮未验证容器内依赖。
先前 2026-09-27 的依赖错误只作为历史诊断，不能当作今天重新运行的结果。

## 性能测试口径（待执行）

先通过真实模型加载、有限 loss/梯度、一次参数更新、训推权重同步和版本一致性检查。
再验证一次 EHP 加入/退出与 GRP 计划实际执行。各阶段保存峰值 HBM、错误日志、
实际进程和设备映射；不以模拟器结果或前向成功宣称训练适配完成。

固定 96 张卡预算，同一初始权重、数据、上下文预算、global batch、采样参数和随机种子。
DSpark 开关必须在所有组相同。主工作负载需有真实工具调用及返回状态，避免仅用单轮
GSM8K 掩盖 C-MLFQ 的因果调度机制；GSM8K 可先用于适配回归。

| 组 | C-MLFQ | EHP | GRP |
| --- | --- | --- | --- |
| F（完整 Libra） | 开 | 开 | 开 |
| F-C | 关，使用匹配的负载均衡对照 | 开 | 开 |
| F-E | 开 | 关 | 开 |
| F-G | 开 | 开 | 关，使用预先固定的合法布局 |

这四组给出其他两项开启时的条件贡献：提升_i = (吞吐_F / 吞吐_F-i - 1) × 100%。
贡献不可相加。若要报告平均主效应及交互作用，扩展为全部 2^3=8 组，而非逐项累加。
所有组合先检查可执行性；例如 GRP 关闭时 EHP 的调度策略必须明确，不能让 EHP 开关
成为无效开关。固定预算内的闲置资源仍计入卡时。

主要指标为有效完成轨迹/s，补充输出 token/s、训练 step 时间、TTFT/TPOT、延迟
P50/P95/P99、失败率、policy staleness、奖励、卡时及重配置总耗时。同步/迁移/重配置
耗时计入端到端窗口。各组从相同初始 checkpoint 独立启动，轮换执行顺序，至少 3 次
配对重复；先用 pilot 判断需要的预热、测量长度和重复数，再冻结正式实验方案。

当前没有任何可用于计算三项性能提升的正式样本。
