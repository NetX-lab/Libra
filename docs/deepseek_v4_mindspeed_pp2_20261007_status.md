# Libra × DeepSeek-V4-Flash: MindSpeed PP=2 状态

日期：2026-10-07

## 当前结果

128 卡全模型集成门禁通过：16 台 910B3 服务器，每台 8 卡；MindSpeed、TP=4、PP=2、EP=32、ETP=1、全局 batch=48，2048-token 定长轨迹，CPU optimizer offload。128/128 个 rank 完成初始化、logprob 前向、GRPO 前向/反向和 optimizer 更新，16 个节点退出码均为 0，128 个 rank 结果文件均已生成。

门禁汇总指标：loss -0.33531、policy loss -0.33549、KL 0.18236、48 个样本、参数更新检查通过；每卡峰值 allocated HBM 约 42.82 GiB，reserved 约 47.53 GiB。该门禁验证集成与一次合成更新，不是吞吐基准。

共享结果目录：`/data/qianzhirong/runtime_sources/Libra_MindSpeed_PP2_20261007/full_v4_gate/`，包括 `logs/node_*.log`、`logs/node_*.exit` 和 `rank_*.json`。已清理已退出的命名容器，日志和结果文件仍保留。

## 修复记录

- CANN 9.1 toolkit 缺少 `libcann_kb.so`：使用 automodelwire beta.3 的库补齐 KB 路径，`.to(NPU)` 与全量权重加载通过。
- MindSpeed CPU optimizer offload 需要 DP-with-CP 与 Expert-DP Gloo 组：运行时补丁只创建 optimizer 所需 Gloo 组，保留其他 Gloo 镜像关闭。
- PP=2 variable-sequence pipeline 曾向后续 stage 传递零长度 activation：门禁先将 microbatch 对齐为定长，再关闭 `variable_seq_lengths`。
- CANN Sparse Flash MLA 要求 `cmp_topk` 为 0/512/1024：门禁改用 2048 token 序列，压缩 KV 索引达到 512。
- 多 rank HCCL host socket 争用：按本地 rank 隔离 base port，并设 `HCCL_HOST_SOCKET_PORT_RANGE=auto`。
- 合成轨迹按数据并行副本切分：全局 batch 48、DP=16，每个副本提交 3 条样本。

## 性能与卡数结论

还没有 C-MLFQ、EHP、GRP 相对基线的性能提升数字。当前结果不能替代有 rollout、固定窗口和匹配配置的对照基准。

此前 96 卡尝试在当前并行与 optimizer 设置下因 DDP 梯度缓冲区 OOM 失败；128 卡是首个通过全模型训练更新门禁的已测配置。因未系统测试中间卡数与其他可行并行拓扑，128 不能称为理论最小卡数。

当前 MindSpeed 引擎的 `configure_elastic_training` 明确拒绝 EHP：EP 分片模型尚未实现独立 dense/expert replica domains。因此 EHP 测试需要先补齐这一分布式训练域能力。C-MLFQ/GRP 性能对照也仍需接入真实 rollout 与计时基准；本轮未启动这些对照。
