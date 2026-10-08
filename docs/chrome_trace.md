# Chrome Trace 事件采集

训练进程可以导出 Chrome Trace Event JSON，用于查看 rollout、训练阶段、权重同步、C-MLFQ 路由/迁移和 GRP 决策/重配置的时间关系。

在配置文件中启用：

```yaml
enable_chrome_trace: true
chrome_trace_output_dir: "./logs/chrome_trace"
```

也可以在启动命令中临时覆盖配置：

```bash
python examples/r2e_gym_async_rl.py \
  --config configs/your_config.yaml \
  --enable-chrome-trace \
  --chrome-trace-output-dir ./logs/chrome_trace
```

关闭采集使用 `--disable-chrome-trace`。命令行开关优先于配置文件，且启用与关闭选项不能同时使用。

每个训练进程会生成一个独立文件：

```text
trace.rank_<rank>.pid_<pid>.json
```

训练结束后合并：

```bash
python scripts/merge_chrome_traces.py \
  ./logs/chrome_trace \
  ./logs/chrome_trace_merged.json
```

然后用 Perfetto 或 Chrome 的 `chrome://tracing` 打开合并后的 JSON。主要事件类别如下：

- `rollout`：请求入队、提交、路由、生成、结果返回和 rollout 总耗时。
- `train`：采样批次、分发、优势计算、重算 log-prob、GRPO 更新等阶段。
- `weight_sync`：权重同步阶段和同步耗时。
- `cmlfq`：初始路由、tool return、迁移判断、实际迁移和请求完成。
- `grp`：资源规划决策和运行时重配置。

采集默认关闭；打开后会增加少量本地 JSON 写入开销。配置只对新启动的训练进程生效，正在运行的训练不会自动加载新代码或新配置。
