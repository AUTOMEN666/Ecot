# 验证范围

CPU suite保留源分支7模块并按叠碗更新，新增任务、布局、证据与视频检查。
结果见../provenance/cpu_validation.json；测试中的合成fixture不是实际API/GPU/native接受证据。

覆盖因果图片/State匹配、93+3均匀历史、792帧99条执行历史、ACK后入记忆、
失败重试隔离、reset、RNG不变、私有RPC等待参数、回环HTTP超时/子进程回收、
原训练身份/EMA/step限制、Random30/source SHA/私有SDK、全场景冻结、800与拒绝128、
实际文本边界、完整视频解码、缺失/伪造ACK与中断拒绝、30回合统计准入。

新只读校验器已在H20-6核对真实EMA10000、训练源码/norm/manifest/seal和49文件清单通过。
保留源eval_ecot=false。本次只读核查未对所有正式tensor重新hash/restore，没有初始化GPU/API/仿真。

实际运行前仍需新主机资源/环境/renderer准入、完整checkpoint绑定、真实策略加载、
一次完整工程回合和最终30回合；Prompt连续性不保证LLM识别/跟踪/动作遵循。

另在H20-6实际SDK上用临时私有目录完成Random30配置接入：30布局、800上限、2100秒私有RPC核对通过，共享SDK未修改，临时目录已清理；未启动原生仿真。详见REMOTE_CPU_VALIDATION.json。
