# 验证范围

CPU套件保留源分支7模块并按叠碗更新，加入任务、实际导入来源、布局、证据与视频检查。
最新46项全部通过，0失败、0错误、0跳过，见../provenance/cpu_validation.json。
测试中的合成fixture不是实际API/GPU/native接入证据。

覆盖因果图片/State匹配、93+3均匀历史、792帧99条执行历史、ACK后入记忆、
失败重试隔离、reset、RNG不变、私有RPC等待参数、回环HTTP超时/子进程回收、
真实训练formatter调用与不一致拒绝、实际模块来源/缓存外来模块拒绝、
原训练身份/EMA/step限制、Random30/source SHA/私有SDK、全场景冻结、800与拒绝128、
实际文本边界、完整视频解码、缺失/伪造ACK与中断拒绝、30回合统计准入。

H20-6已只读核对真实EMA10000、训练源码/norm/manifest/seal和49文件清单。
源eval_ecot=false保持不变；这不是正式checkpoint全tensor恢复验收。
实际SDK的Random30私有CPU准备亦已通过；早期收据见REMOTE_CPU_VALIDATION.json。

用户随后授权环境检查与H20修复。A800两节点有限seed42 JAX HIGHEST/Torch CUDA核函数通过；
HIGHEST只用于数值核验，没有改变策略精度。H20修复显式策略库路径后核函数通过，
私有SDK改正旧support导入后，实际scene1/layout0原生reset、300步稳定性、
零物理推进的三路640×480 RGB采集均通过，图像已收回目视检查。
共享SDK/环境、物理/奖励规则和已有任务未修改。原失败与新通过收据分别保留。

上述GPU/原生初始化不包含策略权重加载、DeepSeek调用或完整在线ECoT回合，不能作为正式30回合准入。
两台A800本次采样未见运行中的64路GPU进程；没有完成64并发容量验收。
真实策略加载、完整在线单回合工程验收及正式30回合仍需分别执行与审核。
Prompt连续性不能保证LLM识别、跟踪或动作遵循正确。
