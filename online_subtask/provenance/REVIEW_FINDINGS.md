# 用户复查与修复

针对PR#3按已确认要求逐项复查，发现并修复了以下实际接入缺口：
1. 训练源SHA校验没有约束Python实际导入。现在训练vendor/OpenPI、严格文本与SDK/RPC源被显式固定，并检查真实module.__file__及SHA，CPU准备先验证。
2. 主机解释器可用性与原生SDK/IsaacLab导入先前依赖外部隐式设置。现在准备时拒绝断链解释器，固定私有SDK/XPolicyLab/curobo/IsaacLab路径。
3. 只传device_id不能约束AppLauncher。现在同时传device cuda:<id>，关闭多卡渲染，启动前复核。
4. 原计划仅准备时检查。现在在原生启动前重验代码/Prompt/候选、SDK、布局与独立工程验收。
5. 完整证据准入增加逐次等待/10Hz历史与录像帧时钟完整性，不能用单条冻结记录覆盖整回合。
6. 原SDK成功包含回位；Prompt补充三碗完成后保持栈并释放/撤臂，不重启抓取。
7. 实际H20 SDK残留旧项目support.probe_readback依赖，私有副本改为本分支逐字节相同的readback helper，避免重复安装；原reward规则不改。
8. 训练Prompt逐字节一致性现在实际调用绑定训练源码的compose_training_prompt比较，而不是与自身formatter互证。
9. H20的实际GPU库路径与训练主机不同。新增策略专用库路径配置及SHA，不改共享环境，不改变模型精度或原生进程环境。

固定科学设置保持EMA10000、Subtask-only/dropout0、seed1现有Random30/noise42、50预测/8执行/10flow、原生800。
策略仍只消费当前三图、State14、Instruction和Subtask；没有未来图或专家动作进入在线生成。
目标连续性仍由Prompt和已执行历史约束，不代表已有接触/完成真值；语义跟踪与真实ECoT-follow必须在闭环中另测。
环境核函数/独立原生初始化不是完整工程回合，不能进入正式分母或直接解锁30回合。
