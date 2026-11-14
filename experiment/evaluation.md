# 固定测评定义

目标：Stack Bowls100 Subtask-only π0.5 EMA10000，在原生stack_bowls_random用在线DeepSeek Subtask执行。

- 训练从官方Base独立全参运行15000步，episodes3000–3099，Subtask原文/dropout0。
- 只准入EMA10000，绑定训练源码、identity、norm、数据投影和独立checkpoint seal；不改源eval_ecot=false。
- seed1现有Random布局0–29，每个一次，共30正式回合；不新建或筛选布局。
- 正式前独立完整工程回合，工程通过与任务成功分开，工程回合不进分母。
- 原SDK800动作、reward/termination不变。不用128截断、分类1100步或模型完成判断提前退出。
- noise42，predict50/execute8/flow10，25Hz动作/250Hz物理；等待冻结和实际文本同步录像。
- 每回合独立策略进程与日志，结束关闭后验收/汇总，不在已验收日志追加新回合。

成功率=原SDK success=true的正式回合数/30；原SDK score均值另列，不当成成功率。
API/工程中断、缺录像/文本/时钟或非零退出不能计为完整SDK失败。
先单列中断原因；工具在收齐30个有效回合前拒绝发布完整成功率。
SDK正常失败可通过工程集成验收，不能为通过验收修改结果。

本PR没有实际API/GPU/仿真运行；CPU合成fixture没有科学成功率。
