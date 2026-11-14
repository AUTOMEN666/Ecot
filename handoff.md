# Stack Bowls在线Subtask交接

本次授权是代码修改、新建分支、提交GitHub PR；没有实际API/GPU/仿真启动。
分支ecot_online_generation_subtask_stack_bowls，基线分类分支95188146。
训练来源已只读核对为Stack Bowls100 Subtask-only/dropout0 EMA10000；源eval_ecot=false不改写。
297条候选原文从真实44674有效起点提取，新Prompt针对叠碗阶段完成和目标连续性。

协议/步骤见README、experiment/evaluation.md、online_subtask/docs/integration.md。
先独立完整工程回合，再seed1现有30个Random布局；native800、noise42、execute8。
每回合独立策略进程/日志，封存后验收，视频只显示实际执行动作对应的Subtask。
实际GPU归属/renderer/环境准入由既有外部owner负责；本代码不预约资源或启动parking。
不得恢复已暂停ERVLA、碰其它用户进程或改共享环境。

CPU fixture不能作为真实工程验收。下一步先review PR，再按独立运行授权完成真实接入。
