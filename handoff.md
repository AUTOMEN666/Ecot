# Stack Bowls在线Subtask交接

原代码PR已提交为GitHub PR#3；用户随后授权检查A800/H20环境并修复H20，跨项目仅检查。没有执行DeepSeek在线策略闭环或正式测评。
分支ecot_online_generation_subtask_stack_bowls，基线分类分支95188146。
训练来源已只读核对为Stack Bowls100 Subtask-only/dropout0 EMA10000；源eval_ecot=false不改写。
297条候选原文从真实44674有效起点提取，新Prompt针对叠碗阶段完成和目标连续性。

协议/步骤见README、experiment/evaluation.md、online_subtask/docs/integration.md。
先独立完整工程回合，再seed1现有30个Random布局；native800、noise42、execute8。
每回合独立策略进程/日志，封存后验收，视频只显示实际执行动作对应的Subtask。
实际GPU归属/renderer/环境准入由既有外部owner负责；本代码不预约资源或启动parking。
不得恢复已暂停ERVLA、碰其它用户进程或改共享环境。

CPU fixture不能作为真实工程验收。下一步先review PR，再按独立运行授权完成真实接入。

复查已修：强制实际训练OpenPI/RPC导入与CPU检查；有效解释器；策略专用CUDA库配置；
原生私有SDK/IsaacLab导入；device_id与AppLauncher设备一致且关闭多卡渲染；
启动前重验代码/布局/工程凭证；完整冻结/ACK/录像账本；三碗完成后撤臂回位。
A800两个节点的有限JAX HIGHEST/Torch核函数已实际通过，旧默认精度断言失败和诊断保留；
H20原no-LD方式cuDNN初始化失败，显式已安装库路径后核函数通过，文件保存在本次专用POLICY_ENV.json。
GPU环境检查不证明模型/原生渲染/64并发已通过；当前实测未发现64路运行进程。

H20后续独立原生v2已实际通过：scene1/layout0，原始300步稳定性检查，三路640×480 RGB，
采集期间物理步0，图像已收回目视检查；原生进程退出、公共锁释放已核验。
旧项目support导入只在私有SDK内替换为逐字节相同的本分支readback helper，避免重复安装。
CPU测试46项通过/0跳过。实际训练formatter字节比较与启动前来源重验已纳入。
这些是环境层验收，完整在线ECoT回合、正式30和64并发仍未验收。
