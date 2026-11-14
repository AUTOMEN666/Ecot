# Stack Bowls：π0.5 在线 Subtask 测评

本分支 ecot_online_generation_subtask_stack_bowls 从分类分支
ecot_online_generation_subtask_classify_objects 的 commit
95188146ce7191aa15d8a23a545707712e267d81 独立适配。
DeepSeek生成当前Subtask，π0.5接收文本并预测动作。
**此PR交付代码与CPU验证，未执行真实DeepSeek、GPU推理或仿真，没有成功率结果。**

| 项目 | 固定设置 |
|---|---|
| 权重 | 官方π0.5 Base独立全参训练的Stack Bowls100，EMA10000 |
| 原训练 | episodes3000–3099，44674有效起点，Subtask-only，dropout0，总训练15000步 |
| 任务 | 原生stack_bowls_random，保留SDK成功/终止逻辑，800动作上限 |
| 场景 | seed1，已有30个Random布局0–29，各一次；不生成新布局或重复凑数 |
| 接入 | 先独立一个完整原生工程回合，通过后再正式30；工程回合不进分母 |
| 策略 | 当前三路RGB、State14、Instruction＋Subtask；noise42，predict50/execute8/flow10，25Hz |
| 在线生成 | 当前三图/State14、至多93张严格过去主图及对应State14、已确认执行的Subtask历史 |
| 等待 | 生成、推理、历史采集和ACK等待期间不推进仿真，核对全场景/机器人/物理时钟 |
| 视频 | 25Hz主相机与逐动作实际注入文本同步MP4 |

策略文本：
~~~text
Stack the three bowls together. Current subtask: {原训练Subtask原文}
~~~
不注入Summary、Reasoning、bbox、场景真值、未来帧或专家动作。
源训练eval_ecot=false保持原样，新测评明确记录effective_eval_ecot=true。

- [Prompt](online_subtask/runtime/selector_prompt.md)：移除分类/篮子规则，加入叠碗、支撑栈、分阶段完成与恢复规则。
- [协议](online_subtask/docs/protocol.md)：DeepSeek输出候选ID，宿主精确映射原训练文本。
- [接入步骤](online_subtask/docs/integration.md)：权重、布局、工程验收、30回合、视频与统计。
- [测评定义](experiment/evaluation.md)、[交接](handoff.md)、[验证范围](online_subtask/docs/validation.md)。
- runtime是当前代码；reference/classify_objects仅保留历史分类证据，不是叠碗运行入口。

CPU验证需NumPy、Pillow；视频测试还需ffmpeg/ffprobe。完整套件仅访问临时127.0.0.1 HTTP服务。
~~~sh
python -E -B online_subtask/tests/run_cpu_tests.py --report /tmp/stack_bowls_cpu.json
python -E -B tools/verify_archive.py
python -E -B online_subtask/runtime/native_cpu_smoke.py
~~~
可用 --exclude-http 排除回环HTTP测试；缺少ffmpeg时视频测试明确跳过。
合成fixture和CPU合约验证不证明真实模型识别或动作跟随正确。
仓库不含权重、SDK资产、密钥、真实API请求或视频；依赖已有环境与合法GPU分配。
本代码不做GPU预约或synthetic parking。
