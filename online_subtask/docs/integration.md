# 接入与运行

本PR未启动服务。以下步骤在另行取得运行授权、真实GPU资源与环境/renderer准入后执行。
外部依赖：原训练项目、EMA10000/SEAL、norm、manifest/cache/token audit和训练源码；
匹配OpenPI/JAX、RoboDojo/IsaacSim环境、完整Random资产、DeepSeek授权。
configs/external_paths.example.json仅是占位说明，不下载资产。

## CPU准备

在仓库根目录，使用操作者核实的真实路径：
~~~sh
RUNTIME="$PWD/online_subtask/runtime"
TRAIN_ROOT=/path/to/pi05_stack_bowls100_subtask_fullparam_h20_8_20261113
SDK_ROOT=/path/to/native_sdk
OUT=/path/to/fresh/stack_bowls_random_seed1_online30_ema10000_v1
POLICY_PY=/path/to/verified_policy/bin/python
SIM_PY=/path/to/verified_native/bin/python
mkdir -p "$OUT"
python "$RUNTIME/prepare_policy.py" bind --train-config "$TRAIN_ROOT/configs/train.json" --output "$OUT/EMA10000.binding.json"
python "$RUNTIME/layout_manifest.py" --sdk-root "$SDK_ROOT" --output "$OUT/layouts.json"
~~~
绑定完整读取EMA推理文件并与独立seal比较，不重写/导出/还原权重。
这不等于对正式checkpoint完成了全tensor恢复。
记录实际输出的manifest SHA，后续显式传入，不能重新生成/放宽。

~~~sh
MANIFEST_SHA=实际输出的SHA256
python "$RUNTIME/prepare_policy.py" prepare --parent-root "$TRAIN_ROOT" --export-binding "$OUT/EMA10000.binding.json" --output "$OUT/policies/engineering" --policy-python "$POLICY_PY" --sdk-root "$SDK_ROOT" --port 19071
python "$RUNTIME/prepare_native_sim.py" --layout-manifest "$OUT/layouts.json" --manifest-sha256 "$MANIFEST_SHA" --output "$OUT/engineering" --sim-python "$SIM_PY" --port 19071 --layout 0 --phase engineering
~~~
准备流程先用指定策略解释器完成CPU模块来源检查，再生成policy_launch.json/model.json及sim_launch.json/私有SDK；不启动GPU/API。
原生入口同时固定--device_id 3与AppLauncher的--device cuda:3，并显式关闭多卡渲染；不会只设置环境侧device_id。
默认native物理设备3只是候选，须实际环境/renderer验收支持，不能直接推断可用。
prepare命令只写新私有目录，不改共享SDK和reward规则。

## 独立完整工程回合

由已有的合法GPU owner/调度器覆盖策略与仿真整个生命周期，核验当前全8卡声明范围、
真实进程归属、兼容环境和实际renderer。空CUDA快照不能代替资源授权；
本仓库不实现资源锁、不运行synthetic parking。
沿用现有no-THP等兼容入口，不修改共享环境或强改物理配置。

按policy_launch.json的argv启动策略；API凭据仅放策略进程环境ECOT_AUTHORIZED_API_KEY，
不打印、写配置/日志或命令行。READY必须显示EMA10000、effective_eval_ecot=true。
原生子进程取消CUDA_VISIBLE_DEVICES映射，使用已验收的--device_id；
DeepSeek/GitHub凭据不进入原生子进程。

可由既有GPU owner包住以下真实子进程运行：
~~~sh
python tools/record_native_exit.py --plan "$OUT/engineering/sim_launch.json" --plan-sha256 实际planSHA256 --execute-under-accepted-gpu-owner
~~~
该flag是操作者对前提的声明，不是GPU锁。工具只记录自己启动的真实子进程退出，
超时只停止本组，拒绝覆写或静默重跑。exit0不能替代验收。
回合结束后通过同一调度器关闭策略，封存独立policy audit，不再追加下一回合。

## 视频与实际验收

terminal.jsonl的main_camera_action25hz给出源video、frames账本和原SDK目录。
结果必须是该记录目录下的_result.json，不能用其它回合结果替换。
~~~sh
python tools/render_subtask_video.py --source-video 实际video_path --injection-log "$OUT/engineering/injected_text.jsonl" --output-video "$OUT/engineering/main_camera25hz_with_subtask.mp4" --receipt-path "$OUT/engineering/video_decode.json"
python "$RUNTIME/acceptance.py" --output "$OUT/engineering" --policy-audit "$OUT/policies/engineering/actions.jsonl" --layout-manifest "$OUT/layouts.json" --manifest-sha256 "$MANIFEST_SHA" --native-exit "$OUT/engineering/NATIVE_EXIT.json" --result-path 本回合SDK_result.json --decode-receipt "$OUT/engineering/video_decode.json" --phase engineering --layout-id 0 --receipt "$OUT/engineering/ACCEPTANCE.json"
~~~
源和编码视频完整解码。frame8仍显示动作8使用的旧文本，frame9才显示新重规划文本；
不插值/补帧/裁剪，不显示未执行提议。
验收比对全50预测、前8命令、实际native动作/clock、ACK及实际注入文本。
工程验收不要求SDK任务成功，但要求原生完整结束和证据完整；CPU fixture不是实际验收。

## 正式30回合

~~~sh
python "$RUNTIME/prepare_cohort.py" --layout-manifest "$OUT/layouts.json" --manifest-sha256 "$MANIFEST_SHA" --output "$OUT/formal" --sim-python "$SIM_PY" --port 19071 --phase formal --engineering-acceptance "$OUT/engineering/ACCEPTANCE.json"
~~~
准备前重验工程证据SHA，恰好生成30个sim计划，不启动GPU。
cohort完整生成0–29，不接收筛选布局参数。
每个正式case用prepare_policy.py prepare生成对应新策略目录，例如OUT/policies/layout_000，
由同一GPU owner顺序启动，结束关闭策略并封存audit后渲染/验收。
正式验收用--phase formal、--layout-id实际ID。不向已验收policy audit追加其它回合。
reset清空生成器历史和策略noise状态。

收齐30个正式ACCEPTANCE.json后：
~~~sh
python tools/summarize_random30.py --receipts 30个正式验收文件路径 --output "$OUT/RESULTS.json"
~~~
汇总重验源证据SHA，拒绝缺项、重复、工程回合及中断，不填补失败凑30。
原始API请求、图片、视频、权重与运行日志均在Git仓库外保存。

验收收据是可复核的文件/时钟合约，不是加密硬件证明，不能抵抗伪造整套原始证据；
需要操作者保留原始日志和真实进程来源并作语义质检。

## 复查修复后的环境约束

prepare_policy现在强制从训练项目vendor/openpi及该SDK的XPolicyLab导入，并校验实际module.__file__和文件SHA。
指定--sdk-root是必填；不依赖环境中恰好安装的另一份OpenPI/RPC。CPU_IMPORT_CHECK.json必须exit0。
解释器必须在当前主机存在且可执行；不能把H20-8的断开Python符号链接照搬到H20-6。

若主机需要显式CUDA动态库/编译器路径，可给prepare_policy追加--policy-env-json /path/to/accepted/POLICY_ENV.json。
仅允许LD_LIBRARY_PATH、PATH和单一--xla_gpu_cuda_data_dir的XLA_FLAGS，禁止改GPU选择、凭据、Python导入路径或模型精度。
该文件绑定进model.json的SHA。只用于策略进程，禁止全局export后污染原生进程。
H20-6本次已验证的文件位于本次专用环境目录POLICY_ENV.json；不要修改或覆盖其它项目的环境文件。
policy_launch.json中的required_environment_overrides仅应用于该策略子进程；GPU归属仍由外部owner负责。

原生子进程明确将当前runtime、私有SDK、私有XPolicyLab、curobo及SDK携带的IsaacLab源加入PYTHONPATH。
record_native_exit在Popen前重新检查代码/Prompt/候选库、SDK、布局、策略alias及工程验收SHA。
准备完成后若任何这些来源改变，必须停止并重新准备，不能继续使用旧计划。
逐次生成等待/10Hz历史/执行回执与视频帧账本均要求完整，缺少其中一次也不能验收。

GPU核函数或原生初始化通过不能替代完整在线ECoT单回合接入验收。
原生独立初始化只做reset/稳定性/三图检查，不能生成正式30回合准入的ACCEPTANCE.json。
