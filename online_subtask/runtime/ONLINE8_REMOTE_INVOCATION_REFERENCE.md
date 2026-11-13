# Original A EMA5000, every-eight-frame split pilot

This directory is implementation, **not evidence of a new model call, GPU launch or native episode**. Root owns all SSH, credentials, GPU admission, real invocation, termination and reservation restoration. No shared environment or old source is modified.

The original A policy runtime is not currently established on H20. Local records show H20 Cosmos and a separate Torch/ERVLA simulator runtime, while original A EMA5000/OpenPI/JAX weights and environment are on42132. Root's actual inventory found no H20 A checkpoint/JAX. Do not substitute ERVLA or a different action model.

The proposed split is original A EMA5000 **CPU-only policy on42132**, with original native **H20 simulator**, joined by an SSH loopback relay. This avoids GPU preemption and checkpoint/fingerprint migration. It is pending a genuine CPU model-infer witness. CPU runtime speed is a separate engineering result; do not claim realtime25Hz.

## Shared selector contract

`selector_prompt.md` is the sole hard-rule Prompt file. It explicitly says all same-category objects must enter one same basket, different categories never share, initial basket color is free, observed category-to-color mapping stays consistent, and object color does not change category. Full original Instruction remains `Sort the objects by category into the three baskets.` in every request.

`multiview_protocol.py` sends all288 unchanged catalog entries, original current MAIN/left/right RGB, measured State14 and up to4 whole-episode-uniform strictly past MAIN captures. History is sampled from every previous real8-frame capture, without a64-slot cutoff or future images. Previous generated proposals and human/GT reviews are excluded.

The provider requires exact frames0,8,16,..., rejecting duplicates, skipped frames and cross-episode reuse. A successful selector returns only one legal ID. Strict resolution copies exact catalog text. There is no `NO_MATCH`, null, fallback, keyword-driven answer, Summary or Reasoning generation. A transport/schema failure aborts before policy inference. Legal IDs do not imply semantic correctness.

DeepSeek uses the original memory-only `ECOT_AUTHORIZED_API_KEY` entrance and fixed model `deepseek-v4-1-flash-261017`. No secret is serialized by these preparation tools. Codex POST is loopback `http://127.0.0.1:28781/select`:

```json
{"request":{"model":"__PROVIDER__","messages":[]},"proof":{"messages_sha256":"actual shared message SHA"}}
```

Success body must be strictly `{"subtask_id":"Uxxxx"}`. Headers `X-Completion-ID`, `X-Model` and `X-Messages-SHA256` are required; the last must match actual request messages. Provider timeout360s accommodates the bounded local CLI plus cleanup. Raw body, audit headers, actual ID, exact lookup, captures, SHA and request latency are retained.

## 42132 CPU policy preparation and witness

Existing paths from the original run, to be reverified by Root:

- Policy Python `/mnt/xzt/independent_hcz/pi05_robodojo_20261028/sources/RoboDojo/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python`.
- Original A parent `/mnt/xzt/ecot/pi05_classify_objects100_subtask_natural_dropout05_fullparam_20261111/experiments_20261112/A_subtask_online`.
- SDK `/mnt/xzt/robodojo/src/RoboDojo`.
- Binding must be the original verified A5000 export binding, with original export/norm/fingerprint paths; no relocated/fabricated binding is admitted.

Set actual task-specific values for `CODE`, `PARENT`, `BINDING`, `POLICY_PY`, `SDK`, `POLICY_OUTPUT`, and `PORT`; invoke preparation, which does not launch:

```sh
"$POLICY_PY" -B "$CODE/prepare_policy.py" --parent-root "$PARENT" --export-binding "$BINDING" --sdk-root "$SDK" --policy-python "$POLICY_PY" --provider codex --relay-url http://127.0.0.1:28781/select --port "$PORT" --output "$POLICY_OUTPUT"
```

Root then uses `policy_launch.json` argv/cwd and allowlisted nonsecret environment overrides, including `CUDA_VISIBLE_DEVICES=''` and `JAX_PLATFORMS=cpu`. The inherited environment is never written. For DeepSeek, repeat under a distinct output/provider; do not share native state or generated histories.

Before the live server, run one real CPU action-inference witness using a **real saved frame0 shared request** and a **real model-returned single-ID response**, not a hand-authored ID:

```sh
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu "$POLICY_PY" -B "$CODE/native_cpu_smoke.py" --spec "$POLICY_OUTPUT/model.json" --request "$REAL_FRAME0_REQUEST" --selection "$REAL_SELECTION_JSON" --output "$SMOKE_OUTPUT"
```

Required existing PYTHONPATH/openpi/norm environment is from the prepared launch overrides. This smoke performs original A policy loading, original `policy.infer` and original `to_commands`; it does not call the simulator or execute commands. It replays the exact saved model-generated text only for this witness and explicitly reports zero new selector generations/native actions. Output must be eight commands and CPU-only devices. The source/output checks do not certify that the supplied saved selector output is genuine: Root must bind it to the actual raw completion audit.

After a successful witness, Root may start the policy argv with a finite lifetime and process ownership. SSH tunnels bind loopback only: the policy host needs the local Codex `/select` relay, and H20 needs a loopback endpoint for the policy WebSocket. Do not expose public ports or reuse another experiment's server.

## H20 original native simulator preparation

Root must first read the reservation registry and use its documented stop command on H20-5, verify only the registered supervisor/children exit, then verify selected GPU idle and acquire the public UUID lock. This preparation tool itself never touches GPU or reservation state.

Use the existing original SDK tree and native Python3.11 environment whose actual initialization/capture witness is in `ervla_classify_objects100_ab_20261112/reports/training_diagnosis_20261112/ik_continuity_proposal/H20_5_ENVIRONMENT_v2.md`. Reusing that software/SDK does **not** authorize or use ERVLA weights, loss, periodic control change, or its model. Root supplies the existing library/experience/Pythonpath/Kit settings unchanged.

```sh
"$SIM_PY" -B "$CODE/prepare_native_sim.py" --sdk-root "$H20_ORIGINAL_SDK" --sim-python "$SIM_PY" --sim-gpu-uuid "$SIM_GPU_UUID" --port "$TUNNELED_POLICY_PORT" --seed 0 --layout 0 --command-limit 128 --experience "$EXISTING_EXPERIENCE" --output "$SIM_OUTPUT"
```

The tool verifies and freezes all20 original seed0 layout files, prepares only original layout0 via the unchanged private namespace helper, preserves reward rules/assets and installs the existing narrow private RPC600 timeout adapter. It emits `sim_launch.json`. Root must merge required environment overrides into the existing native environment and append the already-required renderer Kit arguments; the tool does not invent a new simulator build/configuration. It must be a distinct private output under `online8_codex_comparison_20261112`.

`pi05_policy/deploy.py` uses the original `replan`/freeze/physics/action recording logic; only event labels are made selector-generic and an explicit128-command engineering boundary is added. Frames0..120 produce16 updates for128 executed actions. A128 engineering stop is written to `engineering_stop.jsonl` with `actual_environment_terminal=false`; do not report it as an SDK terminal or task success. `--command-limit 0` admits a full native episode only when Root deliberately invokes it. Returned native commands remain original50-step prediction/first8 execution and original gripper clipping.

The original action implementation `pi05_policy/native_model.py`, including `encode_obs` and `to_commands`, is byte-identical to the accepted v4 baseline. All freeze/clock/recorder/RPC/native helpers have baseline SHA in `NATIVE_COPY_BASELINE.json`; `deploy.py` is explicitly the bounded-wrapper exception. No action weights, original action input/decoder, normalization, SDK reward/success conditions or training state are changed.

## Required actual closeout

Retain every current capture/request/response/exact lookup, eight-action injection lineage, freeze before/after proof with zero physics steps during selector/policy wait, per-action clock, original video and engineering/native terminal distinction. Root must verify owned process exit/GPU release/UUID lock and restore the registered idle reservation after the finite native test. None of these actual results are supplied by this implementation document.
