# Stack Bowls online Subtask adaptation

User-authorized implementation and GitHub PR only. Base commit:
`95188146ce7191aa15d8a23a545707712e267d81` on
`ecot_online_generation_subtask_classify_objects`.
Work only on `ecot_online_generation_subtask_stack_bowls`; preserve other branches.

Target: original pi0.5 Stack Bowls100 full-parameter, Subtask-only, dropout0,
EMA10000 from a15000-update training run. Test native `stack_bowls_random`,
scene seed1, original layouts0..29 exactly once, noise42. A separate full-case
engineering acceptance must precede the30 formal cases and is excluded from30.
Native SDK cap800, original reward/termination,25Hz, predict50/execute8/flow10.
Policy input is current3RGB, State14 and original Instruction +
` Current subtask: ` + original selected Subtask. No Summary or Reasoning.

DeepSeek uses current3 and at most93 strictly-past MAIN frames plus their
measured states and acknowledged execution history. Preserve frozen waits,
physics-clock witnesses and exact text/command ACKs. Private SDK scene data is
for freeze auditing only, never generator or policy input.

The original request covered code/PR only. The user subsequently authorized
finite A800/H20 environment checks and H20 repair on2026-11-14; source inspection
across projects is read-only. Preserve existing workloads. Environment kernels
and standalone native initialization are separate from full online acceptance;
no formal30/64-concurrency rollout is launched by this branch preparation.
CPU fixtures are explicitly fixtures, not runtime acceptance. Never
resume paused ERVLA, create synthetic parking, change shared SDK/environments,
read another project's secrets, or publish weights/credentials/raw requests.

Keep `handoff.md` and `experiment/evaluation.md` current. Historical classify
provenance and contract snapshots remain reference material, not active inputs.
