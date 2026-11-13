# Active classify_objects SSR implementation handoff

Branch: ecot_online_generation_subtask_reasoning_and_action_summary_classify_objects.
Active protocol: mainhist93_tok65536_steady_ssr_v1.

The branch now generates public Subtask/Reasoning/Action Summary and actually
injects all three before policy inference. Start at ssr_branch.md, not the
historical steady-only guide. Inherited steady goal/category/basket checks,
96causal images/93pastMAIN, actual8-action ACK history,65536tokens and original
SDK/action/freeze rules remain explicit. Public reasoning is not private thinking.

runtime/ssr_provider.py uses the existing scene observer followed by a joint
allowed-ID/SSR response. Its training style examples are seven exact source-bound
labels; prospective short-term action intent is required by the prompt. Program
checks cannot certify a model's semantic quality or promised future outcomes.

runtime/ssr_model_server.py verifies genuine original noDropout SSR weights and
exact actual training prompt bytes before inference. It reuses unchanged native
camera/action/clock code,predict50/execute8/flow10/noise42/25Hz/native1100. Actual
full-context authorization and action/clock ACK precede durable history commit.
The original training weights/identity are not relabeled as trained-online.

NoDropout attempt2 currently has committed EMA17392 and19566. A checkpoint step
must be selected explicitly with a genuine original binding. The source training
formatter hashes were compared to the actual NAS source; the7golden prompt
comparisons are in SSR_TRAINING_FORMAT_REFERENCE.json.

CPU validation is recorded separately in provenance/ssr_cpu_validation.json;
legacy102-test steady evidence stays historical. Code/CPU checks do not establish
real API/GPU/native acceptance or a higher task success rate. No new GPU, API,
simulator or training job has been launched by this implementation.

Run the CPU suite, tools/verify_archive.py and tools/verify_ssr.py. Real online
acceptance remains a separate isolated run with chosen checkpoint and authorized
resources. Do not modify other branches/main or preempt the independent A800
scheduler. H20 experiments remain paused and synthetic GPU parking retired.
