# Third-party sources and notices

This source archive integrates with external projects and services. It does not
vendor their complete distributions, checkpoints, simulator assets or datasets.
Existing copyright and license notices in preserved files remain in place.
This notice does not assign a new license to the experiment-specific code or
relicense any third-party work.

| Project/component | Relationship to this archive | Notice/version scope |
|---|---|---|
| OpenPI / π0.5 | External policy implementation, transforms, checkpoint loading and inference dependencies | The observed vendor source has an Apache License2.0 text, preserved [here](online_subtask/provenance/licenses/openpi_vendor/LICENSE). Package metadata reports0.1.0; Git commit and complete local changes are unknown. |
| RoboDojo and XPolicyLab integration | External native task, observations/actions, cameras and recording interfaces | The observed SDK has an MIT License, copyright2026 RoboDojo Contributors, preserved [here](online_subtask/provenance/licenses/robodojo_sdk/LICENSE). Full SDK/task/assets are not included; Git commit remains unknown. This observation does not assign that license to every external integration/asset. |
| NVIDIA IsaacSim and simulator assets | External simulation environment required by the native integration | Not distributed here; use under the applicable vendor and asset terms. Exact deployed build is awaiting final inventory. |
| JAX/jaxlib, Flax, Optax, Orbax | External array/model/training-contract/checkpoint ecosystem | Install compatible verified versions and retain their own distribution notices; this archive is not an environment lock. |
| NumPy | CPU array/observation conversion and tests | External dependency; retain its distribution notices. |
| Pillow | Actual-text video generation and CPU video fixture | External dependency required by the video fixture and overlay tool; retain its distribution notices. |
| WebSocket client/server package | External policy RPC transport | Observed openpi-client0.1.0, policy websockets15.0.1 and simulator websockets14.0; retain each actual distribution's notices. |
| DeepSeek model served through the configured API provider | External model service used by the online selector | API/model access is subject to the operator's authorization and service terms; no model weights or raw responses are distributed. |
| Tokenizer, checkpoint and dataset assets | External runtime/training provenance required by strict binding checks | Not included or licensed by this archive; obtain genuine compatible assets and honor their applicable terms. |

The catalogue includes experiment annotation text preserved original Stack Bowls training Subtasks.
Its presence does not license an underlying external dataset or authorize
redistribution of images, trajectories or trained weights. Historical source
paths and hashes describe provenance, not redistribution permission.

The two copied license texts were collected from the actual external source
copies. Their SHA256 values are respectively
`43070e2d4e532684de521b885f385d0841030efa2b1a20bafb76133a5e1379c1`
and `4f2b6958d25115c1bc461044e022889852d9e3a9bb854ab581c416514b9984a0`.
They describe those upstream works, not a new repository-wide license.

Project names alone do not establish an exact source revision or license for a
particular build. [Observed environment evidence](online_subtask/reference/classify_objects/observed_environment.json)
records installed versions and selected source hashes; it cannot recover the
unknown upstream Git commits or attest every third-party file. Missing revisions
remain unpinned rather than guessed. No fully locked third-party reproduction
environment is claimed. See [source provenance](online_subtask/docs/provenance.md).
