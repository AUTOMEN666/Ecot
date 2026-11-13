# Source provenance and external assets

This archive targets only `mainhist93_tok65536_v1`. The runtime preserves the
reference deployed core's file layout and scientific source bytes. The one
test-only adaptation points the prior31-history prompt comparison to
the package's explicit fixture rather than another experiment directory.
Original and packaged hashes and the exact changed-file list are recorded in
[SOURCE_MAP.json](../provenance/SOURCE_MAP.json):48 mapped sources,47 byte-identical
and one test-only path adaptation. `ARCHIVE_ADAPTATIONS.patch` records that diff.

## Known source identities

| Object | Original SHA256 |
|---|---|
| Current complete selector prompt | `f18ee5fa2310d8aba963171c8d8846a6d92e7f05841fd75bcc9782c33a8e0561` |
| Frozen288 catalogue | `97734cb3db4e92fcf62beec96d635896d09d8f23c99c646170bf7736f8b48600` |
| Current model_server.py | `ac89372dfc41ffccf27e3ae23828837cb45232ecc435dce7841488a42400d224` |
| Prior31-history prompt test fixture | `971417b2eac74372c1411cd4e6d8828dc5bb7a07debbda84e34cf022ab633226` |

The two reference cores had34 identical files and one machine-attestation
difference in `export_binding.py`. That difference validates genuine
machine-specific asset/stat evidence; it is not a different prompt or scientific
configuration. Alternate source identity belongs in provenance and is not a
portable receipt that a new machine may reuse.

The11 preserved parent source/support files are a statically identified local
validation closure. Some imports are lazy and include training-contract
definitions. Their inclusion does not start training and does not establish a
complete third-party software or external-asset closure.

`NATIVE_COPY_BASELINE.json`, `ONLINE8_SOURCE_BASELINE.json`, `REMOTE_INVOCATION.md`
and `ONLINE8_REMOTE_INVOCATION_REFERENCE.md` inside the preserved runtime are
historical snapshots. Their original hashes and invocation examples are not
this release's source map or a supported current startup command.

## Observed external environments

[observed_environment.json](../provenance/observed_environment.json) records a
read-only metadata/source observation on2026-11-13. Policy and simulator
interpreters are separate environments; do not merge their package columns into
one installation command.

| Package | Policy environment | Simulator environment |
|---|---|---|
| Python | 3.11.17 | 3.11.17 |
| NumPy | 1.26.4 | 1.26.0 |
| JAX / jaxlib | 0.5.3 / 0.5.3 | Not observed in package metadata |
| Flax | 0.10.2 | Not observed in package metadata |
| Optax | 0.2.4 | Not observed in package metadata |
| orbax-checkpoint | 0.11.13 | Not observed in package metadata |
| OpenPI / openpi-client | 0.1.0 / 0.1.0 | Not observed in package metadata |
| websockets | 15.0.1 | 14.0 |
| Pillow | 11.2.1 | 11.3.0 |
| PyTorch | 2.10.0 | 2.7.0+cu128 |
| Transformers | 4.53.2 | 4.57.6 |
| sentencepiece | 0.2.0 | Not observed in package metadata |

These observed distribution versions are not a complete environment lock or an
instruction to install latest versions. An OpenPI package version does not
identify all local source changes. The actual OpenPI and SDK source copies did
not expose usable Git HEAD or tracked-change metadata; their commits and full
patch sets remain **unknown**. The receipt contains hashes of10 selected OpenPI
files and3 selected SDK files, including license/build metadata, rather than a
complete source-tree attestation. It also records all11 parent source hashes,
which matched the local preserved sources.

The SDK task sample has SHA256
`aef4333c088766375e21a928d53a9a93aaee285fecae41e22385acdc74dd11ac`.
IsaacSim build, all simulator/assets, tokenizer/checkpoint identities and the
complete third-party source closure still require genuine external evidence.
No missing revision is inferred from a directory name or fabricated as a SHA.
The archive-owned source hashes and observed package metadata cannot establish
that a clean GPU deployment with compatible assets was reproduced.

## Distribution boundary

The public archive includes source, full selector prompt, catalogue and test
fixtures. The catalogue records known training lexical descriptions and derived
descriptor/basket pairings; its annotations are not test labels or a guarantee
that every candidate was observed verbatim in training. The archive does not
grant rights to an external dataset or its images.

Checkpoints, normalizers and other model assets; training or expert trajectories;
RGB/state request captures; real videos; raw API responses and private reasoning;
credentials; SSH configuration; environment dumps; live host/PID logs; and entire
experiment report trees are excluded. Historical references inside preserved
source are not instructions to distribute those external assets.

See [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md) for notice and license
scope. No new license is assigned to experiment-specific code in this archive.
