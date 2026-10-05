# Analysis outputs

The analysis suite is reproducible and intentionally separate from the normal smoke test.

Run:

```bash
make install-analysis
make analysis
```

The generated directory contains CSV diagnostics and PNG figures. The CI workflow runs the same analysis and uploads the generated directory as the `flowoe-analysis-full-results` artifact.

A verified historical snapshot from CI run #136 is documented in [CI_RUN_136.md](CI_RUN_136.md). It is synthetic-only and is retained for regression reference; it is not real-market evidence.
