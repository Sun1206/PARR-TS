# PARR-TS ICDM Remote Experiment Scripts

Remote project root:

```bash
Time-Series-Library
```

Backbone plan:

- Main modern backbones: `TimeMixer`, `TimeXer`, `iTransformer`.
- Anchor baselines: `PatchTST`, `DLinear`.
- Optional foundation-model comparison: `Chronos`, `TimesFM`, `Moirai`, `TimeMoE` in zero-shot or frozen mode only.

First-stage hypothesis checks:

1. Does the PARR score correlate with future residual after a properly trained model?
2. Does PARR reduce degradation under synthetic corruption?
3. Does predictability-binned calibration improve bin-wise coverage compared with global conformal?

The smoke tests are intentionally tiny and only validate code paths.

