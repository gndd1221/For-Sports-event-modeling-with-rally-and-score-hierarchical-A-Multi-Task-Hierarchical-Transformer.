# Third-Party Notices

The MIT License in this repository covers the original MT-HTA project code. It does not replace the licenses or terms that apply to third-party code, model designs, datasets, or annotations.

## ShuttleNet (adapted)

`src/models/baseline_shuttlenet_full.py` adapts the ShuttleNet architecture for this repository's single-step multi-task classification interface. The implementation follows the encoder-decoder, player branches, and position-aware gated fusion design published with CoachAI Projects.

- Upstream: <https://github.com/wywyWang/CoachAI-Projects>
- Upstream license: MIT
- Paper: *ShuttleNet: Position-Aware Fusion of Rally Progress and Player Styles for Stroke Forecasting in Badminton*

## PatchTST (adapted)

`src/models/baseline_patchtst.py` adapts PatchTST's channel-independent temporal patching design to discrete racket-sport event classification.

- Upstream: <https://github.com/yuqinie98/PatchTST>
- Upstream license: Apache License 2.0
- Paper: *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers*

## Datasets

Dataset files retain their original ownership, citation requirements, and distribution terms. They are not relicensed by this repository's MIT License. See [`data/README.md`](data/README.md) for provenance and experiment usage.
