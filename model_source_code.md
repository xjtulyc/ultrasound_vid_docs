# UltraDet模型源代码

配置文件中关于模型的部分摘录如下


``configs/RDN-LSTM/BUS-RDN_LSTM.yaml``
```yaml
MODEL:
  WEIGHTS: "pretrained_models/r34.pkl"
  PIXEL_MEAN: [123.675, 116.280, 103.530]
  PIXEL_STD: [58.395, 57.120, 57.375]
  META_ARCHITECTURE: "TemporalRCNN"
  ROI_HEADS:
    NAME: "Res5TemporalROIBoxHeads"
    PROPOSAL_APPEND_GT: True
  ROI_BOX_HEAD:
    INTERVAL_PRE_TEST: 12
    INTERVAL_AFTER_TEST: 3
  RESNETS:
    DEPTH: 34
    STRIDE_IN_1X1: False
    RES2_OUT_CHANNELS: 64
  RPN:
    POST_NMS_TOPK_TEST: 16
  ANCHOR_GENERATOR:
    ASPECT_RATIOS: [[0.33, 0.5, 0.66, 1.0, 2.0, 3.0]]
    SIZES: [[32, 64, 128, 256, 512]]
```

``configs/RDN-LSTM/BUS_BasicConfig_StaticFrame.yaml``
```yaml
_BASE_: "MIXED-RDN_LSTM.yaml"
MODEL:
  ORGAN_SPECIFIC:
    ENABLE: ("cls", "rpn_cls")
  BACKBONE:
    FREEZE_AT: 1
    NAME: "build_resnet_backbone_mix_style"
  RESNETS:
    HALF_CHANNEL: True
  USE_LSTM: True
  PROPOSAL_GENERATOR:
    NAME: "DeFCN"
  DeFCN:
    NMS_THRESH_TEST: 0.7
    NMS_TYPE: "normal"
    FOCAL_LOSS_GAMMA: 2.0
    FOCAL_LOSS_ALPHA: 0.9
    IN_FEATURES: [ "res4" ]
    FPN_STRIDES: [ 16 ]
    NUM_PROPOSALS: 12
  ROI_BOX_HEAD:
    INTERVAL_PRE_TEST: 11
    INTERVAL_AFTER_TEST: 0
```

## Meta-arch

