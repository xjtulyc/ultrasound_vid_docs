# UltraDet模型源代码

在d2中调用某一个模块，首先需要注册模块，之后根据与模块同名的字符串来调用该模块。配置文件中关于模型的部分摘录如下


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

## 1. META_ARCHITECTURE

UltraDet支持两种meta-arch：``TemporalRCNN``和``TemporalRetinaNet``，可以在``ultrasound_vid/modeling/meta_arch``中找到对应的代码。
<details>
<summary>TemporalRCNN</summary>

```python
import logging
from collections import deque
from itertools import chain

import torch
from detectron2.modeling import build_roi_heads, build_backbone, META_ARCH_REGISTRY
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.utils.logger import log_first_n
from torch import nn
from ultrasound_vid.utils import imagelist_from_tensors
from ultrasound_vid.modeling.layers import LSTMSampleModule, ContextFusion


@META_ARCH_REGISTRY.register()
class TemporalRCNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(
            cfg, self.backbone.output_shape()
        )
        self.num_frames = cfg.INPUT.TRAIN_FRAME_SAMPLER.NUM_OUT_FRAMES
        self.roi_heads = build_roi_heads(cfg, self.backbone.output_shape())
        self.frame_delay = getattr(self.roi_heads, "interval_after_test", None)

        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        self.register_buffer(
            "pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1)
        )
        self.register_buffer(
            "pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1)
        )

        self.use_lstm = cfg.MODEL.USE_LSTM
        if self.use_lstm:
            self.sample_module = LSTMSampleModule(cfg)

        buffer_length = (
            cfg.MODEL.ROI_BOX_HEAD.INTERVAL_PRE_TEST
            + cfg.MODEL.ROI_BOX_HEAD.INTERVAL_AFTER_TEST
            + 1
        )
        self.feature_buffer = deque(maxlen=buffer_length)

        self.to(self.device)

        # organ-specific loss factor for data balance
        self.organ_loss_factor = {
            "thyroid": cfg.MODEL.ORGAN_SPECIFIC.THYROID_LOSS_WEIGHT,
            "breast": cfg.MODEL.ORGAN_SPECIFIC.BREAST_LOSS_WEIGHT,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, inputs):
        if self.training:
            return self.forward_train(inputs)
        else:
            return self.forward_infer(inputs)

    def forward_train(self, batched_inputs):
        batchsize = len(batched_inputs)
        batched_inputs = list(chain.from_iterable(batched_inputs))
        assert len(batched_inputs) == batchsize * self.num_frames

        # organ switching
        dataset_name = batched_inputs[0]["dataset"]
        organ = dataset_name.split("@")[0].split("_")[0]
        assert organ == "breast" or organ == "thyroid"
        self.roi_heads.box_predictor.switch(organ)
        self.proposal_generator.head.switch(organ)

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            raise AttributeError("Failed to get 'instances' from training image.")

        features = self.backbone(images.tensor)

        if self.use_lstm:
            assert len(features) == 1, f"Support only one level features now!"
            k = list(features.keys())[0]
            sampled_feat = self.sample_module(features[k])
            features[k] = sampled_feat

        proposals, proposal_losses = self.proposal_generator(
            images, features, gt_instances
        )

        _, detector_losses = self.roi_heads(
            batched_inputs, features, proposals, gt_instances
        )

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return {k: v * self.organ_loss_factor[organ] for k, v in losses.items()}

    def forward_infer(self, frame_input):
        # for test mode
        assert not self.training
        # organ switching
        dataset_name = frame_input["dataset"]
        organ = dataset_name.split("@")[0].split("_")[0]
        assert organ == "breast" or organ == "thyroid"
        self.roi_heads.box_predictor.switch(organ)
        self.proposal_generator.head.switch(organ)

        images = self.preprocess_image([frame_input])
        features = self.backbone(images.tensor)

        if self.use_lstm:
            assert len(features) == 1, f"Support only one level features now!"
            k = list(features.keys())[0]
            sampled_feat = self.sample_module.step(features[k])
            features[k] = sampled_feat

        proposals, _ = self.proposal_generator(images, features, None)

        results, _ = self.roi_heads(images, features, proposals)

        if results is None:
            return None
        else:
            assert len(results) == 1
            r = results[0]
            return self.postprocess(r, frame_input, images.image_sizes[0])

    def reset(self):
        """
        Reset caches to inference on a new video.
        """
        # self.cur_mean_feat = None
        if self.use_lstm:
            self.sample_module.reset()
        self.feature_buffer.clear()
        reset_op = getattr(self.roi_heads, "reset", None)
        if callable(reset_op):
            reset_op()
        else:
            log_first_n(logging.WARN, 'Roi Heads doesnt have function "reset"')

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device, non_blocking=True) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = imagelist_from_tensors(images, self.backbone.size_divisibility)
        return images

    def postprocess(self, instances, frame_input, image_size):
        """
        Rescale the output instance to the target size.
        """
        height = frame_input.get("height", image_size[0])
        width = frame_input.get("width", image_size[1])
        return detector_postprocess(instances, height, width)


@META_ARCH_REGISTRY.register()
class TemporalProposalNetwork(TemporalRCNN):
    def forward_infer(self, frame_input):
        # for test mode
        assert not self.training
        images = self.preprocess_image([frame_input])
        features = self.backbone(images.tensor)
        proposals, _ = self.proposal_generator(images, features, None)
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            proposals, [frame_input], images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append(r)
        assert (
            len(processed_results) == 1
        ), "We only support proposals from 1 image for now"
        return processed_results[0]
```
</details>

<details>
<summary>TemporalRetinaNet</summary>

```python
from collections import deque
from itertools import chain
import math
from typing import List
from torch import nn

import torch
from detectron2.layers import ShapeSpec
from detectron2.modeling import (
    META_ARCH_REGISTRY,
    detector_postprocess,
    build_anchor_generator,
)
from detectron2.modeling import RetinaNet
from ultrasound_vid.utils import imagelist_from_tensors


def permute_to_N_HWA_K(tensor, K):
    """
    Transpose/reshape a tensor from (N, (Ai x K), H, W) to (N, (HxWxAi), K)
    """
    assert tensor.dim() == 4, tensor.shape
    N, _, H, W = tensor.shape
    tensor = tensor.view(N, -1, K, H, W)
    tensor = tensor.permute(0, 3, 4, 1, 2)
    tensor = tensor.reshape(N, -1, K)  # Size=(N,HWA,K)
    return tensor


@META_ARCH_REGISTRY.register()
class TemporalRetinaNet(RetinaNet):
    """
    Implement RetinaNet in :paper:`RetinaNet`.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.inference_pre_test = cfg.MODEL.RETINANET.INTERVAL_PRE_TEST
        self.inference_after_test = cfg.MODEL.RETINANET.INTERVAL_AFTER_TEST

        # regenerate head
        backbone_shape = self.backbone.output_shape()
        feature_shapes = [backbone_shape[f] for f in self.in_features]
        self.head = TemporalRetinaHead(cfg, feature_shapes)

        # frame buffer for inference usage
        buffer_length = self.inference_pre_test + self.inference_after_test + 1
        self.feature_buffer = deque(maxlen=buffer_length)

    def forward(self, inputs):
        if self.training:
            return self.forward_train(inputs)
        else:
            return self.forward_infer(inputs)

    def forward_train(self, batched_inputs):
        batched_inputs = list(chain.from_iterable(batched_inputs))
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        features = [features[f] for f in self.in_features]

        # concate current features and mean feature
        B = features[0].shape[0]
        mean_feature = [f.mean(0, keepdim=True) for f in features]
        mean_features = [f.repeat(B, 1, 1, 1) for f in mean_feature]

        anchors = self.anchor_generator(features)
        pred_logits, pred_anchor_deltas = self.head(features, mean_features)
        # Transpose the Hi*Wi*A dimension to the middle:
        pred_logits = [permute_to_N_HWA_K(x, self.num_classes) for x in pred_logits]
        pred_anchor_deltas = [permute_to_N_HWA_K(x, 4) for x in pred_anchor_deltas]

        assert (
            "instances" in batched_inputs[0]
        ), "Instance annotations are missing in training!"
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

        gt_labels, gt_boxes = self.label_anchors(anchors, gt_instances)
        losses = self.losses(
            anchors, pred_logits, gt_labels, pred_anchor_deltas, gt_boxes
        )

        return losses

    def forward_infer(self, frame_input):
        assert not self.training
        images = self.preprocess_image([frame_input])
        features = self.backbone(images.tensor)
        features = [features[f] for f in self.in_features]
        self.feature_buffer.append(features)

        if len(self.feature_buffer) <= self.inference_after_test:
            return None

        buffered_features = list(self.feature_buffer)
        mean_features = [
            torch.cat(f).mean(0, keepdim=True) for f in zip(*buffered_features)
        ]
        features = buffered_features[-self.inference_after_test - 1]

        anchors = self.anchor_generator(features)
        pred_logits, pred_anchor_deltas = self.head(features, mean_features)
        # Transpose the Hi*Wi*A dimension to the middle:
        pred_logits = [permute_to_N_HWA_K(x, self.num_classes) for x in pred_logits]
        pred_anchor_deltas = [permute_to_N_HWA_K(x, 4) for x in pred_anchor_deltas]

        results = self.inference(
            anchors, pred_logits, pred_anchor_deltas, images.image_sizes
        )
        frame_result = results[0]
        image_size = images.image_sizes[0]
        height = frame_input.get("height", image_size[0])
        width = frame_input.get("width", image_size[1])
        ret = detector_postprocess(frame_result, height, width)
        return ret

    def reset(self):
        """
        Reset caches to inference on a new video.
        """
        self.feature_buffer.clear()

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = imagelist_from_tensors(images, self.backbone.size_divisibility)
        return images


class TemporalRetinaHead(nn.Module):
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()
        # fmt: off
        in_channels      = input_shape[0].channels
        num_classes      = cfg.MODEL.RETINANET.NUM_CLASSES
        num_convs        = cfg.MODEL.RETINANET.NUM_CONVS
        prior_prob       = cfg.MODEL.RETINANET.PRIOR_PROB
        num_anchors      = build_anchor_generator(cfg, input_shape).num_cell_anchors
        # fmt: on
        assert (
            len(set(num_anchors)) == 1
        ), "Using different number of anchors between levels is not currently supported!"
        num_anchors = num_anchors[0]

        cls_subnet = []
        bbox_subnet = []
        # to merge mean feature and current feature
        cls_subnet.append(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, stride=1, padding=1)
        )
        cls_subnet.append(nn.ReLU())
        bbox_subnet.append(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, stride=1, padding=1)
        )
        bbox_subnet.append(nn.ReLU())

        for _ in range(num_convs - 1):
            cls_subnet.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            cls_subnet.append(nn.ReLU())
            bbox_subnet.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            bbox_subnet.append(nn.ReLU())

        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.bbox_subnet = nn.Sequential(*bbox_subnet)
        self.cls_score = nn.Conv2d(
            in_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1
        )
        self.bbox_pred = nn.Conv2d(
            in_channels, num_anchors * 4, kernel_size=3, stride=1, padding=1
        )

        # Initialization
        for modules in [
            self.cls_subnet,
            self.bbox_subnet,
            self.cls_score,
            self.bbox_pred,
        ]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)

        # Use prior in model initialization to improve stability
        bias_value = -(math.log((1 - prior_prob) / prior_prob))
        torch.nn.init.constant_(self.cls_score.bias, bias_value)

    def forward(self, features, mean_features):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns:
            logits (list[Tensor]): #lvl tensors, each has shape (N, AxK, Hi, Wi).
                The tensor predicts the classification probability
                at each spatial position for each of the A anchors and K object
                classes.
            bbox_reg (list[Tensor]): #lvl tensors, each has shape (N, Ax4, Hi, Wi).
                The tensor predicts 4-vector (dx,dy,dw,dh) box
                regression values for every anchor. These values are the
                relative offset between the anchor and the ground truth box.
        """
        features = [torch.cat([f, mf], dim=1) for f, mf in zip(features, mean_features)]

        logits = []
        bbox_reg = []
        for feature in features:
            logits.append(self.cls_score(self.cls_subnet(feature)))
            bbox_reg.append(self.bbox_pred(self.bbox_subnet(feature)))
        return logits, bbox_reg

```
</details>

## 2. BACKBONE

UltraDet的backbone可以描述为**ResNet34+FPN提取单帧特征，ConvLSTM结合多帧信息**。在代码上``ultrasound_vid/modeling/backbone/native_resnet.py``和``ultrasound_vid/modeling/backbone/resnet_mix_style.py``分别是纯粹的resnet和resnet+mix-style。

<details>
<summary>native_resnet</summary>

```python
from torchvision.models.resnet import (
    ResNet,
    BasicBlock,
    Bottleneck,
    model_urls,
)
from detectron2.modeling import ShapeSpec
from torch.hub import load_state_dict_from_url


class NativeResNet(ResNet):
    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return {"res4": x}

    @property
    def size_divisibility(self):
        return 0

    def output_shape(self):
        return {"res4": ShapeSpec(channels=256, height=None, width=None, stride=16)}


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = NativeResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained: bool = False, progress: bool = True, **kwargs):
    return _resnet("resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


def resnet34(pretrained: bool = False, progress: bool = True, **kwargs):
    return _resnet("resnet34", BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet50(pretrained: bool = False, progress: bool = True, **kwargs):
    return _resnet("resnet50", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)

```
</details>

<details>
<summary>resnet_mix_style</summary>

```python
# Copyright (c) Facebook, Inc. and its affiliates.
import torch
from torch.distributions import Beta
from torch.distributed import all_gather
from detectron2.modeling.backbone import BACKBONE_REGISTRY
from detectron2.modeling.backbone.resnet import (
    BasicBlock,
    BasicStem,
    BottleneckBlock,
    DeformBottleneckBlock,
    ResNet,
)


class MixStyle(torch.nn.Module):
    def __init__(self, alpha=0.3):
        super(MixStyle, self).__init__()
        self.alpha = alpha
        self.process_group = torch.distributed.group.WORLD
        self.world_size = torch.distributed.get_world_size(self.process_group)

    def forward(self, x):
        if not self.training:
            return x
        mu = x.mean(dim=[0, 2, 3], keepdim=True)
        var = x.var(dim=[0, 2, 3], keepdim=True)
        sig = (var + 1e-6).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x - mu) / sig
        lmda = Beta(self.alpha, self.alpha).sample((1,)).item()
        combined = torch.cat([mu, sig], dim=0)
        combined_list = [torch.empty_like(combined) for _ in range(self.world_size)]
        all_gather(combined_list, combined, self.process_group, async_op=False)
        idx = torch.randint(self.world_size, (1,)).item()
        mu_2, sig_2 = torch.split(combined_list[idx], [1, 1], dim=0)
        mu_mix = mu * lmda + mu_2 * (1 - lmda)
        sig_mix = sig * lmda + sig_2 * (1 - lmda)
        return x_normed * sig_mix + mu_mix


class ResNetMixStyle(ResNet):
    def __init__(self, *args, **kwargs):
        super(ResNetMixStyle, self).__init__(*args, **kwargs)
        self.mix = MixStyle()

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.

        Returns:
            dict[str->Tensor]
        """
        outputs = {}
        x = self.stem(x)
        x = self.mix(x)
        for stage, name in zip(self.stages, self.stage_names):
            x = stage(x)
            if name == "res2" or name == "res3":
                x = self.mix(x)
            if name in self._out_features:
                outputs[name] = x
        return outputs


@BACKBONE_REGISTRY.register()
def build_resnet_backbone_mix_style(cfg, input_shape):
    """
    Create a ResNet instance from config.

    Returns:
        ResNet: a :class:`ResNet` instance.
    """
    # need registration of new blocks/stems?
    norm = cfg.MODEL.RESNETS.NORM
    stem = BasicStem(
        in_channels=input_shape.channels,
        out_channels=cfg.MODEL.RESNETS.STEM_OUT_CHANNELS,
        norm=norm,
    )

    # fmt: off
    freeze_at = cfg.MODEL.BACKBONE.FREEZE_AT
    out_features = cfg.MODEL.RESNETS.OUT_FEATURES
    depth = cfg.MODEL.RESNETS.DEPTH
    num_groups = cfg.MODEL.RESNETS.NUM_GROUPS
    width_per_group = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
    bottleneck_channels = num_groups * width_per_group
    in_channels = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
    out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    stride_in_1x1 = cfg.MODEL.RESNETS.STRIDE_IN_1X1
    res5_dilation = cfg.MODEL.RESNETS.RES5_DILATION
    deform_on_per_stage = cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE
    deform_modulated = cfg.MODEL.RESNETS.DEFORM_MODULATED
    deform_num_groups = cfg.MODEL.RESNETS.DEFORM_NUM_GROUPS
    # fmt: on
    assert res5_dilation in {1, 2}, "res5_dilation cannot be {}.".format(res5_dilation)

    num_blocks_per_stage = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
    }[depth]

    if depth in [18, 34]:
        assert (
            out_channels == 64
        ), "Must set MODEL.RESNETS.RES2_OUT_CHANNELS = 64 for R18/R34"
        assert not any(
            deform_on_per_stage
        ), "MODEL.RESNETS.DEFORM_ON_PER_STAGE unsupported for R18/R34"
        assert (
            res5_dilation == 1
        ), "Must set MODEL.RESNETS.RES5_DILATION = 1 for R18/R34"
        assert num_groups == 1, "Must set MODEL.RESNETS.NUM_GROUPS = 1 for R18/R34"

    stages = []

    # Avoid creating variables without gradients
    # It consumes extra memory and may cause allreduce to fail
    out_stage_idx = [
        {"res2": 2, "res3": 3, "res4": 4, "res5": 5}[f] for f in out_features
    ]
    max_stage_idx = max(out_stage_idx)
    for idx, stage_idx in enumerate(range(2, max_stage_idx + 1)):
        dilation = res5_dilation if stage_idx == 5 else 1
        first_stride = 1 if idx == 0 or (stage_idx == 5 and dilation == 2) else 2
        stage_kargs = {
            "num_blocks": num_blocks_per_stage[idx],
            "stride_per_block": [first_stride] + [1] * (num_blocks_per_stage[idx] - 1),
            "in_channels": in_channels,
            "out_channels": out_channels,
            "norm": norm,
        }
        # Use BasicBlock for R18 and R34.
        if depth in [18, 34]:
            stage_kargs["block_class"] = BasicBlock
        else:
            stage_kargs["bottleneck_channels"] = bottleneck_channels
            stage_kargs["stride_in_1x1"] = stride_in_1x1
            stage_kargs["dilation"] = dilation
            stage_kargs["num_groups"] = num_groups
            if deform_on_per_stage[idx]:
                stage_kargs["block_class"] = DeformBottleneckBlock
                stage_kargs["deform_modulated"] = deform_modulated
                stage_kargs["deform_num_groups"] = deform_num_groups
            else:
                stage_kargs["block_class"] = BottleneckBlock
        blocks = ResNetMixStyle.make_stage(**stage_kargs)
        in_channels = out_channels
        out_channels *= 2
        bottleneck_channels *= 2
        stages.append(blocks)
    return ResNetMixStyle(stem, stages, out_features=out_features).freeze(freeze_at)

```
</details>

## 3. PROPOSAL_GENERATOR

UltraDet使用DeFCN的策略（POTO）来实现训练时的one-to-one assignment。代码位于``ultrasound_vid/modeling/proposal_generator/defcn.py``

<details>
<summary>defcn</summary>

```python
import logging
import math
from typing import List
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from torch import nn

from fvcore.nn import sigmoid_focal_loss_jit

from detectron2.layers import ShapeSpec, cat, batched_nms
from detectron2.modeling.meta_arch.retinanet import permute_to_N_HWA_K
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY

from ultrasound_vid.modeling.anchor_generator import ShiftGenerator
from ultrasound_vid.modeling.box_regression import Shift2BoxTransform
from ultrasound_vid.modeling.losses import iou_loss
from ultrasound_vid.utils import comm


def focal_loss(
    probs,
    targets,
    alpha: float = -1,
    gamma: float = 2,
    reduction: str = "none",
):
    ce_loss = F.binary_cross_entropy(probs, targets, reduction="none")
    p_t = probs * targets + (1 - probs) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


focal_loss_jit = torch.jit.script(focal_loss)  # type: torch.jit.ScriptModule


def permute_all_cls_and_box_to_N_HWA_K_and_concat(
    box_cls, box_delta, box_center, num_classes=80
):
    """
    Rearrange the tensor layout from the network output, i.e.:
    list[Tensor]: #lvl tensors of shape (N, A x K, Hi, Wi)
    to per-image predictions, i.e.:
    Tensor: of shape (N x sum(Hi x Wi x A), K)
    """
    # for each feature level, permute the outputs to make them be in the
    # same format as the labels. Note that the labels are computed for
    # all feature levels concatenated, so we keep the same representation
    # for the objectness, the box_delta and the centerness
    box_cls_flattened = [permute_to_N_HWA_K(x, num_classes) for x in box_cls]
    box_delta_flattened = [permute_to_N_HWA_K(x, 4) for x in box_delta]
    box_center_flattened = [permute_to_N_HWA_K(x, 1) for x in box_center]
    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    box_cls = cat(box_cls_flattened, dim=1).reshape(-1, num_classes)
    box_delta = cat(box_delta_flattened, dim=1).reshape(-1, 4)
    box_center = cat(box_center_flattened, dim=1).reshape(-1, 1)
    return box_cls, box_delta, box_center


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


@PROPOSAL_GENERATOR_REGISTRY.register()
class DeFCN(nn.Module):
    def __init__(self, cfg, input_shape):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        # fmt: off
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.in_features = cfg.MODEL.DeFCN.IN_FEATURES
        self.fpn_strides = cfg.MODEL.DeFCN.FPN_STRIDES
        # Loss parameters:
        self.focal_loss_alpha = cfg.MODEL.DeFCN.FOCAL_LOSS_ALPHA
        self.focal_loss_gamma = cfg.MODEL.DeFCN.FOCAL_LOSS_GAMMA
        self.iou_loss_type = cfg.MODEL.DeFCN.IOU_LOSS_TYPE
        self.reg_weight = cfg.MODEL.DeFCN.REG_WEIGHT
        # Inference parameters:
        self.score_threshold = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
        self.nms_threshold = cfg.MODEL.DeFCN.NMS_THRESH_TEST
        self.nms_type = cfg.MODEL.DeFCN.NMS_TYPE
        self.pre_nms_topk = cfg.MODEL.DeFCN.PRE_NMS_TOPK
        self.post_nms_topk = cfg.MODEL.DeFCN.NUM_PROPOSALS

        # fmt: on

        feature_shapes = [input_shape[f] for f in self.in_features]
        self.in_channels = feature_shapes[0].channels
        self.head = FCOSHead(cfg, feature_shapes)
        self.shift_generator = ShiftGenerator(cfg, feature_shapes)

        # Matching and loss
        self.shift2box_transform = Shift2BoxTransform(
            weights=cfg.MODEL.DeFCN.BBOX_REG_WEIGHTS
        )
        self.poto_alpha = cfg.MODEL.POTO.ALPHA
        self.center_sampling_radius = cfg.MODEL.POTO.CENTER_SAMPLING_RADIUS
        self.poto_aux_topk = cfg.MODEL.POTO.AUX_TOPK

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def forward(self, images, features, gt_instances):
        if self.training:
            return self.forward_train(images, features, gt_instances)
        else:
            return self.forward_infer(images, features)

    def forward_train(self, images, features, gt_instances):
        """
        Params:
            images:
            features:
            gt_instances:
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        features = [features[f] for f in self.in_features]
        box_cls, box_delta, box_filter = self.head(features)
        shifts = self.shift_generator(features)
        gt_classes, gt_shifts_reg_deltas = self.get_ground_truth(
            shifts, gt_instances, box_cls, box_delta, box_filter
        )
        losses = self.losses(
            gt_classes, gt_shifts_reg_deltas, box_cls, box_delta, box_filter
        )
        gt_classes_aux = self.get_aux_ground_truth(
            shifts, gt_instances, box_cls, box_delta
        )
        losses.update(self.aux_losses(gt_classes_aux, box_cls))
        with torch.no_grad():
            results = self.inference(
                box_cls,
                box_delta,
                box_filter,
                features,
                shifts,
                images,
                gt_classes,
                gt_shifts_reg_deltas,
            )
        results = [x.to(self.device) for x in results]
        return results, losses

    def forward_infer(self, images, features):
        assert not self.training
        features = [features[f] for f in self.in_features]
        box_cls, box_delta, box_filter = self.head(features)
        shifts = self.shift_generator(features)
        results = self.inference(
            box_cls, box_delta, box_filter, features, shifts, images, None, None
        )
        frame_result = results[0]
        height, width = images[0].shape[-2:]
        processed_result = detector_postprocess(frame_result, height, width)
        return [processed_result], {}

    def losses(
        self,
        gt_classes,
        gt_shifts_deltas,
        pred_class_logits,
        pred_shift_deltas,
        pred_filtering,
    ):
        """
        Args:
            For `gt_classes` and `gt_shifts_deltas` parameters, see
                :meth:`FCOS.get_ground_truth`.
            Their shapes are (N, R) and (N, R, 4), respectively, where R is
            the total number of shifts across levels, i.e. sum(Hi x Wi)
            For `pred_class_logits`, `pred_shift_deltas` and `pred_fitering`, see
                :meth:`FCOSHead.forward`.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a scalar tensor
                storing the loss. Used during training only. The dict keys are:
                "loss_cls" and "loss_box_reg"
        """
        (
            pred_class_logits,
            pred_shift_deltas,
            pred_filtering,
        ) = permute_all_cls_and_box_to_N_HWA_K_and_concat(
            pred_class_logits, pred_shift_deltas, pred_filtering, self.num_classes
        )  # Shapes: (N x R, K) and (N x R, 4), respectively.

        gt_classes = gt_classes.flatten()
        gt_shifts_deltas = gt_shifts_deltas.view(-1, 4)

        valid_idxs = gt_classes >= 0
        foreground_idxs = (gt_classes >= 0) & (gt_classes != self.num_classes)
        num_foreground = foreground_idxs.sum()

        gt_classes_target = torch.zeros_like(pred_class_logits)
        gt_classes_target[foreground_idxs, gt_classes[foreground_idxs]] = 1
        pred_class_logits = pred_class_logits.sigmoid() * pred_filtering.sigmoid()
        num_foreground = comm.all_reduce(num_foreground) / float(comm.get_world_size())

        # logits loss
        loss_cls = focal_loss_jit(
            pred_class_logits[valid_idxs],
            gt_classes_target[valid_idxs],
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        ) / max(1.0, num_foreground)

        # regression loss
        loss_box_reg = (
            iou_loss(
                pred_shift_deltas[foreground_idxs],
                gt_shifts_deltas[foreground_idxs],
                box_mode="ltrb",
                loss_type=self.iou_loss_type,
                reduction="sum",
            )
            / max(1.0, num_foreground)
            * self.reg_weight
        )

        return {
            "loss_rpn_cls": loss_cls,
            "loss_rpn_reg": loss_box_reg,
        }

    def aux_losses(self, gt_classes, pred_class_logits):
        pred_class_logits = cat(
            [permute_to_N_HWA_K(x, self.num_classes) for x in pred_class_logits], dim=1
        ).view(-1, self.num_classes)

        gt_classes = gt_classes.flatten()

        valid_idxs = gt_classes >= 0
        foreground_idxs = (gt_classes >= 0) & (gt_classes != self.num_classes)
        num_foreground = foreground_idxs.sum()

        gt_classes_target = torch.zeros_like(pred_class_logits)
        gt_classes_target[foreground_idxs, gt_classes[foreground_idxs]] = 1

        num_foreground = comm.all_reduce(num_foreground) / float(comm.get_world_size())

        # logits loss
        loss_cls_aux = sigmoid_focal_loss_jit(
            pred_class_logits[valid_idxs],
            gt_classes_target[valid_idxs],
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        ) / max(1.0, num_foreground)

        return {"loss_rpn_aux": loss_cls_aux}

    @torch.no_grad()
    def get_ground_truth(self, shifts, targets, box_cls, box_delta, box_filter):
        """
        Args:
            shifts (list[list[Tensor]]): a list of N=#image elements. Each is a
                list of #feature level tensors. The tensors contains shifts of
                this image on the specific feature level.
            targets (list[Instances]): a list of N `Instances`s. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.  Specify `targets` during training only.
        Returns:
            gt_classes (Tensor):
                An integer tensor of shape (N, R) storing ground-truth
                labels for each shift.
                R is the total number of shifts, i.e. the sum of Hi x Wi for all levels.
                Shifts in the valid boxes are assigned their corresponding label in the
                [0, K-1] range. Shifts in the background are assigned the label "K".
                Shifts in the ignore areas are assigned a label "-1", i.e. ignore.
            gt_shifts_deltas (Tensor):
                Shape (N, R, 4).
                The last dimension represents ground-truth shift2box transform
                targets (dl, dt, dr, db) that map each shift to its matched ground-truth box.
                The values in the tensor are meaningful only when the corresponding
                shift is labeled as foreground.
        """
        gt_classes = []
        gt_shifts_deltas = []

        box_cls = torch.cat(
            [permute_to_N_HWA_K(x, self.num_classes) for x in box_cls], dim=1
        )
        box_delta = torch.cat([permute_to_N_HWA_K(x, 4) for x in box_delta], dim=1)
        box_filter = torch.cat([permute_to_N_HWA_K(x, 1) for x in box_filter], dim=1)
        box_cls = box_cls.sigmoid_() * box_filter.sigmoid_()

        for (
            shifts_per_image,
            targets_per_image,
            box_cls_per_image,
            box_delta_per_image,
        ) in zip(shifts, targets, box_cls, box_delta):
            shifts_over_all_feature_maps = torch.cat(shifts_per_image, dim=0)

            gt_boxes = targets_per_image.gt_boxes
            gt_classes_i = shifts_over_all_feature_maps.new_full(
                (len(shifts_over_all_feature_maps),), self.num_classes, dtype=torch.long
            )
            gt_shifts_reg_deltas_i = shifts_over_all_feature_maps.new_zeros(
                len(shifts_over_all_feature_maps), 4
            )
            if len(gt_boxes) == 0:
                gt_classes.append(gt_classes_i)
                gt_shifts_deltas.append(gt_shifts_reg_deltas_i)
                continue

            prob = box_cls_per_image[:, targets_per_image.gt_classes].t()
            boxes = self.shift2box_transform.apply_deltas(
                box_delta_per_image, shifts_over_all_feature_maps
            )
            iou = pairwise_iou(gt_boxes, Boxes(boxes))
            quality = prob ** (1 - self.poto_alpha) * iou**self.poto_alpha

            deltas = self.shift2box_transform.get_deltas(
                shifts_over_all_feature_maps, gt_boxes.tensor.unsqueeze(1)
            )

            if self.center_sampling_radius > 0:
                centers = gt_boxes.get_centers()
                is_in_boxes = []
                for stride, shifts_i in zip(self.fpn_strides, shifts_per_image):
                    radius = stride * self.center_sampling_radius
                    center_boxes = torch.cat(
                        (
                            torch.max(centers - radius, gt_boxes.tensor[:, :2]),
                            torch.min(centers + radius, gt_boxes.tensor[:, 2:]),
                        ),
                        dim=-1,
                    )
                    center_deltas = self.shift2box_transform.get_deltas(
                        shifts_i, center_boxes.unsqueeze(1)
                    )
                    is_in_boxes.append(center_deltas.min(dim=-1).values > 0)
                is_in_boxes = torch.cat(is_in_boxes, dim=1)
            else:
                # no center sampling, it will use all the locations within a ground-truth box
                is_in_boxes = deltas.min(dim=-1).values > 0

            quality[~is_in_boxes] = -1
            # row_ind, col_ind
            gt_idxs, shift_idxs = linear_sum_assignment(
                quality.cpu().numpy(), maximize=True
            )

            assert len(targets_per_image) > 0
            # ground truth classes
            gt_classes_i[shift_idxs] = targets_per_image.gt_classes[gt_idxs]
            # ground truth box regression
            gt_shifts_reg_deltas_i[shift_idxs] = self.shift2box_transform.get_deltas(
                shifts_over_all_feature_maps[shift_idxs], gt_boxes[gt_idxs].tensor
            )

            gt_classes.append(gt_classes_i)
            gt_shifts_deltas.append(gt_shifts_reg_deltas_i)

        gt_classes = torch.stack(gt_classes).type(torch.LongTensor).to(self.device)
        gt_shifts_deltas = torch.stack(gt_shifts_deltas).to(self.device)

        return gt_classes, gt_shifts_deltas

    @torch.no_grad()
    def get_aux_ground_truth(self, shifts, targets, box_cls, box_delta):
        """
        Args:
            shifts (list[list[Tensor]]): a list of N=#image elements. Each is a
                list of #feature level tensors. The tensors contains shifts of
                this image on the specific feature level.
            targets (list[Instances]): a list of N `Instances`s. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.  Specify `targets` during training only.

        Returns:
            gt_classes (Tensor):
                An integer tensor of shape (N, R) storing ground-truth
                labels for each shift.
                R is the total number of shifts, i.e. the sum of Hi x Wi for all levels.
                Shifts in the valid boxes are assigned their corresponding label in the
                [0, K-1] range. Shifts in the background are assigned the label "K".
                Shifts in the ignore areas are assigned a label "-1", i.e. ignore.
        """
        gt_classes = []

        box_cls = torch.cat(
            [permute_to_N_HWA_K(x, self.num_classes) for x in box_cls], dim=1
        )
        box_delta = torch.cat([permute_to_N_HWA_K(x, 4) for x in box_delta], dim=1)
        box_cls = box_cls.sigmoid_()

        for (
            shifts_per_image,
            targets_per_image,
            box_cls_per_image,
            box_delta_per_image,
        ) in zip(shifts, targets, box_cls, box_delta):
            shifts_over_all_feature_maps = torch.cat(shifts_per_image, dim=0).to(
                self.device
            )

            gt_boxes = targets_per_image.gt_boxes
            if len(gt_boxes) == 0:
                gt_classes_i = self.num_classes + torch.zeros(
                    len(shifts_over_all_feature_maps), device=self.device
                )
                gt_classes.append(gt_classes_i)
                continue
            prob = box_cls_per_image[:, targets_per_image.gt_classes].t()
            boxes = self.shift2box_transform.apply_deltas(
                box_delta_per_image, shifts_over_all_feature_maps
            )
            iou = pairwise_iou(gt_boxes, Boxes(boxes))
            quality = prob ** (1 - self.poto_alpha) * iou**self.poto_alpha

            candidate_idxs = []
            st, ed = 0, 0
            for shifts_i in shifts_per_image:
                ed += len(shifts_i)
                _, topk_idxs = quality[:, st:ed].topk(self.poto_aux_topk, dim=1)
                candidate_idxs.append(st + topk_idxs)
                st = ed
            candidate_idxs = torch.cat(candidate_idxs, dim=1)

            is_in_boxes = (
                self.shift2box_transform.get_deltas(
                    shifts_over_all_feature_maps, gt_boxes.tensor.unsqueeze(1)
                )
                .min(dim=-1)
                .values
                > 0
            )

            candidate_qualities = quality.gather(1, candidate_idxs)
            quality_thr = candidate_qualities.mean(
                dim=1, keepdim=True
            ) + candidate_qualities.std(dim=1, keepdim=True)
            is_foreground = torch.zeros_like(is_in_boxes).scatter_(
                1, candidate_idxs, True
            )
            is_foreground &= quality >= quality_thr

            quality[~is_in_boxes] = -1
            quality[~is_foreground] = -1

            # if there are still more than one objects for a position,
            # we choose the one with maximum quality
            positions_max_quality, gt_matched_idxs = quality.max(dim=0)

            # num_fg += (positions_max_quality != -1).sum().item()
            # num_gt += len(targets_per_image)

            # ground truth classes
            assert len(targets_per_image) > 0
            gt_classes_i = targets_per_image.gt_classes[gt_matched_idxs]
            # Shifts with quality -1 are treated as background.
            gt_classes_i[positions_max_quality == -1] = self.num_classes
            gt_classes.append(gt_classes_i)
        gt_classes = torch.stack(gt_classes).type(torch.LongTensor).to(self.device)
        return gt_classes

    @torch.no_grad()
    def inference(
        self,
        box_cls,
        box_delta,
        box_filter,
        box_feature,
        shifts,
        images,
        gt_classes=None,
        gt_shifts_reg_deltas=None,
    ):
        """
        Arguments:
            gt_classes: Tensor of shape (N, nr_boxes_all_level)
            gt_shifts_reg_deltas: Tensor of shape (N, nr_boxes_all_level, 4)
            box_cls, box_delta, box_filter: Same as the output of :meth:`FCOSHead.forward`
            shifts (list[list[Tensor]): a list of #images elements. Each is a
                list of #feature level tensor. The tensor contain shifts of this
                image on the specific feature level.
            images (ImageList): the input images

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(shifts) == len(images)
        results = []

        box_cls = [permute_to_N_HWA_K(x, self.num_classes) for x in box_cls]
        box_delta = [permute_to_N_HWA_K(x, 4) for x in box_delta]
        box_filter = [permute_to_N_HWA_K(x, 1) for x in box_filter]
        box_feature = [permute_to_N_HWA_K(x, self.in_channels) for x in box_feature]

        if self.training:
            assert gt_classes is not None and gt_shifts_reg_deltas is not None
        if self.training:
            feat_level_num = [x.shape[1] for x in box_cls]
            st = ed = 0
            gt_class, gt_delta = [], []
            for n in feat_level_num:
                ed += n
                gt_class.append(gt_classes[:, st:ed, None])
                gt_delta.append(gt_shifts_reg_deltas[:, st:ed, :])
                st = ed

        # list[Tensor], one per level, each has shape (N, Hi x Wi x A, K or 4)

        for img_idx, shifts_per_image in enumerate(shifts):
            image_size = images.image_sizes[img_idx]
            box_cls_per_image = [
                box_cls_per_level[img_idx] for box_cls_per_level in box_cls
            ]
            box_reg_per_image = [
                box_reg_per_level[img_idx] for box_reg_per_level in box_delta
            ]
            box_filter_per_image = [
                box_filter_per_level[img_idx] for box_filter_per_level in box_filter
            ]
            box_feature_per_image = [
                bbox_feature_per_level[img_idx]
                for bbox_feature_per_level in box_feature
            ]
            if self.training:
                gt_class_per_image = [
                    gt_class_per_level[img_idx] for gt_class_per_level in gt_class
                ]
                gt_delta_per_image = [
                    gt_delta_per_level[img_idx] for gt_delta_per_level in gt_delta
                ]
            else:
                gt_class_per_image = gt_delta_per_image = None
            results_per_image = self.inference_single_image(
                box_cls_per_image,
                box_reg_per_image,
                box_filter_per_image,
                box_feature_per_image,
                shifts_per_image,
                tuple(image_size),
                gt_class_per_image,
                gt_delta_per_image,
            )
            results.append(results_per_image)
        return results

    def inference_single_image(
        self,
        box_cls,
        box_delta,
        box_filter,
        box_feature,
        shifts,
        image_size,
        gt_class=None,
        gt_delta=None,
    ):
        """
        Single-image inference. Return bounding-box detection results by thresholding
        on scores and applying non-maximum suppression (NMS).

        Arguments:
            box_cls (list[Tensor]): list of #feature levels. Each entry contains
                tensor of size (H x W, K)
            box_delta (list[Tensor]): Same shape as 'box_cls' except that K becomes 4.
            box_filter (list[Tensor]): Same shape as 'box_cls' except that K becomes 1.
            shifts (list[Tensor]): list of #feature levels. Each entry contains
                a tensor, which contains all the shifts for that
                image in that feature level.
            image_size (tuple(H, W)): a tuple of the image height and width.
            gt_class (Tensor(H x W, K))
            gt_delta (Tensor(H x W, 4))

        Returns:
            Same as `inference`, but for only one image.
        """
        if self.training:
            assert gt_class is not None and gt_delta is not None
        if self.training:
            gt_classes_all = []
            gt_boxes_all = []
        boxes_all = []
        scores_all = []
        class_idxs_all = []
        features_all = []

        # Iterate over every feature level
        for i in range(len(box_cls)):
            box_cls_i = box_cls[i]
            box_reg_i = box_delta[i]
            box_filter_i = box_filter[i]
            box_feat_i = box_feature[i]
            shifts_i = shifts[i]
            if self.training:
                gt_class_i = gt_class[i]
                gt_delta_i = gt_delta[i]
            # (HxWxK,)
            box_cls_i = (box_cls_i.sigmoid_() * box_filter_i.sigmoid_()).flatten()

            # Keep top k top scoring indices only.
            num_topk = min(self.pre_nms_topk, box_reg_i.size(0))
            # torch.sort is actually faster than .topk (at least on GPUs)
            predicted_prob, topk_idxs = box_cls_i.sort(descending=True)
            predicted_prob = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[:num_topk]

            # filter out the proposals with low confidence score
            # NOTE: For RPN, we don't discard low confidence proposals
            # keep_idxs = predicted_prob > self.score_threshold
            # predicted_prob = predicted_prob[keep_idxs]
            # topk_idxs = topk_idxs[keep_idxs]

            shift_idxs = torch.div(topk_idxs, self.num_classes, rounding_mode='floor')
            classes_idxs = topk_idxs % self.num_classes

            box_reg_i = box_reg_i[shift_idxs]
            shifts_i = shifts_i[shift_idxs]
            # predict boxes
            predicted_boxes = self.shift2box_transform.apply_deltas(box_reg_i, shifts_i)
            # box_features
            box_features = box_feat_i[shift_idxs]

            boxes_all.append(predicted_boxes)
            scores_all.append(predicted_prob)
            class_idxs_all.append(classes_idxs)
            features_all.append(box_features)
            if self.training:
                gt_classes = gt_class_i[shift_idxs]
                gt_delta_i = gt_delta_i[shift_idxs]
                gt_boxes = self.shift2box_transform.apply_deltas(gt_delta_i, shifts_i)
                gt_classes_all.append(gt_classes)
                gt_boxes_all.append(gt_boxes)

        boxes_all, scores_all, class_idxs_all, features_all = [
            cat(x) for x in [boxes_all, scores_all, class_idxs_all, features_all]
        ]
        if self.training:
            gt_classes_all, gt_boxes_all = [
                cat(x) for x in [gt_classes_all, gt_boxes_all]
            ]

        if self.nms_type is None:
            # strategies above (e.g. pre_nms_topk and score_threshold) are
            # useless for POTO, just keep them for debug and analysis
            keep = scores_all.argsort(descending=True)
        else:
            keep = batched_nms(
                boxes_all, scores_all, class_idxs_all, self.nms_threshold
            )
        keep = keep[: self.post_nms_topk]

        result = Instances(image_size)
        boxes_all = boxes_all[keep]
        scores_all = scores_all[keep]
        features_all = features_all[keep]
        result.proposal_boxes = Boxes(boxes_all)
        result.objectness_logits = scores_all
        result.proposal_features = features_all
        if self.training:
            gt_classes_all = gt_classes_all[keep]
            gt_boxes_all = gt_boxes_all[keep]
            result.gt_classes = gt_classes_all
            result.gt_boxes = Boxes(gt_boxes_all)
        return result

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    def reset(self):
        """
        Reset caches to inference on a new video.
        """
        return


class FCOSHead(nn.Module):
    """
    The head used in FCOS for object classification and box regression.
    It has two subnets for the two tasks, with a common structure but separate parameters.
    """

    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()
        # fmt: off
        in_channels = input_shape[0].channels
        num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        num_convs = cfg.MODEL.DeFCN.NUM_CONVS
        prior_prob = cfg.MODEL.DeFCN.PRIOR_PROB
        num_shifts = ShiftGenerator(cfg, input_shape).num_cell_shifts
        self.fpn_strides = cfg.MODEL.DeFCN.FPN_STRIDES
        self.norm_reg_targets = cfg.MODEL.DeFCN.NORM_REG_TARGETS
        # fmt: on
        assert (
            len(set(num_shifts)) == 1
        ), "using differenct num_shifts value is not supported"
        num_shifts = num_shifts[0]

        cls_subnet = []
        bbox_subnet = []
        for _ in range(num_convs):
            cls_subnet.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            cls_subnet.append(nn.GroupNorm(32, in_channels))
            cls_subnet.append(nn.ReLU())
            bbox_subnet.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            bbox_subnet.append(nn.GroupNorm(32, in_channels))
            bbox_subnet.append(nn.ReLU())

        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.bbox_subnet = nn.Sequential(*bbox_subnet)

        self.organ_specific = cfg.MODEL.ORGAN_SPECIFIC.ENABLE
        if "rpn_cls" in self.organ_specific:
            # organ-specific classification layers
            print("enable rpn organ-specific classification!")
            self.cls_score = None
            self.breast_cls = nn.Conv2d(
                in_channels,
                num_shifts * num_classes,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            self.thyroid_cls = nn.Conv2d(
                in_channels,
                num_shifts * num_classes,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            init_cls = [self.breast_cls, self.thyroid_cls]
        else:
            self.cls_score = nn.Conv2d(
                in_channels,
                num_shifts * num_classes,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            init_cls = [self.cls_score]

        self.bbox_pred = nn.Conv2d(
            in_channels, num_shifts * 4, kernel_size=3, stride=1, padding=1
        )

        self.max3d = MaxFiltering(
            in_channels,
            kernel_size=cfg.MODEL.POTO.FILTER_KERNEL_SIZE,
            tau=cfg.MODEL.POTO.FILTER_TAU,
        )
        self.filter = nn.Conv2d(
            in_channels, num_shifts * 1, kernel_size=3, stride=1, padding=1
        )

        # Initialization
        for modules in [
            self.cls_subnet,
            self.bbox_subnet,
            self.bbox_pred,
            self.max3d,
            self.filter,
        ] + init_cls:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)
                if isinstance(layer, nn.GroupNorm):
                    torch.nn.init.constant_(layer.weight, 1)
                    torch.nn.init.constant_(layer.bias, 0)

        # Use prior in model initialization to improve stability
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        if "rpn_cls" in self.organ_specific:
            torch.nn.init.constant_(self.breast_cls.bias, bias_value)
            torch.nn.init.constant_(self.thyroid_cls.bias, bias_value)
        else:
            torch.nn.init.constant_(self.cls_score.bias, bias_value)

        self.scales = nn.ModuleList(
            [Scale(init_value=1.0) for _ in range(len(self.fpn_strides))]
        )

    def switch(self, organ):
        self.organ = organ
        assert self.organ == "thyroid" or self.organ == "breast"
        if "rpn_cls" in self.organ_specific:
            self.cls_score = self.thyroid_cls if organ == "thyroid" else self.breast_cls

    def forward(self, features):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns:
            logits (list[Tensor]): #lvl tensors, each has shape (N, K, Hi, Wi).
                The tensor predicts the classification probability
                at each spatial position for each of the K object classes.
            bbox_reg (list[Tensor]): #lvl tensors, each has shape (N, 4, Hi, Wi).
                The tensor predicts 4-vector (dl,dt,dr,db) box
                regression values for every shift. These values are the
                relative offset between the shift and the ground truth box.
            filter (list[Tensor]): #lvl tensors, each has shape (N, 1, Hi, Wi).
                The tensor predicts the centerness at each spatial position.
        """
        logits, bbox_reg = [], []
        filter_subnet = []
        for level, feature in enumerate(features):
            cls_subnet = self.cls_subnet(feature)
            bbox_subnet = self.bbox_subnet(feature)

            logits.append(self.cls_score(cls_subnet))

            bbox_pred = self.scales[level](self.bbox_pred(bbox_subnet))
            if self.norm_reg_targets:
                bbox_reg.append(F.relu(bbox_pred) * self.fpn_strides[level])
            else:
                bbox_reg.append(torch.exp(bbox_pred))
            filter_subnet.append(bbox_subnet)

        filters = [self.filter(x) for x in self.max3d(filter_subnet)]
        return logits, bbox_reg, filters


class MaxFiltering(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int = 3, tau: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )
        self.norm = nn.GroupNorm(32, in_channels)
        self.nonlinear = nn.ReLU()
        self.max_pool = nn.MaxPool3d(
            kernel_size=(tau + 1, kernel_size, kernel_size),
            padding=(tau // 2, kernel_size // 2, kernel_size // 2),
            stride=1,
        )
        self.margin = tau // 2

    def forward(self, inputs):
        features = []
        for l, x in enumerate(inputs):
            features.append(self.conv(x))

        outputs = []
        for l, x in enumerate(features):
            func = lambda f: F.interpolate(
                f, size=x.shape[2:], mode="bilinear", align_corners=True
            )
            feature_3d = []
            for k in range(
                max(0, l - self.margin), min(len(features), l + self.margin + 1)
            ):
                feature_3d.append(func(features[k]) if k != l else features[k])
            feature_3d = torch.stack(feature_3d, dim=2)
            max_pool = self.max_pool(feature_3d)[:, :, min(l, self.margin)]
            output = max_pool + inputs[l]
            outputs.append(self.nonlinear(self.norm(output)))
        return outputs

```
</details>

## 4. ROI_HEADS

UltraDet的RoI Head为``Res5TemporalROIBoxHeads``，代码同时提供了四种可选的RoI Head，``Res5TemporalROIBoxHeads``、``TemporalROIHeads``、``Res5ROIHeads34``、``DynamicHead``

<details>
<summary>TemporalROIHeads and Res5TemporalROIBoxHeads</summary>

```python
from collections import Counter, deque, namedtuple
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from detectron2.layers import ShapeSpec

# from detectron2.modeling.backbone.resnet import BottleneckBlock, BasicBlock, make_stage
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY
from detectron2.modeling.sampling import subsample_labels
from detectron2.structures import Boxes, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from torch import nn

from ultrasound_vid.modeling.proposal_generator.proposal_utils import (
    add_ground_truth_to_proposals,
)
from ultrasound_vid.modeling.layers import (
    ContextFusion,
    ROIRelationLayers, 
    ROIRelationLayersWithContext
)
from ultrasound_vid.modeling.heads.fast_rcnn import FastRCNNOutputLayers
from ultrasound_vid.modeling.backbone.resnet import (
    BottleneckBlock,
    BasicBlock,
    make_stage,
)

frame_cache = namedtuple("frame_cache", ["proposal", "feature"])


class TemporalROIHeads(torch.nn.Module):
    """
    ROIHeads perform all per-region computation in an R-CNN.

    It contains logic of cropping the regions, extract per-region features,
    and make per-region predictions.

    It can have many variants, implemented as subclasses of this class.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super(TemporalROIHeads, self).__init__()

        # fmt: off
        self.batch_size_per_image = cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE
        self.positive_sample_fraction = cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
        self.test_score_thresh = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
        self.test_nms_thresh = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
        self.test_detections_per_img = cfg.TEST.DETECTIONS_PER_IMAGE
        self.in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.proposal_append_gt = cfg.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT
        self.feature_strides = {k: v.stride for k, v in input_shape.items()}
        self.feature_channels = {k: v.channels for k, v in input_shape.items()}
        self.cls_agnostic_bbox_reg = cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG
        self.smooth_l1_beta = cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA
        self.interval_pre_test = cfg.MODEL.ROI_BOX_HEAD.INTERVAL_PRE_TEST
        self.interval_after_test = cfg.MODEL.ROI_BOX_HEAD.INTERVAL_AFTER_TEST
        # fmt: on

        # Matcher to assign box proposals to gt boxes
        self.proposal_matcher = Matcher(
            cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS,
            cfg.MODEL.ROI_HEADS.IOU_LABELS,
            allow_low_quality_matches=False,
        )

        # Box2BoxTransform for bounding box regression
        self.box2box_transform = Box2BoxTransform(
            weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS
        )

    def _sample_proposals(
        self,
        matched_idxs: torch.Tensor,
        matched_labels: torch.Tensor,
        gt_classes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.

        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.

        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        """
        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            gt_classes[matched_labels == 0] = self.num_classes
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes

        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
            gt_classes,
            self.batch_size_per_image,
            self.positive_sample_fraction,
            self.num_classes,
        )

        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        return sampled_idxs, gt_classes[sampled_idxs]

    @torch.no_grad()
    def label_and_sample_proposals(
        self, proposals: List[Instances], targets: List[Instances]
    ) -> List[Instances]:
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
        boxes, with a fraction of positives that is no larger than
        ``self.positive_sample_fraction``.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:

                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)

                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """
        gt_boxes = [x.gt_boxes for x in targets]
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head, mask head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            # We index all the attributes of targets that start with "gt_"
            # and have not been added to proposals yet (="gt_classes").
            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(
                        trg_name
                    ):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
                )
                proposals_per_image.gt_boxes = gt_boxes

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt

    def reorganize_proposals_by_video(
        self,
        batched_inputs: List[Dict],
        proposals: List[Instances],
        box_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare data for relation modules.
        Reorganize proposals by grouping those from same video, and these proposals will
        be regarded as "related inputs" in relation modules. The meaning of batchsize is
        number of videos in relation module.
        We use video_folder to check whether some frames are from the same video.

        Boxes and features of each image are paded to have the same size, so that it can
        be processed in a batch manner.
        """
        video_folders = [b["video_folder"] for b in batched_inputs]
        counter = Counter(video_folders)
        assert (
            len(set(counter.values())) == 1
        ), "videos with different numbers of frames"
        num_frames_per_video = list(set(counter.values()))[0]
        num_videos = len(counter)
        assert num_videos == 1, "Support only one video per GPU!"

        # box type: x1, y1, x2, y2
        boxes = [p.proposal_boxes.tensor for p in proposals]
        num_proposals = [len(b) for b in boxes]
        box_features = list(box_features.split(num_proposals, dim=0))
        out_num_proposals = max(num_proposals)
        device = boxes[0].device
        valid_boxes = []
        valid_boxes_exceptgt = []

        for i in range(len(boxes)):
            boxes[i] = F.pad(boxes[i], [0, 0, 0, out_num_proposals - num_proposals[i]])
            box_features[i] = F.pad(
                box_features[i], [0, 0, 0, out_num_proposals - num_proposals[i]]
            )
            valid_boxes.append(
                torch.arange(out_num_proposals, device=device) < num_proposals[i]
            )
            basemask = torch.arange(out_num_proposals, device=device) < num_proposals[i]
            gtmask = proposals[i].notgt_bool
            basemask[0 : num_proposals[i]] = basemask[0 : num_proposals[i]] & gtmask
            valid_boxes_exceptgt.append(basemask)

        boxes = torch.stack(boxes, dim=0)
        box_features = torch.stack(box_features, dim=0)
        valid_boxes = torch.stack(valid_boxes, dim=0)
        valid_boxes_exceptgt = torch.stack(valid_boxes_exceptgt, dim=0)

        boxes = boxes.reshape(num_videos, -1, 4)
        box_features = box_features.reshape(num_videos, -1, box_features.shape[-1])
        valid_boxes = valid_boxes.reshape(num_videos, -1)
        valid_boxes_exceptgt = valid_boxes_exceptgt.reshape(num_videos, -1)

        return boxes, box_features, valid_boxes, valid_boxes_exceptgt

    def reorganize_proposals_for_single_video(
        self, proposals: List[Instances], box_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare data for relation modules.
        This function is designed for test mode, which means all proposals are
        from the same video
        """

        # box type: x1, y1, x2, y2
        boxes = [p.proposal_boxes.tensor for p in proposals]
        boxes = torch.cat(boxes, dim=0)
        boxes = boxes.reshape(1, -1, 4)
        box_features = box_features.reshape(1, -1, box_features.shape[-1])
        return boxes, box_features

    def forward(
        self,
        batched_inputs: List[Dict],
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        Args:
            batched_inputs (List[Dict]):
            features (dict[str: Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            proposals (list[Instances]): length `N` list of `Instances`s. The i-th
                `Instances` contains object proposals for the i-th input image,
                with fields "proposal_boxes" and "objectness_logits".
            targets (list[Instances], optional): length `N` list of `Instances`s. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.  Specify `targets` during training only.
                It may have the following fields:

                - gt_boxes: the bounding box of each instance.
                - gt_classes: the label for each instance with a category ranging in [0, #class].
                - gt_masks: PolygonMasks or BitMasks, the ground-truth masks of each instance.
                - gt_keypoints: NxKx3, the groud-truth keypoints for each instance.

        Returns:
            results (list[Instances]): length `N` list of `Instances`s containing the
            detected instances. Returned during inference only; may be [] during training.

            losses (dict[str->Tensor]):
            mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        raise NotImplementedError()


@ROI_HEADS_REGISTRY.register()
class Res5TemporalROIBoxHeads(TemporalROIHeads):
    """
    The ROIHeads in a typical "C4" R-CNN model, where
    the box and mask head share the cropping and
    the per-region feature computation by a Res5 block.

    Only box head. Mask head not supported.
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)

        assert len(self.in_features) == 1

        # fmt: off
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        pooler_scales = (1.0 / self.feature_strides[self.in_features[0]],)
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        # fmt: on
        assert not cfg.MODEL.KEYPOINT_ON
        assert not cfg.MODEL.MASK_ON

        self.pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        self.res5, out_channels = self._build_res5_block(cfg)
        self.relation, out_channels = self._build_relation_module(cfg, out_channels)
        self.box_predictor = FastRCNNOutputLayers(
            out_channels,
            box2box_transform=self.box2box_transform,
            num_classes=self.num_classes,
            cls_agnostic_bbox_reg=self.cls_agnostic_bbox_reg,
            smooth_l1_beta=self.smooth_l1_beta,
            test_score_thresh=self.test_score_thresh,
            test_nms_thresh=self.test_nms_thresh,
            test_topk_per_image=self.test_detections_per_img,
            organ_specific=cfg.MODEL.ORGAN_SPECIFIC.ENABLE,
        )
        buffer_length = self.interval_pre_test + self.interval_after_test + 1
        self.history_buffer = deque(maxlen=buffer_length)
        self.key_frame_id = 0
        self.d_model = out_channels

    def _build_caches(self):
        self.history_buffer.clear()
        self.key_frame_id = 0

    def reset(self):
        self._build_caches()

    def _build_res5_block(self, cfg):
        # fmt: off
        stage_channel_factor = 2 ** 3  # res5 is 8x res2
        num_groups = cfg.MODEL.RESNETS.NUM_GROUPS
        width_per_group = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
        bottleneck_channels = num_groups * width_per_group * stage_channel_factor
        out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * stage_channel_factor
        half_channel = cfg.MODEL.RESNETS.HALF_CHANNEL
        res5_out_channel = cfg.MODEL.RESNETS.RES5_OUT_CHANNEL
        if half_channel: # deprecated, using res5_out_channel to set RDN channels
            res5_out_channel = 256
        stride_in_1x1 = cfg.MODEL.RESNETS.STRIDE_IN_1X1
        norm = cfg.MODEL.RESNETS.NORM
        assert not cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE[-1], \
            "Deformable conv is not yet supported in res5 head."
        # fmt: on

        if cfg.MODEL.RESNETS.DEPTH >= 50:
            # if use ResNet-50
            blocks = make_stage(
                BottleneckBlock,
                3,
                stride_per_block=[2, 1, 1],
                in_channels=out_channels // 2,
                bottleneck_channels=bottleneck_channels,
                out_channels=out_channels,
                num_groups=num_groups,
                norm=norm,
                stride_in_1x1=stride_in_1x1,
            )
        else:
            # if use ResNet-18 and 34
            if res5_out_channel != 512:
                blocks = make_stage(
                    BasicBlock,
                    3,
                    stride_per_block=[2, 1, 1],
                    in_channels=out_channels // 2,
                    out_channels=res5_out_channel,
                    norm=norm,
                    short_cut_per_block=[True, False, False],
                )
                out_channels = res5_out_channel
            else:
                blocks = make_stage(
                    BasicBlock,
                    3,
                    stride_per_block=[2, 1, 1],
                    in_channels=out_channels // 2,
                    # bottleneck_channels=bottleneck_channels,
                    out_channels=out_channels,
                    # num_groups=num_groups,
                    norm=norm,
                    # stride_in_1x1=stride_in_1x1,
                )
        return nn.Sequential(*blocks), out_channels

    def _build_relation_module(self, cfg, in_channels):
        return ROIRelationLayers(cfg, in_channels), in_channels

    def _shared_roi_transform(self, features, boxes):
        x = self.pooler(features, boxes)
        return self.res5(x)

    def forward(self, batched_inputs, features, proposals, targets=None):
        if self.training:
            return self.forward_train(batched_inputs, features, proposals, targets)
        else:
            return self.forward_test(batched_inputs, features, proposals, targets)

    def forward_train(self, batched_inputs, features, proposals, targets=None):
        """
        See :class:`ROIHeads.forward`.
        The input images is replaced by batched_inputs, to get video informations.
        """
        proposals = self.label_and_sample_proposals(proposals, targets)
        proposal_boxes = [x.proposal_boxes for x in proposals]
        num_boxes = sum([len(p) for p in proposal_boxes])
        if num_boxes == 0:
            losses = {
                "loss_cls": torch.tensor(0.0, device=proposal_boxes[0].tensor.device),
                "loss_box_reg": torch.tensor(0.0, device=proposal_boxes[0].tensor.device)
            }
            return [], losses

        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])
        (
            boxes,
            box_features,
            is_valid,
            valid_boxes_exceptgt,
        ) = self.reorganize_proposals_by_video(
            batched_inputs, proposals, feature_pooled
        )
        relation_mask = valid_boxes_exceptgt.unsqueeze(1).repeat(
            1, is_valid.shape[1], 1
        )
        box_features = self.relation(boxes, box_features, mask=relation_mask)
        box_features = box_features[is_valid]
        box_features = torch.flatten(box_features, start_dim=0, end_dim=-2)
        predictions = self.box_predictor(box_features)

        del feature_pooled, box_features
        losses = self.box_predictor.losses(predictions, proposals)
        del features
        return [], losses

    def forward_test(self, batched_inputs, features, proposals, targets=None):

        assert len(batched_inputs) == 1
        assert len(proposals) == 1
        assert targets is None

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
        self.history_buffer.append(frame_cache(proposals[0], feature_pooled))

        proposals = [x.proposal for x in self.history_buffer]
        box_features = [x.feature for x in self.history_buffer]
        box_features = torch.cat(box_features, dim=0)

        boxes, box_features = self.reorganize_proposals_for_single_video(
            proposals, box_features
        )
        box_features = self.relation(boxes, box_features)
        box_features = torch.flatten(box_features, start_dim=0, end_dim=-2)

        box_features = box_features.split([len(p) for p in proposals])

        proposals = [proposals[-1]]
        box_features = box_features[-1]

        predictions = self.box_predictor(box_features)
        pred_instances, _ = self.box_predictor.inference(predictions, proposals)

        return pred_instances, {}


@ROI_HEADS_REGISTRY.register()
class Res5TemporalContextROIBoxHeads(Res5TemporalROIBoxHeads):
    """
    The ROIHeads in a typical "C4" R-CNN model, where
    the box and mask head share the cropping and
    the per-region feature computation by a Res5 block.

    Only box head. Mask head not supported.
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        self.relation = ROIRelationLayersWithContext(cfg, self.d_model)
        self.context_fusion = ContextFusion(cfg, self.d_model)

    def reset(self):
        self._build_caches()
        self.context_fusion.reset()

    def reorganize_proposals_by_frame(
        self,
        proposals: List[Instances],
        box_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # box type: x1, y1, x2, y2
        boxes = [p.proposal_boxes.tensor for p in proposals]
        num_proposals = [len(b) for b in boxes]
        box_features = list(box_features.split(num_proposals, dim=0))
        out_num_proposals = max(num_proposals)
        device = boxes[0].device
        valid_boxes = []
        valid_boxes_exceptgt = []

        for i in range(len(boxes)):
            boxes[i] = F.pad(boxes[i], [0, 0, 0, out_num_proposals - num_proposals[i]])
            box_features[i] = F.pad(
                box_features[i], [0, 0, 0, out_num_proposals - num_proposals[i]]
            )
            valid_boxes.append(
                torch.arange(out_num_proposals, device=device) < num_proposals[i]
            )
            basemask = torch.arange(out_num_proposals, device=device) < num_proposals[i]
            if self.training:
                gtmask = proposals[i].notgt_bool
                basemask[0 : num_proposals[i]] = basemask[0 : num_proposals[i]] & gtmask
            valid_boxes_exceptgt.append(basemask)

        boxes = torch.stack(boxes, dim=0)
        box_features = torch.stack(box_features, dim=0)
        valid_boxes = torch.stack(valid_boxes, dim=0)
        valid_boxes_exceptgt = torch.stack(valid_boxes_exceptgt, dim=0)

        return boxes, box_features, valid_boxes, valid_boxes_exceptgt
    
    def forward_train(self, batched_inputs, features, proposals, targets=None):
        """
        See :class:`ROIHeads.forward`.
        The input images is replaced by batched_inputs, to get video informations.
        """
        frame_features = self.context_fusion(features)
        proposals = self.label_and_sample_proposals(proposals, targets)
        proposal_boxes = [x.proposal_boxes for x in proposals]
        num_boxes = sum([len(p) for p in proposal_boxes])
        if num_boxes == 0:
            losses = {
                "loss_cls": torch.tensor(0.0, device=proposal_boxes[0].tensor.device),
                "loss_box_reg": torch.tensor(0.0, device=proposal_boxes[0].tensor.device)
            }
            return [], losses
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])
        (
            boxes,                  # (B, N, 4)
            box_features,           # (B, N, D)
            is_valid,               # (B, N)
            valid_boxes_exceptgt,   # (B, N)
        ) = self.reorganize_proposals_by_frame(
            proposals, feature_pooled
        )
        box_features = self.relation(
            boxes, box_features, frame_features, mask=valid_boxes_exceptgt
        )
        box_features = box_features[is_valid]
        box_features = torch.flatten(box_features, start_dim=0, end_dim=-2)
        predictions = self.box_predictor(box_features)

        del feature_pooled, box_features
        losses = self.box_predictor.losses(predictions, proposals)
        del features

        k = list(losses.keys())[0]
        losses[k] += frame_features.sum() * 0.0
        # losses.update(self.context_losses(frame_features, targets))
        return [], losses

    def forward_test(self, batched_inputs, features, proposals, targets=None):

        assert len(batched_inputs) == 1
        assert len(proposals) == 1
        assert targets is None

        frame_features = self.context_fusion.step(features)
        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
        self.history_buffer.append(frame_cache(proposals[0], feature_pooled))

        proposals = [x.proposal for x in self.history_buffer]
        box_features = [x.feature for x in self.history_buffer]
        box_features = torch.cat(box_features, dim=0)

        boxes, box_features, _, _ = self.reorganize_proposals_by_frame(
            proposals, box_features
        )
        box_features = self.relation(boxes, box_features, frame_features)
        box_features = box_features[-1, :len(proposals[-1])]
        proposals = [proposals[-1]]

        predictions = self.box_predictor(box_features)
        pred_instances, _ = self.box_predictor.inference(predictions, proposals)

        return pred_instances, {}
```
</details>

<details>
<summary>Res5ROIHeads34</summary>

```python
from torch import nn

from detectron2.modeling.backbone.resnet import BottleneckBlock, BasicBlock, make_stage
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY, Res5ROIHeads


@ROI_HEADS_REGISTRY.register()
class Res5ROIHeads34(Res5ROIHeads):
    """
    The ROIHeads in a typical "C4" R-CNN model, where
    the box and mask head share the cropping and
    the per-region feature computation by a Res5 block.
    """

    def _build_res5_block(self, cfg):
        # fmt: off
        stage_channel_factor = 2 ** 3  # res5 is 8x res2
        num_groups = cfg.MODEL.RESNETS.NUM_GROUPS
        width_per_group = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
        bottleneck_channels = num_groups * width_per_group * stage_channel_factor
        out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * stage_channel_factor
        stride_in_1x1 = cfg.MODEL.RESNETS.STRIDE_IN_1X1
        norm = cfg.MODEL.RESNETS.NORM
        assert not cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE[-1], \
            "Deformable conv is not yet supported in res5 head."
        # fmt: on

        if cfg.MODEL.RESNETS.DEPTH >= 50:
            # if use ResNet-50
            blocks = make_stage(
                BottleneckBlock,
                3,
                stride_per_block=[2, 1, 1],
                in_channels=out_channels // 2,
                bottleneck_channels=bottleneck_channels,
                out_channels=out_channels,
                num_groups=num_groups,
                norm=norm,
                stride_in_1x1=stride_in_1x1,
            )
        else:
            # if use ResNet-18 and 34
            blocks = make_stage(
                BasicBlock,
                3,
                stride_per_block=[2, 1, 1],
                in_channels=out_channels // 2,
                out_channels=out_channels,
                norm=norm,
            )
        return nn.Sequential(*blocks), out_channels

```
</details>

<details>
<summary>DynamicHead</summary>

```python
#
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import copy
import math
from typing import Optional, List

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY
from detectron2.modeling.poolers import ROIPooler, cat
from detectron2.structures import Boxes

_DEFAULT_SCALE_CLAMP = math.log(100000.0 / 16)


@ROI_HEADS_REGISTRY.register()
class DynamicHead(nn.Module):
    def __init__(self, cfg, roi_input_shape):
        super().__init__()

        # Build RoI.
        box_pooler = self._init_box_pooler(cfg, roi_input_shape)
        self.box_pooler = box_pooler

        # Build heads.
        num_classes = cfg.MODEL.SparseRCNN.NUM_CLASSES
        d_model = cfg.MODEL.SparseRCNN.HIDDEN_DIM
        dim_feedforward = cfg.MODEL.SparseRCNN.DIM_FEEDFORWARD
        nhead = cfg.MODEL.SparseRCNN.NHEADS
        dropout = cfg.MODEL.SparseRCNN.DROPOUT
        activation = cfg.MODEL.SparseRCNN.ACTIVATION
        num_heads = cfg.MODEL.SparseRCNN.NUM_HEADS
        self.efficient = cfg.MODEL.SparseRCNN.EFFICIENT

        rcnn_head = RCNNHead(
            cfg, d_model, num_classes, dim_feedforward, nhead, dropout, activation
        )
        self.head_series = _get_clones(rcnn_head, num_heads)
        self.return_intermediate = cfg.MODEL.SparseRCNN.DEEP_SUPERVISION

        # Init parameters.
        self.use_focal = cfg.MODEL.SparseRCNN.USE_FOCAL
        self.num_classes = num_classes
        if self.use_focal:
            prior_prob = cfg.MODEL.SparseRCNN.PRIOR_PROB
            self.bias_value = -math.log((1 - prior_prob) / prior_prob)
        self._reset_parameters()

    def _reset_parameters(self):
        # init all parameters.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

            # initialize the bias for focal loss.
            if self.use_focal:
                if p.shape[-1] == self.num_classes:
                    nn.init.constant_(p, self.bias_value)

    @staticmethod
    def _init_box_pooler(cfg, input_shape):

        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        return box_pooler

    def forward(self, features, init_bboxes, init_features):
        inter_class_logits = []
        inter_pred_bboxes = []
        proposal_features = None
        # batchsize = len(features[0])
        bboxes = init_bboxes
        # init_features = init_features[None].repeat(1, batchsize, 1)
        proposal_features = init_features.clone()
        # query = init_features.clone()

        for rcnn_head in self.head_series:
            class_logits, pred_bboxes, proposal_features = rcnn_head(
                features, bboxes, proposal_features, self.box_pooler
            )

            if self.return_intermediate:
                inter_class_logits.append(class_logits)
                inter_pred_bboxes.append(pred_bboxes)
            bboxes = pred_bboxes.detach()

        if self.return_intermediate:
            return torch.stack(inter_class_logits), torch.stack(inter_pred_bboxes)

        return class_logits[None], pred_bboxes[None]


class RCNNHead(nn.Module):
    def __init__(
        self,
        cfg,
        d_model,
        num_classes,
        dim_feedforward=2048,
        nhead=8,
        dropout=0.1,
        activation="relu",
        scale_clamp: float = _DEFAULT_SCALE_CLAMP,
        bbox_weights=(2.0, 2.0, 1.0, 1.0),
    ):
        super().__init__()

        self.d_model = d_model
        # self.use_relation = cfg.MODEL.SparseRCNN.POSITION_INFO
        self.temporal = cfg.MODEL.SparseRCNN.TEMPORAL_ATTN
        self.attn_mask = cfg.MODEL.SparseRCNN.ATTN_MASK
        self.roi_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        self.efficient = cfg.MODEL.SparseRCNN.EFFICIENT
        self.use_pos = cfg.MODEL.SparseRCNN.USE_POS
        self.frame_idx_embed = cfg.MODEL.SparseRCNN.FRAME_IDX_EMBED
        self.temperature = cfg.MODEL.SparseRCNN.POS_TEMPERATURE
        self.embed = None

        # dynamic.
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.inst_interact = DynamicConv(cfg)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        if self.temporal:
            self.temporal_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            self.norm_t = nn.LayerNorm(d_model)
            self.dropout_t = nn.Dropout(dropout)

        # cls.
        num_cls = cfg.MODEL.SparseRCNN.NUM_CLS
        cls_module = list()
        for _ in range(num_cls):
            cls_module.append(nn.Linear(d_model, d_model, False))
            cls_module.append(nn.LayerNorm(d_model))
            cls_module.append(nn.ReLU(inplace=True))
        self.cls_module = nn.ModuleList(cls_module)

        # reg.
        num_reg = cfg.MODEL.SparseRCNN.NUM_REG
        reg_module = list()
        for _ in range(num_reg):
            reg_module.append(nn.Linear(d_model, d_model, False))
            reg_module.append(nn.LayerNorm(d_model))
            reg_module.append(nn.ReLU(inplace=True))
        self.reg_module = nn.ModuleList(reg_module)

        # pred.
        self.use_focal = cfg.MODEL.SparseRCNN.USE_FOCAL
        if self.use_focal:
            self.class_logits = nn.Linear(d_model, num_classes)
        else:
            self.class_logits = nn.Linear(d_model, num_classes + 1)
        self.bboxes_delta = nn.Linear(d_model, 4)
        self.scale_clamp = scale_clamp
        self.bbox_weights = bbox_weights

    def with_frame_idx_embed(self, feature, shape):
        """
        Params:
            feature: Tensor of shape [N * nr_boxes, 1, d_model]
            shape: (N, nr_boxes)
        Return:
            feature: feature plus frame index embedding
        """
        if not self.frame_idx_embed:
            return feature
        elif self.embed is not None and self.embed.shape == feature.shape:
            return feature + self.embed
        N, nr_boxes = shape
        eps = 1e-6
        embed = torch.arange(N, dtype=torch.float32)
        embed = embed / (N + eps) * 2 * math.pi

        dim_t = torch.arange(self.d_model, dtype=torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.d_model)

        pos = embed[:, None] / dim_t[None]
        # (N, d_model)
        pos = torch.stack((pos[:, 0::2].sin(), pos[:, 1::2].cos()), dim=1).flatten(1)
        embed = pos.repeat_interleave(nr_boxes, dim=0)[:, None, :].to(feature.device)
        self.embed = embed
        return feature + embed

    def forward(self, features, bboxes, pro_features, pooler):
        """
        Params:
            bboxes: (N, nr_boxes, 4)
            pro_features: (N, nr_boxes, d_model)
        """
        N, nr_boxes = bboxes.shape[:2]

        # roi_feature.
        proposal_boxes = list()
        for b in range(N):
            proposal_boxes.append(Boxes(bboxes[b]))
        roi_features = pooler(features, proposal_boxes)
        roi_features = roi_features.view(N * nr_boxes, self.d_model, -1).permute(
            2, 0, 1
        )

        # self_att.
        # [nr_boxes, N, d_model]
        pro_features = pro_features.reshape(N, nr_boxes, self.d_model).permute(1, 0, 2)
        pro_features2 = self.self_attn(pro_features, pro_features, value=pro_features)[
            0
        ]
        pro_features = pro_features + self.dropout1(pro_features2)
        pro_features = self.norm1(pro_features)

        # inst_interact.
        # [1, N * nr_boxes, d_model]
        pro_features = pro_features.permute(1, 0, 2).reshape(
            1, N * nr_boxes, self.d_model
        )
        pro_features2 = self.inst_interact(pro_features, roi_features)
        pro_features = pro_features + self.dropout2(pro_features2)
        obj_features = self.norm2(pro_features)

        # cross_att.
        if self.temporal:
            mask = None
            if self.attn_mask:
                mask = torch.eye(nr_boxes).repeat(N, N).bool().to(obj_features.device)
            obj_features = obj_features.transpose(0, 1).reshape(
                N * nr_boxes, 1, self.d_model
            )
            # [N * nr_boxes, 1, d_model]
            obj_features = self.with_frame_idx_embed(obj_features, (N, nr_boxes))
            obj_features2 = self.temporal_attn(
                obj_features, obj_features, value=obj_features, attn_mask=mask
            )[0]
            obj_features = obj_features + self.dropout_t(obj_features2)
            # [1, N * nr_boxes, d_model]
            obj_features = self.norm_t(obj_features).transpose(0, 1)

        # obj_feature.
        obj_features2 = self.linear2(
            self.dropout(self.activation(self.linear1(obj_features)))
        )
        obj_features = obj_features + self.dropout3(obj_features2)
        obj_features = self.norm3(obj_features)

        fc_feature = obj_features.transpose(0, 1).reshape(N * nr_boxes, -1)
        cls_feature = fc_feature.clone()
        reg_feature = fc_feature.clone()
        for cls_layer in self.cls_module:
            cls_feature = cls_layer(cls_feature)
        for reg_layer in self.reg_module:
            reg_feature = reg_layer(reg_feature)
        class_logits = self.class_logits(cls_feature).view(N, nr_boxes, -1)
        bboxes_deltas = self.bboxes_delta(reg_feature)
        pred_bboxes = self.apply_deltas(bboxes_deltas, bboxes.view(-1, 4)).view(
            N, nr_boxes, -1
        )

        return class_logits, pred_bboxes, obj_features

    def apply_deltas(self, deltas, boxes):
        """
        Apply transformation `deltas` (dx, dy, dw, dh) to `boxes`.

        Args:
            deltas (Tensor): transformation deltas of shape (N, k*4), where k >= 1.
                deltas[i] represents k potentially different class-specific
                box transformations for the single box boxes[i].
            boxes (Tensor): boxes to transform, of shape (N, 4)
        """
        boxes = boxes.to(deltas.dtype)

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.bbox_weights
        dx = deltas[:, 0::4] / wx
        dy = deltas[:, 1::4] / wy
        dw = deltas[:, 2::4] / ww
        dh = deltas[:, 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.scale_clamp)
        dh = torch.clamp(dh, max=self.scale_clamp)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w  # x1
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h  # y1
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w  # x2
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h  # y2

        return pred_boxes


class DynamicConv(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.hidden_dim = cfg.MODEL.SparseRCNN.HIDDEN_DIM
        self.dim_dynamic = cfg.MODEL.SparseRCNN.DIM_DYNAMIC
        self.num_dynamic = cfg.MODEL.SparseRCNN.NUM_DYNAMIC
        self.num_params = self.hidden_dim * self.dim_dynamic
        self.dynamic_layer = nn.Linear(
            self.hidden_dim, self.num_dynamic * self.num_params
        )

        self.norm1 = nn.LayerNorm(self.dim_dynamic)
        self.norm2 = nn.LayerNorm(self.hidden_dim)

        self.activation = nn.ReLU(inplace=True)

        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        num_output = self.hidden_dim * pooler_resolution**2
        self.out_layer = nn.Linear(num_output, self.hidden_dim)
        self.norm3 = nn.LayerNorm(self.hidden_dim)

    def forward(self, pro_features, roi_features):
        """
        Params:
            pro_features: (1,  N * nr_boxes, d_model)
            roi_features: (49, N * nr_boxes, d_model)
        Return:
            features: (N * nr_boxes, 1, d_model)
        """
        # (N * nr_boxes, 49, d_model)
        features = roi_features.permute(1, 0, 2)
        parameters = self.dynamic_layer(pro_features).permute(1, 0, 2)

        param1 = parameters[:, :, : self.num_params].view(
            -1, self.hidden_dim, self.dim_dynamic
        )
        param2 = parameters[:, :, self.num_params :].view(
            -1, self.dim_dynamic, self.hidden_dim
        )

        # (N * nr_boxes, 49, dim_dynamic)
        features = torch.bmm(features, param1)
        features = self.norm1(features)
        features = self.activation(features)

        # (N * nr_boxes, 49, d_model)
        features = torch.bmm(features, param2)
        features = self.norm2(features)
        features = self.activation(features)

        # (N * nr_boxes, 49 * d_model)
        features = features.flatten(1)
        features = self.out_layer(features)
        features = self.norm3(features)
        features = self.activation(features)

        # (N * nr_boxes, d_model)
        return features


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")

```
</details>