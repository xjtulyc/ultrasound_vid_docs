# d2使用默认模型和自定义模型

## 1. 默认模型

### 1.1. 从Yacs配置中建立模型

从 yacs 配置对象，模型（及其子模型）可以通过 ``build_model``、``build_backbone``、``build_roi_heads`` 等函数构建：

```python
from detectron2.modeling import build_model
model = build_model(cfg)  # returns a torch.nn.Module
```

``build_model`` 只构建模型结构并用随机参数填充它。请参阅下文了解如何将现有检查点加载到模型以及如何使用``model``对象。

### 1.2. 加载/保存检查点

```python
from detectron2.checkpoint import DetectionCheckpointer
DetectionCheckpointer(model).load(file_path_or_url)  # load a file, usually from cfg.MODEL.WEIGHTS

checkpointer = DetectionCheckpointer(model, save_dir="output")
checkpointer.save("model_999")  # save to output/model_999.pth
```

Detectron2 的检查点识别 pytorch 的 ``.pth`` 格式的模型，以及我们模型库中的 ``.pkl`` 文件。有关其用法的更多详细信息，请参阅 [API 文档](https://detectron2.readthedocs.io/en/latest/modules/checkpoint.html#detectron2.checkpoint.DetectionCheckpointer)。
可以使用 ``torch.{load,save}`` 对 ``.pth`` 文件或 ``pickle.{dump,load}`` 对 ``.pkl`` 文件任意操作模型文件。

### 1.3. 使用模型

可以通过 ``outputs = model(inputs)`` 调用模型，其中 ``inputs`` 是一个``list[dict]``。每个字典对应一张图片，所需的键取决于模型的类型，以及模型是处于训练模式还是评估模式。例如，为了进行推理，所有现有模型都需要“图像”键，以及可选的“高度”和“宽度”。现有模型的输入和输出的详细格式解释如下。

**训练**：在训练模式下，所有模型都需要在 ``EventStorage`` 下使用。训练统计数据将入库：

```python
from detectron2.utils.events import EventStorage
with EventStorage() as storage:
  losses = model(inputs)
```

**推理**：如果您只想使用现有模型进行简单推理，``DefaultPredictor`` 是提供此类基本功能的模型包装器。它包括默认行为，包括模型加载、预处理和对单个图像而不是批处理的操作。请参阅其文档了解使用情况。
您也可以像这样直接运行推理：

```python
model.eval()
with torch.no_grad():
  outputs = model(inputs)
```

### 1.4. 模型输入格式
用户可以实现支持任意输入格式的自定义模型。这里我们描述了 detectron2 中所有内置模型都支持的标准输入格式。他们都以 ``list[dict]`` 作为输入。每个字典对应一张图片的信息。
该字典可能包含以下键：

- “image”：（C，H，W）格式的张量。通道的含义由 ``cfg.INPUT.FORMAT`` 定义。图像标准化（如果有）将使用 ``cfg.MODEL.PIXEL_{MEAN,STD}`` 在模型内部执行。
- “height”、“width”：inference中需要的输出高度和宽度，不一定与图像域的高度或宽度相同。例如，如果将调整大小用作预处理步骤，则图像字段包含调整大小的图像。但您可能希望输出为原始分辨率。如果提供，模型将以该分辨率产生输出，而不是以图像的分辨率作为模型的输入。这样更有效和准确。
- “instances”：用于训练的 Instances 对象，具有以下字段：
    - “gt_boxes”：一个 Boxes 对象，存储 N 个box，每个实例一个。
    - “gt_classes”：long type张量，N个标签的向量，范围[0, num_categories)。
    - “gt_masks”：存储 N 个掩码的 PolygonMasks 或 BitMasks 对象，每个实例一个。
    - “gt_keypoints”：一个关键点对象，存储N个关键点集，每个实例一个。
- “sem_seg”：（H，W）格式的 Tensor[int]。用于训练的语义分割基本事实。值表示从 0 开始的类别标签。
- “proposals”：仅在 Fast R-CNN 样式模型中使用的 Instances 对象，具有以下字段：
    - “proposal_boxes”：存储 P 个建议框的 Boxes 对象。
    - “objectness_logits”：张量，P 分数的向量，每个proposal一个。

对于内置模型的推理，只需要“image”键，“width/height”是可选的。

d2目前没有为全景分割训练定义标准输入格式，因为模型现在使用自定义数据加载器生成的自定义格式。

#### 1.4.1. 如何连接到数据加载器：
默认 ``DatasetMapper`` 的输出是一个遵循上述格式的字典。数据加载器执行批处理后，成为内置模型支持的``list[dict]``。


### 1.5. 模型输出格式

在训练模式下，内置模型输出带有所有损失的 ``dict[str->ScalarTensor]``。
在推理模式下，内置模型输出一个``list[dict]``，每个图像一个字典。根据模型正在执行的任务，每个字典可能包含以下字段：

- “instances”：具有以下字段的实例对象：
  - “pred_boxes”：存储 N 个框的框对象，每个框对应一个检测到的实例。
  - “scores”：张量，N 个置信度分数的向量。
  - “pred_classes”：张量，范围为 [0, num_categories) 的 N 个标签的向量。
  - “pred_masks”：形状为 (N, H, W) 的张量，每个检测到的实例的掩码。
  - “pred_keypoints”：形状为 (N, num_keypoint, 3) 的张量。最后一个维度中的每一行都是 (x, y, score)。置信度分数大于 0。
- “sem_seg”：(num_categories, H, W)的张量，语义分割预测。
- “proposals”：具有以下字段的实例对象：
  - “proposal_boxes”：存储N个框的框对象。
  - “objectness_logits”：N 个置信度分数的 torch 向量。
- “panoptic_seg”：``(pred: Tensor, segments_info: Optional[list[dict]])`` 的元组。 pred 张量的形状为 (H, W)，包含每个像素的段 ID。
   - 如果 segments_info 存在，则每个 dict 描述 pred 中的一个 segment id 并具有以下字段：
      - “id”：段id
      - “isthing”：段是一个东西还是东西
      - “category_id”：该段的类别id。
   - 如果一个像素的 id 不存在于 ``segments_info`` 中，则认为它是 Panoptic Segmentation 中定义的无效标签。
   - 如果 segments_info 为 None，则 pred 中的所有像素值必须≥-1。值为 -1 的像素分配了空标签。否则，每个像素的类别id通过category_id = pixel // metadata.label_divisor获得。


### 1.6. 部分执行模型
有时你可能想获得一个模型内部的中间张量，比如某一层的输入，后处理前的输出。由于通常有数百个中间张量，因此没有 API 可以为您提供所需的中间结果。您有以下选择：
1. 写一个（子）模型。按照本教程，您可以重写模型组件（例如模型的头部），使其与现有组件执行相同的操作，但返回您需要的输出。
2. 部分执行模型。您可以像往常一样创建模型，但使用自定义代码来执行它而不是 ``forward()``。比如下面的代码在mask head之前获取mask特征。

```python
images = ImageList.from_tensors(...)  # preprocessed input tensor
model = build_model(cfg)
model.eval()
features = model.backbone(images.tensor)
proposals, _ = model.proposal_generator(images, features)
instances, _ = model.roi_heads(images, features, proposals)
mask_features = [features[f] for f in model.roi_heads.in_features]
mask_features = model.roi_heads.mask_pooler(mask_features, [x.pred_boxes for x in instances])
```

3. 使用[forward hooks](https://pytorch.org/tutorials/beginner/former_torchies/nnft_tutorial.html#forward-and-backward-function-hooks)。前向钩子可以帮助您获取某个模块的输入或输出。如果它们不是你想要的，它们至少可以与部分执行一起使用以获得其他张量。

所有选项都需要您阅读现有模型的文档，有时还需要阅读现有模型的代码以了解内部逻辑，以便编写代码来获取内部张量。

