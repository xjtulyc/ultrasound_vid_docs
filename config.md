# d2的配置文件

网络参数可以通过配置文件或者直接在代码里写入两种方式进行配置，一些公用的就写在代码里，需要调整的就卸载配置文件里，修改起来较为方便。

所有的网络配置基本都是可以通过配置文件进行完成，个人认为，这是学习Detectron2框架过程中最需要学习和掌握的一个重要的环节。

## 1. yaml语法
yaml文件在detectron2整个框架中作为配置文件进行参数配置的

YAML以数据为中心，比json、xml等更适合做配置文件

1. 基本语法规则

- 大小写敏感
- 使用缩进表示层级关系
- 缩进时不允许使用Tab键，只允许使用空格
- 缩进的空格数目不重要，只要相同层级的元素左对齐即可。

2. 支持的数据结构

- 对象：键值对的集合，又称为映射（mapping）/ 哈希（hashes） / 字典（dictionary）
- 数组：一组按次序排列的值，又称为序列（sequence） / 列表（list）
- 字面量（数字、字符串、布尔值）：单个的、不可再分的值

3. 具体写法

**字面量（数字、字符串、布尔值）**
- k: v :字面量直接写，字符串默认不用加上单引号或者双引号，冒号后面必须有一个空格
- 双引号：会转义特殊字符
- 单引号：不会转义字符串里边的特殊字符；特殊字符会作为本身想表示的意思

如

```yaml
name: zhangsan
age: 20
isBoss: false
```

**数组**
多行写法使用``- 值``来表示数组中的一个元素，需要注意缩进；单行使用``[值，值]``来表示一个数组

```yaml
#多行结构
friends：
 - zhangsan
 - lisi
 - wangwu

#单行结构
friend:[zhangsan,lisi,wangwu]
```

**对象**
- 多行写法：key：value的形式，使用多行写法需要注意缩进
- 单行写法：使用{key:value}的形式书写

```yaml
#多行结构
friend:
 name:zhangsan
 age:20

#单行结构
friend:{name:zhangsan,age:20}
```

## 2. 如何在d2下使用配置文件
在d2中，配置文件放置于项目的``config\``文件夹下面。每次实验之前，需要确保配置文件正确。基本流程是

1. 根据``.yaml``文件的路径获得cfg；
2. 将cfg解析为**树状**的数据结构；
3. 在文件中调用配置参数。

### 2.1. 根据``.yaml``文件的路径获得cfg

在Python脚本中代码如下，便可获得原始配置文件

```python
from detectron2.engine import default_argument_parser   # 按照d2默认参数输入格式获取配置，可以参考源代码或官方文档
from detectron2.config import get_cfg   # 获取cfg的备份，返回一个d2 CfgNode实例

args = default_argument_parser().parse_args()   # 获取d2默认的参数

cfg = get_cfg()
# 将配置文件解析
add_ultrasound_config(cfg)  # 一个复杂的检测系统由多个模块组成，分模块解析有助于代码可读；下面几步是模型层面的参数配置
add_defcn_config(cfg)
add_sparsercnn_config(cfg)
add_deformable_detr_config(cfg)
cfg.merge_from_file(args.config_file)   # 配置文件也可以继承基类文件，优化器、数据集等的配置在下面进行
cfg.merge_from_list(args.opts)
if cfg.AUTO_DIR:
    cfg.OUTPUT_DIR = os.path.join(
        "outputs", os.path.splitext(os.path.basename(args.config_file))[0]
    )

cfg.freeze()    # 冻结配置文件
default_setup(cfg, args)
setup_logger(
    output=cfg.OUTPUT_DIR,
    distributed_rank=comm.get_rank(),
    name="ultrasound_vid",
    abbrev_name="vid",
)
```

在bash脚本中，应该按照d2的参数名称来写。请查阅[d2文档](https://detectron2.readthedocs.io/en/latest/tutorials/)

### 2.2. 将cfg解析为**树状**的数据结构
配置文件其实是指定参数用的，需要解析为d2可以使用的数据结构（CfgNode）才能在代码中真正调用。

```python
from detectron2.config import CfgNode as CN # create node

def add_ultrasound_config(cfg):
    """
    Add config for ultrasound videos.
    """
    # 可以理解为默认参数
    # We use n_fold cross validation for evaluation.
    # Number of folds
    cfg.DEVICE_WISE = False
    cfg.DATASETS.NUM_FOLDS = 5
    cfg.DATASETS.JSON_NUM_FOLDS = 5
    # IDs of test folds, every entry in [0, num_folds - 1]
    cfg.DATASETS.TEST_FOLDS = (0,)
    cfg.DATASETS.JSON_TEST_FOLDS = (0,)
    # which sampler?
    cfg.DATASETS.FRAMESAMPLER = "HardMiningFrameSampler"
    # which data version?
    cfg.DATASETS.BUS_TIMESTAMP = ""
    cfg.DATASETS.BUS_TRAIN = ()
    cfg.DATASETS.BUS_TEST = ()
    cfg.DATASETS.TUS_TIMESTAMP = ""
    cfg.DATASETS.TUS_TRAIN = ()
    cfg.DATASETS.TUS_TEST = ()
    cfg.DATASETS.GROUP = ()
    # False Positive
    cfg.DATASETS.SAMPLE_FP_BY_VIDEO = False
    cfg.DATASETS.FP_VIDEO_SAMPLE_RATE = 0.5
    cfg.DATASETS.FP_FRAMES_PATH = "notebook/fp_frames.json"

    # Segments per batch for training
    cfg.SOLVER.SEGS_PER_BATCH = 16
    cfg.SOLVER.CHECKPOINT_PERIOD = 5000
    cfg.TEST.EVAL_PERIOD = 50000

    # Frame sampler for training
    cfg.INPUT.TRAIN_FRAME_SAMPLER = CN()
    # Sample interval
    cfg.INPUT.TRAIN_FRAME_SAMPLER.INTERVAL = 30
    # Number of output frames
    cfg.INPUT.TRAIN_FRAME_SAMPLER.NUM_OUT_FRAMES = 10
    # HardMining False Positive
    cfg.INPUT.TRAIN_FRAME_SAMPLER.FP_JSON_DIR = ""
    cfg.INPUT.TRAIN_FRAME_SAMPLER.FP_SAMPLE_RATE = 1.5
    
    # Fixed area
    cfg.INPUT.FIXED_AREA_TRAIN = 0
    cfg.INPUT.FIXED_AREA_TEST = 0

    # Parameters for backbone
    cfg.MODEL.USE_LSTM = True
    cfg.MODEL.RESNETS.HALF_CHANNEL = False
    cfg.MODEL.RESNETS.RES5_OUT_CHANNEL = 512

    # Reset NMS parameters
    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 256
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 256
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 128
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 128
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.33, 0.5, 0.66, 1.0, 2.0, 3.0]]

    # Config for temporal retinanet
    cfg.MODEL.RETINANET.INTERVAL_PRE_TEST = 12
    cfg.MODEL.RETINANET.INTERVAL_AFTER_TEST = 3

    # Config for temporal relation
    cfg.MODEL.ROI_BOX_HEAD.RELATION_HEAD_NUMS = 8
    cfg.MODEL.ROI_BOX_HEAD.RELATION_LAYER_NUMS = 2
    cfg.MODEL.ROI_BOX_HEAD.INTERVAL_PRE_TEST = 12
    cfg.MODEL.ROI_BOX_HEAD.INTERVAL_AFTER_TEST = 3

    # To evaluate mAP
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    # ROI heads sampler
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 16
    cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.5

    cfg.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT = True
    cfg.MODEL.MASK_ON = False

    cfg.DATALOADER.ASPECT_RATIO_GROUPING = False
    cfg.DATALOADER.REFRESH_PERIOD = 0

    cfg.INPUT.SCALE_TRAIN = (1.0, 1.0)
    cfg.INPUT.SCALE_TEST = 1.0

    cfg.MODEL.SAMPLE_BLOCK_OUT_CHANNEL = 256
    cfg.MODEL.SAMPLE_BLOCK_ALPHA = 0.9

    # organ-specific
    cfg.MODEL.ORGAN_SPECIFIC = CN()
    cfg.MODEL.ORGAN_SPECIFIC.ENABLE = ()  # 'cls', 'reg'
    cfg.MODEL.ORGAN_SPECIFIC.BREAST_LOSS_WEIGHT = 1.0
    cfg.MODEL.ORGAN_SPECIFIC.THYROID_LOSS_WEIGHT = 1.0

    # misc
    cfg.AUTO_DIR = False

    # solver
    cfg.SOLVER.ADAM_BETA = (0.9, 0.999)

    # static frame
    cfg.STATIC_FRAME = CN() # 需要创建新节点
    cfg.STATIC_FRAME.RATE = 0.0
```

### 2.3. 在文件中调用配置参数

如下所示：

```python
static_frame_rate = cfg.STATIC_FRAME.RATE
```