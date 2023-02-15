# d2中的训练流程

从前面的教程来看，你现在可能有一个自定义模型和一个数据加载器。为了运行训练，用户通常有以下两种风格之一的偏好。

## 1. 自定义训练循环

准备好模型和数据加载器后，编写训练循环所需的一切都可以在 PyTorch 中找到，您可以自由地自己编写训练循环。这种风格让研究人员可以更清晰地管理整个训练逻辑并拥有完全的控制权。 ``tools/plain_train_net.py`` 中提供了一个这样的例子。


然后，用户可以轻松控制对训练逻辑的任何定制。

## 2. Trainer Abstraction

我们还提供了一个标准化的“trainer”抽象和一个hook系统，有助于简化标准培训行为。它包括以下两个实例化：

- **SimpleTrainer** 为单成本单优化器单数据源训练提供了一个最小的训练循环，没有别的。其他任务（检查点、日志记录等）可以使用挂钩系统来实现。
- **DefaultTrainer** 是从 yacs 配置初始化的，由 ``tools/train_net.py`` 和许多脚本使用。它包括更多人们可能想要选择加入的标准默认行为，包括优化器的默认配置、学习率计划、日志记录、评估、检查点等。

自定义：``DefaultTrainer``
1. 对于简单的自定义（例如更改优化器、评估器、LR 调度器、数据加载器等），在子类中覆盖其方法，就像 ``tools/train_net.py`` 一样。
2. 对于训练期间的额外任务，请检查[hook系统](https://detectron2.readthedocs.io/en/latest/modules/engine.html#detectron2.engine.HookBase)以查看是否支持它。
例如，在训练期间打印 hello：

```python
class HelloHook(HookBase):
  def after_step(self):
    if self.trainer.iter % 100 == 0:
      print(f"Hello at iteration {self.trainer.iter}!")
```

<details><summary><em>hook的使用</em></summary>

<br>
每个钩子可以实现 4 个方法。它们的调用方式在以下代码片段中进行了演示：

```python
hook.before_train()
for iter in range(start_iter, max_iter):
    hook.before_step()
    trainer.run_step()
    hook.after_step()
iter += 1
hook.after_train()
```

**笔记**
1. 在 hook 方法中，用户可以访问 ``self.trainer`` 以访问有关上下文的更多属性（例如，模型、当前迭代或配置，如果使用 ``DefaultTrainer``）。
2. 在 ``before_step()`` 中执行某些操作的hook通常可以在 ``after_step()`` 中等效地实现。如果 hook 花费了很多时间，强烈建议在 ``after_step()`` 而不是 ``before_step()`` 中实现 hook。惯例是 ``before_step()`` 应该只花费可以忽略不计的时间。

遵循此约定将允许关心 ``before_step()`` 和 ``after_step()`` 之间差异的挂钩（例如，计时器）正常运行。

</details>

3. 使用 trainer+hook 系统意味着总会有一些不规范的行为无法得到支持，尤其是在研究中。出于这个原因，我们有意保持trainer+hook 系统最小化，而不是强大。如果这样的系统无法实现任何目标，那么从 ``tools/plain_train_net.py`` 开始手动实现自定义训练逻辑会更容易。


## 3. 衡量标准的日志

在训练期间，detectron2 模型和训练器将指标放入集中式 [EventStorage](https://detectron2.readthedocs.io/en/latest/modules/utils.html#detectron2.utils.events.EventStorage)。您可以使用以下代码访问它并向其记录指标：

```python
from detectron2.utils.events import get_event_storage

# inside the model:
if self.training:
  value = # compute the value from inputs
  storage = get_event_storage()
  storage.put_scalar("some_accuracy", value)
```

有关详细信息，请参阅其文档。
然后使用 ``EventWriter`` 将指标写入各种目的地。 ``DefaultTrainer`` 启用一些具有默认配置的 ``EventWriter``。有关如何自定义它们的信息，请参见上文。


## 4. Ultrasound VID的训练流程

这部分基于项目代码介绍Ultrasound VID中的训练流程。相关代码位于``tools/train_net.py``中。

首先，获取bash输入的参数，启动分布式训练。

```python
from detectron2.engine import launch

args = default_argument_parser().parse_args()
print("Command Line Args:", args)
# e.g. Command Line Args: Namespace(config_file='configs/RDN-LSTM/BUS_BasicConfig_20221108_hardmining_fp_thresh0.6_by_video_rate0.1_fold0_iter_10w.yaml', dist_url='tcp://127.0.0.1:59166', eval_only=False, machine_rank=0, num_gpus=8, num_machines=1, opts=[], resume=False)
launch(
    main,
    args.num_gpus,
    num_machines=args.num_machines,
    machine_rank=args.machine_rank,
    dist_url=args.dist_url,
    args=(args,),
)   # 在分布式训练的情况下，执行main(args)
```

函数``main``根据参数配置，进行训练前的准备

```python
from detectron2.utils.comm import is_main_process
from detectron2.checkpoint import DetectionCheckpointer
from ultrasound_vid.utils.miscellaneous import backup_code
from datetime import datetime

def main(args):
    cfg = setup(args)   # 解析配置文件，根据配置文件初始化训练所需要的参数
    output_dir = cfg.OUTPUT_DIR
    if is_main_process():   # 只在主进程中执行
        hash_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_code(
            os.path.abspath(os.path.curdir),
            os.path.join(output_dir, "code_" + hash_tag),
        )   # 备份代码
    if args.eval_only:  # 验证模式
        model = Trainer.build_model(cfg)    # 这里没有对Trainer进行实例化，只调用了其中的方法，因为Trainer.test中写了测试需要的一切
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )   # 载入checkpoint
        res = Trainer.test(cfg, model)  # 测试模式
        return res

    trainer = Trainer(cfg)  # 训练模式
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()
```

类``Trainer``继承于``DefaultTrainer``，包括训练所需要的所有内容；``classmethod`` 修饰符对应的函数不需要实例化，不需要 ``self`` 参数，但第一个参数需要是表示自身类的 ``cls`` 参数，可以来调用类的属性，类的方法，实例化对象等。

```python
from detectron2.engine import DefaultTrainer
import detectron2.utils.comm as comm
from detectron2.evaluation import DatasetEvaluator

from ultrasound_vid.data import (
    build_video_detection_train_loader,
    build_video_detection_test_loader,
    build_video_detection_train_hardmining_fp_loader,
)

from torch.nn.parallel import DistributedDataParallel
class Trainer(DefaultTrainer):
    def __init__(self, cfg):
        """
        Set "find_unused_parameters=True" to prevent empty gradient bug.
        Set "refresh period" to refresh dataloader periodicly when datasets are
        modified during training.
        """
        super().__init__(cfg)
        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                self.model.module,
                device_ids=[comm.get_local_rank()],
                broadcast_buffers=False,
                find_unused_parameters=True,
                check_reduction=False,
            )
            self.model = model

        self.refresh_period = self.cfg.DATALOADER.REFRESH_PERIOD
        if self.refresh_period > 0:
            self.register_hooks(
                [RefreshDataloaderHook(self.cfg.DATALOADER.REFRESH_PERIOD)]
            )

            # Ugly! Trying to solve this.
            self.sample_name = cfg.DATALOADER.SAMPLER_TRAIN
            self.frame_sampler = (
                self.data_loader.sampler.data_source._map_func._obj.frame_sampler
            )

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return DatasetEvaluator()

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_video_detection_test_loader(cfg, dataset_name)

    @classmethod
    def build_train_loader(cls, cfg):
        if cfg.DATASETS.SAMPLE_FP_BY_VIDEO:
            return build_video_detection_train_hardmining_fp_loader(cfg)
        return build_video_detection_train_loader(cfg)

    @classmethod
    def build_optimizer(cls, cfg, model):
        optimizer_type = cfg.SOLVER.get("OPTIMIZER", "SGD")
        if is_main_process():
            print(f"Using optimizer {optimizer_type}")
        if optimizer_type == "SGD":
            optimizer = super().build_optimizer(cfg, model)
            return optimizer

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for key, value in model.named_parameters(recurse=True):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "backbone" in key and "BACKBONE_MULTIPLIER" in cfg.SOLVER:
                lr = lr * cfg.SOLVER.BACKBONE_MULTIPLIER
            if "sample_module" in key and "LSTM_MULTIPLIER" in cfg.SOLVER:
                lr = lr * cfg.SOLVER.LSTM_MULTIPLIER
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        if optimizer_type.upper() == "ADAMW":
            optimizer = torch.optim.AdamW(
                params,
                cfg.SOLVER.BASE_LR,
                betas=cfg.SOLVER.ADAM_BETA,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                `cfg.DATASETS.TEST`.

        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger("ultrasound_vid")
        output_dir = cfg.OUTPUT_DIR
        os.makedirs(os.path.join(output_dir, "predictions"), exist_ok=True)
        rpn_only = cfg.MODEL.META_ARCHITECTURE == "TemporalProposalNetwork"
        # save_folder = os.path.join(output_dir, "predictions")
        results = OrderedDict()
        for _, hospital_name in enumerate(cfg.DATASETS.BUS_TEST):
            if cfg.DEVICE_WISE:
                device_list = []
                for k in DatasetCatalog.list():
                    if "@" not in k:
                        continue
                    parts = k.split("@")
                    if (
                        parts[0] == hospital_name
                        and len(parts) == 3
                        and parts[1] == cfg.DATASETS.BUS_TIMESTAMP
                    ):
                        device_list.append(parts[-1])

                # 将部分机型进行聚合，避免机型太过于分散
                group_mapping = {}
                for idx, group in enumerate(cfg.DATASETS.GROUP):
                    for item in group:
                        group_mapping[item] = idx

                device_list_group = []
                vis_group = set()
                mem = defaultdict(list)
                for device in device_list:
                    if hospital_name + "@" + device in group_mapping:
                        idx = group_mapping[hospital_name + "@" + device]
                        mem[idx].append(device)
                    else:
                        if str(device) not in vis_group:
                            device_list_group.append([device])
                            vis_group.add(str(device))
                for k in mem:
                    if str(mem[k]) not in vis_group:
                        device_list_group.append(mem[k])
                        vis_group.add(str(mem[k]))
                logger.info(pformat((hospital_name, device_list_group)))

                for device in device_list_group:
                    dataset_name = [
                        "@".join([hospital_name, cfg.DATASETS.TIMESTAMP, d])
                        for d in device
                    ]
                    data_loader = cls.build_test_loader(cfg, dataset_name)
                    if data_loader is None:
                        continue
                    results_i = inference_on_video_dataset(
                        model,
                        data_loader,
                        dataset_name,
                        save_folder=output_dir,
                        rpn_only=rpn_only,
                    )
                    if isinstance(dataset_name, list):
                        dataset_name = sorted(dataset_name)[0]
                    results[dataset_name] = results_i
                    if comm.is_main_process():
                        assert isinstance(
                            results_i, dict
                        ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                            results_i
                        )
                        logger.info(
                            "Evaluation results for {} in csv format:".format(
                                dataset_name
                            )
                        )
                        print_csv_format(results_i)
            else:
                dataset_name = "@".join(
                    ["breast_" + hospital_name, cfg.DATASETS.BUS_TIMESTAMP]
                )
                data_loader = cls.build_test_loader(cfg, [dataset_name])
                results_i = inference_on_video_dataset(
                    model,
                    data_loader,
                    dataset_name,
                    save_folder=output_dir,
                    rpn_only=rpn_only,
                )
                results[dataset_name] = results_i
                if comm.is_main_process():
                    assert isinstance(
                        results_i, dict
                    ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                        results_i
                    )
                    logger.info(
                        "Evaluation results for {} in csv format:".format(dataset_name)
                    )
                    print_csv_format(results_i)

        for _, hospital_name in enumerate(cfg.DATASETS.TUS_TEST):
            if cfg.DEVICE_WISE:
                device_list = []
                for k in DatasetCatalog.list():
                    if "@" not in k:
                        continue
                    parts = k.split("@")
                    if (
                        parts[0] == hospital_name
                        and len(parts) == 3
                        and parts[1] == cfg.DATASETS.TUS_TIMESTAMP
                    ):
                        device_list.append(parts[-1])

                # 将部分机型进行聚合，避免机型太过于分散
                group_mapping = {}
                for idx, group in enumerate(cfg.DATASETS.GROUP):
                    for item in group:
                        group_mapping[item] = idx

                device_list_group = []
                vis_group = set()
                mem = defaultdict(list)
                for device in device_list:
                    if hospital_name + "@" + device in group_mapping:
                        idx = group_mapping[hospital_name + "@" + device]
                        mem[idx].append(device)
                    else:
                        if str(device) not in vis_group:
                            device_list_group.append([device])
                            vis_group.add(str(device))
                for k in mem:
                    if str(mem[k]) not in vis_group:
                        device_list_group.append(mem[k])
                        vis_group.add(str(mem[k]))
                logger.info(pformat((hospital_name, device_list_group)))

                for device in device_list_group:
                    dataset_name = [
                        "@".join([hospital_name, cfg.DATASETS.TIMESTAMP, d])
                        for d in device
                    ]
                    data_loader = cls.build_test_loader(cfg, dataset_name)
                    if data_loader is None:
                        continue
                    results_i = inference_on_video_dataset(
                        model,
                        data_loader,
                        dataset_name,
                        save_folder=output_dir,
                        rpn_only=rpn_only,
                    )
                    if isinstance(dataset_name, list):
                        dataset_name = sorted(dataset_name)[0]
                    results[dataset_name] = results_i
                    if comm.is_main_process():
                        assert isinstance(
                            results_i, dict
                        ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                            results_i
                        )
                        logger.info(
                            "Evaluation results for {} in csv format:".format(
                                dataset_name
                            )
                        )
                        print_csv_format(results_i)
            else:
                dataset_name = "@".join(
                    ["thyroid_" + hospital_name, cfg.DATASETS.TUS_TIMESTAMP]
                )
                data_loader = cls.build_test_loader(cfg, [dataset_name])
                results_i = inference_on_video_dataset(
                    model,
                    data_loader,
                    dataset_name,
                    save_folder=output_dir,
                    rpn_only=rpn_only,
                )
                results[dataset_name] = results_i
                if comm.is_main_process():
                    assert isinstance(
                        results_i, dict
                    ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                        results_i
                    )
                    logger.info(
                        "Evaluation results for {} in csv format:".format(dataset_name)
                    )
                    print_csv_format(results_i)

        return results
```