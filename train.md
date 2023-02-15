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

这部分基于项目代码介绍Ultrasound VID中的训练流程