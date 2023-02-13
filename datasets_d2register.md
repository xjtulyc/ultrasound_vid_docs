# d2的注册机制

在Detectron2 中，经常会对一个类或者函数进行注册：

```python
@ROI_HEADS_REGISTRY.register()
class DynamicHead(nn.Module):
```

关于这种操作，必须要明确两点：

## 1. 目的

### 1.1. 注册机制的使用方法
首先来看一下注册机制是如何使用的：
```python
registry_machine = Registry('registry_machine')

@registry_machine.register()
def print_hello_world(word):
    print('hello {}'.format(word))


@registry_machine.register()
def print_hi_world(word):
    print('hi {}'.format(word))

if __name__ == '__main__':

    cfg1 = 'print_hello_word'
    registry_machine.get(cfg1)('world')

    cfg2 = 'print_hi_word'
    registry_machine.get(cfg2)('world')
```

可以看到，如果创建了一个Registry的对象，并在方法/类定义的时候用装饰器装饰它，则可以通过 registry_machine.get(方法名)的 办法来间接的调用被注册的函数。

注册机可以使用和函数名相同的字符串（如上例中的`print_hello_word`和`print_hi_world`）来返回一个函数对象，我们可以传入参数执行函数的内容。

### 1.2. 为什么使用注册类
对于detectron2这种，需要支持许多不同的模型的大型框架，理想情况下所有的模型的参数都希望写在配置文件中，那问题来了，如果我希望根据我的配置文件，决定我是需要用VGG还是用ResNet ，我要怎么写呢？

如果是我，我可能会写出这种可扩展性超级低的暴搓的代码：

```python
if class_name == 'VGG':
    model = build_VGG(args)
elif class_name == 'ResNet':
    model = build_ResNet(args)
```

但是如果用了注册类，代码就是这样的：

```python
class_name = 'VGG' # 'ResNet'
model = model_registry(class_name)(args)
```

## 2. 具体实现细节

这部分就直接展示注册类的代码了，有兴趣的朋友可以研究一下其中的细节，个人觉得对装饰器的应用是非常的好了

```python
# from detectron2.utils.registry import Registry
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
class Registry(object):

    def __init__(self, name):
        """
        Args:
            name (str): the name of this registry
        """
        self._name = name

        self._obj_map = {}

    def _do_register(self, name, obj):
        assert (
            name not in self._obj_map
        ), "An object named '{}' was already registered in '{}' registry!".format(name, self._name)
        self._obj_map[name] = obj

    def register(self, obj=None):
        """
        Register the given object under the the name `obj.__name__`.
        Can be used as either a decorator or not. See docstring of this class for usage.
        """
        if obj is None:
            # used as a decorator
            def deco(func_or_class):
                name = func_or_class.__name__
                self._do_register(name, func_or_class)
                return func_or_class

            return deco

        # used as a function call
        name = obj.__name__
        self._do_register(name, obj)

    def get(self, name):
        ret = self._obj_map.get(name)
        if ret is None:
            raise KeyError("No object named '{}' found in '{}' registry!".format(name, self._name))
        return ret



registry_machine = Registry('registry_machine')

registry_machine.register()
def print_hello_world(word):
    print('hello {}'.format(word))


registry_machine.register()
def print_hi_world(word):
    print('hi {}'.format(word))

if __name__ == '__main__':

    cfg1 = 'print_hello_word'
    registry_machine.get(cfg1)('world')

    cfg2 = 'print_hi_word'
    registry_machine.get(cfg2)('world')
```


## 3. d2的数据集的注册
### 3.1. 数据集注册
数据集注册的实际意义：将数据集的一些信息提前保存在类中，方便全局调用。

在Detectron2中，类`DatasetCatalog`管理所有的数据集。类中的只有一个字典类型变量`_REGISTERED`，保存数据集的名称以及对应的加载函数。

类`MetadataCatalog`保存了数据集具体的一些信息。类中只有一个字典类型变量`_NAME_TO_META`，保存数据集名称与对应的类`Metadata`。

类`Metadata`保存我们实际需要的数据集所有信息，如数据集中类别ID与连续ID的对应，类别名称，数据集路径，novel类，base类等等。

**实际注册的步骤：**

1. 将数据集名称和对应的加载函数注册到类``DatasetCatalog``中


```python
DatasetCatalog.register(
        name,
        lambda: load_coco_json(annofile, imgdir, metadata, name,
                               extra_annotation_keys=['id']),
    )
```
2. 将数据集的具体信息保存在类``MetadataCatalog``中

```python
MetadataCatalog.get(name).set(
        json_file=annofile,
        image_root=imgdir,
        evaluator_type="coco",
        dirname="datasets/coco",
        **metadata,
    )
```

代码保存在``data/builtin``,通过函数``register_all_coco``注册所有的数据集
### 3.2. 加载数据集
在Detectron2中，训练或者测试使用到的数据集定义在``config.yaml``中，如下形式：

```yaml
DATASETS:
  TRAIN: ('coco_trainval_base',)
```

代码中加载数据集在类``Trainer``的初始化部分，具体写法如下：

```python
def __init__(self, cfg):
        data_loader = self.build_train_loader(cfg)
```

实例化类Trainer后，其中的属性data_loader就已经加载好了数据集。所以实质上加载数据集的代码就是在``self.build_train_loader()``函数中。 函数里面的操作有很多，实际上加载数据的代码就是

```python
dataset_dicts = [DatasetCatalog.get(dataset_name) for dataset_name in dataset_names]
```

之前注册的时候提到过，类``DatasetCatalog``里面保存的是数据集名称以及对应的加载函数。所以``dataset_dicts``保存的是函数``load_coco_json()``的返回值。

**注意**：函数``load_coco_json()``就是负责读取数据集的图片以及标签json文件，要仔细阅读。

``dataset_dicts``保存了数据集中的所有图片和相对应的标签数据，下一步就是根据需求进行数据增强的操作，这一部分的代码写法如下：

```python
if mapper is None:
    mapper = DatasetMapper(cfg, True)
dataset = MapDataset(dataset, mapper)
```
所以数据增强部分的代码在类``DatasetMapper``中，如果对图片或者标签要进行修改的话，就应该关注类``DatasetMapper``。

经过上述描述之后，实例化类``Trainer``的时候就已经加载好了数据集，保存在``Trainer.data_loader``中。

### 3.3. 读取数据参加训练
``Trainer.data_loader``中数据是可以直接参加训练或者推理。

Detectron2框架中循环迭代数据的代码在类``SimpleTrainer.run_step(self)``，具体写法如下：

```python
data = next(self._data_loader_iter)  # 获取当前batch的训练数据
loss_dict = self.model(data)  # 数据输入到模型，输出损失
```

至此，就已经成功加载并读取了指定数据集中的数据，并将数据送入模型获取输出。