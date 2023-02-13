# d2的注册机制

在Detectron2 中，经常会对一个类或者函数进行注册：

```python
@ROI_HEADS_REGISTRY.register()
class DynamicHead(nn.Module):
```

关于这种操作，必须要明确两点：

## 1. 目的

## 1.1. 注册机制的使用方法
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

## 1.2. 为什么使用注册类
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