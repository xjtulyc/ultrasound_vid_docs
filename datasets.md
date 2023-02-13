# 数据集

## 1. 可使用的数据集

目前有乳腺和甲状腺数据集，目录如下

```shell
/projects/US/ProjectDatasets/db/breast
/projects/US/ProjectDatasets/db/thyroid
```

## 2. 软连接到数据集

d2的数据集默认在`datasets\`文件夹下，处于数据管理和企业隐私上的方便，需要使用软连接到项目数据集的位置

```shell
ln -s /projects/US/ProjectDatasets/db/breast datasets/bus_data
ln -s /projects/US/ProjectDatasets/db/thyroid datasets/tus_data
```

软连接成功后，`datasets\`文件夹下输入`ls -l`的输出应该如下所示

![](file/datasets_soft_link.png)


## 3. 数据集的注册
学习使用一个框架，首先要了解框架的基本原理，之后才应该阅读项目的代码。

d2所使用的数据集需要经过注册才能使用，下面先从一般角度介绍[d2的数据集注册机制](datasets_d2register.md)。

和其他基于d2开发的项目一样，Ultrasound VID的数据集注册相关的文件在``ultrasound_vid/data``下面。在``ultrasound_vid/data/datasets.py``中，进行TUS和BUS的数据集注册，注册函数为``register_dataset``，如下所示

```python
def register_dataset(
    jpg_root, pkl_root, anno_temp_path, lesion_classes, us_processed_data, organ
):
    dataset_to_build = {}  # dataset_name -> csv_file, num_videos
    for csv_file in pkl_root.files("*.csv"):
        timestamp = csv_file.basename().splitext()[0]
        if re.match(r"\d{8}-\d{6}.csv", csv_file.basename()) is None:
            continue
        menu = pd.read_csv(csv_file)
        for dataset in set(menu["hospital"].values):
            mask = menu["hospital"] == dataset
            device_cnt = Counter()
            for i, row in menu[mask].iterrows():
                device = get_device(row)
                device_cnt.update([device])
            # dataset_name: organ_hospital@timestamp
            dataset_to_build["@".join([organ + "_" + dataset, timestamp])] = (
                csv_file.abspath(),
                sum(device_cnt.values()),
            )
            for device in device_cnt:
                dataset_to_build[
                    "@".join([organ + "_" + dataset, timestamp, device])
                ] = (csv_file.abspath(), device_cnt[device])

        # add ALL dataset to train all hospital data
        device_cnt = Counter()
        for i, row in menu.iterrows():
            device = get_device(row)
            device_cnt.update([device])
        dataset_to_build["@".join([organ + "_ALL", timestamp])] = (
            csv_file.abspath(),
            sum(device_cnt.values()),
        )

    for dataset_name, (csv_file, num_videos) in dataset_to_build.items():
        DatasetCatalog.register(
            dataset_name,
            lambda dataset_name=dataset_name, csv_file=csv_file: load_ultrasound_annotations(
                dataset_name,
                csv_file,
                anno_temp_path,
                jpg_root,
                lesion_classes,
                us_processed_data,
            ),
        )
        MetadataCatalog.get(dataset_name).set(
            thing_classes=THING_CLASSES, num_videos=num_videos
        )
```

