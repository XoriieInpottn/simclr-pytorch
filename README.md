# SimCLR Implementation (PyTorch)

This is a pytorch implementation of "SimCLR - A Simple Framework for Contrastive Learning of Visual Representations".

## Prepare Dataset

### Data Format

In this implementation, all data is stored in DocSet (*.ds) files. To support ds files in your system, install the "dcoset" package for your python environment.

```bash
pip3 install docset
```

You can directly download the preprocessed STL10 dataset from Baidu net disk (Link: https://pan.baidu.com/s/1Xog3Rz_8tBqFrnGjd8aiYQ, Password: jb6b). After downloading, you get 3 files:

```
stl10/
----unlabeled.ds
----train.ds
----test.ds
```

execute the following command in your command line:

```bash
docset stl10/train.ds
```

you can preview the dataset:

```
stl10/train.ds
Count: 5000, Size: 132.1 MB, Avg: 27.1 KB/sample

Sample 0
    "feature": ndarray(dtype=uint8, shape=(96, 96, 3))
    "label": 1
Sample 1
    "feature": ndarray(dtype=uint8, shape=(96, 96, 3))
    "label": 5
...
Sample 4999
    "feature": ndarray(dtype=uint8, shape=(96, 96, 3))
    "label": 5
```

## Start Training

```
usage: train.py [-h] [--gpu GPU] --data-path DATA_PATH
                [--batch-size BATCH_SIZE] [--num-epochs NUM_EPOCHS]
                [--max-lr MAX_LR] [--weight-decay WEIGHT_DECAY]
                [--optimizer OPTIMIZER] [--base-model BASE_MODEL]
                [--emb-size EMB_SIZE] [--proj-size PROJ_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  --gpu GPU             Which GPU to use.
  --data-path DATA_PATH
                        Path of the directory that contains the data files.
  --batch-size BATCH_SIZE
                        Batch size.
  --num-epochs NUM_EPOCHS
                        The number of epochs to train.
  --max-lr MAX_LR       The maximum value of learning rate.
  --weight-decay WEIGHT_DECAY
                        The weight decay value.
  --optimizer OPTIMIZER
                        Name of the optimizer to use.
  --base-model BASE_MODEL
                        The base model.
  --emb-size EMB_SIZE   The embedding dimension.
  --proj-size PROJ_SIZE
                        The projection head dimension.
```

For example, the following command is to train the model using ResNet-18 with STL10 dataset on GPU: 0.

```bash
python3 train.py --data-path /path/of/data/stl10/ --base-model resnet18 --gpu 0
```

## Results on STL10

| Base Net  | Project Head | Feature | Batch Size | Optimizer | Learning Rate | Weight Decay | Epochs | Top 1 Accuracy |
| :-------: | :----------: | :-----: | :--------: | :-------: | :-----------: | :----------: | :----: | :------------: |
|  VGG-16   |     128      |   512   |    256     |   AdamW   |   1e-3 to 0   |     0.3      |  100   |     68.56%     |
| ResNet-18 |     128      |   512   |    256     |   AdamW   |   1e-3 to 0   |     0.3      |  100   |     78.94%     |
| ResNet-34 |     128      |   512   |    256     |   AdamW   |   1e-3 to 0   |     0.3      |  100   |     79.34%     |

