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

## Results on STL10

| Base Net  | Project Head Size | Feature Size | Optimizer |   Learning Rate   | Weight Decay | Epochs | Top 1 Accuracy |
| :-------: | :---------------: | :----------: | :-------: | :---------------: | :----------: | :----: | :------------: |
| ResNet-18 |        128        |     512      |   AdamW   | max: 1e-3, min: 0 |     0.3      |  100   |     78.94%     |

