# SimCLR Implementation (PyTorch)

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

