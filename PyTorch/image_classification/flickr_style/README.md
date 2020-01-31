# Finetuning for Image Classification

## Setup


### Data

Get the data:
```bash
# Usage: python assemble_data.py image_path train_file test_file images_per_style
$ python assemble_data.py images train.txt test.txt 500
```

Split Data:

The previous step only creates train and test sets. We'll also create a validation set using the following command:
```bash
python helper_scripts/split_data.py /path/to/train.txt split_ratio /path/to/output/fldr/
python helper_scripts/split_data.py /hpctmp/`whoami`/project/flickr_style/train.txt 0.3 /hpctmp/`whoami`/project/flickr_style/
```

### Pretrained Weights

As there is no internet connection on Volta nodes, use_pretrained will not be able to download the weights.
On login nodes, download the pretrained weights first.

For ResNet:
```python
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

```

For example:
```bash
$ mkdir -p /hpctmp/`whoami`/torch_models
$ cd /hpctmp/`whoami`/torch_models
$ wget https://download.pytorch.org/models/resnet50-19c8e357.pth
```

You'd then need to set the environment variable `TORCH_MODEL_ZOO` to
````
TORCH_MODEL_ZOO=/hpctmp/`whoami`/torchmodels
```


## Finetuning:

```bash
# python /path/to/train.csv /path/to/test.csv /path/to/val.csv /path/to/root/image_fldr learning_rate batch_size epochs checkpoint_fldr use_pretrained top_only
$ python /path/to/train.csv /path/to/test.csv /path/to/val.csv /path/to/root/images 0.0001 32 100 checkpoint_fldr True True  
```