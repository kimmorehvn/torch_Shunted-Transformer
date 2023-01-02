# MAF Framework validated

### Installation 

```bash
conda create --name shunted python=3.8.0
conda activate shunted

pip install --user bcolz mxnet tensorboardX matplotlib easydict opencv-python einops --no-cache-dir -U | cat
pip install --user scikit-image imgaug PyTurboJPEG --no-cache-dir -U | cat
pip install --user scikit-learn --no-cache-dir -U | cat
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html  --no-cache-dir -U | cat
pip install --user  termcolor imgaug prettytable --no-cache-dir -U | cat
pip install --user timm==0.3.4 --no-cache-dir -U | cat
pip install --user mmcv --no-cache-dir -U | cat

# update-moreh --force --target 22.8.2

git clone https://github.com/OliverRensu/Shunted-Transformer
cd Shunted-Transformer

```
### Data

This repo downloaded data, preprocess and train in 1 file. So we do not need to download data ourself

### Fix some error

Edit the `main.py`:
* Comment line number 179 `utils.init_distributed_mode(args)`
* Add this line    `args.distributed = False` **after** line 179


Edit the `datasets.py`:
* Line 61 from `dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)` to `dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform, download = True)`
Training

```bash
python main.py --config configs/Shunted/shunted_S.py --data-path /data/dungvo/ --batch-size 128 --epochs 2 --data-set CIFAR
```



--------------------
# Original README
--------------------
# Shunted Transformer

This is the offical implementation of [Shunted Self-Attention via Multi-Scale Token Aggregation](https://arxiv.org/abs/2111.15193)
by Sucheng Ren, Daquan Zhou, Shengfeng He, Jiashi Feng, Xinchao Wang
### Training from scratch

## Training
```shell
bash dist_train.sh
```

## Model Zoo
The checkpoints can be found at [Goolge Drive](https://drive.google.com/drive/folders/15iZKXFT7apjUSoN2WUMAbb0tvJgyh3YP?usp=sharing), [Baidu Pan](https://pan.baidu.com/s/1a9nVWpw2SzP0csCBCF8DNw) (code:hazr) (Checkpoints of the large models are coming soon.)

| Method           | Size | Acc@1 | #Params (M) |
|------------------|:----:|:-----:|:-----------:|
| Shunted-T        |  224 |  79.8 |    11.5     |
| Shunted-S        |  224 |  82.9 |    22.4     |
| Shunted-B        |  224 |  84.0 |    39.6     |


## Citation
```shell
@misc{ren2021shunted,
      title={Shunted Self-Attention via Multi-Scale Token Aggregation}, 
      author={Sucheng Ren and Daquan Zhou and Shengfeng He and Jiashi Feng and Xinchao Wang},
      year={2021},
      eprint={2111.15193},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```