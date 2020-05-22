# DGAM-Weakly-Supervised-Action-Localization
Code for our paper "[Weakly-Supervised Action Localization by Generative Attention Modeling](https://arxiv.org/abs/2003.12424)" by [Baifeng Shi](https://bfshi.github.io), 
[Qi Dai](https://scholar.google.com/citations?hl=en&user=NSJY12IAAAAJ), [Yadong Mu](http://www.muyadong.com/index.html),
[Jingdong Wang](https://jingdongwang2017.github.io/), **CVPR2020**.

## Requirements
Required packges are listed in `requirements.txt`. You can install by running:
```bash
pip install -r requirements.txt
```

## Dataset
We provide extracted features and corresponsing annotations for THUMOS14 (available [here](https://drive.google.com/open?id=1SuyUdug6bb5HG0rnpDkdyIVdVp119LcV))
and ActivityNet1.2 ([here](https://drive.google.com/open?id=1zwdF72z_y5TWAAHyZyMVcU6Bz_KFv5oL)). 

Before running the code, please download the target dataset and unzip it under `/data`.

## Running
You can train your own model by running:
```bash
python train_all.py
```
Note that you can configure the hyperparameters in `/lib/core/config.py`.

To test your model, you shall first go to the file `/lib/core/config.py` and change the entries `config.TEST.STATE_DICT_RGB` and `config.TEST.STATE_DICT_FLOW`,
then run:
```bash
python test.py
```

## Citation
If you find our code useful, please consider citing:
```
@article{shi2020weakly,
  title={Weakly-Supervised Action Localization by Generative Attention Modeling},
  author={Shi, Baifeng and Dai, Qi and Mu, Yadong and Wang, Jingdong},
  journal={arXiv preprint arXiv:2003.12424},
  year={2020}
}
```
