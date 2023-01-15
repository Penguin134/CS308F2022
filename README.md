# CV Project -- TransUNet
This repo is a CV project on CS308 2022F, SUSTech.

## Usage

### 1. Download Google pre-trained ViT models
* [Get models in this link](https://console.cloud.google.com/storage/vit_models/): R50-ViT-B_16, ViT-B_16, ViT-L_16...
```bash
wget https://storage.googleapis.com/vit_models/imagenet21k/{MODEL_NAME}.npz &&
mkdir model/vit_checkpoint/imagenet21k &&
mv {MODEL_NAME}.npz model/vit_checkpoint/imagenet21k/{MODEL_NAME}.npz
```

### 2. Prepare data

You need to download [VOC2012 Dataset For Segmentation](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit)
The directory structure of the whole project is as follows:
```bash
.
├── TransUNet
│   ├──datasets
│   │       └── dataset_*.py
│   ├──train.py
│   ├──test.py
│   └──...
├── model
│   └── vit_checkpoint
│       └── imagenet21k
│           ├── R50+ViT-B_16.npz
│           └── *.npz
└── data
    └──VOC2012
        ├── JPEGImages
        │   ├── 2007_000027.jpg
        │   └── *.jpg
        ├── SementationClass
        │   ├── 2007_000032.png
        │   └── *.png
        ├── train.txt
        ├── val.txt   
```


### 3. Environment

Please prepare an environment with python=3.7, and then use the command "pip install -r requirements.txt" for the dependencies.

### 4. Train/Test

- Run the train script on synapse dataset. 

```bash
python train.py --dataset VOC2012 --vit_name R50-ViT-B_16
```

- Run the test script on synapse dataset. 

```bash
python test.py --dataset VOC2012 --vit_name R50-ViT-B_16
```

## Reference
* [TransUnet](https://github.com/Beckschen/TransUNet)
