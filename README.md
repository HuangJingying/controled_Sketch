# controled_Sketch

## Sketch Dataset

1. Abstract sketch dataset from TUB (20k, only abstract sketches): https://drive.google.com/file/d/1tFc2dNEToTLwzoAK_UCVIjJPXRLC2PAa/view?usp=drive_link

2. Abstract sketch dataset with abstract sketch - real image pairs **[sketchy](https://github.com/CDOTAD/SketchyDatabase) (70k)**: https://drive.google.com/file/d/0B7ISyeE8QtDdTjE1MG9Gcy1kSkE/view?usp=sharing&resourcekey=0-r6nB4crmdU-LK7H38xnOUw

   use images in **tx_000000000000** for both sketches and photoes

   

## Install vqgan

```
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

pip install --upgrade diffusers[torch] -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

```

```
git from https://github.com/huggingface/diffusers/tree/main/examples/vqgan

pip install -r requirements.txt  -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# Successfully installed requests-2.32.3 timm-1.0.14 tokenizers-0.20.3 tqdm-4.67.1 transformers-4.46.3

pip install . -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

```
## Set Config

1. Replace train_vqgan.py
2. Modify parameters in **vqgan_config_@hjy.yaml**

## Download Pretrained vgg19

Download vgg19 from https://download.pytorch.org/models/vgg19-dcbb9e9d.pth

mv to download_weights/

## Train VQGAN

```shell

accelerate launch train_vqgan.py \
  --train_data_dir="./sketchdataset/sketches_png.TUB/train" \
  --model_config_name_or_path="vqgan_config_@hjy.yaml" \
  --image_column=image \
  --validation_images ./sketchdataset/sketches_png.TUB/test/panda/11519.png ./sketchdataset/sketches_png.TUB/test/laptop/9544.png   \
  --resolution=256 \
  --center_crop \
  --train_batch_size=16 \
  --gradient_accumulation_steps=1 \
  --learning_rate=4e-6 \
  --use_ema \
  --dataloader_num_workers 4 \
```
