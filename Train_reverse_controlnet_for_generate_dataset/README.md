
### Train a Reverse ControlNet to generate abstract sketches from concrete sketch or nature image

#### **Step1: Training Stable Diffusion (VAE-based) for Reverse ControlNet** – Ongoing

**Goal**: Fine-tune a pre-trained Stable Diffusion model to generate abstract sketches, which will be used as the backbone for Reverse ControlNet.
  
**Training**: Running on 70k sketches from the Sketchy dataset, using a Linux GPU (A100).

- **Option 1: directly finetune whole weights of sdv1.5**
    - **1. Prepare Dataset**:  **Sketchy**, `/Users/jingyinghuang/Dataset/sketch/sketchy/rendered_256x256/256x256/sketch/tx_000000000000`
  
    - **2. Create Metadata for dataset**: Necessary metadata for the local dataset is being created.
    
    `python write_meta_json_for_diffuser.py ./diffusers/examples/vqgan/sketchdataset/sketchy/tx_000000000000`
    
    Example output:
    
    ```json
    {"file_name": "ape/n02480495_12444-4.png", "caption": "abstract sketch of an ape"}
    {"file_name": "ape/n02483708_5763-6.png", "caption": "abstract sketch of an ape"}
    {"file_name": "ape/n02481823_9892-2.png", "caption": "abstract sketch of an ape"}
    ```
    
    - **3. Training Script used**: `/Users/jingyinghuang/Code/AIGC/diffusers-main/examples/text_to_image/train_text_to_image_saveimg.py` , with small modification including saving validation images
    
    - **4. Training settings**:
      - **Batch size**: 16
      - **Steps**: 160,000
      - **Pre-trained model**: Stable Diffusion v1.5
      - **Training command**:
    
```bash
python ./diffusers/examples/text_to_image/train_text_to_image_saveimg.py \
    --pretrained_model_name_or_path="./stable-diffusion-v1-5" \
    --train_data_dir="./diffusers/examples/vqgan/sketchdataset/sketchy/tx_000000000000" \
    --caption_column="caption" \
    --use_ema \
    --resolution=256 --center_crop --random_flip \
    --train_batch_size=16 \
    --gradient_accumulation_steps=1 \
    --gradient_checkpointing \
    --max_train_steps=160000 \
    --learning_rate=1e-05 \
    --max_grad_norm=1 \
    --lr_scheduler="constant" --lr_warmup_steps=0 \
    --checkpointing_steps=2000 \
    --validation_prompt "abstract sketch of a elephant" "abstract sketch of a flower" "abstract sketch of a house" \
    --output_dir="./sdv1-5_finetune_forsketch_v2"

```
    

  - **Generated Output**: The model produces abstract sketch images based on prompts.
      - **Observation**: While the images match the white background and basic sketch format, the ***output is less "abstract" compared to the training data, featuring more lines and curves.***

- **Option 2: use LoRA finetune sdv1.5**
    - **4. Training settings**:
      - **Rans**: 8 or 16, also can try 32
      - **Batch size**: 8
      - **Steps**: recommand between (80k to 15k)
      - **Pre-trained model**: Stable Diffusion v1.5
      - **Training command**:

```shell
export MODEL_NAME="./stable-diffusion-v1-5"
export OUTPUT_DIR="./lora_forsketch_sdv1-5"
export DATASET_NAME="./diffusers/examples/vqgan/sketchdataset/sketchy/tx_000000000000"
  accelerate launch ./diffusers/examples/text_to_image/train_text_to_image_lora_saveimg.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --caption_column="caption"\
  --dataloader_num_workers=8 \
  --resolution=512 \
  --center_crop \
  --random_flip \
  --train_batch_size=8 \
  --rank=16 \
  --max_train_steps=50000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --checkpointing_steps=3000 \
  --validation_epochs=200 \
  --num_validation_images=18 \
  --validation_prompt "abstract sketch of a elephant" "abstract sketch of a flower" "abstract sketch of a house" "abstract sketch of a car" "abstract sketch of a desk" "abstract sketch of a tree"
 ```