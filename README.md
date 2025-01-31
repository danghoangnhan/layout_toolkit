# Document Layout Toolkit

## Installation

``` bash
conda create --name layout_toolkit python=3.7
conda activate layout_toolkit
git clone https://github.com/danghoangnhan/layout_toolkit
cd layout_toolkit
pip install -r requirements.txt
# install pytorch, torchvision refer to https://pytorch.org/get-started/locally/
pip install torch==1.10.0+cu111 torchvision==0.11.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
# install detectron2 refer to https://detectron2.readthedocs.io/en/latest/tutorials/install.html
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html
pip install -e .
```

## Pre-trained Models

| Model            | Model Name (Path)                                                               |
|------------------|---------------------------------------------------------------------------------|
| LayoutLM-Base, Uncased  |  [microsoft/layoutlm-base-uncased](https://huggingface.co/microsoft/layoutlm-base-uncased) |
| LayoutLM-Base, Cased    |  [microsoft/layoutlm-base-cased](https://huggingface.co/microsoft/layoutlm-base-cased) |
| LayoutLM-Large, Uncased |  [layoutlm-large-uncased](https://huggingface.co/microsoft/layoutlm-large-uncased) |
layoutxlm-base | [microsoft/layoutxlm-base](https://huggingface.co/microsoft/layoutxlm-base)
| layoutlmv2-base-uncased | [microsoft/layoutlmv2-base-uncased](https://huggingface.co/microsoft/layoutlmv2-base-uncased)   |
| layoutlmv3-base  | [microsoft/layoutlmv3-base](https://huggingface.co/microsoft/layoutlmv3-base)   |
| layoutlmv3-large | [microsoft/layoutlmv3-large](https://huggingface.co/microsoft/layoutlmv3-large) |
| layoutlmv3-base-chinese | [microsoft/layoutlmv3-base-chinese](https://huggingface.co/microsoft/layoutlmv3-base-chinese) |


### Form Understanding on FUNSD

* Train

  ``` bash
  python -m torch.distributed.launch \
    --nproc_per_node=8 --master_port 4398 examples/run_funsd_cord.py \
    --dataset_name funsd \
    --do_train --do_eval \
    --model_name_or_path microsoft/layoutlmv3-base \
    --output_dir /path/to/layoutlmv3-base-finetuned-funsd \
    --segment_level_layout 1 --visual_embed 1 --input_size 224 \
    --max_steps 1000 --save_steps -1 --evaluation_strategy steps --eval_steps 100 \
    --learning_rate 1e-5 --per_device_train_batch_size 2 --gradient_accumulation_steps 1 \
    --dataloader_num_workers 8
  ```

* Test

  ``` bash
  python -m torch.distributed.launch \
    --nproc_per_node=8 --master_port 4398 examples/run_funsd_cord.py \
    --dataset_name funsd \
    --do_eval \
    --model_name_or_path HYPJUDY/layoutlmv3-base-finetuned-funsd \
    --output_dir /path/to/layoutlmv3-base-finetuned-funsd \
    --segment_level_layout 1 --visual_embed 1 --input_size 224 \
    --dataloader_num_workers 8
  ```

### Form Understanding on XFUND

An example for the LayoutLMv3 Chinese model to train and evaluate model.

#### Data Preparation

Download the chinese data in XFUND from this [link](https://github.com/doc-analysis/XFUND/releases/tag/v1.0).
The resulting directory structure looks like the following:

```
│── data
│   ├── zh.train.json
│   ├── zh.val.json
│   └── images
│      ├── zh_train_*.jpg
│      └── zh_val_*.jpg
```

* Train

  ``` bash
    python -m torch.distributed.launch \
      --nproc_per_node=8 --master_port 4398 examples/run_xfund.py \
      --data_dir data --language zh \
      --do_train --do_eval \
      --model_name_or_path microsoft/layoutlmv3-base-chinese \
      --output_dir path/to/output \
      --segment_level_layout 1 --visual_embed 1 --input_size 224 \
      --max_steps 1000 --save_steps -1 --evaluation_strategy steps --eval_steps 20 \
      --learning_rate 7e-5 --per_device_train_batch_size 2 --gradient_accumulation_steps 1 \
      --dataloader_num_workers 8
  ```

* Test

  ``` bash
  python -m torch.distributed.launch \
    --nproc_per_node=8 --master_port 4398 examples/run_xfund.py \
    --data_dir data --language zh \
    --do_eval \
    --model_name_or_path path/to/model \
    --output_dir /path/to/output \
    --segment_level_layout 1 --visual_embed 1 --input_size 224 \
    --dataloader_num_workers 8
  ```
