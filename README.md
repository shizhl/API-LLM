# API-LLM
on the limitation of LLM: enhance the LLM with external tools.

## Introduction

The existing foundation model has shown remarkable intrinsic performance in most downstream tasks of NLP. However, how to extend the extrinsic ability of LLM via external APIs still needs to explore. Specifically, we first collect a tool-embedded dataset to support our task via SELF-INSTRUCT. Based on this dataset, we propose a multi-level learning method to supervise the model learning to use API.

Our dataset is about 61k, which includes about 100 APIs, e.g., SEARCH, CALCULATOR, and WEATHER SYSTEM. Each example in the dataset contains multiple API calling to solve the specific task. We also mix some text generation and commonsense reasoning datasets to maintain the performance of our foundation models. We will publish our dataset as soon as possible.


## Methodology

## Dataset
### Download
The dataset has been shared on the Google Drive, which can be found/downloaded in API-data (will publish recently).
The `v_` append in the filename indicate the version of our dataset (`v2` for current version, please select the latest version for training and evaluation).

### Data preparation


### Data cleaning and filter


### Data description
```json
{
  "api": "The api for solve the specific task",
  "number": "The number for calling API",
  "prompt": "The prompt for generating this example",
  "task": "The task name",
  "question": "The specific query based on the API in the this task",
  "_answer": "The solution to solve problem in the format of chain of thought (COT), where the above APIs are called back"
}
```
A concrete example:

```json
{
    "api": [
        [
            "CAL",
            "expression: 2500/5",
            "%s1",
            "CAL(expression: e)->float: calculate the result of expression `e`, e.g. 1+2, 1/3, 4*5 and 7-1."
        ],
        [
            "CAL",
            "expression: 2*%s1",
            "%s2",
            "CAL(expression: e)->float: calculate the result of expression `e`, e.g. 1+2, 1/3, 4*5 and 7-1."
        ],
        [
            "CAL",
            "expression: %s2-200",
            "%s3",
            "CAL(expression: e)->float: calculate the result of expression `e`, e.g. 1+2, 1/3, 4*5 and 7-1."
        ]
    ],
    "number": 3,
    "prompt": "According to the ratio, for every 5 parts that Johnson gets, Mike gets 2 parts.Since Johnson got $2500, each part is therefore $2500/5 = $<<2500/5=500>>500.Mike will get 2*$500 = $<<2*500=1000>>1000.After buying the shirt he will have $1000-$200 = $<<1000-200=800>>800 left. ### 800",
    "question": "The profit from a business transaction is shared among 2 business partners, Mike and Johnson in the ratio 2:5 respectively. If Johnson got $2500, how much will Mike have after spending some of his share on a shirt that costs $200?",
    "_answer": "According to the ratio, for every 5 parts that Johnson gets, Mike gets 2 parts. Since Johnson got $2500, each part is therefore [CAL(2500/5) -> %s1].Mike will get 2*$%s1 = [CAL(2*%s1) -> %s2]. After buying the shirt, he will have $%s2-$200 = [CAL(%s2-200) -> %s3] left. ### 800",
    "task": "cal"
}
```
## Set up
1. python 3.9 
2. pytorch lightning (1.9.0)
3. Deepspeed (deepspeed in pytorch lightning)
4. transformer (install from source)
5. pytorch (torch 1.11)
6. tqdm
7. openai (only for collecting data)

<span style="color: red">**Note:**</span>
1. 如何配置deepspeed +transformers: https://github.com/stas00/porting/blob/master/transformers/deepspeed/DeepSpeed_on_colab_CLI.ipynb
2. transformer 必须从源构建，不能直接pip install transformer
3. pytorch lightning需要1.9版本，不能直接pip(否则安装的是2.0+版本)
4. deepspeed配置可能需要高版本gcc

## Usage
### How to train the model?
details of our parameters.
```text
torchrun --nnodes <节点数量> --nproc_per_node <每个节点gpu数量>  --master_port <端口号>  \
train.py --per_device_train_batch_size <训练时的batchsize> \
      --num_device_per_node <节点数量>  \
      --num_works <dataloader的进程数量> \
      --strategy <pytorch lightning的加速策略> \
      --gradient_accumulation_steps <梯度累计大小> \
      --model_name_or_path <huggingface模型的路径> \
      --train_data_path <训练数据的路径，支持json以及jsonl格式> \
      --eval_data_path <测试数据的路径，支持json以及jsonl格式>  \
      --do_train <True/False, 是/否训练> \
      --do_eval <True/False, 是/否做测试> \
      --naive <int, 测试模式1的数量> \
      --in_domain <int, 测试模式2的数量> \
      --cross_domain <int, 测试模式3的数量> 
```
To reproduce our training process, we provide the following steps and commands as a reference.
- [ ] Download our dataset and place it in the same directory as the `train.py` file.
- [ ] prepare the initial weight of the backbone model (download from the [huggingface](https://huggingface.co/models) or our provided [checkpoint](https://drive.google.com/drive/folders/1UzFU-SEYOKWe-Oiu4aAW1qeFDxpgVSlE?usp=sharing)).
- [ ] Check out and config the environment based on the `Set up` section, e.g., PyTorch, PyTorch lightning, Deepspeed, and gcc version.
- [ ] check out the number and size of GPU in your environment to modify the above command line arguments, e.g., `per_device_train_batch_size`, `nproc_per_node`, and `num_device_per_node`.


Due to the resource limitation, we use the `EleutherAI/gpt-neo-2.7B` s backbone. And the real commend that run in our environment (4 $\times$ 24G 3090) is:
```text
torchrun --nnodes 1 --nproc_per_node 4  --master_port 9999  \
train.py --per_device_train_batch_size 2 \
      --num_device_per_node 4  \
      --num_works 12 \
      --strategy deepspeed_stage_3_offload \
      --gradient_accumulation_steps 32 \
      --model_name_or_path EleutherAI/gpt-neo-2.7B \
      --train_data_path <path> \
      --eval_data_path <path> \
      --do_train True \
      --do_eval False \
      --checkpoint_every_n_steps 50 \
      --checkpoint_every_n_epochs 0
```

We employ the `deepspeed_stage_3_offload`  strategy to accelerate the training process.

For better performance, we suggest to alternative the `EleutherAI/gpt-neo-2.7B`  with the other large language models, e.g., `decapoda-research/llama-7b-hf` and `decapoda-research/llama-13b-hf`.

### How to evaluate or inference with the ckpt?

```text
domain: the zero (domain=0) or few (domain=1) shot 
n: the example provided in the prompt when conducting the few shot.
```

running the following command.

1. To evaluate the gpt-based model, e.g., gpt-large, gpt-xl
```text
python inference.py \
      --model_name_or_path gpt-neo \
      --data_path <path> \
      --domain 0 \
      --n 3 \
```

2. To evaluate the ChatGLM
```text
python inference.py \
      --model_name_or_path chatglm \
      --data_path <path> \
      --domain 0 \
      --n 3 \
```

3. To evaluate the Llama-based model
```text
python inference.py \
      --model_name_or_path llama \
      --data_path <path> \
      --domain 0 \
      --n 3 \
```

4. To evaluate the `checkpoint` saved in the training process

```text
python inference.py \
      --model_name_or_path EleutherAI/gpt-neo-2.7B \
      --data_path <path> \
      --save_ckpt_path <path> \
      --huggingface_ckpt_path <path> \
      --domain 1 \
      --n 3
```

6. To evaluate the ChatGPT/Davinci

```text
python inference.py \
      --model_name_or_path chatgpt/davinci \
      --data_path <path> \
      --domain 1 \
      --n 3
```


## Todo
We will soon complete the following work:
- [ ] Add the details of dataset, e.g., category, collecting , filter and cleaning.
- [ ] Add the learning process in methodology section.
- [ ] Add the motivation for this work.


