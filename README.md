# API-LLM
on the limitation of LLM: enhance the LLM with external tools.

# ToolLLM

Enhance the capability of large language model (llm) with the external tool.

## Introduction


## Methodology

## Dataset

### Download

The dataset has been shared on the Google Drive, which can be found/downloaded in [API-data](https://drive.google.com/drive/folders/1UzFU-SEYOKWe-Oiu4aAW1qeFDxpgVSlE?usp=sharing).
The `v_` append in the filename indicate the version of our dataset (`v0` for current version).

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

## Todo

- [ ] Add the details of dataset, e.g., category, collecting , filter and cleaning.
- [ ] Add the learning process in methodology section (waiting for license).
- [ ] open all code for this work (recently).

