# CILMP

The Pytorch implementation of _Medical Knowledge Intervention Prompt Tuning for Medical Image Classification._


## Environment

```
conda create -n cilmp python==3.8
conda activate cilmp
git clone https://github.com/usr922/cilmp.git
cd cilmp
pip install -r requirements.txt
```


## Data Preparation

Follow each public dataset in the paper for preparation.

## Usage

### 1. Training

```bash
bash scripts/train.sh
```



### 2. Validation

```bash
bash scripts/eval.sh
```


## Acknowledgement

We sincerely thank Prof. Kaiyang Zhou for the great work [CoOp](https://github.com/KaiyangZhou/CoOp), [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch) and Dr. Muhammad Uzair Khattak for the great work [PromptSRC](https://github.com/muzairkhattak/PromptSRC). We borrow codes heavily from their repositories.

