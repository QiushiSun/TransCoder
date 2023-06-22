# TransCoder

Code for EMNLP2023 Submission

## Environment & Preparing

```shell
conda create --name transcoder python=3.7
conda activate transcoder
pip install -r requirements.txt
cd TransCoder/evaluator/CodeBLEU/parser
bash build.sh
cd ../../../
cp evaluator/CodeBLEU/parser/my-languages.so build/
#make sure git-lfs installed like 'apt-get install git-lfs'
bash get_models.sh
```

## Preparing data

The dataset comes from [CodeXGLUE](https://github.com/microsoft/CodeXGLUE).

```shell
mkdir data
cd data
pip install gdown
gdown https://drive.google.com/uc?export=download&id=1BBeHFlKoyanbxaqFJ6RRWlqpiokhDhY7
unzip data.zip
rm data.zip
```

### Preparing local path

Direct WORKDIR, HUGGINGFACE_LOCALS in run.sh, run_few_shot.sh to your path.

## Finetune

```bash
export MODEL_NAME=
export TASK=
export SUB_TASK=
# to run one task
bash run.sh $MODEL_NAME $TASK $SUB_TASK
# to run few shot
bash run_few_shot.sh $MODEL_NAME $TASK $SUB_TASK
```

  `MODEL_NAME` can be any one of `["roberta", "codebert", "graphcodebert", "unixcoder","t5","codet5","bart","plbart"]`.

  `TASK` can be any one of `['summarize', 'translate', 'refine', 'generate', 'defect', 'clone']`. 

| Category | Dataset   | Task              | Sub_task(LANG)                                     | Type           | Category | Description                                                                                                                  |
| -------- | --------- | ----------------- | -------------------------------------------------- | -------------- | -------- | ---------------------------------------------------------------------------------------------------------------------------- |
| C2C      | BCB       | clone             | [] (java)                                          | bi-directional | encoder  | code summarization task on[CodeSearchNet](https://arxiv.org/abs/1909.09436) data with six PLs                                   |
| C2C      | Devign    | defect            | [] (c)                                             | bi-directional | encoder  | text-to-code generation on[Concode](https://aclanthology.org/D18-1192.pdf) data                                                 |
| C2C      | CodeTrans | translate         | ['java-cs', 'cs-java’]                            | end2end        | en2de    | code-to-code translation between[Java and C#](https://arxiv.org/pdf/2102.04664.pdf)                                             |
| C2C      | Bugs2Fix  | refine(repair)    | ['small','medium'] (java)                          | end2end        | en2de    | code refinement on[code repair data](https://arxiv.org/pdf/1812.08693.pdf) with small/medium functions                          |
| C2T      | CodeSN    | summarize         | ['java', 'python', 'javascript','php','ruby','go'] | end2end        | en2de    | code defect detection in[C/C++ data](https://proceedings.neurips.cc/paper/2019/file/49265d2447bc3bbfe9e76306ce40a31f-Paper.pdf) |
| T2C      | CONCODE   | generate(concode) | [] (java)                                          | end2end        | en2de    | code clone detection in[Java data](https://arxiv.org/pdf/2002.08653.pdf)TransCoder                                              |

## Run TransCoder

```bash
export MODEL_NAME=
export TASK=
bash run_transcoder.sh $MODEL_NAME $TASK 
```

`TASK` can be any one of `['cls2translate','translate2cls','cls2summarize','summarize2cls','translate2summarize','summarize2translate','cross2java','cross2php','cross2ruby','cross2python','cross2go','cross2javascript']`
