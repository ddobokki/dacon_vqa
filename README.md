# dacon_vqa
월간 데이콘 이미지 기반 질의 응답 AI 경진대회

# system info
```
os:
Ubuntu 18.04.6 LTS

GPU:
Driver Version: 525.60.13
CUDA Version: 12.0 
NVIDIA GeForce RTX 3090(x4)
```
# folders
```
├── arguments
│   ├── __init__.py
│   ├── DatasetsArguments.py
│   ├── ModelArguments.py
│   └── MyTrainingArguments.py
├── data
│   └── open.zip
├── shell_scripts
│   └── run_train.sh
├── inference.ipynb
├── literal.py
├── preprocess.py
├── train.py
├── utils.py
├── README.md
└── requirements.txt
```
# unzip data
```
$ cd data
data$ unzip open.zip
```

# code formatter
```
black *.py --line-length 119
```


# run
```
python=3.8
``````
## env setting
```
$ python3.8 -m venv .venv
$ source .venv/bin/activate
$ pip install --upgrade pip
$ pip install -r requiremets.txt
```

## preprocess(for datasets)
```
$ python preprocess.py
```

## train
```
$ bash shell_scripts/run_train.sh
```

## inference
inference.ipynb 실행