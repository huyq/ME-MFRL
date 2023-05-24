## Requirements
tensorflow > 2.0
magent

## Install environments
# install magent
```shell
pip install magent
```
# install mpe
```shell
cd env/mpe
pip install -e .
```

## Scripts
# train in tag
```shell
python train_tag --algo me_mfppo --order 4
```

# train in battle
```shell
python train_battle --algo me_mfq --order 4
```

# train in spread
```shell
python train_spread --algo me_mfppo --order 4 
```

# test in tag
```shell
python test_tag --pred me_mfppo --prey ppo --path xxx xxx --idx  xxx xxx
```

# test in battle
```shell
python test_battle --algo me_mfq --oppo mfq --path xxx xxx --idx xxx xxx
```


