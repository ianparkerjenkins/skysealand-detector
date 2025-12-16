# SkySeaLand Detector


## Developer Install

Close the repository
```
git clone https://github.com/ianparkerjenkins/skysealand-detector.git
```

Navigate into the repository
```
cd skysealand-detector
```

TODO: Use dev environment yml file

```
conda create -n skysealand
```

```
conda activate skysealand
```

```
conda install matplotlib pyyaml pytorch
```

For YOLO baseline
```
conda install -c conda-forge ultralytics
```

Install pytorch and torchvision.

(Use CPU only if no access to GPU)
```
conda install pytorch torchvision cpuonly -c pytorch
```

Launch VSCode
```
code .
```

## Training the model

### Dataset Download

TODO: Convert these to typer CLI commands

Download the data
```
python ./src/skysealand/dataset/download.py
```

Ensure that a `data` directory has been created in the root directory and that it is not empty.

Validate that the data was downloaded correctly
```
python ./src/skysealand/dataset/validation.py
```

Then inspect the newly created `validation_report.json` in the downloaded data folder.