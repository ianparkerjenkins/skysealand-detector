# SkySeaLand Detector

TODO: Pretty picture

## Developer Install

Close the repository
```
git clone https://github.com/ianparkerjenkins/skysealand-detector.git
```

Navigate into the repository
```
cd skysealand-detector
```

Install mamba (if not already installed)
```
conda install -c conda-forge mamba
```

Create the conda environment.
```
mambda env create -f dev_environment.yml
```

Activate the new environment.
```
conda activate skysealand
```

Run the `devinstall.py` script.
```
python devinstall.py
```

Launch VSCode.
```
code .
```


## Dataset Download

To download the data, first ensure that you performed the [developer install](#developer-install), then run:
```
skysealand download
```

Ensure that a `data` directory has been created in the root directory and that it is not empty.

This also checks for any issues with the downloaded data. To confirm that everything was correctly downloaded, inspect the newly created `validation_report.json` in the downloaded data folder.


## Training the model

To train the baseline version of the model, first ensure that you performed the [developer install](#developer-install) and have [downloaded the dataset](#dataset-download), then run:
```
skysealand train
```

Then you can check the output of this in the `runs` directory.


## Using a model for inference

To run inference on some images you can use `skysealand infer` with the image paths to analyze:
```
skysealand infer path/to/my/img1.jpg path/to/my/img2.jpg
```

By default this uses the model that is output from running `skysealand train`, called `yolov8n.pt`, but this can be changed by passing `--model-path` with your model.

The CLI inference writes the output into a json file with the following structure:
```json
[
    {
        "filename": "path/to/my/img1.jpg",
        "inference": [
            {
                "class_id": 2,
                "confidence": 0.46020451188087463,
                "bbox": [
                    651.6776733398438,
                    542.8021850585938,
                    739.9046630859375,
                    610.9060668945312
                ]
            },
            {
                "class_id": 0,
                "confidence": 0.30361688137054443,
                "bbox": [
                    600.8245849609375,
                    517.4049072265625,
                    685.6116333007812,
                    587.8324584960938
                ]
            }
        ]
    },
    {
        "filename": "path/to/my/img2.jpg",
        "inference": [
            {
                "class_id": 13,
                "confidence": 0.2531780004501343,
                "bbox": [
                    0.0,
                    0.0,
                    245.4083251953125,
                    191.41256713867188
                ]
            }
        ]
    }
]
```

By default this file is called `inference.json`, but you can change this with `--output_path`.

In the event that an image cannot be loaded a warning will be logged and inference will be skipped for that image.

If you want to analyze all of the images in a particular directory, then you can use the `--images-dir` option:

```
skysealand infer --images-dir path/to/all/my/images/
```
