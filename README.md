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
{
    "results": [
        {
            "filename": "path/to/my/img1.jpg",
            "inference": [
                {
                    "class_id": 2,
                    "confidence": 0.4285559058189392,
                    "bbox": [
                        1028.7703857421875,
                        417.98046875,
                        1210.3580322265625,
                        538.3756713867188
                    ]
                },
                {
                    "class_id": 0,
                    "confidence": 0.4281531572341919,
                    "bbox": [
                        1192.9344482421875,
                        555.4136352539062,
                        1224.4884033203125,
                        627.125732421875
                    ]
                }
            ]
        },
        {
            "filename": "path/to/my/img2.jpg",
            "inference": [
                {
                    "class_id": 13,
                    "confidence": 0.5388724207878113,
                    "bbox": [
                        0.2129051238298416,
                        0.0,
                        245.41836547851562,
                        193.52255249023438
                    ]
                }
            ]
        }
    ],
    "metrics": {
        "num_images": 2,
        "inference_time_sec": 0.28876129999844125
    }
}
```

By default this file is called `inference.json`, but you can change this with `--output_path`.

In the event that an image cannot be loaded a warning will be logged and inference will be skipped for that image.

If you want to analyze all of the images in a particular directory, then you can use the `--images-dir` option:

```
skysealand infer --images-dir path/to/all/my/images/
```
