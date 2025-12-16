import json
import logging
import pathlib

import yaml
from PIL import Image

logger = logging.getLogger(__name__)


def validate_image(image_path):
    try:
        with Image.open(image_path) as img:
            img.verify()
        with Image.open(image_path) as img:
            width, height = img.size
        return True, width, height, None
    except Exception as e:
        return False, None, None, str(e)


# Each annotation must have the corners of the box and the class.
_NUM_ANNOTATION_VALS = 5


def validate_annotation(ann_path, img_width, img_height, num_classes):
    errors = []
    with ann_path.open() as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) != _NUM_ANNOTATION_VALS:
            errors.append(f"Line {i}: malformed")
            continue

        class_id, x, y, w, h = map(float, parts)
        if not (0 <= class_id < num_classes):
            errors.append(f"Line {i}: invalid class {class_id}")
        if w <= 0 or h <= 0:
            errors.append(f"Line {i}: zero or negative area")

        x_min = (x - w / 2) * img_width
        y_min = (y - h / 2) * img_height
        x_max = (x + w / 2) * img_width
        y_max = (y + h / 2) * img_height

        if x_min < 0 or y_min < 0 or x_max > img_width or y_max > img_height:
            errors.append(f"Line {i}: bbox out of bounds")
    return errors


def validate_split(split_dir, num_classes):
    images_dir = split_dir
    # Roboflow-style: ../images --> ../labels/images
    ann_dir = split_dir.parent / "labels" / split_dir.name

    results = {
        "missing_annotation": [],
        "missing_image": [],
        "corrupt_images": [],
        "annotation_errors": {},
    }

    for img_path in images_dir.glob("*.jpg"):
        ann_path = ann_dir / (img_path.stem + ".txt")
        if not ann_path.exists():
            results["missing_annotation"].append(str(img_path))
            continue

        ok, w, h, err = validate_image(img_path)
        if not ok:
            results["corrupt_images"].append({"image": str(img_path), "error": err})
            continue

        errors = validate_annotation(ann_path, w, h, num_classes)
        if errors:
            results["annotation_errors"][str(ann_path)] = errors

    for ann_path in ann_dir.glob("*.txt"):
        img_path = images_dir / (ann_path.stem + ".jpg")
        if not img_path.exists():
            results["missing_image"].append(str(ann_path))

    return results


def validate_all_data():
    # Load YAML
    with pathlib.Path("data/data.yaml").open() as f:
        cfg = yaml.safe_load(f)

    splits = {
        "train": pathlib.Path(cfg["train"]).resolve(),
        "val": pathlib.Path(cfg["val"]).resolve(),
        "test": pathlib.Path(cfg["test"]).resolve(),
    }
    num_classes = cfg["nc"]

    full_report = {}
    for split_name, split_dir in splits.items():
        logger.info("Validating %s split...", split_name)
        report = validate_split(split_dir, num_classes)
        full_report[split_name] = report

    with pathlib.Path("data/validation_report.json").open("w") as f:
        json.dump(full_report, f, indent=2)

    logger.info("Validation complete. Report saved to validation_report.json")
