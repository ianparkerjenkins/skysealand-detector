import json
import logging
import pathlib
from typing import TypedDict

from PIL import Image

from skysealand.dataset import load

logger = logging.getLogger(__name__)


def validate_image(image_path: pathlib.Path) -> tuple[int, int, str]:
    """
    Validates that the image at the given file path can be opened by pillow.

    If so, then the width and height in pixels are extracted for downstream calculations.

    Args:
        image_path: The path to the image to validate.

    Returns:
        A tuple of: the width of the image, the height of the image, and an error string if any occurred.
        (if no error occured this string will be empty string).
    """
    try:
        with Image.open(image_path) as img:
            img.verify()
        with Image.open(image_path) as img:
            width, height = img.size
        return width, height, ""
    except Exception as e:
        # These int values will not be used.
        return -1, -1, str(e)


# Each annotation must have the corners of the box and the class.
_NUM_ANNOTATION_VALS = 5


def validate_annotation(
    ann_path: pathlib.Path,
    img_width: int,
    img_height: int,
    num_classes: int,
) -> list[str]:
    """
    Validates that the given annotation file path contains a correctly formatted annoation for bounding boxes.

    This means that all bounding boxes are contained within the bounds of the annotations respective image,
    each bounding box is specified by two corners plus a class, and that each class is in the expected range of class ids.

    Args:
        ann_path: The path to the annotation file to validate.
        img_width: The horizontal width of the image in pixels.
        img_height: The verticle height of the image in pixels.
        num_classes: The max number of classes in the dataset.

    Returns:
        A list of the encountered bounding box parsing errors for the given file.
    """
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


class ValidationErrorResults(TypedDict):
    """A Json summary of potential validation errors."""

    missing_annotation: list[str]
    missing_image: list[str]
    corrupt_images: list[dict[str, str]]
    annotation_errors: dict[str, list[str]]


def validate_split(split_dir: pathlib.Path, num_classes: int) -> ValidationErrorResults:
    """
    Run a validation check for every (image, label) pair in a given directory.

    Asumes that the directory contains a sub-folder full of images, called ``images``,
    and another sub-folder of corresponding labels, called ``labels``.

    Args:
        split_dir: The path to the *images* directory.
        num_classes: The expected number of object classes to find in the dataset.

    Returns:
        A json-dict summart of any validation issues that were encountered with the given directory.
    """
    images_dir = split_dir
    # Roboflow-style: ../images --> ../labels/images
    ann_dir = split_dir.parent / "labels"

    results: ValidationErrorResults = {
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

        w, h, err = validate_image(img_path)
        if err != "":
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


class ValidationReport(TypedDict):
    train: ValidationErrorResults
    test: ValidationErrorResults
    valid: ValidationErrorResults


def validate_all_data(
    dataset_config_path: pathlib.Path = pathlib.Path("data/data.yaml"),
    report_path: pathlib.Path | None = pathlib.Path("data/validation_report.json"),
) -> ValidationReport:
    """
    Validates all splits in the given dataset config yaml file path.

    The results are dumped to a json file at the given report path.

    Args:
        dataset_config_path: The path to the config summary yaml file for the dataset.
            This should be included with the dataset download.
            Defaults to ``data/data.yaml``.
            This directory must contain a ``train``, ``val``, *and* ``test`` directory.
        report_path: The path to the output validation report json file.
            Defaults to ``data/validation_report.json``
            (within the downloaded dataset directory).
            If None is given, then no dump will occur.
    """
    dataset_spec = load.load_dataset_config(dataset_config_path)

    splits = {
        "train": dataset_spec["train"],
        "val": dataset_spec["val"],
        "test": dataset_spec["test"],
    }
    num_classes = dataset_spec["nc"]
    full_report: ValidationReport = {}
    for split_name, split_dir in splits.items():
        logger.info("Validating %s split at %s...", split_name, split_dir)
        report = validate_split(split_dir, num_classes)
        full_report[split_name] = report

    if report_path is not None:
        with report_path.open("w") as f:
            json.dump(full_report, f, indent=2)

    logger.info("Validation complete. Report saved to validation_report.json")
    return full_report
