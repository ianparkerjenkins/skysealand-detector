import pathlib

from skysealand.dataset import validation


def test_validate_all_data():
    good_data_dir = pathlib.Path(__file__).parent / "dummy-data"
    good_report = validation.validate_all_data(
        dataset_config_path=good_data_dir / "dummy-data.yaml",
        report_path=None,
    )
    assert good_report == {
        "train": {
            "missing_annotation": [],
            "missing_image": [],
            "corrupt_images": [],
            "annotation_errors": {},
        },
        "val": {
            "missing_annotation": [],
            "missing_image": [],
            "corrupt_images": [],
            "annotation_errors": {},
        },
        "test": {
            "missing_annotation": [],
            "missing_image": [],
            "corrupt_images": [],
            "annotation_errors": {},
        },
    }

    bad_data_dir = pathlib.Path(__file__).parent / "dummy-data-with-errors"
    bad_report = validation.validate_all_data(
        dataset_config_path=bad_data_dir / "dummy-data.yaml",
        report_path=None,
    )
    assert len(bad_report["train"]["missing_annotation"]) == 1
    train_ann_errs = bad_report["train"]["annotation_errors"]
    assert len(train_ann_errs) == 1
    assert list(train_ann_errs.values()).pop() == ["Line 2: malformed"]
    test_ann_errs = bad_report["test"]["annotation_errors"]
    assert len(test_ann_errs) == 1
    assert list(test_ann_errs.values()).pop() == ["Line 27: invalid class 100.0"]
    assert bad_report["val"] == {
        "missing_annotation": [],
        "missing_image": [],
        "corrupt_images": [],
        "annotation_errors": {},
    }
