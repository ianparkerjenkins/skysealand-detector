import pathlib

import numpy as np

from skysealand import inference


class DummyBox:
    def __init__(self):
        self.cls = np.array([2])
        self.conf = np.array([0.75])
        self.xyxy = np.array([[1, 2, 3, 4]])


class DummyResult:
    def __init__(self, num_boxes=1):
        self.boxes = [DummyBox() for _ in range(num_boxes)]


def test_process_single_result():
    results = [DummyResult()]

    output = inference.process_ultralytics_yolo_batched_detections(results)

    assert len(output) == 1
    assert len(output[0]) == 1

    det = output[0][0]
    assert det["class_id"] == 2
    assert det["confidence"] == 0.75
    assert det["bbox"] == (1.0, 2.0, 3.0, 4.0)


def test_process_empty_boxes():
    results = [DummyResult(num_boxes=0)]

    output = inference.process_ultralytics_yolo_batched_detections(results)

    assert output == [[]]


class DummyModel:
    def __call__(self, images):
        # Return one dummy "result" per image
        return [DummyResult() for _ in images]


def test_run_model_with_timing():
    model = DummyModel()
    images = [np.zeros((32, 32, 3)), np.zeros((32, 32, 3))]
    filenames = ["a.jpg", "b.jpg"]
    # Ignore because we're mocking the model.
    output = inference.run_model_with_timing(model, images, filenames)  # pyright: ignore [reportArgumentType]

    assert "results" in output
    assert "metrics" in output
    assert output["metrics"]["num_images"] == 2
    assert output["metrics"]["inference_time_sec"] >= 0

    assert len(output["results"]) == 2
    assert output["results"][0]["filename"] == "a.jpg"


def test_load_ultralytics_yolo_model(monkeypatch):
    calls = {}

    class DummyYOLO:
        def __init__(self, path):
            calls["path"] = path

        def to(self, device):
            calls["device"] = device

    monkeypatch.setattr("skysealand.inference.YOLO", DummyYOLO)

    inference.load_ultralytics_yolo_model(pathlib.Path("model.pt"), device="cpu")

    assert calls["path"].name == "model.pt"
    assert calls["device"] == "cpu"
