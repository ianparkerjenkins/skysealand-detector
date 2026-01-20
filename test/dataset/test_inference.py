import pathlib
from unittest.mock import MagicMock, patch

import numpy as np

from skysealand import inference


def test_load_ultralytics_yolo_model_calls_yolo_and_to():
    fake_model = MagicMock()

    with patch.object(inference, "YOLO", return_value=fake_model) as mock_yolo:
        model_path = pathlib.Path("fake_model.pt")
        device = "cuda"

        model = inference.load_ultralytics_yolo_model(model_path, device=device)

    mock_yolo.assert_called_once_with(model_path)
    fake_model.to.assert_called_once_with(device)
    assert model is fake_model


def test_load_images_successful_load(tmp_path):
    img_path = tmp_path / "image.jpg"

    fake_img = np.zeros((10, 10, 3), dtype=np.uint8)

    with patch("cv2.imread", return_value=fake_img):
        images, names = inference.load_images(img_path)

    assert len(images) == 1
    assert len(names) == 1
    assert names[0] == "image.jpg"
    assert np.array_equal(images[0], fake_img)


def test_load_images_skips_unloadable_images(tmp_path):
    img_path = tmp_path / "bad.jpg"

    with patch("cv2.imread", return_value=None):
        images, names = inference.load_images(img_path)

    assert images == []
    assert names == []


def test_load_images_partial_success(tmp_path):
    good = tmp_path / "good.jpg"
    bad = tmp_path / "bad.jpg"

    fake_img = np.ones((5, 5, 3), dtype=np.uint8)

    def imread_side_effect(path):
        if path.endswith("good.jpg"):
            return fake_img
        return None

    with patch("cv2.imread", side_effect=imread_side_effect):
        images, names = inference.load_images(good, bad)

    assert len(images) == 1
    assert names == ["good.jpg"]


class FakeTensor:
    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value

    def tolist(self):
        return self._value


class FakeBox:
    def __init__(self, cls, conf, xyxy):
        self.cls = FakeTensor(cls)
        self.conf = FakeTensor(conf)
        self.xyxy = [FakeTensor(xyxy)]


class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


def test_process_ultralytics_single_result_multiple_boxes():
    boxes = [
        FakeBox(cls=0, conf=0.9, xyxy=[10, 20, 30, 40]),
        FakeBox(cls=1, conf=0.75, xyxy=[50, 60, 70, 80]),
    ]

    results = [FakeResult(boxes)]

    output = inference.process_ultralytics_yolo_batched_detections(results)

    assert len(output) == 1
    assert len(output[0]) == 2

    assert output[0][0] == {
        "class_id": 0,
        "confidence": 0.9,
        "bbox": (10, 20, 30, 40),
    }


def test_process_ultralytics_multiple_results():
    results = [
        FakeResult([FakeBox(0, 0.5, [0, 0, 10, 10])]),
        FakeResult([]),  # no detections
    ]

    output = inference.process_ultralytics_yolo_batched_detections(results)

    assert len(output) == 2
    assert len(output[0]) == 1
    assert output[1] == []


def test_process_ultralytics_empty_results():
    output = inference.process_ultralytics_yolo_batched_detections([])

    assert output == []
