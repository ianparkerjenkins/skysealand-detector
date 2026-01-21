import io

import numpy as np
import pytest
from fastapi import UploadFile
from PIL import Image

from skysealand import inference


def test_load_and_verify_valid_image():
    img = Image.new("RGB", (16, 16))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")

    arr = inference._load_and_verify_image_data(buf.getvalue())

    assert isinstance(arr, np.ndarray)
    assert arr.shape == (16, 16, 3)


def test_load_and_verify_invalid_format():
    img = Image.new("RGB", (16, 16))
    buf = io.BytesIO()
    img.save(buf, format="BMP")

    with pytest.raises(ValueError, match="Unsupported format"):
        inference._load_and_verify_image_data(buf.getvalue())


def test_load_images_from_paths(tmp_path):
    img1 = tmp_path / "a.jpg"
    img2 = tmp_path / "b.png"

    Image.new("RGB", (8, 8)).save(img1)
    Image.new("RGB", (8, 8)).save(img2)

    images, names = inference.load_images(img1, img2)

    assert len(images) == 2
    assert names == ["a.jpg", "b.png"]


def test_load_images_skips_invalid_when_configured(tmp_path):
    valid = tmp_path / "ok.jpg"
    invalid = tmp_path / "bad.jpg"

    Image.new("RGB", (8, 8)).save(valid)
    invalid.write_text("not an image")

    images, names = inference.load_images(valid, invalid, skip_errors=True)

    assert len(images) == 1
    assert names == ["ok.jpg"]


def test_load_images_raises_on_invalid_by_default(tmp_path):
    bad = tmp_path / "bad.jpg"
    bad.write_text("not an image")

    with pytest.raises(ValueError, match="Invalid image"):
        inference.load_images(bad)


def make_image_bytes(fmt="JPEG", size=(32, 32)):
    img = Image.new("RGB", size, color="blue")
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def make_upload_file(filename="test.jpg", fmt="JPEG"):
    data = make_image_bytes(fmt=fmt)
    return UploadFile(filename=filename, file=io.BytesIO(data))


def test_load_images_from_uploadfile():
    upload = make_upload_file()

    images, names = inference.load_images(upload)

    assert len(images) == 1
    assert names == ["test.jpg"]
