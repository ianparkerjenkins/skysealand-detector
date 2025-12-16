import logging
import sys
from logging import handlers

_ROOT_LOGGER = logging.getLogger()


def setup_logging():
    handler = handlers.RotatingFileHandler(
        "skysealand.log", maxBytes=5 * 1024 * 1024, backupCount=3
    )
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    stream_handler.setFormatter(stream_formatter)

    _ROOT_LOGGER.setLevel(logging.INFO)
    _ROOT_LOGGER.addHandler(handler)
    _ROOT_LOGGER.addHandler(stream_handler)
    _ROOT_LOGGER.info("Starting SkySeaLand Detector Logging.")
