import logging
import warnings


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )


def silence_superfluous_warnings():
    warnings.filterwarnings(
        "ignore",
        message="The pynvml package is deprecated. Please install nvidia-ml-py*",
        category=FutureWarning,
    )
