import logging
import colorlog
from mmhb.utils.config import Config

def setup_logging():
    log_conf = Config("config/logging.yml").read()
    log_level = logging.DEBUG if log_conf.level == "debug" else logging.INFO

    handler = logging.StreamHandler()
    handler.setFormatter(_colour_formatter(log_conf))
    logging.basicConfig(level=logging.INFO, handlers=[handler])
    logger = logging.getLogger(__name__)


    return logger

def _colour_formatter(config: Config):
    formatter = colorlog.ColoredFormatter(
        config.format,
        datefmt="%Y-%m-%d %H:%M:%S",
        reset=True,
        log_colors={
            'DEBUG':    'white',
            'INFO':     'cyan',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    return formatter

def _check_config(config: Config):
    valid_levels = ["debug", "info"]
    assert config.level in valid_levels, f"Invalid logging level, must be one of {valid_levels}"

