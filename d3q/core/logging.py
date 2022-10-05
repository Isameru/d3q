#
#         .o8    .oooo.
#        "888  .dP""Y88b
#    .oooo888        ]8P'  .ooooo oo
#   d88' `888      <88b.  d88' `888
#   888   888       `88b. 888   888
#   888   888  o.   .88P  888   888
#   `Y8bod88P" `8bd88P'   `V8bod888
#                               888.
#                               8P'
#                               "

import logging
import os

DEFAULT_LOGGER_NAME = 'd3q'
DEFAULT_LOG_LEVEL = 'info'

logger = None


def configure_logger(logger_name=DEFAULT_LOGGER_NAME, log_level=None):
    if log_level is None:
        log_level = os.environ.get('LOG_LEVEL', None)
    if log_level is None:
        log_level = DEFAULT_LOG_LEVEL
    log_level = log_level.upper()

    os.environ['LOG_LEVEL'] = log_level

    global logger
    logger = logging.getLogger(logger_name)

    logger.setLevel(logging.DEBUG)
    streamhandler = logging.StreamHandler()
    streamhandler.setLevel(log_level)
    logger.addHandler(streamhandler)

    if logger_name != DEFAULT_LOGGER_NAME:
        formatter = logging.Formatter('[%(name)s] %(message)s')
        streamhandler.setFormatter(formatter)


class log:
    @staticmethod
    def _get_logger():
        global logger
        assert logger is not None, 'Configure the logger first by calling: configure_logger(...)'
        return logger

    @staticmethod
    def debug(*args, **kwargs):
        log._get_logger().debug(*args, **kwargs)

    @staticmethod
    def info(*args, **kwargs):
        log._get_logger().info(*args, **kwargs)

    @staticmethod
    def warning(*args, **kwargs):
        log._get_logger().warning(*args, **kwargs)

    @staticmethod
    def error(*args, **kwargs):
        log._get_logger().error(*args, **kwargs)

    @staticmethod
    def critical(*args, **kwargs):
        log._get_logger().critical(*args, **kwargs)
