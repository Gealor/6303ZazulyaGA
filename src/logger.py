import logging

import config

logging.basicConfig(
    level=logging.INFO,
    format=config.FORMAT,
    datefmt=config.DATEFMT,
)

log = logging.getLogger(__name__)
