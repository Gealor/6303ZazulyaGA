import logging

import memetl.config as config

logging.basicConfig(
    level=logging.INFO,
    format=config.FORMAT,
    datefmt=config.DATEFMT,
)

log = logging.getLogger(__name__)
