import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(module)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H-%M-%S",
)

log = logging.getLogger(__name__)
