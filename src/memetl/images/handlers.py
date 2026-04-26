from pathlib import Path

from memetl.images.artwork import ArtworkColorful
from memetl.images.image_processors.image_processor import ImageProcessor
from memetl.logger import log


def handle_one_image(file_path: Path, file_dir: Path):
    '''
    Функция обработчик одного изображения.
    '''
    file_name = file_path.stem
    parts = file_name.split("_")
    id_image = int(parts[0])
    artwork = ArtworkColorful(path=file_path)
    log.debug("[%d] Получено изображение: %s", id_image, artwork)
    image_processor = ImageProcessor(artwork=artwork, save_path=file_dir, id_image=id_image)
    log.info("[%d] Начало обработки изображения %s...", id_image, file_path.stem)
    image_processor.process_artwork()
