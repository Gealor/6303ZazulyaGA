import random

from core.artwork import ArtworkColorful
from core.files_processor import CSVFileProcessor
from core.image_processor import ImageProcessor
from logger import log

random.seed(52)


def main():
    file_processor = CSVFileProcessor()

    log.info("Начало подготовки данных...")
    saved_file_path, saved_file_dir = file_processor.start_pipeline()

    # artwork = ArtworkGrayscale(path=saved_file_path)
    artwork = ArtworkColorful(path=saved_file_path)
    log.info("Получено изображение: %s", artwork)
    image_processor = ImageProcessor(artwork=artwork, save_path=saved_file_dir)
    log.info("Начало обработки изображения...")
    image_processor.process_artwork()


if __name__ == "__main__":
    main()
