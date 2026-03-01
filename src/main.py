import random

from core.artwork import ArtworkColorful, ArtworkGrayscale
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

    artwork1 = ArtworkColorful(path=saved_file_path)
    log.info("Тест сложения с выделенными границами...")
    artwork2 = ArtworkGrayscale(path=saved_file_path, img = artwork1.handmade_highlight_borders())
    result = artwork1 + artwork2
    new_path = result.path.with_name("original_plus_highlight_borders.jpg")
    result.save_image(path = new_path)

    log.info("Тест сложения с размытием Гаусса...")
    artwork3 = ArtworkGrayscale(path=saved_file_path, img=artwork1.handmade_gaussian_blur())
    result = artwork1 + artwork3
    new_path = result.path.with_name("original_plus_gaussian_blur.jpg")
    result.save_image(path=new_path)

if __name__ == "__main__":
    main()
