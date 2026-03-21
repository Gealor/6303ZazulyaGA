import random

from analysis.pipeline import analyze_file, run_pipeline
from core.artwork import ArtworkColorful, ArtworkGrayscale
from core.files_processor import CSVFileProcessor
from core.image_processor import ImageProcessor
from logger import log

random.seed(52)

def analyze_csv():
    df_clean = run_pipeline()
    stats_df, timeline_df = analyze_file(df_clean)
    # print(stats_df[:10])

def main(only_analize: bool = True):
    log.info("Начало аналитики...")
    analyze_csv()
    if only_analize:
        return
    log.info("Данные проанализированны.\n")

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
    artwork2 = ArtworkGrayscale(img = artwork1.handmade_highlight_borders())
    result = artwork1 + artwork2
    result.save_image(path = saved_file_dir / "original_plus_highlight_borders.jpg")

    log.info("Тест сложения с размытием Гаусса...")
    artwork3 = ArtworkGrayscale(img=artwork1.handmade_gaussian_blur())
    result = artwork1 + artwork3
    result.save_image(path = saved_file_dir / "original_plus_gaussian_blur.jpg")

    log.info("Тест сложения grayscale изображения и выделенные границы")
    artwork_gray = ArtworkGrayscale(path=saved_file_path)
    artwork_sobel = ArtworkGrayscale(img = artwork_gray.handmade_highlight_borders())
    result = artwork_gray + artwork_sobel
    result.save_image(path = saved_file_dir / "grayscale_plus_highlight_borders.jpg")
    result = artwork_sobel + artwork_gray
    result.save_image(path = saved_file_dir / "highlight_borders_plus_grayscale.jpg")

if __name__ == "__main__":
    main(only_analize=True)
