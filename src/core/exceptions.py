class BaseArtworkException(Exception):
    pass


class ShapeArtworkColorfulException(BaseArtworkException):
    def __str__(self):
        return "Количество каналов не соответствует цветному изображению"
