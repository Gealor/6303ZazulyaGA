# Artwork Exceptions
class BaseArtworkException(Exception):
    pass


class ConstructorArtworkException(BaseArtworkException):
    def __str__(self):
        return "Нужно передать либо путь к файлу, либо массив с изображением"


class ShapeArtworkColorfulException(BaseArtworkException):
    def __str__(self):
        return "Количество каналов не соответствует цветному изображению"


class AddImagesException(BaseArtworkException):
    pass



# Read file Exceptions

class BaseReadFileException(Exception):
    pass


class IncorrectFormatCSVException(BaseReadFileException):
    pass


class IncorrectFormatJSONException(BaseReadFileException):
    pass

