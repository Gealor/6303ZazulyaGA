# Read file Exceptions


from pathlib import Path


class BaseReadFileException(Exception):
    pass


class IncorrectFormatCSVException(BaseReadFileException):
    pass


class IncorrectFormatJSONException(BaseReadFileException):
    pass


class FileNotFoundException(BaseReadFileException):
    def __init__(self, path: Path):
        self._path = path

    def __str__(self):
        return f"Файл не найден: {self._path}"
