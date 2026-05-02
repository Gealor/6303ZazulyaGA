# Сборка пакета

## Локально

Символ . означает относительный путь до pyproject.toml, т.е. путь должен указываться относительно места, где выполняется команда:

```cli
pip install -e .
```

## TestPyPI

1. Сначала соберем пакет в форматы sdist и wheel, в рабочей директории проекта, где лежит pyproject.toml прописываем:

```cli
python -m build
```

2. После этого устанавливаем twine

```cli
pip install twine
```

3. После этого надо зарегистрироваться на сайте TestPyPI и получить API-токен

4. Опубликовать пакеты в директории dist/* в тестовый индекс

```cli
python -m twine upload --repository testpypi dist/* --verbose
```

Надо будет ввести API-токен В ТОЧНОСТИ как он представлен на сайте TestPyPI

https://test.pypi.org/project/memetl-zazulya6303/0.1.2/

5. После этого можно скачать пакет с тестового индекса

```cli
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ memetl-zazulya6303==0.1.2
```
