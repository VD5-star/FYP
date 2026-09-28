# library

This folder holds third-party and generated libraries — not source code.

- `library/python/`  -> Python virtual environment (.venv) and site-packages
- `library/dart/`    -> Dart pub cache and .dart_tool
- `library/fit/`     -> placeholder for fit generated caches

Source lives in `python/` and `dart/`, assets in `fit/`.
Deleting `python/` never touches `dart/` or `fit/`.
