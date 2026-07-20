"""
Локальный запуск: подхватывает переменные из файла .env и стартует бота.
Именно этот файл жми «Run» в Cursor.
"""

import os
from pathlib import Path


def load_env(path: str = ".env") -> None:
    env_file = Path(__file__).parent / path
    if not env_file.exists():
        raise SystemExit(
            "Не нашёл файл .env. Скопируй .env.example в .env и заполни его."
        )
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


if __name__ == "__main__":
    load_env()
    from bot import main

    main()
