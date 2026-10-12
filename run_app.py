#!/usr/bin/env python3
"""Запуск интерфейса одной командой: python run_app.py

Делает то же, что `streamlit run app.py`, плюс:
 - отключает слежение за файлами (иначе Streamlit сыплет сотнями ошибок про torchvision из-за transformers);
 - красит интерфейс в фиолетовый, как шапку;
 - не отправляет статистику использования.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")
os.environ.setdefault("STREAMLIT_THEME_PRIMARY_COLOR", "#8B5CF6")
os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")

from streamlit.web import cli  # noqa: E402

if __name__ == "__main__":
    sys.argv = ["streamlit", "run", str(Path(__file__).with_name("app.py")), *sys.argv[1:]]
    sys.exit(cli.main())
