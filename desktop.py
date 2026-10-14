#!/usr/bin/env python3
"""
desktop.py - десктопная обёртка: Streamlit в фоновом потоке + окно pywebview.

Запуск для разработки:   python desktop.py
Сборка exe:              pyinstaller desktop.spec
"""
import os
import sys
import signal
import socket
import threading
import time


# ── пути ──────────────────────────────────────────────────────────────────────
def base_dir() -> str:
    """Папка проекта. В PyInstaller - папка, где лежит exe."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


# ── свободный порт ────────────────────────────────────────────────────────────
def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ── Streamlit в отдельном потоке ──────────────────────────────────────────────
def _patch_signals():
    """Streamlit ставит обработчики сигналов; в побочном потоке это падает."""
    _orig = signal.signal

    def _safe(sig, handler):
        try:
            return _orig(sig, handler)
        except (ValueError, OSError):
            return None

    signal.signal = _safe


def start_streamlit(port: int, script: str) -> threading.Thread:
    os.environ.update({
        "STREAMLIT_SERVER_HEADLESS":            "true",
        "STREAMLIT_SERVER_PORT":                str(port),
        "STREAMLIT_SERVER_FILE_WATCHER_TYPE":   "none",
        "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
        "STREAMLIT_THEME_PRIMARY_COLOR":        "#8B5CF6",
        # вот это главное: принудительно выключаем режим разработчика
        "STREAMLIT_GLOBAL_DEVELOPMENT_MODE":    "false",
    })

    def _run():
        _patch_signals()
        old_argv = sys.argv[:]
        # Порт и headless уже в переменных окружения -
        # через CLI их НЕ передаём, чтобы не было конфликта
        sys.argv = ["streamlit", "run", script]
        try:
            from streamlit.web import cli
            cli.main()
        except SystemExit:
            pass
        except Exception:
            log("Streamlit упал:\n" + traceback.format_exc())
        finally:
            sys.argv = old_argv

    t = threading.Thread(target=_run, daemon=True, name="streamlit")
    t.start()
    return t


# ── ждём, пока сервер поднимется ─────────────────────────────────────────────
def wait_ready(port: int, timeout: float = 90) -> bool:
    import urllib.request
    url = f"http://127.0.0.1:{port}/_stcore/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.3)
    return False


# ── главное ───────────────────────────────────────────────────────────────────
def main():
    script = os.path.join(base_dir(), "app.py")
    if not os.path.isfile(script):
        _error(f"Не найден {script}\nПоложите app.py и mood.py рядом с exe.")
        return

    port = free_port()
    start_streamlit(port, script)

    if not wait_ready(port):
        _error("Streamlit не запустился за 90 секунд. Проверьте консоль.")
        return

    import webview
    win = webview.create_window(
        title="Настроение текста",
        url=f"http://127.0.0.1:{port}",
        width=1280,
        height=850,
        min_size=(900, 600),
        text_select=True,
    )
    # Блокирует до закрытия окна; демон-поток Streamlit умрёт сам
    webview.start(debug=False)


def _error(msg: str):
    """Показываем ошибку системным диалогом (без консоли)."""
    try:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Настроение текста", msg)
        root.destroy()
    except Exception:
        print(msg, file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()