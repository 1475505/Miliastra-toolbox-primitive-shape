"""
Shaper Desktop — pywebview 桌面应用入口
在原生窗口中打开 Flask Web UI
Windows 上使用 EdgeChromium (WebView2) 渲染引擎
"""

import sys
import os
import threading
import socket

# 确保从项目根目录导入（上级目录）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
os.chdir(BASE_DIR)


def find_free_port():
    """找一个空闲端口"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def start_flask(port):
    """在后台线程中启动 Flask"""
    import server
    server.app.run(host='127.0.0.1', port=port,
                   debug=False, threaded=True, use_reloader=False)


def main():
    import webview

    port = find_free_port()
    url = f'http://127.0.0.1:{port}'

    # 后台启动 Flask
    flask_thread = threading.Thread(
        target=start_flask, args=(port,), daemon=True)
    flask_thread.start()

    # 等待 Flask 就绪
    import time
    for _ in range(50):  # 最多等 5 秒
        try:
            with socket.create_connection(('127.0.0.1', port), timeout=0.1):
                break
        except (ConnectionRefusedError, OSError):
            time.sleep(0.1)

    # 创建原生窗口
    window = webview.create_window(
        title='Shaper — 轮廓描边工具',
        url=url,
        width=1280,
        height=800,
        min_size=(960, 600),
        resizable=True,
        text_select=True,
    )

    # 启动 GUI 事件循环 (Windows 上自动使用 EdgeChromium)
    webview.start(
        gui='edgechromium' if sys.platform == 'win32' else None,
        debug=False,
    )


if __name__ == '__main__':
    main()
