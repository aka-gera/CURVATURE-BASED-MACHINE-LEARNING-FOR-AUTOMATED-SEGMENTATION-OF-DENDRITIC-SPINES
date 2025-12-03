import webbrowser, threading

def post_fork(server, worker):
    def open_browser():
        webbrowser.open("http://127.0.0.1:8050")
    threading.Timer(1.25, open_browser).start()
