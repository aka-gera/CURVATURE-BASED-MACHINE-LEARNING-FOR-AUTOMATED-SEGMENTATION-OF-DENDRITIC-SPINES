import threading
import webbrowser
import time
from waitress import serve
from wsgi import server  # your Dash/Flask app

# URL of your app
url = "http://127.0.0.1:8050/DSA-2"
 
def open_browser():
    time.sleep(.25)  # wait for server to start
    webbrowser.open(url, new=0)

threading.Thread(target=open_browser).start()

# Start Waitress server
serve(server, listen="0.0.0.0:8050")
