import http.server
import socketserver
import socket
import os
PORT = 8100
cur_path = os.path.dirname(__file__)
class HttpRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        cur_path = os.path.dirname(__file__)
        if self.path == '/':
            self.path = '/index.html'
        ctype = self.guess_type(self.path)

        if self.path.endswith(".html"):
            f = open(cur_path + self.path[0:]).read()
            self.send_response(200)
            self.send_header('Content-type',ctype)
            self.end_headers()
            self.wfile.write(bytes(f, 'utf-8'))

httpd = socketserver.TCPServer(("0.0.0.0", PORT), HttpRequestHandler, bind_and_activate=False)
httpd.allow_reuse_address = True
httpd.daemon_threads = True
try:
 HOST_NAME = socket.gethostname()
 IP = socket.gethostbyname(socket.getfqdn())
 print(HOST_NAME)
 print('IP:'+IP)
 print(f"serving at <{IP}>:{PORT}")
 httpd.server_bind()
 httpd.server_activate()
 httpd.serve_forever()
except KeyboardInterrupt:
 pass
finally:
 httpd.server_close()