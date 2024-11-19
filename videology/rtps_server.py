import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')

from gi.repository import Gst, GstRtspServer, GObject

class MyRtspServer(GstRtspServer.RTSPServer):
    def __init__(self):
        super().__init__()
        self.factory = GstRtspServer.RTSPMediaFactory()
        self.set_service("25512")
        self.set_address("0.0.0.0")
        self.factory.set_launch(
            "( videotestsrc is-live=true ! x264enc speed-preset=ultrafast tune=zerolatency ! rtph264pay name=pay0 pt=96 )"
        )
        self.factory.set_shared(True)
        self.get_mount_points().add_factory("/test", self.factory)

def main():
    Gst.init(None)
    server = MyRtspServer()

    loop = GObject.MainLoop()
    server.attach(None)
    print("RTSP server is running at rtsp://127.0.0.1:8554/test")
    loop.run()

if __name__ == "__main__":
    main()
