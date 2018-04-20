import sys
import rpyc

sys.path.append('/home/pi/pyfpm')
from pyfpm.devices import LedMatrixRGB
obj = LedMatrixRGB()

class MyService(rpyc.Service):
    def exposed_get_instance(self):
        return obj

if __name__ == "__main__":
    from rpyc.utils.server import ThreadedServer
    t = ThreadedServer(MyService, port=18861,
                       protocol_config={"allow_public_attrs": True, "allow_all_attrs": True})
    t.start()
