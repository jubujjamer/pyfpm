# -*- coding: utf-8 -*-

import io
import os
import subprocess
import time
import threading

try:
    import picamera
except ImportError:
    picamera = None

current_path = os.path.dirname(os.path.abspath(__file__))


def _prop(name):

    def _get(self):
        return self.camprops[name]

    def _set(self, value):
        self.camprops[name] = value
        self.update_mode()

    return property(_get, _set)

class BaseCamera(object):
    """Base class for cameras
    """

    with open(os.path.join(current_path, 'web', 'static', 'img', 'noimage.jpg')) as f:
        NO_IMAGE = f.read()

    def __init__(self, **kwargs):
        self.camprops = dict()
        for name, value in kwargs.items():
            setattr(self, name, value)

    def acquire(self):
        """Acquire an image with the current settings and return it.

        :rtype: bytes
        """
        return b''

    def update_mode(self):
        pass

    saturation = _prop('saturation')
    brightness = _prop('brightness')
    contrasts = _prop('contrasts')
    sharpness = _prop('sharpness')
    shutter_speed = _prop('shutter_speed')
    iso = _prop('iso')
    integration_time = shutter_speed


class SimCamera(BaseCamera):
    """Simulated camera
    """

    def acquire(self):
        return self.NO_IMAGE


class RPICamera(BaseCamera):
    """Rasberry PI camera.

    camera.sharpness = 0
    camera.contrast = 0
    camera.brightness = 50
    camera.saturation = 0
    camera.ISO = 0
    camera.video_stabilization = False
    camera.exposure_compensation = 0
    camera.exposure_mode = 'auto'
    camera.meter_mode = 'average'
    camera.awb_mode = 'auto'
    camera.image_effect = 'none'
    camera.color_effects = None
    camera.rotation = 0
    camera.hflip = False
    camera.vflip = False
    camera.crop = (0.0, 0.0, 1.0, 1.0)

    """

    def __init__(self, **kwargs):
        if picamera is None:
            raise ValueError('Please Install picamera')
        super(RPICamera, self).__init__(**kwargs)
        c = picamera.PiCamera()
        for name, value in kwargs.items():
            setattr(c, name, value)

        time.sleep(2)
        #c.start_preview()
        self.camera = c

    def acquire(self):
        """Acquire an image with the current settings and return it.

        :rtype: bytes
        """
        with threading.Lock():
            print('Acquiring %s' % threading.current_thread())
            stream = io.BytesIO()
            self.camera.capture(stream, 'jpeg')
            return stream.getvalue()


class RaspiStill(BaseCamera):

    _mapping = {
        'sharpness': '-sh',
        'contrast': '-co',
        'brightness': '-br',
        'saturation': '-sa',
        'iso': '-ISO',
        'shutter_speed': '-ss',
        'exposure': '-ex',
        'awb': '-awb',
        'format': '-e',
        'width': '-w',
        'height': '-h',
        'nopreview': '-n',
        'timeacq': '-t'
    }

    def __init__(self, **kwargs):
        super(RaspiStill, self).__init__(**kwargs)
        self.tmpfile = kwargs.pop('tmpfile')
        self.bin = kwargs.pop('bin')
        for key, value in kwargs.items():
            if key in self._mapping:
                self.camprops[key] = value
        self.update_mode()

    def acquire(self):
        """Acquire an image with the current settings and return it.

        :rtype: bytes
        """
        with threading.Lock():
            print('Acquiring %s' % threading.current_thread())
            try:
                subprocess.check_call(self.cmd)
            except subprocess.CalledProcessError as e:
                print(e)
                return self.NO_IMAGE

            with open(self.tmpfile) as fi:
                out = fi.read()

            os.remove(self.tmpfile)

            return out

    def update_mode(self):
        cmd = [self.bin]
        for key, value in self.camprops.items():
            cmd.append(self._mapping[key])
            cmd.append(str(value))
        cmd.append('-o')
        cmd.append(self.tmpfile)
        self.cmd = cmd
        print("New command %s" % (' '.join(cmd)))
