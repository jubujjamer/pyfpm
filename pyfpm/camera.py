# -*- coding: utf-8 -*-

import io
import os
import subprocess
import time
import threading

try:
    import picamera
    import picamera.array

except ImportError:
    picamera = None

current_path = os.path.dirname(os.path.abspath(__file__))
print(current_path)

def yuv_to_rgb(input_stream=None, total_width=2592, total_height=1944,
               box_width=None, box_height=None, yoff=None, xoff=None):
    import numpy as np
    if xoff is not None:
        xoff = int(xoff)
    else:
        xoff = 0
    if yoff is not None:
        yoff = int(yoff)
    else:
        yoff = 0
    fwidth = (total_width + 31) // 32 * 32
    fheight = (total_height + 15) // 16 * 16
    # Load the Y (luminance) data from the stream
    # Y = np.fromfile(input_stream, dtype=np.uint8, count=fwidth*fheight).reshape((fheight, fwidth))
    Y = np.fromfile(input_stream, dtype=np.uint8, count=fwidth*fheight).reshape((fheight, fwidth))
    # Load the UV (chrominance) data from the stream, and double its size
    # U = np.fromfile(stream, dtype=np.uint8, count=(fwidth//2)*(fheight//2)).\
    #         reshape((fheight//2, fwidth//2)).\
    #         repeat(2, axis=0).repeat(2, axis=1)
    # V = np.fromfile(stream, dtype=np.uint8, count=(fwidth//2)*(fheight//2)).\
    #         reshape((fheight//2, fwidth//2)).\
    #         repeat(2, axis=0).repeat(2, axis=1)
    # Stack the YUV channels together, crop the actual resolution, convert to
    # floating point for later calculations, and apply the standard biases
    # YUV = np.dstack((Y, U, V))[:height, :width, :].astype(np.float)
    # YUV[:, :, 0]  = YUV[:, :, 0]  - 16   # Offset Y by 16
    # YUV[:, :, 1:] = YUV[:, :, 1:] - 128  # Offset UV by 128
    # # YUV conversion matrix from ITU-R BT.601 version (SDTV)
    # #              Y       U       V
    # M = np.array([[1.164,  0.000,  1.596],    # R
    #               [1.164, -0.392, -0.813],    # G
    #               [1.164,  2.017,  0.000]])   # B
    # # Take the dot product with the matrix to produce RGB output, clamp the
    # # results to byte range and convert to bytes
    # RGB = YUV.dot(M.T).clip(0, 255).astype(np.uint8)
    print(xoff, yoff)
    boxed_image = Y[yoff:yoff+box_height, xoff:xoff+box_width]
    return boxed_image.ravel()

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

    with open(os.path.join(current_path, 'web', 'static', 'img', 'no-image.png'), 'rb') as f:
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
            with open(self.tmpfile, 'rb') as fi:
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


class RaspiYUV(BaseCamera):
    tmpfile = 'tmp.yuv'
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
        super(RaspiYUV, self).__init__(**kwargs)
        # self.tmpfile = 'tmp.yuv'
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
            with open(self.tmpfile, 'r+b') as fi:
                # out = fi.read()
                out = yuv_to_rgb(input_stream=fi, box_height=self.patch_size,
                                 box_width=self.patch_size, xoff=self.xoff, yoff=self.yoff)
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
