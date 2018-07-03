##########################################################
# File client.py
# Author:
# Date:
#
##########################################################

import base64
import json
import socket
import os
# import queue
# import guid
# import threading

import numpy as np
import yaml
from flask import Flask, Response, render_template, request
from flask.json import jsonify

from .. import local

FOLDER = os.path.abspath(os.path.dirname(__file__))

def create_server(client):

    # cache_lock = threading.RLock()
    # q = queue.Queue()
    # cache = {}
    #
    # def worker():
    #     while True:
    #         settings = q.get()
    #         if settings is None:
    #             break
    #         guid = settings.pop('uuid')
    #         img = client.acquire(**settings)
    #         with cache_lock:
    #             cache[guid] = img
    #         q.task_done()
    #
    # t = threading.Thread(target=worker)
    # t.start()

    app = Flask("FPM", template_folder=os.path.join(FOLDER, 'templates'),
                       static_folder=os.path.join(FOLDER, 'static'))
    # app.config.update(PROPAGATE_EXCEPTIONS = True)

    @app.route("/")
    def init():
        return Response("Testing")

    @app.route('/index')
    def index():
        image_title = {'calibration': 'Image for calibration'}  # fake user
        image = client.capture_image()
        return render_template('index.html',
                               title='Home',
                               image_title=image_title, image=image)
        print("Template folder is set in ", app.template_folder)

    @app.route('/action', methods=['GET', 'POST'])
    def action():
        def move_servo_up():
            client.move_phi(1, mode='relative')

        def move_servo_down():
            client.move_phi(-1, mode='relative')

        def move_theta_up():
            client.move_theta(40, mode='relative')
            print("moved")

        def move_theta_down():
            client.move_theta(-40, mode='relative')

        def move_shift_up():
            client.move_shift(40, mode='relative')

        def move_shift_down():
            client.move_shift(-40, mode='relative')

        def ls_up():
            client.set_power(10)

        def ls_down():
            client.set_power(0)

        def save_parameters():
            config_dict = yaml.load(open('config.yaml', 'r'))
            parameters = client.get_cal_parameters()
            np.save(config_dict['output_cal'], parameters)

        def append_parameter():
            client.append_parameter()

        def textboxes_input():
            phi =  float(request.form['input_phi'])
            theta =  float(request.form['input_theta'])
            shift =  float(request.form['input_shift'])
            power = 200
            client.set_parameters(theta, phi, shift, power,  mode='corrected')

        actions_dict = {'phi_up': move_servo_up, 'phi_down': move_servo_down,
                        'theta_up': move_theta_up, 'theta_down': move_theta_down,
                        'shift_up': move_shift_up, 'shift_down': move_shift_down,
                        'ls_up': ls_up, 'ls_down': ls_down,
                        'save_parameters': save_parameters,
                        'append_parameter': append_parameter,
                        'textboxes_input': textboxes_input}

        if request.method == 'POST':
            print("Select action", request.form['controls'])
            actions_dict[request.form['controls']]()
            theta, phi, shift = client.get_parameters()
        return render_template('index.html',
                               title='Home', theta=theta, phi=phi,
                               shift=shift)

    @app.route('/calibrate', methods=['GET', 'POST'])
    def temptest():
        print(app.template_folder)
        theta, phi, shift = client.get_parameters()
        return render_template('index.html',
                                title='Home', theta=theta, phi=phi,
                                shift=shift)

    @app.route("/testcam")
    def testcam():
        return Response(client.capture_image(), mimetype='image/png')

    @app.route("/acquire/<theta>/<phi>/<shift>/<power>/<color>/<shutter_speed>/<iso>")
    def acquire(theta, phi, shift, power, color, shutter_speed, iso):
        print ("app", float(theta), float(phi), color)
        try:
            return Response(client.acquire(theta, phi, shift, power, color, shutter_speed, iso),
                            mimetype='image/png')
        except socket.error:
            print("An error")
            pass

    @app.route("/acquire_ledmatrix/<nx>/<ny>/<power>/<color>/<shutter_speed>/<iso>/<xoff>/<yoff>")
    def acquire_ledmatrix(nx, ny, power, color, shutter_speed, iso, xoff, yoff):
        try:
            # return Response(client.acquire(nx, ny, power, color,
            #                 shutter_speed, iso), mimetype='image/png')
            y_values = client.acquire(nx, ny, power, color, shutter_speed, iso, xoff, yoff)
            # return Response(y_values, mimetype='image/png')
            # print('sending', jsonify(y_values.tolist()), y_values.tolist())
            # json_stream = jsonify(y_values.tolist())
            json_stream = json.dumps(y_values.tolist())
            print('json ready')
            return json_stream
        except socket.error:
            print("An error")
            pass

    @app.route("/acquire_ledmatrix_pattern/<pattern>/<power>/<color>/<shutter_speed>/<iso>/<xoff>/<yoff>")
    def acquire_ledmatrix_pattern(pattern, power, color, shutter_speed, iso, xoff, yoff):
        try:
            y_values = client.acquire_pattern(pattern, power, color, shutter_speed, iso, xoff, yoff)
            json_stream = json.dumps(y_values.tolist())
            return json_stream
        except socket.error:
            print("An error")
            pass

    @app.route("/set_pixel/<nx>/<ny>/<power>/<color>")
    def set_pixel(nx, ny, power, color):
        try:
            return Response(client.set_pixel(nx, ny, power, color))
        except socket.error:
            print("An error")
            pass


    # @app.route("/acquire/<theta>/<phi>/<shift>/<power>/<color>/<shutter_speed>/<iso>")
    # def acquire_async(theta, phi, shift, power, color, shutter_speed, iso):
    #     print ("app", float(theta), float(phi), color)
    #     try:
    #         if len(cache) > 10:
    #             return Response('no hay lugar')
    #
    #         # TODO COMO DICT
    #         setings = theta, phi, shift, power, color, shutter_speed, iso
    #         settings['uuid'] = out = uuid.uuid4().hex
    #         return Response(out)
    #     except socket.error:
    #         print("An error")
    #         pass

    @app.route("/get_img/<uuid>")
    def get_img(uuid):
        with cache_lock:
            return Response(cache.pop(uuid),
                            mimetype='image/png')

    @app.route("/just_move/<theta>/<phi>/<shift>/<power>/<color>")
    def just_move(theta=None, phi=None, shift=None, power=None, color=None):
        try:
            return Response(client.just_move(theta, phi, shift, power, color),
                            mimetype='image/png')
        except socket.error:
            print("An error")
            pass

    @app.route("/complete_scan/<color>")
    def complete_scan(color):
        print ("app", color)
        try:
            return Response(client.complete_scan(color), mimetype='image/png')
        except socket.error:
            pass

    @app.route("/metadata")
    def metadata():
        return json.dumps(dict(pupil_size=client.get_pupil_size()))
    return app


def create_sim_server(mic_client, sim_client):
    app = Flask("FPM")

    @app.route("/")
    def hello():
        return "Hello World!"

    @app.route("/testcam")
    def testcam():
    	return Response(client.acquire(), mimetype='image/png')

    @app.route("/compare/<theta>/<phi>/<power>")
    def compare(theta, phi, power):
        return r'<html><body><img src="/acquire_mic/%s/%s/%s"><img src="/acquire_sim/%s/%s/%s"></body></html>' % (theta, phi, power, theta, phi, power)

    @app.route("/acquire_mic/<theta>/<phi>/<power>")
    def acquire_mic(theta, phi, power):
        return Response(mic_client.acquire(), mimetype='image/png')

    @app.route("/acquire_sim/<theta>/<phi>/<power>")
    def acquire_sim(theta, phi, power):
        return Response(sim_client.acquire(), mimetype='image/png')

    return app
