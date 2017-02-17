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

from flask import Flask, Response, render_template, request

from .. import local

def create_server(client):
    app = Flask("FPM", template_folder='/home/pi/pyfpm/pyfpm/web/templates/',
                       static_folder='/home/pi/pyfpm/pyfpm/web/static')
    # app.config.update(PROPAGATE_EXCEPTIONS = True)

    @app.route("/")
    def init():
        return Response("Testing")

    @app.route('/index')
    def index():
        image_title = {'calibration': 'Image for calibration'}  # fake user
        image = client.acquire()
        return render_template('index.html',
                               title='Home',
                               image_title=image_title, image=image)

    @app.route('/action', methods=['GET', 'POST'])
    def action():
        if request.method == 'POST':
            user = request.form['nm']
        if request.form['controls'] == 'phi++':
            client.move_servo(1, mode='relative')
        if request.form['controls'] == 'phi--':
            client.move_servo(-1, mode='relative')
        if request.form['controls'] == 'toggle led':
            client.set_power(1)
        theta, phi, shift = client.get_parameters()
        return render_template('index.html',
                               title='Home', theta=theta, phi=phi,
                               shift=shift)

    @app.route('/calibrate', methods=['GET', 'POST'])
    def temptest():
        print(app.template_folder)
        theta, phi, shift = client.get_parameters()
        return render_template('index.html',
                                title='Home', theta = theta, phi=phi,
                                shift = shift)

    @app.route("/testcam")
    def testcam():
        return Response(client.acquire(), mimetype='image/png')

    @app.route("/acquire/<theta>/<phi>/<power>/<color>")
    def acquire(theta, phi, power, color):
        print ("app", float(theta), float(phi), color)
        try:
            return Response(client.acquire(theta, phi, power, color),
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
