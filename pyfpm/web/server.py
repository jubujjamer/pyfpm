##########################################################
# File client.py
# Author:
# Date:
#
##########################################################

import base64
import json

from flask import Flask, Response

from .. import local

def create_server(client):
    app = Flask("FPM")

    @app.route("/")
    def hello():
        return "Hello World!"

    @app.route("/testcam")
    def testcam():
    	return Response(client.acquire(), mimetype='image/png')

    @app.route("/acquire/<theta>/<phi>/<power>/<color>")
    def acquire(theta, phi, power,color):
        print ("app", float(theta), float(phi), color)
        return Response(client.acquire(theta, phi, power, color), mimetype='image/png')

    @app.route("/show_pupil/<theta>/<phi>/<power>")
    def show_pupil(theta, phi, power):
        print ("app", float(theta), float(phi))
        return Response(client.show_pupil(theta, phi, power), mimetype='image/png')

    @app.route("/show_filtered_image/<theta>/<phi>/<power>")
    def show_filtered_image(theta, phi, power):
        print ("app", float(theta), float(phi))
        return Response(client.show_pupil(theta, phi, power), mimetype='image/png')

    @app.route("/compare/<theta>/<phi>/<power>")
    def compare(theta, phi, power):
        return r'<html><div style="width:640px;height:480px;padding:2px;border:5px solid yellowgreen;"><body><img src="/acquire/%s/%s/%s"></div><img src="/show_pupil/%s/%s/%s"></body></html>' % (theta, phi, power, theta, phi, power)
    return app

    @app.route("/metadata")
    def metadata():
        return json.dumps(dict(pupil_size=client.get_pupil_size()))


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
