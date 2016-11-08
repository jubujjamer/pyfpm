##########################################################
# File client.py
# Author:
# Date:
#
##########################################################

import base64
import json
import socket

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
    def acquire(theta, phi, power, color):
        print ("app", float(theta), float(phi), color)
        try:
            return Response(client.acquire(theta, phi, power, color),
                            mimetype='image/png')
        except socket.error:
            print("Error")
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
