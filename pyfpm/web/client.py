##########################################################
# File client.py
# Author:
# Date:
#
##########################################################
import requests

from ..local import BaseClient

class Client(BaseClient):

    def __init__(self, url):
        self.url = url
        # self.metadata = requests.get(self.url + '/metadata').json()

    def acquire(self, theta, phi, power, color):
        print(self.url + '/%d/%d/%d/%s' % (theta, phi, power, color))
        response = requests.get(self.url +
                                '/%d/%d/%d/%s' % (theta, phi, power, color),
                                stream=True)
        if response.status_code == 200:
            return response.raw
        else:
            print("Failed to load webpage")

    def complete_scan(self, color):
        print(self.url + '/%s' % (color))
        response = requests.get(self.url + '/%s' % (color), stream=True)
        if response.status_code == 200:
            return response.raw
        else:
            print("Failed to load webpage")

    def get_pupil_size(self):
        return self.metadata['pupil_size']

    def get_camera_picture(self):
        print(self.url + '/testcam')
        response = requests.get(self.url + '/testcam', stream=True)
        if response.status_code == 200:
            return response.raw
        else:
            print("Failed to load webpage")
