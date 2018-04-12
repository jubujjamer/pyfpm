#!/bin/bash
# Make remote server run
ssh -X pi@10.99.39.174 'cd /home/pi/pyfpm; python serve.py;  python ./sampling/set_parameters.py'
