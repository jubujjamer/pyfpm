#!/bin/bash
# Upload server files and make remote server run
rsync -avg ./etc/ pi@10.99.39.174:/home/pi/pyfpm/etc/
rsync -avg ./serve.py pi@10.99.39.174:/home/pi/pyfpm/serve.py
rsync -avg ./pyfpm/ pi@10.99.39.174:/home/pi/pyfpm/pyfpm/
rsync -avg ./sampling/ pi@10.99.39.174:/home/pi/pyfpm/sampling/
# ssh -X pi@10.99.39.174 'cd /home/pi/pyfpm; python serve.py'
