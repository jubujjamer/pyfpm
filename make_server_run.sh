#!/bin/bash
# Upload server files and make remote server run
sshpass -p raspberry rsync -avg ./etc/ pi@10.99.39.174:/home/pi/git/pyfpm/etc/
sshpass -p raspberry rsync -avg ./led_service.py pi@10.99.39.174:/home/pi/git/pyfpm/led_service.py
sshpass -p raspberry rsync -avg ./serve.py pi@10.99.39.174:/home/pi/git/pyfpm/serve.py
sshpass -p raspberry rsync -avg ./pyfpm/ pi@10.99.39.174:/home/pi/git/pyfpm/pyfpm/
sshpass -p raspberry rsync -avg ./sampling/ pi@10.99.39.174:/home/pi/git/pyfpm/sampling/
sshpass -p raspberry rsync -avg ./sampling/ pi@10.99.39.174:/home/pi/git/pyfpm/sampling/
# ssh -X pi@10.99.39.174 'cd /home/pi/pyfpm; python serve.py'
