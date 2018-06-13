#!/bin/bash
# Upload server files and make remote server run
sudo python led_service.py &
python serve.py
# ssh -X pi@10.99.39.174 'cd /home/pi/pyfpm; python serve.py'
