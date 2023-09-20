#! /usr/bin/python3

import logging
import sys
logging.basicConfig(stream=sys.stderr)
sys.path.insert(0, '/home/ahmadmojiri/GreenH2/GUI/Dash_codes')
from GreenH2 import server as application
application.secret_key = 'monkeys'