#!/usr/bin/env python3
import os

pwd = os.path.dirname(os.path.realpath(__file__))
venv_path = os.path.join(pwd, '.venv', 'bin', 'python')

if os.path.exists(venv_path):
    # Need to pass python interpreter as first arg and script as second arg
    os.execv(venv_path, [venv_path, os.path.join(pwd, 'chat.py')])
else:
    print(f"{venv_path} does not exist")
