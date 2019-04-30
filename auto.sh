#!/usr/bin/env bash

python3 setup.py bdist_egg
python3 setup.py install --record record.txt

