#!/usr/bin/env bash

set -e

pipenv run python3 src/download_data.py
pipenv run python3 src/splitter.py
pipenv run python3 src/train.py
