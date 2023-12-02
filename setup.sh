#!/bin/sh
# Install system dependencies
sudo apt-get update
sudo apt-get install -y python3.10 python3-pip python3-venv

# Create a new virtual environment and activate it
python3 -m venv venv
. venv/bin/activate

# Install Python dependencies
pip install poetry
poetry install

# Export the necessary environment variables
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
