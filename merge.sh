#!/bin/bash

# Check if Python is installed
if ! command -v python3 &> /dev/null
then
    echo "Python3 could not be found"
    exit
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null
then
    echo "pip3 could not be found"
    exit
fi

# Install requirements
pip3 install argparse torch copy safetensors

# Run merger.py with arguments
python3 merger.py --model1 $1 --model2 $2 --output $3 --range $4 $5
