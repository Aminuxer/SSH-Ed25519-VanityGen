#!/bin/bash

set -e

echo "Preparing Python environment for OpenCL version ..."

if [[ $EUID -eq 0 ]]; then
    echo "!! NO RUNNING from root;"
    exit 1
fi

if ! python3 -m pip --version > /dev/null 2>&1; then
    echo "pip not found, bootstrapping from get-pip.py..."
    curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
    python3 /tmp/get-pip.py --user
    rm -f /tmp/get-pip.py
fi

python3 -m pip --version

python3 -m pip install --user "pyopencl>=2026.1"
python3 -m pip install --user "cryptography>=3.1"

echo "Verification:"
python3 << PYTHON
import pyopencl
import cryptography
print("pyopencl: " + pyopencl.__version__)
print("cryptography: " + cryptography.__version__)
PYTHON

echo "Environment prepared successfully!"
