#!/bin/bash
set -e

# Ne PAS upgrader pip (il est géré par rpm sur Amazon Linux 2023)
# On utilise pip tel qu'il est, ça suffit largement

sudo python3 -m pip install --ignore-installed \
    "tensorflow-cpu==2.15.0" \
    pillow \
    pandas \
    pyarrow \
    numpy \
    matplotlib