#!/bin/bash
set -e

# Pass all arguments exactly as received to the python script
python translate.py "$@"
