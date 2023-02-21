#!/bin/bash

set -o errexit
set -o xtrace

# require image name as argument
if [ $# -ne 1 ]; then
    echo "Usage: $0 <image name>"
    exit 1
fi
img=$1

docker build -t $img .

docker push $img