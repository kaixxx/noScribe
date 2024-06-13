#!/bin/bash

docker buildx build --platform linux/amd64 -t noscribe .
docker run --rm -it --mount type=bind,src=.,dst=/usr/src/app noscribe pyinstaller noScribe-linux.spec