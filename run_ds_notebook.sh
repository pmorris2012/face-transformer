#!/bin/bash

docker run --rm -it -p 7778:8888 -e JUPYTER_ENABLE_LAB=yes -v /media/quadserver2/data/paul/face:/data -v "$PWD":/home/joyvan/work jupyter/datascience-notebook