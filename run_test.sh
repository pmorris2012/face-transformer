#!/bin/bash

docker run --shm-size 32G -v /home/quadserver2/Documents/paul/face_transformer/:/face_transformer -v /media/quadserver2/data/paul/face/:/media --rm face_transformer python3 finetune.py