#!/bin/bash
docker run -it -d --rm --runtime=nvidia --gpus=all --entrypoint /bin/bash -p 2222:22 -v .:/home/root --name ctranslate2_container acf42ccdbf56
