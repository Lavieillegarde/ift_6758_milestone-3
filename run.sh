#!/bin/bash

docker run -it -p 8501:8501/tcp --env-file ./env.list ift6758/serving:1.0.0