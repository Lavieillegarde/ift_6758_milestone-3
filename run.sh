#!/bin/bash

docker run -it -p 5000:5000/tcp --env-file ./env.list ift6758/serving:1.0.0