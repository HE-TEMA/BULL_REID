Copyright © 2026 Bull® SAS. All rights reserved

# BULL_REID

## Overview
This repository provides a Docker container image based on `nvidia/cuda:12.1.0-devel-ubuntu22.04`. The container is pre-configured with CUDA 12.1.0, PyTorch, LLAVA, and essential dependencies for running machine learning and object tracking tasks. It includes additional tools and scripts for specific applications such as YOLO tracking and person re-identification.

## Features
- **Base Image**: NVIDIA CUDA 12.1.0 development environment
- **ML Frameworks**: PyTorch, torchvision, and torchaudio with CUDA support
- **Pre-installed Tools**: YOLO tracking, ultralytics, and related dependencies
- **Custom Scripts**: Includes `tfa_tech_07.py` as the main application script
- **Environment Variables**: Flexible configuration for runtime parameters

## Prerequisites
- NVIDIA GPU with compatible drivers
- Docker installed on the host machine

## Get the Image from the registry
```bash
docker pull ghcr.io/he-tema/atos_reid_image:1.0
```

## Run the Container
Run the container with the necessary environment variables:

```bash
docker run --rm -it \
  -e PORT=8002 \
  -e BASE_PATH='/reid' \
  -e BROKER_URL='https://orion.tema.digital-enabler.eng.it' \
  -e BROKER_TYPE_ID='PersonREID' \
  -e DEBUG=False \
  -e PROCESSING_UNIT=gpu \
  -e endpoint=<endpoint> \
  -e access_key=<access-key> \
  -e secret_key=<secret-key> \
  -p 8002:8002 \
  atos_reid_image
```

### Persistent Storage
To persist application data:

```bash
docker run --rm -it \
  -v /local/path/to/data:/home/atos/app/data \
  atos_reid_image
```

## Included Components
### Installed Packages
- **System Utilities**:
  - Python 3 and associated tools (`python3`, `pip`, `setuptools`)
  - OpenCV, GCC, and MPICH for parallel computing
- **Python Libraries**:
  - PyTorch and torchvision with CUDA support
  - Libraries required for YOLO tracking (`boxmot`, `ultralytics`)
- **Custom Scripts**:
  - The `tfa_tech_07.py` application script

### Environment Variables
#### Mandatory
- `PORT`: The port on which the application listens
- `BASE_PATH`: Base path for API routes
- `BROKER_URL`: URL of the message broker
- `BROKER_TYPE_ID`: Entity type for interaction
- `endpoint`: URL of the S3-compatible object storage service.
- `access_key`: Access key used to authenticate with the object storage service.
- `secret_key`: Secret key associated with the access key, used for authentication with the object storage service.

#### Optional
- `DEBUG`: Enable debugging mode (`True` or `False`, default: `False`)
- `PROCESSING_UNIT`: Specify the processing unit (`cpu` or `gpu`, default: `gpu`)

### Application Entry Point
The application is launched using the following command:

```bash
CMD ["python3", "tfa_tech_07.py"]
```


## Maintainer
For any questions or issues, please contact: Advanced Computing & AI. Strategy and Innovation

# BULL_REID
# BULL_REID
