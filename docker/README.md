# Docker

This directory contains Dockerfiles for running LeRobot in containerized environments. Both images are **built nightly from `main`** and published to Docker Hub with the full environment pre-baked — no dependency setup required.

## Pre-built Images

```bash
# CPU-only image (based on Dockerfile.user)
docker pull huggingface/lerobot-cpu:latest

# GPU image with CUDA support (based on Dockerfile.internal)
docker pull huggingface/lerobot-gpu:latest
```

## Quick Start

The fastest way to start training is to pull the GPU image and run `lerobot-train` directly. This is the same environment used for all of our CI, so it is a well-tested, batteries-included setup.

```bash
docker run -it --rm --gpus all --shm-size 16gb huggingface/lerobot-gpu:latest

# inside the container:
lerobot-train --policy.type=act --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human
```

## Dockerfiles

### `Dockerfile.user` (CPU)

A lightweight image based on `python:3.12-slim`. Includes all Python dependencies and system libraries but does not include CUDA — there is no GPU support. Useful for exploring the codebase, running scripts, or working with robots, but not practical for training.

### `Dockerfile.internal` (GPU)

A CUDA-enabled image based on `nvidia/cuda`. This is the image for training — mostly used for internal interactions with the GPU cluster.

## Usage

### Running a pre-built image

```bash
# CPU
docker run -it --rm huggingface/lerobot-cpu:latest

# GPU
docker run -it --rm --gpus all --shm-size 16gb huggingface/lerobot-gpu:latest
```

### Building locally

From the repo root:

```bash
# CPU
docker build -f docker/Dockerfile.user -t lerobot-user .
docker run -it --rm lerobot-user

# GPU
docker build -f docker/Dockerfile.internal -t lerobot-internal .
docker run -it --rm --gpus all --shm-size 16gb lerobot-internal
```

### Multi-GPU training

To select specific GPUs, set `CUDA_VISIBLE_DEVICES` when launching the container:

```bash
# Use 4 GPUs
docker run -it --rm --gpus all --shm-size 16gb \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  huggingface/lerobot-gpu:latest
```

### USB device access (e.g. robots, cameras)

```bash
docker run -it --device=/dev/ -v /dev/:/dev/ --rm huggingface/lerobot-cpu:latest
```


# Cloud Automation Guide: LeRobot on Vast.ai

This guide describes the automated deployment of a Vast.ai instance, its secure connection to our private GitHub repository, and the execution of headless VLA training and evaluations.

---

## 1. Git Repo Connection

To grant the cloud instance secure, isolated access to the Git repository, we use Deploy Keys instead of personal GitHub tokens.

### 1.1 Generate Key Pair Locally
Create a dedicated key pair in the local terminal:
```bash
ssh-keygen -t ed25519 -C "vast_lerobot_deploy" -f ~/.ssh/id_vast_lerobot -N ""
```

### 1.2 Add Public Key to GitHub
Read the public key:
```bash
cat ~/.ssh/id_vast_lerobot.pub
```
Copy the output and add it in the repository settings:
URL: [https://github.com/DarioRepoRuler/lerobot/settings/keys](https://github.com/DarioRepoRuler/lerobot/settings/keys)
Click on "Add deploy key". Check the box for "Allow write access" if the instance needs to push changes.

### 1.3 Format Private Key for Vast.ai
The private key must be formatted as a continuous Base64 string so it can be passed as an environment variable:
```bash
cat ~/.ssh/id_vast_lerobot | base64 -w 0
```
Copy this string. It will be required in the Vast.ai startup command as `<YOUR_BASE64_DEPLOY_KEY>`.

---

## 2. Vast.ai Connection

### 2.1 Generate Local Vast Key
This SSH key is required to log into the Vast instance from the local PC:
```bash
ssh-keygen -t ed25519 -C "your_email@example.com" -f ~/.ssh/id_vast_cloud -N ""
```

### 2.2 Add Public Key to Vast.ai
Read the generated key:
```bash
cat ~/.ssh/id_vast_cloud.pub
```
Go to the Vast.ai Console ([https://console.vast.ai/account/](https://console.vast.ai/account/)) and add the copied key under "SSH Keys". New instances will automatically inherit this key.

### 2.3 Instance Selection Policy
For compliance reasons, only select providers from the "Secure Cloud" category located in Europe (EU) in the search filters.

### 2.4 Start Instance (CLI Command)
Before starting an instance on Vast.ai, you need to adjust the template. Here is the [link](https://cloud.vast.ai?ref_id=575083&template_id=ad4cab34ff262489ca0e06eaa2fd2aca) to the Vast.ai template to easily integrate it into the environment.

The final command for the Vast.ai instance in the terminal window should ultimately look like this:

```bash
vastai create instance <OFFER_ID> \
  --image docker1dario/lerobot-internal_cloud:latest \
  --env '-p 1111:1111 -p 6006:6006 -p 8080:8080 -e SSH_PRIVATE_KEY="<YOUR_BASE64_DEPLOY_KEY>" -e GIT_REPO_URL="git@github.com:DarioRepoRuler/lerobot.git" -e GIT_BRANCH="cloud_feature"' \
  --onstart-cmd 'git config --global init.defaultBranch main && export GIT_SSH_COMMAND="ssh -i /home/user_lerobot/.ssh/id_ed25519 -o StrictHostKeyChecking=no" && git config --global --add safe.directory /lerobot && /lerobot/entrypoint_cloud.sh' \
  --disk 128 \
  --ssh \
  --direct
```

---

## 3. LeRobot Test Inside the Environment

Connect to the started instance via SSH and switch to the working directory:
```bash
cd /lerobot
```

### 3.1 Test Evaluation (Headless via EGL)
To compensate for the lack of a physical monitor, `MUJOCO_GL=egl` forces rendering on the GPU. The `--rename_map` argument aligns the differing camera names between the Libero environment and the SmolVLA policy.

```bash
 MUJOCO_GL=egl lerobot-eval \
  --policy.path=lerobot/smolvla_base \
  --env.type=libero \
  --env.task=libero_object \
  --eval.batch_size=2 \
  --env.task_ids=[0] \
  --eval.n_episodes=3 \
  --rename_map='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'
```

### 3.2 Test Offline Training
When training on public datasets, camera names must also be mapped.

```bash
# Get Hugging Face User ID for the policy repo ID
HF_USER=$(NO_COLOR=1 hf auth whoami | awk -F': *' '/user:/ {print $2}')

lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=lerobot/aloha_static_coffee \
  --output_dir=outputs/train/smolvla_coffee \
  --job_name=smolvla_coffee_run \
  --policy.repo_id=${HF_USER}/policytest \
  --policy.device=cuda \
  --batch_size=64 \
  --steps=20000 \
  --wandb.enable=false \
  --rename_map='{"observation.images.cam_high": "observation.images.camera1", "observation.images.cam_low": "observation.images.camera2", "observation.images.cam_left_wrist": "observation.images.camera3"}'
```

Happy training!