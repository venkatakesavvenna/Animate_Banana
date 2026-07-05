#!/bin/bash

# Docker configuration variables
IMAGE_NAME="img-2-svg-pretraining-singlenode"
USER_NAME="venkat.kesav"
CONTAINER_NAME="img-2-svg-pretraining-singlenode-${USER_NAME}"
CODE_MOUNT="/fsxvision_new/${USER_NAME}/img_2_svg_pretraining"
DATA_MOUNT="/home"
DATA_MOUNT2="/fsxvision_new"
DATA_MOUNT3="/fsxvision"
HF_CACHE="/fsxvision_new/venkat.kesav/backup/hf_cache"
ENVIRONMENT_MOUNT="/fsxvision_new/${USER_NAME}/environments"
ENV_NAME="img_2_svg_pretraining"
VIEWER_PORT=7860

DOCKERFILE_NAME="Dockerfile"

echo "=== img_2_svg_pretraining Docker Initialization ==="

# Build Docker image if it doesn't exist
if docker image inspect $IMAGE_NAME >/dev/null 2>&1; then
  echo "✓ Image $IMAGE_NAME already exists."
else
  echo "Building image $IMAGE_NAME..."
  docker build -f $DOCKERFILE_NAME -t $IMAGE_NAME .
  echo "✓ Image built successfully."
fi


# Check if container already exists
if docker ps -aq --filter "name=$CONTAINER_NAME" | grep -q .; then
    echo "✓ Container $CONTAINER_NAME already exists."
    
    # Check if container is running
    if docker ps --filter "name=$CONTAINER_NAME" | grep -q .; then
        echo "✓ Container is already running."
    else
        echo "Starting existing container..."
        docker start $CONTAINER_NAME
        echo "✓ Container started."
    fi
else
    echo "Creating new container $CONTAINER_NAME..."
    docker run --shm-size=512g -dit --gpus all \
        -v $CODE_MOUNT:/code \
        -v $DATA_MOUNT:$DATA_MOUNT \
        -v $DATA_MOUNT2:$DATA_MOUNT2 \
        -v $DATA_MOUNT3:$DATA_MOUNT3 \
        -v /opt/dlami/nvme:/opt/dlami/nvme \
        -v $HF_CACHE:/root/.cache/huggingface \
        -v $ENVIRONMENT_MOUNT:/environments \
        -p $VIEWER_PORT:$VIEWER_PORT \
        --name $CONTAINER_NAME \
        -w /code \
        -it \
        $IMAGE_NAME \
        bash -c "/bin/bash"
    echo "✓ Container created and started."
fi

# # SSH Setup (Original - kept for reference)
# echo "Setting up SSH keys..."
# if [ -d ~/.ssh ]; then
#     docker cp ~/.ssh $CONTAINER_NAME:/root/
#     docker exec $CONTAINER_NAME bash -c "chmod 700 /root/.ssh && chmod 600 /root/.ssh/* && chown -R root:root /root/.ssh"
#     echo "✓ SSH keys configured."
# else
#     echo "⚠ No SSH directory found at ~/.ssh, skipping SSH setup."
# fi
# SSH Setup (root)
echo "Setting up SSH keys..."
if [ -d ~/.ssh ]; then
    docker cp ~/.ssh $CONTAINER_NAME:/root/
    docker exec $CONTAINER_NAME bash -c "chmod 700 /root/.ssh && chmod 600 /root/.ssh/* && chown -R root:root /root/.ssh"
    echo "✓ SSH keys configured."
else
    echo "⚠ No SSH directory found at ~/.ssh, skipping SSH setup."
fi

# Copy init_env.sh script into container
echo "Setting up Python environment..."
docker exec $CONTAINER_NAME chmod +x /code/docker/init_env.sh

# Run environment initialization
echo "Initializing Python environment (this may take a while on first run)..."
docker exec $CONTAINER_NAME /code/docker/init_env.sh $ENV_NAME

# Set up auto-activation of environment in .bashrc
echo "Configuring auto-activation of environment..."
docker exec $CONTAINER_NAME bash -c "grep -q 'source /environments/${ENV_NAME}/bin/activate' /root/.bashrc || echo 'source /environments/${ENV_NAME}/bin/activate' >> /root/.bashrc"
echo "✓ Environment will auto-activate on container entry."

echo "=== Initialization Complete ==="
echo "To access the container, run: docker exec -it $CONTAINER_NAME bash"
echo "Python environment '${ENV_NAME}' will be automatically activated."
echo "Container port $VIEWER_PORT is published to the host — run the viewer with --port $VIEWER_PORT and open http://<host>:$VIEWER_PORT"