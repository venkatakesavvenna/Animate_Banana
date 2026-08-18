FROM nvcr.io/nvidia/pytorch:25.02-py3

ENV DEBIAN_FRONTEND=noninteractive TZ=UTC

# -----------------------------
# System dependencies
# -----------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libsm6 \
        libxext6 \
        poppler-utils \
        ca-certificates \
        curl \
        git \
        vim \
        tzdata \
        libgl1 \
        software-properties-common \
        libglx-mesa0\
        htop \
        ncdu \
        tmux \
        python3.12-venv \
        postgresql \
        postgresql-contrib \
        sudo \
        wget && \
    ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update


# -----------------------------
# User creation (CRITICAL)
# -----------------------------
ARG UID
ARG GID
ARG USER_NAME

RUN if [ -z "$UID" ] || [ -z "$GID" ] || [ -z "$USER_NAME" ]; then \
      echo "UID, GID, USER_NAME must be provided" >&2 && exit 1; \
    fi && \
    groupadd -g ${GID} ${USER_NAME} && \
    useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USER_NAME} && \
    passwd -d ${USER_NAME} && \
    usermod -aG sudo ${USER_NAME}

# Create sudoers file as a separate step for clarity
RUN echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" | tee /etc/sudoers.d/${USER_NAME} && \
    chmod 0440 /etc/sudoers.d/${USER_NAME} && \
    visudo -c -f /etc/sudoers.d/${USER_NAME}

# -----------------------------
# Global python tools (root-only)
# -----------------------------
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install psutil nvitop

# -----------------------------
# Switch to non-root
# -----------------------------
USER ${USER_NAME}
WORKDIR /code

CMD ["/bin/bash"]