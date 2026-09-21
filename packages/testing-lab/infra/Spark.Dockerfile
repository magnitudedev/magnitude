FROM ubuntu:24.04@sha256:008173c23f95b170204355c12626cb5a965d779a7e1283b09e9cffbb1bf33ca3
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -qq && apt-get install -y --no-install-recommends \
    sudo curl ca-certificates python3 python3-venv git xz-utils tar build-essential \
    binutils lsof openssl xvfb xauth dbus-x11 openbox libgtk-3-0t64 libnss3 \
    libasound2t64 libgbm1 libxss1 libxtst6 libffi-dev python3-dev \
    libnotify4 xdg-utils libsecret-1-0 libglib2.0-bin \
    && rm -rf /var/lib/apt/lists/* \
    && userdel ubuntu && useradd --create-home --uid 1000 --shell /bin/bash labworker \
    && printf 'labworker ALL=(ALL) NOPASSWD:ALL\n' > /etc/sudoers.d/labworker \
    && chmod 0440 /etc/sudoers.d/labworker \
    && mkdir -p /opt/lab /lab && chown labworker:labworker /opt/lab /lab
USER labworker
ENV HOME=/home/labworker
ENV PATH=/home/labworker/.local/bin:/opt/lab/node/bin:/opt/lab/tooling/node_modules/.bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
COPY --chown=labworker:labworker runtime.tar.gz /opt/lab/runtime.tar.gz
COPY --chown=labworker:labworker spark-prepare.sh /opt/lab/prepare.sh
WORKDIR /opt/lab
RUN bash /opt/lab/prepare.sh && rm /opt/lab/runtime.tar.gz /opt/lab/prepare.sh
USER root
RUN apt-get update -qq && apt-get install -y --no-install-recommends nftables polkitd pkexec \
    && rm -rf /var/lib/apt/lists/*
COPY spark-update.rules /etc/polkit-1/rules.d/49-magnitude-lab-update.rules
RUN chmod 644 /etc/polkit-1/rules.d/49-magnitude-lab-update.rules
USER labworker
ENV LAB_TERMINAL_NODE_EXECUTABLE=/opt/lab/node/bin/node
ENV LAB_PI_EXECUTABLE=/home/labworker/.local/bin/pi
ENV LAB_OPENCODE_EXECUTABLE=/home/labworker/.local/bin/opencode
ENV LAB_HERMES_EXECUTABLE=/home/labworker/.local/bin/hermes
WORKDIR /lab
CMD ["sleep", "120"]
