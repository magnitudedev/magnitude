FROM oven/bun:1.4.2 AS bun
FROM mcr.microsoft.com/azure-cli:2.90.0
COPY --from=bun /usr/local/bin/bun /usr/local/bin/bun
RUN tdnf install -y tar gzip openssh-clients && tdnf clean all && command -v tar && command -v gzip && command -v ssh
RUN curl -fsSL https://pkgs.tailscale.com/stable/tailscale_1.102.4_amd64.tgz -o /tmp/tailscale.tgz \
    && echo '50748df1045e60b5b695f19f4c56b0da36c019948b440fb456b6584a50f0d8b9  /tmp/tailscale.tgz' | sha256sum -c - \
    && tar -xzf /tmp/tailscale.tgz -C /usr/local/bin --strip-components=1 tailscale_1.102.4_amd64/tailscale tailscale_1.102.4_amd64/tailscaled \
    && rm /tmp/tailscale.tgz
RUN curl -fsSL https://download.docker.com/linux/static/stable/x86_64/docker-28.1.1.tgz -o /tmp/docker.tgz \
    && echo 'ff34bd799b5e4c3738a3b1a383cf4067c672add48c7203fcd003fcb0aacebc94  /tmp/docker.tgz' | sha256sum -c - \
    && tar -xzf /tmp/docker.tgz -C /usr/local/bin --strip-components=1 docker/docker \
    && rm /tmp/docker.tgz
# Pin the same Devbox release qualified by the local provider adapter.
RUN curl -fsSL https://get.namespace.so/packages/devbox/v0.0.189/devbox_0.0.189_linux_amd64.tar.gz -o /tmp/devbox.tar.gz \
    && echo '8de02bbc595550f52cdf724979f5d3be0ad8a963681ccba90464ef317b60b6b3  /tmp/devbox.tar.gz' | sha256sum -c - \
    && python3 -c "import tarfile; tarfile.open('/tmp/devbox.tar.gz').extract('devbox', path='/usr/local/bin', filter='data')" \
    && chmod 755 /usr/local/bin/devbox \
    && /usr/local/bin/devbox version \
    && rm /tmp/devbox.tar.gz
WORKDIR /opt/lab
COPY coordinator.js /opt/lab/coordinator.js
COPY entrypoint.sh /opt/lab/entrypoint.sh
COPY tailscale-start.sh spark-docker.sh /opt/lab/
COPY spark-ssh.sh /opt/lab/spark-bin/ssh
RUN chmod 755 /opt/lab/spark-docker.sh /opt/lab/spark-bin/ssh
RUN mkdir -p /root/.ssh && chmod 700 /root/.ssh \
    && printf 'Host *\n  BatchMode yes\n  ConnectTimeout 30\n  ServerAliveInterval 15\n  ServerAliveCountMax 2\n  ForwardAgent no\n' > /root/.ssh/config \
    && chmod 600 /root/.ssh/config
COPY linux-worker.sh /opt/lab/linux-worker.sh
COPY macos-worker.sh /opt/lab/macos-worker.sh
COPY windows-tools.ps1 windows-runtime.ps1 windows-desktop.ps1 /opt/lab/
EXPOSE 8080
ENTRYPOINT ["/bin/sh", "/opt/lab/entrypoint.sh"]
