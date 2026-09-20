FROM oven/bun:1.4.2 AS bun
FROM mcr.microsoft.com/azure-cli:2.90.0
COPY --from=bun /usr/local/bin/bun /usr/local/bin/bun
RUN tdnf install -y tar gzip && tdnf clean all && command -v tar && command -v gzip
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
COPY linux-worker.sh /opt/lab/linux-worker.sh
COPY macos-worker.sh /opt/lab/macos-worker.sh
COPY windows-tools.ps1 windows-runtime.ps1 windows-desktop.ps1 /opt/lab/
EXPOSE 8080
ENTRYPOINT ["/bin/sh", "/opt/lab/entrypoint.sh"]
