FROM oven/bun:1.4.2 AS bun
FROM mcr.microsoft.com/azure-cli:2.90.0
COPY --from=bun /usr/local/bin/bun /usr/local/bin/bun
WORKDIR /opt/lab
COPY coordinator.js /opt/lab/coordinator.js
COPY entrypoint.sh /opt/lab/entrypoint.sh
COPY linux-worker.sh /opt/lab/linux-worker.sh
COPY windows-tools.ps1 windows-runtime.ps1 windows-desktop.ps1 /opt/lab/
EXPOSE 8080
ENTRYPOINT ["/bin/sh", "/opt/lab/entrypoint.sh"]
