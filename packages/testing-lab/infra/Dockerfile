FROM oven/bun:1.4.2 AS bun
FROM mcr.microsoft.com/azure-cli:2.90.0
COPY --from=bun /usr/local/bin/bun /usr/local/bin/bun
WORKDIR /opt/lab
COPY coordinator.js /opt/lab/coordinator.js
COPY entrypoint.sh /opt/lab/entrypoint.sh
COPY ubuntu-worker.sh /opt/lab/ubuntu-worker.sh
EXPOSE 8080
ENTRYPOINT ["/bin/sh", "/opt/lab/entrypoint.sh"]
