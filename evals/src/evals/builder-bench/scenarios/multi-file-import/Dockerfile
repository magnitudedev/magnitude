FROM oven/bun:1
WORKDIR /workspace
COPY project/ .
RUN bun install --frozen-lockfile 2>/dev/null || bun install
