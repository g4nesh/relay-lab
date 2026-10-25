# Studio control plane only: agent hosts install their own inference runtime.
FROM node:22-bookworm-slim AS subscriptions
WORKDIR /bridge
COPY subscriptions/package.json subscriptions/package-lock.json ./
RUN npm ci
COPY subscriptions/tsconfig.json ./
COPY subscriptions/src/ ./src/
RUN npm run build && npm prune --omit=dev

FROM python:3.13-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /opt/relay
COPY deploy/studio/pyproject.toml ./
COPY src/ ./src/
RUN pip install . \
    && groupadd --gid 10001 relay && useradd --uid 10001 --gid relay --no-create-home relay \
    && mkdir -p /data && chown relay:relay /data \
    && rm -rf /opt/relay/src
COPY deploy/init-state.py /opt/relay/init-state.py
COPY --from=subscriptions /usr/local/bin/node /usr/local/bin/node
COPY --from=subscriptions /bridge/src /opt/relay/subscriptions/src
COPY --from=subscriptions /bridge/dist /opt/relay/subscriptions/dist
COPY --from=subscriptions /bridge/node_modules /opt/relay/subscriptions/node_modules
COPY --from=subscriptions /bridge/package.json /opt/relay/subscriptions/package.json
USER 10001:10001
WORKDIR /data
ENV RELAY_STUDIO_ROOT=/data RELAY_STUDIO_PORT=8770 RELAY_STUDIO_EXPERIMENTS=0
ENV RELAY_STUDIO_SUBSCRIPTION_BRIDGE=/opt/relay/subscriptions/src/bridge.cjs
EXPOSE 8770
CMD ["python", "-m", "relay_lab.studio.asgi"]
