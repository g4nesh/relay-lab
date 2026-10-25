# Studio control plane only: agent hosts install their own inference runtime.
FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /opt/relay
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-deps . && pip install uvicorn==0.53.0 click==8.5.0 h11==0.16.0 \
    && groupadd --gid 10001 relay && useradd --uid 10001 --gid relay --no-create-home relay \
    && mkdir -p /data && chown relay:relay /data \
    && rm -rf /opt/relay/src
COPY deploy/init-state.py /opt/relay/init-state.py
USER 10001:10001
WORKDIR /data
ENV RELAY_STUDIO_ROOT=/data RELAY_STUDIO_PORT=8770
EXPOSE 8770
CMD ["python", "-m", "relay_lab.studio.asgi"]
