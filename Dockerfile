# Build and run kuntadata. Multi-stage so the runtime image carries no build
# tooling and no test dependencies.
FROM python:3.13-slim AS build

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --prefix=/install .

FROM python:3.13-slim AS runtime

# Run unprivileged: nothing here needs root.
RUN useradd --create-home --uid 10001 app
COPY --from=build /install /usr/local
WORKDIR /home/app
USER app

ENV PORT=8000 STATFIN_CACHE_DIR=/tmp/statfin-cache
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status==200 else 1)"

CMD ["uvicorn", "kuntadata.api:app", "--host", "0.0.0.0", "--port", "8000"]
