ARG PYTHON_IMAGE=python:3.14-slim@sha256:ce40764625a4ff50df3548277632e7f96c4e77fe75fa848aae9885476e7df5a4

FROM ${PYTHON_IMAGE} AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore

WORKDIR /build

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN python -m pip wheel --no-deps --wheel-dir /wheelhouse .


FROM ${PYTHON_IMAGE}

ARG ACTEVAL_UID=10001
ARG ACTEVAL_GID=10001

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN groupadd --gid "${ACTEVAL_GID}" acteval \
    && useradd --uid "${ACTEVAL_UID}" --gid acteval --create-home \
        --shell /usr/sbin/nologin acteval \
    && install -d --owner=acteval --group=acteval /data

COPY --from=builder /wheelhouse /wheelhouse

RUN python -m pip install /wheelhouse/*.whl \
    && rm -rf /wheelhouse

USER acteval:acteval
WORKDIR /data

ENTRYPOINT ["acteval"]
CMD ["--help"]
