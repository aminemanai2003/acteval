# Docker image

ActEval publishes its command-line interface as a public, multi-platform image
at `ghcr.io/aminemanai2003/acteval`. The image is a one-shot command rather
than a persistent service: it opens no ports and exits when the evaluation is
complete.

## Run the CLI

Check the installed ActEval version:

```bash
docker run --rm ghcr.io/aminemanai2003/acteval:latest --version
```

Mount a directory containing `predictions.csv` read-only and run an evaluation:

```bash
docker run --rm \
  --mount type=bind,src="$PWD",dst=/data,readonly \
  ghcr.io/aminemanai2003/acteval:latest \
  evaluate /data/predictions.csv --task claim_frequency
```

The CSV column and metric options are identical to the native
[`acteval` command](cli.md).

To write a report through a bind mount on Linux, run with the host user's IDs
so the output is not owned by another user:

```bash
docker run --rm \
  --user "$(id -u):$(id -g)" \
  --mount type=bind,src="$PWD",dst=/data \
  ghcr.io/aminemanai2003/acteval:latest \
  evaluate /data/predictions.csv \
  --task claim_frequency --output /data/evaluation.json
```

The image defaults to the unprivileged `acteval` user with UID and GID 10001.
Overriding the user is only needed when bind-mount ownership requires it.

## Tags and platforms

- `latest` and `main` track successful builds from the default branch;
- `sha-<full-commit>` identifies the source commit used for a build;
- future GitHub releases publish full and major/minor semantic-version tags.

Published images support `linux/amd64` and `linux/arm64`. A registry digest is
the strongest immutable reference. Display the current digest with:

```bash
docker buildx imagetools inspect ghcr.io/aminemanai2003/acteval:latest
```

Then replace the tag separator with `@`, for example
`ghcr.io/aminemanai2003/acteval@sha256:...`, when reproducibility matters.

## Supply-chain evidence

The workflow pins every third-party action to a full commit, tests the packaged
command before registry login, emits an SBOM, and signs build provenance with
GitHub's OIDC-backed attestation service. Verify the published provenance with:

```bash
gh attestation verify \
  oci://ghcr.io/aminemanai2003/acteval:latest \
  --repo aminemanai2003/acteval
```

The Python base image is pinned by its multi-platform digest. Updating that
digest is an explicit maintenance change rather than an implicit rebuild.
