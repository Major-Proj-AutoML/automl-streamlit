# automl-streamlit

Streamlit UI for the AutoML stack. Upload a CSV, pick a prompt condition, and get back the LLM-generated scikit-learn pipeline plus the trained `model.joblib`.

Port **8501**. Talks to the stack via the gateway (`GATEWAY_URL`, default `http://gateway:8000`).

## What it does

1. Registers (or reuses) the uploaded dataset in `automl-data-service`.
2. Extracts meta-features via `automl-metafeatures-service`.
3. Enqueues a generation run against the selected prompt condition (`b0_naive`, `b1_schema`, `b2_metafeature`) and LLM backend.
4. Polls until the worker finishes, then downloads the generated script and trained model from MinIO through the gateway.

## Run standalone

```bash
pip install streamlit==1.38.0 requests==2.32.3
GATEWAY_URL=http://localhost:8000 \
MINIO_CONSOLE_URL=http://localhost:9001 \
streamlit run app.py
```

## Run via docker-compose (recommended)

Ships with the full stack in [`automl-infra`](https://github.com/Major-Proj-AutoML/automl-infra):

```bash
cd ../automl-infra
docker compose -f docker-compose.yml -f docker-compose.full.yml up -d --build
```

Then open http://localhost:8501.

## Environment

| Var | Default | Purpose |
|---|---|---|
| `GATEWAY_URL` | `http://gateway:8000` | AutoML gateway base URL |
| `MINIO_CONSOLE_URL` | `http://localhost:9001` | Link shown in the sidebar to open the MinIO console |

## Layout

- `app.py` — the whole app (single file).
- `Dockerfile` — `python:3.11-slim` + streamlit + requests.

## License

Dual-licensed under either of

- Apache License 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option. Unless you explicitly state otherwise, any contribution you
intentionally submit for inclusion in this work shall be dual-licensed as
above, without any additional terms or conditions.

## Citing

Part of the VESIT AutoML project. For citation metadata and (once released) a
DOI, see the [`automl-infra`](https://github.com/Major-Proj-AutoML/automl-infra)
repository's `CITATION.cff`.
