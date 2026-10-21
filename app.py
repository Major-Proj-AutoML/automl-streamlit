# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright 2026 The VESIT AutoML Project contributors

"""Streamlit demo for the AutoML gateway.

Upload CSV -> full-run workflow -> poll -> download script + model.
Reuses an existing dataset if the same (name, target) already lives in the registry.
"""

from __future__ import annotations

import os
import time
from typing import Any

import requests
import streamlit as st

GATEWAY = os.environ.get("GATEWAY_URL", "http://gateway:8000")
MINIO_CONSOLE = os.environ.get("MINIO_CONSOLE_URL", "http://localhost:9001")
POLL_INTERVAL = 4
POLL_TIMEOUT = 900

st.set_page_config(
    page_title="AutoML Studio",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
:root {
    --accent: #6366f1;
    --accent-2: #8b5cf6;
    --ok: #10b981;
    --warn: #f59e0b;
    --err: #ef4444;
    --bg-card: rgba(99, 102, 241, 0.06);
    --border-card: rgba(99, 102, 241, 0.25);
}
.stApp header { background: transparent; }
h1, h2, h3 { letter-spacing: -0.02em; }
.hero {
    padding: 1.5rem 2rem;
    border-radius: 18px;
    background: linear-gradient(120deg, rgba(99,102,241,0.15), rgba(139,92,246,0.10));
    border: 1px solid var(--border-card);
    margin-bottom: 1.5rem;
}
.hero h1 { margin: 0; font-size: 2.1rem; }
.hero p { margin: 0.35rem 0 0; opacity: 0.8; }
.card {
    padding: 1.25rem 1.5rem;
    border-radius: 14px;
    background: var(--bg-card);
    border: 1px solid var(--border-card);
    margin-bottom: 1rem;
}
.step-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
    background: rgba(99,102,241,0.12);
    color: var(--accent);
    margin-bottom: 0.6rem;
    letter-spacing: 0.02em;
    text-transform: uppercase;
}
.score-hero {
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(120deg, var(--accent), var(--accent-2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1;
}
.stDownloadButton > button {
    width: 100%;
    padding: 0.8rem 1rem;
    border-radius: 10px;
    font-weight: 600;
}
.stDownloadButton > button:hover {
    border-color: var(--accent);
    color: var(--accent);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(99,102,241,0.05), transparent);
}
</style>
    """,
    unsafe_allow_html=True,
)


def _get(path: str, **kw) -> requests.Response:
    return requests.get(f"{GATEWAY}{path}", timeout=30, **kw)


def _post(path: str, **kw) -> requests.Response:
    return requests.post(f"{GATEWAY}{path}", timeout=120, **kw)


def _find_existing_dataset(name_stem: str, target: str) -> dict | None:
    r = _get("/datasets")
    if r.status_code != 200:
        return None
    for d in r.json():
        if d["name"] == name_stem and d["target_col"] == target and d["source"] == "custom":
            return d
    return None


def _find_run(dataset_id: int, condition_upper: str, backend: str, seed_val: int) -> dict | None:
    r = _get("/runs", params={
        "dataset_id": dataset_id, "condition": condition_upper,
        "llm_backend": backend, "limit": 50,
    })
    if r.status_code != 200:
        return None
    latest = None
    for row in r.json():
        if row["seed"] == seed_val:
            if latest is None or row["id"] > latest["id"]:
                latest = row
    return latest


# ---------- Header ----------
st.markdown(
    """
<div class="hero">
    <h1>AutoML Studio</h1>
    <p>Upload a CSV. An LLM writes the scikit-learn pipeline, we run it, and hand you back the script and the trained model.</p>
</div>
    """,
    unsafe_allow_html=True,
)

# ---------- Sidebar config ----------
with st.sidebar:
    st.markdown("### Configuration")
    file = st.file_uploader("Training CSV", type=["csv"])
    target_col = st.text_input("Target column", placeholder="e.g. final_grade")

    task_type_choice = st.selectbox(
        "Task type",
        ["auto-detect", "binary_classification", "multiclass_classification", "regression"],
    )

    st.markdown("---")
    st.markdown("### LLM & prompt")
    condition = st.selectbox(
        "Prompt condition",
        ["b0_naive", "b1_schema", "b2_metafeature"],
        index=1,
        help="B0: plain • B1: + schema • B2: + meta-features and reasoning trace",
    )
    llm_backend = st.text_input("LLM backend", value="gpt-oss:120b-cloud")

    with st.expander("Advanced"):
        seed = st.number_input("Seed", value=1, step=1)
        max_iter = st.number_input("Max iterations", value=3, min_value=1, max_value=10)
        timeout_seconds = st.number_input("Per-iter timeout (s)", value=300, min_value=30, max_value=1800)
        reuse_existing = st.checkbox(
            "Reuse existing dataset if same name + target",
            value=True,
            help="Avoids the unique-constraint 500 on re-upload.",
        )

    st.markdown("---")
    submitted = st.button("🚀 Run pipeline", type="primary", use_container_width=True)
    st.markdown(f"<small><a href='{MINIO_CONSOLE}' target='_blank'>Open MinIO console ↗</a></small>", unsafe_allow_html=True)


# ---------- Main flow ----------
def run_workflow() -> None:
    if not file or not target_col:
        st.warning("Upload a CSV and enter the target column name to continue.")
        return

    name_stem = os.path.splitext(file.name)[0]
    task_type = None if task_type_choice == "auto-detect" else task_type_choice

    step1, step2, step3 = st.container(), st.container(), st.container()

    # ---- Step 1: dataset ----
    with step1:
        st.markdown('<div class="card"><span class="step-pill">Step 1 · Dataset</span>', unsafe_allow_html=True)
        existing = _find_existing_dataset(name_stem, target_col) if reuse_existing else None
        if existing:
            st.success(f"Reusing existing dataset **#{existing['id']} — {existing['name']}** ({existing['task_type']})")
            dataset = existing
            meta_features_payload: dict | None = None
        else:
            with st.spinner("Uploading CSV..."):
                files = {"file": (file.name, file.getvalue(), "text/csv")}
                data: dict[str, Any] = {"target_col": target_col}
                if task_type:
                    data["task_type"] = task_type
                r = _post("/datasets", files=files, data=data)
            if r.status_code >= 300:
                st.error(f"Upload failed ({r.status_code}): {r.text}")
                st.markdown("</div>", unsafe_allow_html=True)
                return
            dataset = r.json()
            st.success(f"Created dataset **#{dataset['id']} — {dataset['name']}** ({dataset['task_type']})")
            meta_features_payload = None

        cols = st.columns(4)
        cols[0].metric("Rows", dataset["n_rows"])
        cols[1].metric("Columns", dataset["n_cols"])
        cols[2].metric("Task", dataset["task_type"].replace("_", " "))
        cols[3].metric("Target", dataset["target_col"])
        st.markdown("</div>", unsafe_allow_html=True)

    # ---- Step 2: meta-features + enqueue ----
    with step2:
        st.markdown('<div class="card"><span class="step-pill">Step 2 · Meta-features & enqueue</span>', unsafe_allow_html=True)
        with st.spinner("Extracting meta-features..."):
            r = _post(f"/meta-features/{dataset['id']}")
        if r.status_code >= 300:
            st.error(f"Meta-features failed ({r.status_code}): {r.text}")
            st.markdown("</div>", unsafe_allow_html=True)
            return
        meta_features_payload = r.json()
        with st.expander("Meta-features JSON"):
            st.json(meta_features_payload.get("features", meta_features_payload))

        with st.spinner("Enqueuing generation job..."):
            r = _post("/runs", json={
                "dataset_id": dataset["id"],
                "condition": condition,
                "llm_backend": llm_backend,
                "seed": int(seed),
                "max_iter": int(max_iter),
                "timeout_seconds": int(timeout_seconds),
            })
        if r.status_code >= 300:
            st.error(f"Enqueue failed ({r.status_code}): {r.text}")
            st.markdown("</div>", unsafe_allow_html=True)
            return
        rq = r.json()
        st.info(f"RQ job **{rq['rq_job_id']}** enqueued.")
        st.markdown("</div>", unsafe_allow_html=True)

    # ---- Step 3: poll + downloads ----
    with step3:
        st.markdown('<div class="card"><span class="step-pill">Step 3 · Generation & artifacts</span>', unsafe_allow_html=True)
        condition_upper = {"b0_naive": "B0", "b1_schema": "B1", "b2_metafeature": "B2"}[condition]
        progress = st.progress(0.0, text="Waiting for LLM + subprocess...")
        run = None
        started = time.time()
        while time.time() - started < POLL_TIMEOUT:
            run = _find_run(dataset["id"], condition_upper, llm_backend, int(seed))
            if run is not None:
                break
            elapsed = time.time() - started
            progress.progress(min(elapsed / POLL_TIMEOUT, 0.95), text=f"Polling... {int(elapsed)}s elapsed")
            time.sleep(POLL_INTERVAL)
        progress.empty()

        if run is None:
            st.error(f"Timed out after {POLL_TIMEOUT}s.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        if run["success"]:
            hero = st.columns([1, 2])
            with hero[0]:
                st.markdown(f'<div class="score-hero">{run["test_score"]:.4f}</div>', unsafe_allow_html=True)
                st.caption(f"balanced-accuracy · run #{run['id']} · {run['runtime_seconds']:.2f}s")
            with hero[1]:
                st.metric("Iterations used", f"{run['iterations_used']}/{run['max_iterations']}")
                st.metric("Prompt tokens", run.get("prompt_tokens") or "—")
                st.metric("Completion tokens", run.get("completion_tokens") or "—")
        else:
            st.error(f"Run **#{run['id']}** failed after {run['iterations_used']} iterations.")
            st.code(f"{run['error_category']}: {run['error_message']}")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        script_r = _get(f"/runs/{run['id']}/script")
        model_r = _get(f"/runs/{run['id']}/model")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Generated pipeline")
            if script_r.status_code == 200:
                st.download_button(
                    "⬇  Download pipeline.py",
                    script_r.content,
                    file_name=f"run{run['id']}.py",
                    mime="text/x-python",
                    use_container_width=True,
                )
                st.caption(f"{len(script_r.content) / 1024:.1f} KiB · text/x-python")
                with st.expander("Preview script"):
                    st.code(script_r.text, language="python")
            else:
                st.warning(f"Script not available ({script_r.status_code})")
        with c2:
            st.markdown("#### Trained model")
            if model_r.status_code == 200:
                st.download_button(
                    "⬇  Download model.joblib",
                    model_r.content,
                    file_name=f"run{run['id']}.joblib",
                    mime="application/octet-stream",
                    use_container_width=True,
                )
                st.caption(f"{len(model_r.content) / 1024:.1f} KiB · joblib")
                st.info("Load with `joblib.load('run%s.joblib')`" % run["id"])
            else:
                st.warning(f"Model not available ({model_r.status_code})")

        st.markdown("</div>", unsafe_allow_html=True)


if submitted:
    run_workflow()
else:
    st.markdown(
        """
<div class="card">
    <span class="step-pill">Ready</span>
    <p style="margin: 0.5rem 0 0;">Configure the run in the sidebar and hit <strong>Run pipeline</strong>. The workflow will:</p>
    <ol style="margin-top: 0.5rem;">
        <li>Register (or reuse) your dataset.</li>
        <li>Compute meta-features and enqueue the LLM job.</li>
        <li>Poll until the pipeline finishes, then serve the script and the trained model from MinIO.</li>
    </ol>
</div>
        """,
        unsafe_allow_html=True,
    )
