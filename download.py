"""Resume TencentGR-1M with persistent HTTP partials and network retries.

Run with .download_venv/bin/python (huggingface_hub 0.36).
Progress is written to download.log and download_status.json.
"""

import fcntl
import json
import os
from pathlib import Path
import threading
import time
from datetime import datetime, timezone

# Override the shell's mirror setting for this process only. HTTP downloads in
# the existing base environment preserve partial files between failed attempts.
os.environ["HF_ENDPOINT"] = "https://huggingface.co"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "30"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import requests
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "TencentGR-1M"
REPO = "TAAC2025/TencentGR-1M"
MANIFEST = DEST / ".cache/download_manifest.json"
STATUS = ROOT / "download_status.json"


def log(message):
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {message}", flush=True)


def main():
    with (ROOT / ".download.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another download.py process is already running.")
        run()


def run():
    attempt = 0
    while not MANIFEST.exists():
        attempt += 1
        try:
            response = requests.get(
                f"https://huggingface.co/api/datasets/{REPO}",
                params={"blobs": "true"}, timeout=30,
            )
            response.raise_for_status()
            info = response.json()
            assert info["id"] == REPO and info["siblings"]
            MANIFEST.parent.mkdir(parents=True, exist_ok=True)
            MANIFEST.write_text(json.dumps(info, indent=2))
        except Exception as exc:
            log(f"Manifest attempt {attempt}: {type(exc).__name__}: {exc}")
            time.sleep(min(30 * attempt, 300))

    info = json.loads(MANIFEST.read_text())
    assert info["id"] == REPO
    files = info["siblings"]
    assert files and all(isinstance(f.get("size"), int) for f in files)
    state = {"repo_id": REPO, "revision": info["sha"], "pid": os.getpid(),
             "phase": "downloading", "attempt": 0,
             "total_files": len(files), "total_bytes": sum(f["size"] for f in files)}
    stop = threading.Event()
    report_lock = threading.Lock()

    def pending():
        return [f for f in files if not (DEST / f["rfilename"]).is_file()
                or (DEST / f["rfilename"]).stat().st_size != f["size"]]

    def report():
        with report_lock:
            missing = pending()
            complete_bytes = state["total_bytes"] - sum(f["size"] for f in missing)
            state.update(completed_files=len(files) - len(missing),
                         completed_bytes=complete_bytes,
                         remaining_bytes=state["total_bytes"] - complete_bytes,
                         updated_at=datetime.now(timezone.utc).isoformat())
            temp = STATUS.with_suffix(".json.tmp")
            temp.write_text(json.dumps(state, indent=2) + "\n")
            temp.replace(STATUS)
            log(f"{state['phase']}: {state['completed_files']}/{len(files)} files, "
                f"{complete_bytes / 1e9:.2f}/{state['total_bytes'] / 1e9:.2f} GB complete")

    def monitor():
        while not stop.wait(30):
            report()

    log(f"Resuming {REPO} at {info['sha']} into {DEST}; 4 HTTP workers")
    report()
    threading.Thread(target=monitor, daemon=True).start()
    try:
        while True:
            missing = pending()
            if not missing:
                break
            state["attempt"] += 1
            state["phase"] = "downloading"
            log(f"Attempt {state['attempt']}: downloading {len(missing)} remaining files")
            try:
                snapshot_download(
                    repo_id=REPO, repo_type="dataset", revision=info["sha"],
                    local_dir=str(DEST), token=False, max_workers=4,
                    allow_patterns=[f["rfilename"] for f in missing],
                )
                if pending():
                    raise RuntimeError("Remote file-size verification found incomplete files")
            except Exception as exc:
                delay = min(30 * state["attempt"], 300)
                state["phase"] = "retrying"
                state["last_error"] = f"{type(exc).__name__}: {exc}"
                log(f"{state['last_error']} — retrying in {delay}s; keeping downloaded files")
                report()
                time.sleep(delay)
        state["phase"] = "complete"
        state.pop("last_error", None)
        report()
        (ROOT / "download.complete").write_text(
            f"Downloaded {REPO}@{info['sha']}\n"
            f"Verified all {len(files)} file sizes against the remote manifest.\n"
            f"Total: {state['total_bytes']} bytes\n"
        )
        log("DOWNLOAD COMPLETE: all remote file sizes verified")
    finally:
        stop.set()


if __name__ == "__main__":
    main()
