"""Model registry: MinIO when reachable, plus a local catalog on disk.

Each artifact is stored as:
  {stage}/{name}/{version}/model   — weights or tracker YAML
  {stage}/{name}/{version}/metadata.json
A combined catalog.json is written locally and, when MinIO is up, to the bucket.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
LOCAL_ROOT = PIPELINE_DIR / "registry"
CATALOG_PATH = LOCAL_ROOT / "catalog.json"
DEFAULT_BUCKET = os.environ.get("MINIO_BUCKET", "mot17-models")
DEFAULT_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "127.0.0.1:19000")
DEFAULT_ACCESS = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
DEFAULT_SECRET = os.environ.get("MINIO_SECRET_KEY", "minioadmin")


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def file_size(path):
    return Path(path).stat().st_size


def object_key(stage, name, version, filename):
    return f"{stage}/{name}/{version}/{filename}"


class ModelRegistry:
    def __init__(self, endpoint=DEFAULT_ENDPOINT, access_key=DEFAULT_ACCESS,
                 secret_key=DEFAULT_SECRET, bucket=DEFAULT_BUCKET, secure=False):
        self.bucket = bucket
        self.local_root = LOCAL_ROOT
        self.local_root.mkdir(parents=True, exist_ok=True)
        self.client = None
        endpoint = endpoint.replace("http://", "").replace("https://", "")
        try:
            from minio import Minio
            client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
            if not client.bucket_exists(bucket):
                client.make_bucket(bucket)
            self.client = client
            print(f"registry: MinIO {endpoint}/{bucket}", flush=True)
        except Exception as error:
            print(f"registry: MinIO unavailable ({error}); local catalog only", flush=True)

    def _local_dir(self, stage, name, version):
        path = self.local_root / "artifacts" / stage / name / version
        path.mkdir(parents=True, exist_ok=True)
        return path

    def put_file(self, stage, name, version, source, filename=None):
        source = Path(source)
        filename = filename or source.name
        local = self._local_dir(stage, name, version) / filename
        if source.resolve() != local.resolve():
            local.write_bytes(source.read_bytes())
        if self.client:
            self.client.fput_object(self.bucket, object_key(stage, name, version, filename), str(local))
        return local

    def put_metadata(self, metadata):
        stage, name, version = metadata["stage"], metadata["name"], metadata["version"]
        payload = json.dumps(metadata, indent=2, sort_keys=True)
        local = self._local_dir(stage, name, version) / "metadata.json"
        local.write_text(payload + "\n", encoding="utf-8")
        if self.client:
            from io import BytesIO
            data = payload.encode("utf-8") + b"\n"
            self.client.put_object(self.bucket, object_key(stage, name, version, "metadata.json"),
                                   BytesIO(data), length=len(data), content_type="application/json")
        self._upsert_catalog(metadata)
        return local

    def _upsert_catalog(self, metadata):
        catalog = self.load_catalog()
        catalog["updated_at"] = utc_now()
        entries = [entry for entry in catalog["models"]
                   if not (entry["stage"] == metadata["stage"] and entry["name"] == metadata["name"]
                           and entry["version"] == metadata["version"])]
        entries.append(metadata)
        entries.sort(key=lambda entry: (entry["stage"], entry["name"], entry["version"]))
        catalog["models"] = entries
        CATALOG_PATH.write_text(json.dumps(catalog, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if self.client:
            from io import BytesIO
            data = CATALOG_PATH.read_bytes()
            self.client.put_object(self.bucket, "catalog.json", BytesIO(data), length=len(data),
                                   content_type="application/json")
        return catalog

    def load_catalog(self):
        if CATALOG_PATH.is_file():
            return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        return {"bucket": self.bucket, "models": [], "updated_at": None}

    def list_models(self):
        return self.load_catalog()["models"]

    def get_metadata(self, stage, name, version="1.0.0"):
        local = self._local_dir(stage, name, version) / "metadata.json"
        if self.client:
            try:
                self.client.fget_object(self.bucket, object_key(stage, name, version, "metadata.json"), str(local))
            except Exception:
                pass
        if local.is_file():
            return json.loads(local.read_text(encoding="utf-8"))
        for entry in self.list_models():
            if entry["stage"] == stage and entry["name"] == name and entry["version"] == version:
                return entry
        raise FileNotFoundError(f"{stage}/{name}/{version}")

    def get_file(self, stage, name, version, filename, dest):
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if self.client:
            self.client.fget_object(self.bucket, object_key(stage, name, version, filename), str(dest))
            return dest
        local = self._local_dir(stage, name, version) / filename
        if not local.is_file():
            raise FileNotFoundError(local)
        if local.resolve() != dest.resolve():
            dest.write_bytes(local.read_bytes())
        return dest


def metadata(*, name, stage, version, source, artifact, extra=None, accuracy=None):
    record = {
        "name": name,
        "stage": stage,
        "version": version,
        "source": source,
        "size_bytes": file_size(artifact),
        "size_mb": round(file_size(artifact) / (1024 * 1024), 3),
        "artifact": Path(artifact).name,
        "registered_at": utc_now(),
    }
    if accuracy:
        record["offline_accuracy"] = accuracy
    if extra:
        record.update(extra)
    return record


def print_catalog(models):
    if not models:
        print("registry is empty")
        return
    header = f"{'stage':<12} {'name':<32} {'ver':<8} {'size':>8} {'HOTA':>8} {'Recall':>8}"
    print(header)
    print("-" * len(header))
    for entry in models:
        acc = entry.get("offline_accuracy") or {}
        hota = acc.get("HOTA")
        recall = acc.get("Recall")
        size = f"{entry['size_bytes']} B" if entry["size_mb"] < 0.01 else f"{entry['size_mb']:.3f}"
        print(f"{entry['stage']:<12} {entry['name']:<32} {entry['version']:<8} "
              f"{size:>8} "
              f"{hota if hota is None else f'{hota:.1f}':>8} "
              f"{recall if recall is None else f'{recall:.1f}':>8}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="List registered model variants")
    parser.parse_args()
    print_catalog(ModelRegistry().list_models())


if __name__ == "__main__":
    main()
