# 01_build_manifest.py
import json, glob, os
from datetime import datetime
from app.config import config

OBJECTS_DIR = "data/objects"          # OK to keep hardcoded for now
OUT_DIR = "output"
KB_VERSION = config.KB_VERSION
MANIFEST_PATH = config.MANIFEST_PATH

os.makedirs(OUT_DIR, exist_ok=True)

kb_build_time = datetime.utcnow().isoformat() + "Z"

def infer_lang(url: str) -> str:
    if not url:
        return "en"
    if "/de/" in url:
        return "de"
    if "/en/" in url:
        return "en"
    return "en"

manifest = []
errors = []

for fp in glob.glob(os.path.join(OBJECTS_DIR, "**/*.json"), recursive=True):
    try:
        with open(fp, "r", encoding="utf-8") as f:
            obj = json.load(f)

        meta = obj.get("metadata", {})
        content = obj.get("content", {})

        object_id = meta.get("object_id")
        object_type = meta.get("object_type")
        url = meta.get("url")
        title = meta.get("title")
        source_html_path = meta.get("source_html_path")

        last_scraped = meta.get("last_scraped") or kb_build_time
        last_processed = meta.get("last_processed") or kb_build_time

        full_text = content.get("full_text") or ""

        # Basic validation
        missing = []
        if not object_id: missing.append("object_id")
        if not object_type: missing.append("object_type")
        if not url: missing.append("url")
        if not full_text.strip(): missing.append("content.full_text")

        if missing:
            errors.append({"file": fp, "missing": missing})

        row = {
            "kb_version": KB_VERSION,  # ✅ ADDED
            "object_id": object_id,
            "object_type": object_type,
            "url": url,
            "title": title,
            "source_html_path": source_html_path,
            "last_scraped": last_scraped,
            "last_processed": last_processed,
            "lang": infer_lang(url),
            "full_text": full_text,
            "source_file": os.path.basename(fp),
        }
        manifest.append(row)

    except Exception as e:
        errors.append({"file": fp, "error": str(e)})

# Write JSONL
jsonl_path = MANIFEST_PATH
with open(jsonl_path, "w", encoding="utf-8") as f:
    for row in manifest:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

# Write summary
summary = {
    "kb_version": KB_VERSION,  # ✅ ADDED (optional but helpful)
    "num_objects": len(manifest),
    "num_errors": len(errors),
    "errors_sample": errors[:20],
    "object_types": {},
}
for row in manifest:
    t = row.get("object_type") or "UNKNOWN"
    summary["object_types"][t] = summary["object_types"].get(t, 0) + 1

summary_path = os.path.join(OUT_DIR, "objects_summary.json")
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("Wrote:", jsonl_path)
print("Summary:", summary_path)
print("Errors:", len(errors))
