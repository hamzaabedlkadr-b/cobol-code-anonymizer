"""Local stage results bound to the source bytes and detector settings."""

from dataclasses import asdict
import hashlib
import json

from .logical_text import LogicalPiece, LogicalText
from .scanner import Finding, iter_all_files, relative_name


def finding_from_dict(data):
    data = dict(data)
    logical = data.get("logical")
    if logical:
        data["logical"] = LogicalText(logical["text"], tuple(LogicalPiece(**piece) for piece in logical["pieces"]), logical["kind"])
    return Finding(**data)


def source_signature(input_path, skip_roots, supplied, settings):
    files = {relative_name(path, input_path): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in iter_all_files(input_path, skip_root=skip_roots)
             if path.resolve() not in supplied}
    return {"input": str(input_path), "files": files, "settings": settings,
            "watchlists": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in supplied}}


def save_checkpoint(path, signature, findings, **state):
    payload = {"version": 1, "signature": signature,
               "findings": [asdict(finding) for finding in findings], **state}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def load_checkpoint(path, signature):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != 1 or payload.get("signature") != signature:
        raise ValueError("saved stage does not match the input or detector settings; run extract again")
    payload["findings"] = [finding_from_dict(item) for item in payload["findings"]]
    return payload
