"""Local stage results bound to the source bytes and detector settings."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from .logical_text import LogicalPiece, LogicalText
from .scanner import Finding, iter_all_files, relative_name


def code_stamp(stage):
    """Invalidate only the stage whose source code changed."""
    modules = ("scanner", "cobol_layout", "logical_text", "source_reader", "text_matching", "extractor", "overlaps", "pipeline") if stage == "extract" else (
        "pipeline", "policy", "judge", "verifier", "decisions", "review_decisions", "llm", "cli", "checkpoints")
    return hashlib.sha256(b"".join(Path(__file__).with_name(name + ".py").read_bytes() for name in modules)).hexdigest()


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
    if any(signature["files"].get(file) != digest for file, digest in state["source_hashes"].items()):
        raise ValueError("source changed during the stage; run extract again")
    payload = {"version": 2, "signature": signature, "extraction_code": code_stamp("extract"),
               "decision_code": code_stamp("decide") if path.stem == "decisions" else None,
               "findings": [asdict(finding) for finding in findings], **state}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def load_checkpoint(path, signature):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != 2 or payload.get("signature") != signature:
        raise ValueError("saved stage does not match the input or detector settings; run extract again")
    if payload.get("extraction_code") != code_stamp("extract"):
        raise ValueError("extraction code changed; run extract again")
    if path.stem == "decisions" and payload.get("decision_code") != code_stamp("decide"):
        raise ValueError("decision code changed; run decide again")
    payload["findings"] = [finding_from_dict(item) for item in payload["findings"]]
    return payload


def model_state(model):
    if model is None:
        return None
    fields = ("model", "calls", "chunk_calls", "canary_calls", "http_attempts", "cache_hits",
              "latency_s", "prompt_tokens", "completion_tokens", "errors",
              "review_items", "comparisons", "complete", "failure_reason", "chunks")
    return {field: getattr(model, field) for field in fields if hasattr(model, field)}
