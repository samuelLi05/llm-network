import argparse
import datetime as dt
import json
import re
from bisect import bisect_left
from pathlib import Path
from typing import Any


RUN_FILE_PATTERNS = {
    "agent_config": r"^log_(\d{8}-\d{6})_\d+_agent_configs\.log$",
    "stance": r"^log_(\d{8}-\d{6})_\d+_stance_configs\.log$",
    "network": r"^log_(\d{8}-\d{6})_\d+\.log$",
    "topology": r"^topology_(\d{8}-\d{6})\.jsonl$",
}


def _extract_run_ts(file_name: str, pattern: str) -> str | None:
    match = re.match(pattern, file_name)
    if not match:
        return None
    return match.group(1)


def _scan_runs(logs_root: Path) -> dict[str, dict[str, Path]]:
    runs: dict[str, dict[str, Path]] = {}

    subdirs = {
        "agent_config": logs_root / "agent_config_logs",
        "stance": logs_root / "stance_logs",
        "network": logs_root / "network_logs",
        "topology": logs_root / "topology_logs",
    }

    for kind, folder in subdirs.items():
        if not folder.exists():
            continue

        for path in folder.iterdir():
            if not path.is_file():
                continue
            run_ts = _extract_run_ts(path.name, RUN_FILE_PATTERNS[kind])
            if run_ts is None:
                continue
            runs.setdefault(run_ts, {})[kind] = path

    return runs


def _parse_datetime_to_epoch(ts_str: str) -> float:
    return dt.datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S").timestamp()


def _parse_agent_config_file(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")

    header_match = re.search(r"^\[(.*?)\]\s+Agent Configurations:\s*$", text, flags=re.MULTILINE)
    run_time = header_match.group(1) if header_match else None

    agent_pattern = re.compile(r"^Agent\s+(agent_\d+):\s*$", flags=re.MULTILINE)
    matches = list(agent_pattern.finditer(text))

    agent_configs: dict[str, dict[str, Any]] = {}
    for i, match in enumerate(matches):
        agent_id = match.group(1)
        body_start = match.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        prompt = text[body_start:body_end].strip()

        stance_sentence = None
        stance_match = re.search(r'The sentence "([^"]+)" reflects your stable perspective', prompt)
        if stance_match:
            stance_sentence = stance_match.group(1)

        agent_configs[agent_id] = {
            "prompt": prompt,
            "stable_perspective_sentence": stance_sentence,
        }

    return {
        "log_created_at": run_time,
        "log_created_at_epoch": _parse_datetime_to_epoch(run_time) if run_time else None,
        "agent_configs": agent_configs,
    }


def _parse_stance_file(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    lines = [line.rstrip("\n") for line in text.splitlines()]

    run_time = None
    topic = None
    weights: dict[str, float] = {}

    header_pattern = re.compile(r"^\[(.*?)\]\s+Baseline stance weights for:\s*(.*?):\s*$")
    weight_pattern = re.compile(r"^Agent\s+(agent_\d+):\s+weight=([-+]?\d*\.?\d+)$")

    for line in lines:
        header_match = header_pattern.match(line)
        if header_match:
            run_time = header_match.group(1)
            topic = header_match.group(2)
            continue

        weight_match = weight_pattern.match(line)
        if weight_match:
            weights[weight_match.group(1)] = float(weight_match.group(2))

    return {
        "log_created_at": run_time,
        "log_created_at_epoch": _parse_datetime_to_epoch(run_time) if run_time else None,
        "topic": topic,
        "stance_weights": weights,
    }


def _parse_network_file(path: Path) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            payload = json.loads(raw)
            if "timestamp" not in payload:
                continue

            entry_agent_ids = [k for k in payload.keys() if k.startswith("agent_")]
            if not entry_agent_ids:
                continue

            agent_id = entry_agent_ids[0]
            entry = payload[agent_id]
            msg_ts = payload.get("timestamp")
            msg_epoch = _parse_datetime_to_epoch(msg_ts)

            metrics = entry.get("metrics", {})
            published = metrics.get("published", {})

            messages.append(
                {
                    "sender_id": agent_id,
                    "timestamp": msg_ts,
                    "timestamp_epoch": msg_epoch,
                    "message": entry.get("message"),
                    "message_id": metrics.get("message_id"),
                    "index": metrics.get("index"),
                    "published": published,
                    "used_indices": published.get("used_indices", metrics.get("used_indices", [])),
                    "recommendation_indices": entry.get("recommendation_indices", []),
                }
            )

    messages.sort(key=lambda m: (m["timestamp_epoch"], m.get("index") or 0))
    return messages


def _parse_topology_file(path: Path) -> dict[str, Any]:
    graph: dict[str, list[str]] = {}
    snapshots: list[dict[str, Any]] = []
    topic = None

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            data = json.loads(raw)

            if data.get("t") == "connection_graph":
                graph = data.get("g", {}) or {}
                continue

            if "tp" in data and topic is None:
                topic = data.get("tp")

            if "ts" in data and "n" in data:
                snapshots.append(
                    {
                        "ts": float(data["ts"]),
                        "profiles": data.get("n", {}) or {},
                    }
                )

    snapshots.sort(key=lambda s: s["ts"])
    snapshot_times = [s["ts"] for s in snapshots]
    return {
        "topic": topic,
        "connection_graph": graph,
        "snapshots": snapshots,
        "snapshot_times": snapshot_times,
    }


def _select_snapshot_for_message(message_ts: float, snapshots: list[dict[str, Any]], snapshot_times: list[float]) -> dict[str, Any] | None:
    if not snapshots:
        return None

    idx = bisect_left(snapshot_times, message_ts)
    if idx >= len(snapshots):
        return snapshots[-1]
    return snapshots[idx]


def _build_agent_timelines(
    messages: list[dict[str, Any]],
    connection_graph: dict[str, list[str]],
    snapshots: list[dict[str, Any]],
    snapshot_times: list[float],
) -> dict[str, list[dict[str, Any]]]:
    all_agents = set(connection_graph.keys())
    for targets in connection_graph.values():
        all_agents.update(targets)
    for message in messages:
        all_agents.add(message["sender_id"])

    timelines: dict[str, list[dict[str, Any]]] = {agent: [] for agent in sorted(all_agents)}

    for message in messages:
        sender = message["sender_id"]
        affected_agents = set(connection_graph.get(sender, []))
        affected_agents.add(sender)

        selected_snapshot = _select_snapshot_for_message(message["timestamp_epoch"], snapshots, snapshot_times)
        selected_snapshot_ts = selected_snapshot["ts"] if selected_snapshot else None
        selected_profiles = selected_snapshot["profiles"] if selected_snapshot else {}

        for affected_agent in sorted(affected_agents):
            event = {
                "agent_id": affected_agent,
                "influenced_by": sender,
                "is_self_influence": affected_agent == sender,
                "message_timestamp": message["timestamp"],
                "message_timestamp_epoch": message["timestamp_epoch"],
                "message_index": message.get("index"),
                "message_id": message.get("message_id"),
                "message": message.get("message"),
                "published": message.get("published", {}),
                "recommendation_indices": message.get("recommendation_indices", []),
                "used_indices": message.get("used_indices", []),
                "matched_topology_snapshot_ts": selected_snapshot_ts,
                "matched_topology_snapshot_delta_s": (
                    selected_snapshot_ts - message["timestamp_epoch"] if selected_snapshot_ts is not None else None
                ),
                "topology_profile_for_agent": selected_profiles.get(affected_agent),
            }
            timelines.setdefault(affected_agent, []).append(event)

    for agent in timelines:
        timelines[agent].sort(key=lambda e: (e["message_timestamp_epoch"], e.get("message_index") or 0))

    return timelines


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def create_cleaned_data(
    logs_root: Path | str | None = None,
    output_root: Path | str | None = None,
    run_filter: str | None = None,
) -> dict[str, Any]:
    root = Path(logs_root) if logs_root is not None else Path(__file__).resolve().parents[1] / "logs"
    out_root = Path(output_root) if output_root is not None else Path(__file__).resolve().parent / "cleaned_data"

    runs = _scan_runs(root)
    if run_filter is not None:
        runs = {k: v for k, v in runs.items() if k == run_filter}

    out_root.mkdir(parents=True, exist_ok=True)

    summary = {
        "logs_root": str(root),
        "output_root": str(out_root),
        "runs_discovered": len(runs),
        "runs_processed": 0,
        "runs_skipped": [],
        "per_run": {},
    }

    required_types = {"agent_config", "stance", "network", "topology"}

    for run_ts in sorted(runs.keys()):
        paths = runs[run_ts]
        missing = sorted(required_types - set(paths.keys()))
        if missing:
            summary["runs_skipped"].append({"run": run_ts, "missing": missing})
            continue

        agent_config = _parse_agent_config_file(paths["agent_config"])
        stance_data = _parse_stance_file(paths["stance"])
        network_messages = _parse_network_file(paths["network"])
        topology_data = _parse_topology_file(paths["topology"])

        timelines = _build_agent_timelines(
            network_messages,
            topology_data["connection_graph"],
            topology_data["snapshots"],
            topology_data["snapshot_times"],
        )

        run_out_dir = out_root / f"run_{run_ts}"
        run_out_dir.mkdir(parents=True, exist_ok=True)

        static_payload = {
            "run_timestamp": run_ts,
            "topic": topology_data.get("topic") or stance_data.get("topic"),
            "agent_config_source": str(paths["agent_config"]),
            "stance_source": str(paths["stance"]),
            "network_source": str(paths["network"]),
            "topology_source": str(paths["topology"]),
            "agent_configs": agent_config["agent_configs"],
            "stance_weights": stance_data["stance_weights"],
        }
        _write_json(run_out_dir / "static_init.json", static_payload)

        run_manifest = {
            "run_timestamp": run_ts,
            "topic": topology_data.get("topic") or stance_data.get("topic"),
            "message_count": len(network_messages),
            "snapshot_count": len(topology_data["snapshots"]),
            "agent_count": len(timelines),
            "agent_ids": sorted(timelines.keys()),
        }
        _write_json(run_out_dir / "run_manifest.json", run_manifest)
        _write_json(run_out_dir / "connection_graph.json", topology_data["connection_graph"])

        _write_jsonl(run_out_dir / "messages_with_alignment.jsonl", [
            {
                **m,
                "matched_topology_snapshot_ts": (
                    _select_snapshot_for_message(m["timestamp_epoch"], topology_data["snapshots"], topology_data["snapshot_times"]) or {}
                ).get("ts"),
            }
            for m in network_messages
        ])

        per_agent_dir = run_out_dir / "per_agent"
        for agent_id, events in timelines.items():
            _write_jsonl(per_agent_dir / f"{agent_id}.jsonl", events)

        summary["runs_processed"] += 1
        summary["per_run"][run_ts] = {
            "output_dir": str(run_out_dir),
            "message_count": len(network_messages),
            "snapshot_count": len(topology_data["snapshots"]),
            "agent_files": len(timelines),
        }

    _write_json(out_root / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse run logs into per-agent cleaned opinion-dynamics data.")
    parser.add_argument("--logs-root", type=str, default=None, help="Path to logs root (default: <repo>/logs)")
    parser.add_argument("--output-root", type=str, default=None, help="Path to output root (default: modeling/cleaned_data)")
    parser.add_argument("--run", type=str, default=None, help="Optional run timestamp filter, e.g. 20260217-223650")
    args = parser.parse_args()

    summary = create_cleaned_data(
        logs_root=Path(args.logs_root) if args.logs_root else None,
        output_root=Path(args.output_root) if args.output_root else None,
        run_filter=args.run,
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
