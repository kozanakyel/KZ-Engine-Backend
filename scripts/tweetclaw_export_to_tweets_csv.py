"""Convert TweetClaw exports into KZEngine tweet CSV files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

WRAPPER_KEYS = ("tweets", "items", "data", "results", "records")
TEXT_KEYS = ("text", "tweet_text", "full_text", "content", "body")
TIME_KEYS = ("created_at", "createdAt", "timestamp", "time", "date")
USER_KEYS = ("username", "user", "author", "screen_name", "handle", "name")


def load_export(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as file:
            return list(csv.DictReader(file))

    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    parsed = json.loads(text)
    return unwrap_records(parsed)


def unwrap_records(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [record for record in value if isinstance(record, dict)]
    if isinstance(value, dict):
        for key in WRAPPER_KEYS:
            records = value.get(key)
            if records is not None:
                return unwrap_records(records)
        return [value]
    return []


def first_text(record: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def first_user(record: dict[str, Any]) -> str:
    for key in USER_KEYS:
        value = record.get(key)
        if isinstance(value, dict):
            nested = first_text(value, USER_KEYS)
            if nested:
                return nested
        elif value is not None and str(value).strip():
            return str(value).strip()
    return "unknown"


def convert_records(records: list[dict[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for record in records:
        text = first_text(record, TEXT_KEYS)
        if not text:
            continue

        row = {
            "created_at": first_text(record, TIME_KEYS),
            "text": text,
            "username": first_user(record),
        }
        key = (row["created_at"], row["username"], row["text"])
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["created_at", "text", "username"])
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert TweetClaw CSV, JSONL, or JSON exports to KZEngine tweet CSV."
    )
    parser.add_argument("input", type=Path, help="TweetClaw export file")
    parser.add_argument("output", type=Path, help="KZEngine tweet CSV output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = convert_records(load_export(args.input))
    write_csv(args.output, rows)
    print(f"Wrote {len(rows)} tweet rows to {args.output}")


if __name__ == "__main__":
    main()
