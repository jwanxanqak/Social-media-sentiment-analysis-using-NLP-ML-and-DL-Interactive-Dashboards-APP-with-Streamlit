from __future__ import annotations

from typing import Any


TEXT_COLUMNS = (
    "text",
    "tweet_text",
    "full_text",
    "content",
    "body",
    "message",
    "comment",
)

ID_COLUMNS = (
    "id",
    "tweet_id",
    "post_id",
    "status_id",
    "source_id",
    "url",
)

USER_COLUMNS = (
    "username",
    "user",
    "author_username",
    "screen_name",
    "handle",
    "name",
)

DATE_COLUMNS = (
    "created_at",
    "date",
    "timestamp",
    "time",
)


def _lookup_key(row: dict[str, Any], aliases: tuple[str, ...]) -> str | None:
    lookup = {str(key).strip().lower(): str(key) for key in row}
    for alias in aliases:
        key = lookup.get(alias)
        if key is not None:
            return key
    return None


def _value(row: dict[str, Any], aliases: tuple[str, ...]) -> str:
    key = _lookup_key(row, aliases)
    if key is None:
        return ""
    value = row.get(key)
    if value is None:
        return ""
    cleaned = str(value).strip()
    if cleaned.lower() == "nan":
        return ""
    return cleaned


def normalize_xquik_records(
    records: list[dict[str, Any]],
    defaults: dict[str, Any],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []

    for index, row in enumerate(records, start=1):
        text = _value(row, TEXT_COLUMNS)
        if not text:
            continue

        source_id = _value(row, ID_COLUMNS) or str(index)
        record = dict(defaults)
        record["tweet_id"] = source_id
        record["text"] = text
        record["name"] = _value(row, USER_COLUMNS) or record["name"]
        record["tweet_created"] = _value(row, DATE_COLUMNS) or record["tweet_created"]
        record["source_id"] = source_id
        normalized.append(record)

    return normalized
