import json
import os
from datetime import date, datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

from filelock import FileLock, Timeout

from .schemas import EpisodeRecord, EpisodicEpisode


class EpisodicMemoryStore:
    """Canonical JSON storage: one lifecycle-record list per calendar day."""

    def __init__(self, episodes_dir: Path | str = "episodic_memory/episodes") -> None:
        self.episodes_dir = Path(episodes_dir)
        self._lock_path = self.episodes_dir / ".episodes.lock"

    def create(
        self,
        episode: EpisodicEpisode,
        timestamp: datetime | None = None,
    ) -> EpisodeRecord:
        record = EpisodeRecord(
            id=str(uuid4()),
            timestamp=self._normalize_timestamp(timestamp),
            action="create",
            episode=episode,
        )
        self._append(record)
        return record

    def update(
        self,
        target_id: str,
        episode: EpisodicEpisode,
        timestamp: datetime | None = None,
    ) -> EpisodeRecord:
        record = EpisodeRecord(
            id=str(uuid4()),
            timestamp=self._normalize_timestamp(timestamp),
            action="update",
            target_id=target_id,
            episode=episode,
        )
        self._append(record, require_active_id=target_id)
        return record

    def delete(
        self,
        target_id: str,
        timestamp: datetime | None = None,
    ) -> EpisodeRecord:
        record = EpisodeRecord(
            id=str(uuid4()),
            timestamp=self._normalize_timestamp(timestamp),
            action="delete",
            target_id=target_id,
        )
        self._append(record, require_active_id=target_id)
        return record

    def read_day(self, day: date) -> list[EpisodeRecord]:
        path = self._path_for_day(day)
        if not path.exists():
            return []
        try:
            raw_records = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Could not read episodic memory file: {path}") from exc
        if not isinstance(raw_records, list):
            raise RuntimeError(f"Episodic memory file must contain a JSON list: {path}")
        try:
            return [EpisodeRecord.model_validate(item) for item in raw_records]
        except Exception as exc:
            raise RuntimeError(f"Invalid episodic records in file: {path}: {exc}") from exc

    def list_records(self) -> list[EpisodeRecord]:
        """Return all lifecycle records, including updates and delete tombstones."""
        self.episodes_dir.mkdir(parents=True, exist_ok=True)
        try:
            with FileLock(str(self._lock_path), timeout=10):
                return self._read_records_unlocked()
        except Timeout as exc:
            raise RuntimeError("Timed out waiting for episodic storage lock.") from exc

    def list_active(self) -> list[EpisodeRecord]:
        """Return current canonical episodes after applying lifecycle records."""
        return self._active_from_records(self.list_records())

    def get_active_by_ids(self, ids: list[str]) -> dict[str, EpisodeRecord]:
        wanted = set(ids)
        return {
            record.id: record
            for record in self.list_active()
            if record.id in wanted
        }

    def _read_records_unlocked(self) -> list[EpisodeRecord]:
        records: list[EpisodeRecord] = []
        if not self.episodes_dir.exists():
            return records

        for path in sorted(self.episodes_dir.glob("EP-????-??-??.json")):
            try:
                raw_records = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"Could not read episodic memory file: {path}") from exc
            if not isinstance(raw_records, list):
                raise RuntimeError(f"Episodic memory file must contain a JSON list: {path}")
            try:
                records.extend(EpisodeRecord.model_validate(item) for item in raw_records)
            except Exception as exc:
                raise RuntimeError(f"Invalid episodic records in file: {path}: {exc}") from exc
        return records

    @classmethod
    def _active_from_records(cls, records: list[EpisodeRecord]) -> list[EpisodeRecord]:
        active: dict[str, EpisodeRecord] = {}
        for record in sorted(records, key=lambda item: cls._sort_timestamp(item.timestamp)):
            if record.action == "create" and record.episode is not None:
                active[record.id] = record
            elif record.action == "update" and record.target_id:
                original = active.get(record.target_id)
                if original is not None and record.episode is not None:
                    active[record.target_id] = original.model_copy(
                        update={"episode": record.episode}
                    )
            elif record.action == "delete" and record.target_id:
                active.pop(record.target_id, None)
        return list(active.values())

    def _append(
        self,
        record: EpisodeRecord,
        require_active_id: str | None = None,
    ) -> None:
        self.episodes_dir.mkdir(parents=True, exist_ok=True)
        try:
            with FileLock(str(self._lock_path), timeout=10):
                if require_active_id is not None:
                    active_ids = {
                        item.id
                        for item in self._active_from_records(
                            self._read_records_unlocked()
                        )
                    }
                    if require_active_id not in active_ids:
                        raise KeyError(
                            f"Active episode '{require_active_id}' does not exist."
                        )

                path = self._path_for_day(record.timestamp.date())
                existing = self.read_day(record.timestamp.date())
                existing.append(record)
                payload = [item.model_dump(mode="json") for item in existing]

                temporary_path: Path | None = None
                try:
                    with NamedTemporaryFile(
                        "w",
                        encoding="utf-8",
                        dir=self.episodes_dir,
                        prefix=f".{path.name}.",
                        suffix=".tmp",
                        delete=False,
                    ) as temporary:
                        json.dump(payload, temporary, ensure_ascii=False, indent=2)
                        temporary.write("\n")
                        temporary.flush()
                        os.fsync(temporary.fileno())
                        temporary_path = Path(temporary.name)
                    os.replace(temporary_path, path)
                    temporary_path = None
                finally:
                    if temporary_path is not None:
                        temporary_path.unlink(missing_ok=True)
        except Timeout as exc:
            raise RuntimeError("Timed out waiting for episodic storage lock.") from exc
        except OSError as exc:
            raise RuntimeError(
                f"Could not persist episodic memory record: {exc}"
            ) from exc

    @staticmethod
    def _normalize_timestamp(timestamp: datetime | None) -> datetime:
        value = timestamp or datetime.now(timezone.utc)
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    @staticmethod
    def _sort_timestamp(timestamp: datetime) -> datetime:
        if timestamp.tzinfo is None:
            return timestamp.replace(tzinfo=timezone.utc)
        return timestamp.astimezone(timezone.utc)

    def _path_for_day(self, day: date) -> Path:
        return self.episodes_dir / f"EP-{day:%Y-%m-%d}.json"
