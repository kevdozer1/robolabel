"""The held-out seal (MEASUREMENT_SPEC 2.3): explicit episode lists and the held-out guard.

Any command that would run a system on, display, or compute a metric on a held-out episode calls
:meth:`HeldoutGuard.check` first. The guard refuses (raises :class:`HeldoutRefused`) unless the call
carries ``final``, the SHA-256 of the preregistration file (``--final <prereg_sha256>``), and that
hash matches the file. Every call that involves a held-out key, refused or allowed, appends one line
to the access log (``eval/heldout_access_log.jsonl``). Dev-only calls log nothing.

``eval/heldout_ids.json`` format (``robolabel/heldout_ids/v1``)::

    {
      "schema_version": "robolabel/heldout_ids/v1",
      "created_utc": "2026-09-27T09:00:00Z",
      "families": {
        "F1": {"repo_id": "lerobot/svla_so101_pickplace", "revision": "<hub commit sha>",
               "rule": "spec 2.2 F1: the legacy test list of eval/so101_split.json",
               "episode_keys": ["F1/2", "F1/6", "F1/7"]},
        "F4": {"repo_id": "armnet/busybox_multitask", "revision": "<hub commit sha>",
               "rule": "spec 2.2 F4: all dev", "episode_keys": []}
      }
    }

- ``episode_keys`` lists every held-out episode of the family as ``"<family>/<episode_index>"``,
  including episodes that are not gold-annotated but inherit a held-out task's split (F2).
- Every family the harness reads should be listed, with an empty ``episode_keys`` list when all of
  its episodes are dev. A key of an unlisted family holds no held-out episode by this file, so by
  default the guard treats it as dev and warns (RuntimeWarning) once per family set;
  ``HeldoutGuard(..., allow_unlisted_families=False)`` raises ValueError for it instead.
- ``repo_id``, ``revision`` and ``rule`` document where the list came from; the loader only
  type-checks them. :func:`write_heldout_ids` writes the file in this format, byte-identically for
  identical input. Its SHA-256 goes into the preregistration (spec 7.2 item 2).

Nothing here makes a network call. The guard runs ``git rev-parse HEAD`` (read-only) when the
caller passes no commit.
"""

from __future__ import annotations

import getpass
import hashlib
import json
import os
import re
import subprocess
import warnings
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .receipts import JsonlWriter, is_sha256_hex, sha256_bytes, utc_now

SCHEMA_VERSION = "robolabel/heldout_ids/v1"
_KEY_RE = re.compile(r"^([A-Za-z][A-Za-z0-9_]*)/(\d+)$")
# Absolute paths that must not reach the access log, which lives in the repo (CI rejects machine paths):
# Windows drive paths, home folders (/home, /Users, /root) and drive mounts as Git Bash, WSL and
# Cygwin write them (/c/..., /mnt/c/..., /cygdrive/c/...).
_WINDOWS_PATH_RE = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/][^\s\"']*")
_POSIX_HOME_RE = re.compile(r"(?<![\w.])(?:/(?:home|Users|root)/|/(?:mnt/|cygdrive/)?[A-Za-z]/)[^\s\"']*")


class HeldoutRefused(PermissionError):
    """A command touched held-out episodes without a valid ``--final <prereg_sha256>``."""


# ---------------------------------------------------------------------------
# episode keys
# ---------------------------------------------------------------------------


def parse_episode_key(key: Any) -> tuple[str, int]:
    """``("F1", 2)`` for ``"F1/2"``. Raises ValueError for anything that is not ``"<family>/<index>"``."""
    if not isinstance(key, str):
        raise ValueError(f"episode keys are strings '<family>/<episode_index>', got {type(key).__name__} {key!r}")
    m = _KEY_RE.match(key.strip())
    if m is None:
        raise ValueError(f"malformed episode key {key!r}; expected '<family>/<episode_index>', e.g. 'F1/0'")
    return m.group(1), int(m.group(2))


def normalize_episode_key(key: Any) -> str:
    """Canonical form of an episode key (``"F1/007"`` becomes ``"F1/7"``)."""
    family, index = parse_episode_key(key)
    return f"{family}/{index}"


def _sort_key(key: str) -> tuple[str, int]:
    return parse_episode_key(key)


def require_explicit_episodes(episodes: Any) -> list[str]:
    """Normalized copy of an explicit, non-empty list (or tuple) of episode keys.

    Raises ValueError for None, an int, a range, a single string, any other non-list, an empty list,
    a malformed key or a repeated key. Every command that takes episodes takes an explicit list:
    ``robolabel run`` took ``list(range(limit))``, and on F1 a prefix of 8 includes held-out
    episodes 2, 6 and 7.
    """
    if episodes is None:
        raise ValueError("episodes must be an explicit list of episode keys, got None")
    if isinstance(episodes, range):
        raise ValueError("episodes must be an explicit list of episode keys, not a range: a contiguous "
                         "prefix can reach held-out episodes")
    if isinstance(episodes, (bool, int)):
        raise ValueError("episodes must be an explicit list of episode keys, not a count")
    if isinstance(episodes, (str, bytes)):
        raise ValueError("episodes must be a list of episode keys, not a single string")
    if not isinstance(episodes, (list, tuple)):
        raise ValueError(f"episodes must be an explicit list of episode keys, got {type(episodes).__name__}")
    if not episodes:
        raise ValueError("episodes must be a non-empty explicit list of episode keys")
    keys = [normalize_episode_key(k) for k in episodes]
    repeated = sorted((k for k, n in Counter(keys).items() if n > 1), key=_sort_key)
    if repeated:
        raise ValueError(f"episode list repeats keys: {', '.join(repeated)}")
    return keys


# ---------------------------------------------------------------------------
# heldout_ids.json
# ---------------------------------------------------------------------------


def _parse_heldout_doc(doc: Any) -> dict[str, list[str]]:
    """Validate a heldout_ids document; return ``{family: sorted normalized held-out keys}``."""
    if not isinstance(doc, dict):
        raise ValueError("heldout_ids must be a JSON object")
    if doc.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"heldout_ids schema_version must be {SCHEMA_VERSION!r}, got {doc.get('schema_version')!r}")
    families = doc.get("families")
    if not isinstance(families, dict) or not families:
        raise ValueError("heldout_ids needs a non-empty 'families' object")
    out: dict[str, list[str]] = {}
    for family, spec in families.items():
        if not isinstance(spec, dict):
            raise ValueError(f"families.{family} must be an object")
        for field in ("repo_id", "revision", "rule"):
            if spec.get(field) is not None and not isinstance(spec[field], str):
                raise ValueError(f"families.{family}.{field} must be a string")
        keys = spec.get("episode_keys")
        if not isinstance(keys, list):
            raise ValueError(f"families.{family}.episode_keys must be a list (empty when every episode is dev)")
        normalized = []
        for key in keys:
            fam, index = parse_episode_key(key)
            if fam != family:
                raise ValueError(f"families.{family}.episode_keys holds {key!r} from another family")
            normalized.append(f"{fam}/{index}")
        out[str(family)] = sorted(set(normalized), key=_sort_key)
    return out


def _read_heldout(path: str | Path) -> tuple[dict[str, list[str]], bytes]:
    raw = Path(path).read_bytes()  # a missing file raises: no file never means "nothing is sealed"
    return _parse_heldout_doc(json.loads(raw.decode("utf-8"))), raw


def load_heldout_keys(path: str | Path) -> set[str]:
    """Every held-out episode key in ``path`` (normalized). Raises for a missing or malformed file."""
    families, _ = _read_heldout(path)
    return {key for keys in families.values() for key in keys}


def write_heldout_ids(path: str | Path, families: dict[str, dict[str, Any]], created_utc: str | None = None) -> str:
    """Write ``heldout_ids.json`` in the v1 format and return its SHA-256.

    ``families`` maps a family id to ``{"repo_id", "revision", "rule", "episode_keys"}``. Keys are
    normalized and sorted by episode index, so identical input gives identical bytes.
    """
    doc: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "created_utc": created_utc or utc_now(),
                           "families": {}}
    for family in sorted(families):
        spec = families[family]
        doc["families"][family] = {
            "repo_id": spec.get("repo_id"),
            "revision": spec.get("revision"),
            "rule": spec.get("rule"),
            "episode_keys": list(spec.get("episode_keys") or []),
        }
    parsed = _parse_heldout_doc(doc)
    for family, keys in parsed.items():
        doc["families"][family]["episode_keys"] = keys
    text = json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return sha256_bytes(text.encode("utf-8"))


def filter_dev(keys: Iterable[str], heldout: Iterable[str] | HeldoutGuard) -> list[str]:
    """``keys`` without the held-out ones, in their original order (for display code).

    ``heldout`` is a set from :func:`load_heldout_keys` or a :class:`HeldoutGuard`. A malformed key
    raises ValueError rather than being shown or silently dropped.
    """
    sealed = heldout.heldout if isinstance(heldout, HeldoutGuard) else {normalize_episode_key(k) for k in heldout}
    return [k for k in keys if normalize_episode_key(k) not in sealed]


# ---------------------------------------------------------------------------
# preregistration hash
# ---------------------------------------------------------------------------


def file_sha256(path: str | Path) -> str:
    """SHA-256 of a file's bytes."""
    return sha256_bytes(Path(path).read_bytes())


def normalized_text_sha256(path: str | Path) -> str:
    """Spec 7.3 hash of a text file: UTF-8 (a leading BOM dropped), LF line endings, trailing whitespace
    stripped from every line, trailing blank lines dropped, one final newline."""
    text = Path(path).read_bytes().decode("utf-8")
    text = text.removeprefix(chr(0xFEFF)).replace("\r\n", "\n").replace("\r", "\n")
    body = "\n".join(line.rstrip() for line in text.split("\n")).rstrip("\n")
    return hashlib.sha256(((body + "\n") if body else "").encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# the guard
# ---------------------------------------------------------------------------


def _redact_paths(text: str) -> str:
    """Replace absolute Windows paths and home-folder paths with ``<abs>/<last part>``."""
    def repl(m: re.Match[str]) -> str:
        last = re.split(r"[\\/]", m.group(0).rstrip("\\/"))[-1]
        return f"<abs>/{last}"

    return _POSIX_HOME_RE.sub(repl, _WINDOWS_PATH_RE.sub(repl, text))


def _current_user() -> str:
    try:
        return getpass.getuser()
    except Exception:  # OSError on 3.13+, KeyError or ImportError on older versions without a user name
        return os.environ.get("USER") or os.environ.get("USERNAME") or "unknown"


def _final_for_log(final: Any) -> str | None:
    """The ``final`` hash as given (lowercased), or a placeholder when it is not a SHA-256 hex digest."""
    if final is None:
        return None
    given = str(final).strip().lower()
    return given if is_sha256_hex(given) else "<not a sha256>"


def _git_head(cwd: Path) -> str | None:
    """HEAD commit of the repository around ``cwd``, or None (read-only, best effort)."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True,
                             timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    sha = out.stdout.strip()
    return sha if out.returncode == 0 and re.fullmatch(r"[0-9a-f]{40,64}", sha) else None


class HeldoutGuard:
    """Refuses held-out episodes unless ``final`` is the preregistration file's SHA-256.

    ``heldout_ids_path`` must exist (a missing file raises, it never means "nothing is sealed").
    ``prereg_path`` is the committed preregistration file; without it every held-out call is refused.
    ``final`` is accepted when it equals the SHA-256 of the file bytes or the spec 7.3 normalized
    hash (:func:`normalized_text_sha256`); both identify the same committed file.

    A key of a family that ``heldout_ids.json`` does not list is not held-out by that file: with
    ``allow_unlisted_families=True`` (the default) the guard treats it as dev and emits a
    RuntimeWarning naming the family; with False it raises ValueError.
    """

    def __init__(self, heldout_ids_path: str | Path, access_log_path: str | Path,
                 prereg_path: str | Path | None = None, *, allow_unlisted_families: bool = True):
        self.heldout_ids_path = Path(heldout_ids_path)
        self.access_log_path = Path(access_log_path)
        self.prereg_path = Path(prereg_path) if prereg_path is not None else None
        self.allow_unlisted_families = allow_unlisted_families
        by_family, raw = _read_heldout(self.heldout_ids_path)
        self.families: frozenset[str] = frozenset(by_family)
        self.heldout: frozenset[str] = frozenset(k for keys in by_family.values() for k in keys)
        self.heldout_ids_sha256 = sha256_bytes(raw)
        self._log = JsonlWriter(self.access_log_path)

    def heldout_in(self, episode_keys: Iterable[str]) -> list[str]:
        """The held-out keys among ``episode_keys``, normalized and sorted."""
        return sorted({k for k in self._normalize(episode_keys) if k in self.heldout}, key=_sort_key)

    def _normalize(self, episode_keys: Iterable[str]) -> list[str]:
        if episode_keys is None:
            raise ValueError("episode_keys must be a list of keys, got None")
        if isinstance(episode_keys, (str, bytes)):
            raise ValueError("episode_keys must be a list of keys, not a single string")
        keys = [normalize_episode_key(k) for k in episode_keys]
        unlisted = sorted({k.split("/", 1)[0] for k in keys} - self.families)
        if unlisted:
            where = self.heldout_ids_path.name
            if not self.allow_unlisted_families:
                raise ValueError(
                    f"families {unlisted} are not listed in {where}, so the guard cannot tell dev from held-out; "
                    "list each family (with an empty episode_keys list if it is all dev)")
            warnings.warn(f"families {unlisted} are not listed in {where}; their episodes are treated as dev. "
                          "List each family (with an empty episode_keys list if it is all dev).",
                          RuntimeWarning, stacklevel=3)
        return keys

    def _final_ok(self, final: str | None) -> tuple[bool, str | None]:
        if final is None:
            return False, "no_final"
        if self.prereg_path is None or not self.prereg_path.is_file():
            return False, "no_prereg_file"
        given = str(final).strip().lower()
        try:
            valid = {file_sha256(self.prereg_path)}
        except OSError:  # unreadable: refuse (and log the attempt) rather than fail open or unlogged
            return False, "prereg_unreadable"
        try:
            valid.add(normalized_text_sha256(self.prereg_path))
        except (UnicodeDecodeError, OSError):
            pass
        return (True, None) if given in valid else (False, "hash_mismatch")

    def check(self, episode_keys: Iterable[str], *, command: str, final: str | None = None,
              commit: str | None = None) -> None:
        """Return if no key is held-out; otherwise log the access and refuse unless ``final`` is valid.

        Raises HeldoutRefused (naming how many held-out keys) when refused, ValueError for None, a
        single string, a malformed key, or (with ``allow_unlisted_families=False``) an unlisted family.
        An empty list involves no held-out key, so it returns without logging. When the access log
        cannot be written, an allowed call raises the OSError instead of proceeding.
        """
        keys = self._normalize(episode_keys)
        involved = sorted({k for k in keys if k in self.heldout}, key=_sort_key)
        if not involved:
            return
        allowed, reason = self._final_ok(final)
        entry = {
            "utc_time": utc_now(),
            "command": _redact_paths(str(command)),
            "commit": commit if commit is not None else _git_head(self.heldout_ids_path.parent),
            "user": _current_user(),
            "heldout_keys": involved,
            "n_heldout": len(involved),
            "n_keys": len(set(keys)),
            "outcome": "allowed_final" if allowed else "refused",
            "reason": reason,
            "final": _final_for_log(final),
            "heldout_ids_sha256": self.heldout_ids_sha256,
        }
        if allowed:
            self._log.write(entry)  # no access without its log line
            return
        message = (f"refused: {len(involved)} held-out episode key(s) in this call; held-out episodes need "
                   f"--final <sha256 of the committed preregistration file> ({reason})")
        try:
            self._log.write(entry)
        except OSError as exc:
            raise HeldoutRefused(message + "; the access log could not be written") from exc
        raise HeldoutRefused(message)
