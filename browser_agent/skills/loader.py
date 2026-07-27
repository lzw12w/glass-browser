"""Local skill discovery and parsing.

Discovery walks ``skill_roots`` (default: ``./.browser-agent/skills`` and
``~/.browser-agent/skills``) for directories containing a ``SKILL.md``
file. The frontmatter mirrors the agentskills.io / Claude Skills shape:

    ---
    name: my-skill              # optional; defaults to directory name
    description: ...            # short, used by skills_list (tier-1)
    version: 0.1.0
    license: MIT
    platforms: [macos, linux]   # optional OS allowlist
    ---

The body of ``SKILL.md`` is the tier-2 payload returned by ``skill_view``.
Files under ``references/``, ``templates/``, ``scripts/``, ``assets/`` are
the tier-3 payloads, also reached via ``skill_view`` with ``file_path=...``.
"""
from __future__ import annotations

import logging
import os
import re
import sys
import tomllib
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterable

from .model import Skill, SkillCommand

logger = logging.getLogger(__name__)


# Map user-friendly platform names to ``sys.platform`` prefixes. Mirrors the
# hermes-agent convention so SKILL.md frontmatter is portable across the two
# ecosystems.
_PLATFORM_MAP: dict[str, str] = {
    "macos": "darwin",
    "linux": "linux",
    "windows": "win32",
}


# ── Lazy YAML loader ──────────────────────────────────────────────────────
# Mirrors hermes-agent's ``yaml_load``. PyYAML's CSafeLoader is the C-accelerated
# safe parser; SafeLoader is the pure-Python fallback. We bind once on first
# call so module import stays cheap even when no skill ever needs YAML.
_yaml_load_fn = None


def _yaml_load(content: str) -> Any:
    global _yaml_load_fn
    if _yaml_load_fn is None:
        import yaml

        loader = getattr(yaml, "CSafeLoader", None) or yaml.SafeLoader

        def _load(value: str):
            return yaml.load(value, Loader=loader)

        _yaml_load_fn = _load
    return _yaml_load_fn(content)


def default_skill_roots() -> list[Path]:
    return [
        Path.cwd() / ".browser-agent" / "skills",
        Path.home() / ".browser-agent" / "skills",
    ]


def _parse_frontmatter(content: str) -> tuple[dict, str]:
    """Parse YAML frontmatter from a markdown string.

    Direct port of hermes-agent's ``parse_frontmatter``: PyYAML with
    ``CSafeLoader`` for full YAML 1.1 support (block scalars, nested
    metadata, lists, anchors, quoting), plus a key:value fallback for
    badly-malformed frontmatter so a single broken SKILL.md never sinks
    the whole skill registry.

    Returns ``(frontmatter_dict, remaining_body)``.
    """
    frontmatter: dict[str, Any] = {}
    body = content
    if not content.startswith("---"):
        return frontmatter, body
    end_match = re.search(r"\n---\s*\n", content[3:])
    if not end_match:
        return frontmatter, body
    yaml_content = content[3 : end_match.start() + 3]
    body = content[end_match.end() + 3 :]
    try:
        parsed = _yaml_load(yaml_content)
        if isinstance(parsed, dict):
            frontmatter = parsed
    except Exception:
        # Fallback: simple key:value parsing for malformed YAML, identical
        # in spirit to hermes-agent's fallback. We do NOT try to recover
        # block scalars here — if a SKILL.md is so broken that PyYAML
        # gives up, the author should fix it.
        for line in yaml_content.strip().split("\n"):
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            frontmatter[key.strip()] = value.strip()
    return frontmatter, body


def _first_body_sentence(body: str) -> str:
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        return line[:160]
    return ""


def _as_str_tuple(value) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list):
        return tuple(str(v).strip() for v in value if str(v).strip())
    return ()


def skill_matches_platform(platforms: Iterable[str]) -> bool:
    """Return True iff this OS is allowed by the skill's ``platforms`` field.

    Empty/missing platforms means "load on every platform" (default).
    Unknown platform tokens are ignored — they neither include nor exclude.
    """
    platforms = list(platforms or ())
    if not platforms:
        return True
    current = sys.platform
    for p in platforms:
        prefix = _PLATFORM_MAP.get(str(p).strip().lower())
        if prefix and current.startswith(prefix):
            return True
    return False


def skill_lookup_path_error(name: str) -> str | None:
    """Return an error message if ``name`` could escape skill search roots.

    Skill names are joined onto each trusted ``skill_roots`` entry to build
    the on-disk lookup path, so they must stay relative and free of ``..``
    segments. Otherwise ``name='../outside'`` or an absolute path could
    select a skill (and its files) from anywhere on disk. Mirrors the
    hermes-agent ``_skill_lookup_path_error`` guard.
    """
    if not isinstance(name, str):
        return "skill name must be a string"
    candidate = name.strip()
    if not candidate:
        return "skill name is empty"
    if (
        PurePosixPath(candidate).is_absolute()
        or PureWindowsPath(candidate).is_absolute()
        or PureWindowsPath(candidate).drive
    ):
        return "skill name must be a relative path within a skill root"
    parts = candidate.replace("\\", "/").split("/")
    if any(part in ("..", "") for part in parts):
        return "skill name cannot contain '..' or empty path components"
    return None


def file_path_within_skill(skill_dir: Path, file_path: str) -> Path | None:
    """Resolve ``file_path`` against ``skill_dir`` and refuse traversal.

    Returns the resolved absolute path on success, or ``None`` when the
    target escapes the skill directory or the path is malformed. Symlinks
    are followed before the containment check, so a symlink pointing
    outside the skill root is rejected.
    """
    if not isinstance(file_path, str) or not file_path.strip():
        return None
    candidate = file_path.strip()
    if PurePosixPath(candidate).is_absolute() or PureWindowsPath(candidate).is_absolute():
        return None
    parts = candidate.replace("\\", "/").split("/")
    if any(part in ("..", "") for part in parts):
        return None
    try:
        resolved = (skill_dir / candidate).resolve()
        skill_root = skill_dir.resolve()
    except OSError:
        return None
    if skill_root != resolved and skill_root not in resolved.parents:
        return None
    if not resolved.is_file():
        return None
    return resolved


def _load_manifest(skill_dir: Path) -> tuple[SkillCommand, ...]:
    path = skill_dir / "skill.toml"
    if not path.exists() or tomllib is None:
        return ()
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return ()
    commands = data.get("commands", [])
    if not isinstance(commands, list):
        return ()

    out: list[SkillCommand] = []
    for item in commands:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        argv_value = item.get("argv", ())
        argv: tuple[str, ...] = ()
        if isinstance(argv_value, list):
            argv = tuple(str(v) for v in argv_value)
        command = item.get("command")
        command_str = str(command).strip() if command is not None else None
        timeout = item.get("timeout_seconds")
        if timeout is not None:
            try:
                timeout = int(timeout)
            except (TypeError, ValueError):
                timeout = None
        cmd = SkillCommand(
            name=name,
            description=str(item.get("description", "")).strip(),
            command=command_str or None,
            argv=argv,
            cwd=str(item.get("cwd")).strip() if item.get("cwd") is not None else None,
            timeout_seconds=timeout,
        )
        if cmd.has_executable():
            out.append(cmd)
    return tuple(out)


def load_skill(skill_dir: Path) -> Skill | None:
    source = skill_dir / "SKILL.md"
    if not source.exists():
        return None
    try:
        raw = source.read_text(encoding="utf-8")
    except OSError:
        return None
    meta, body = _parse_frontmatter(raw)
    name = str(meta.get("name") or skill_dir.name).strip()
    if not name:
        return None
    description = str(meta.get("description") or _first_body_sentence(body)).strip()
    platforms = _as_str_tuple(meta.get("platforms"))
    return Skill(
        name=name,
        path=skill_dir,
        source_path=source,
        description=description,
        body=body,
        commands=_load_manifest(skill_dir),
        platforms=platforms,
        license=str(meta.get("license") or "").strip(),
        version=str(meta.get("version") or "").strip(),
    )


def iter_skill_dirs(root: Path) -> Iterable[Path]:
    if not root.exists():
        return ()
    if (root / "SKILL.md").exists():
        return (root,)
    try:
        children = [p for p in root.iterdir() if p.is_dir()]
    except OSError:
        return ()
    return tuple(sorted(children, key=lambda p: p.name.lower()))


def load_skills(
    roots: Iterable[Path] | None = None,
    *,
    enabled: set[str] | None = None,
    disabled: set[str] | None = None,
    apply_platform_filter: bool = True,
) -> list[Skill]:
    selected_roots = list(roots) if roots is not None else default_skill_roots()
    enabled_norm = {s.lower() for s in enabled} if enabled else None
    disabled_norm = {s.lower() for s in disabled} if disabled else set()
    skills: list[Skill] = []
    seen: set[str] = set()
    for root in selected_roots:
        for skill_dir in iter_skill_dirs(Path(root).expanduser()):
            skill = load_skill(skill_dir)
            if skill is None:
                continue
            key = skill.name.lower()
            if key in seen or key in disabled_norm:
                continue
            if enabled_norm is not None and "*" not in enabled_norm and key not in enabled_norm:
                continue
            if apply_platform_filter and not skill_matches_platform(skill.platforms):
                continue
            seen.add(key)
            skills.append(skill)
    return sorted(skills, key=lambda s: s.name.lower())


def split_path_list(raw: str | None) -> list[Path]:
    if not raw:
        return []
    return [Path(p).expanduser() for p in raw.split(os.pathsep) if p.strip()]
