"""Configuration. Reads env first, then ~/.browser-agent/config.toml if present."""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


CONFIG_PATH = Path.home() / ".browser-agent" / "config.toml"


@dataclass
class Config:
    # ---- browser -----------------------------------------------------
    # Headed by default: the user watches the agent drive the page.
    browser_headless: bool = False
    # When set, attach to an already-running Chrome via CDP instead of
    # launching a fresh Playwright-managed browser.
    browser_cdp_url: str | None = None

    llm_provider: str = "anthropic"
    llm_model: str = "MiMo-V2.5-Pro"
    anthropic_api_key: str | None = None
    anthropic_base_url: str | None = None

    # OpenAI Responses API (used when llm_provider == "openai").
    # llm_model is shared across providers; openai_* fields below are only
    # consumed when this provider is selected. We keep keys per-provider so
    # switching providers via env doesn't accidentally smuggle credentials
    # across the boundary.
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    # None = let the server choose. Reasoning tokens count toward this budget.
    openai_max_output_tokens: int | None = None
    # ``minimal`` / ``low`` / ``medium`` / ``high``; ignored on non-reasoning models.
    openai_reasoning_effort: str | None = "medium"
    # ``auto`` | ``concise`` | ``detailed`` | None. Cosmetic; encrypted_content
    # for cross-turn replay is requested separately.
    openai_reasoning_summary: str | None = "auto"

    workdir_root: Path = field(default_factory=lambda: Path.home() / ".browser-agent" / "runs")
    confirm_for: set[str] = field(default_factory=lambda: {
        "navigate", "run_shell", "run_skill_command"
    })

    # Agent loop budgets (per user turn).
    max_inner_steps: int = 999999
    max_taps: int = 999999

    # Local skill system.
    skill_roots: list[Path] = field(default_factory=lambda: [
        Path.cwd() / ".browser-agent" / "skills",
        Path.home() / ".browser-agent" / "skills",
    ])
    # None = all discovered skills are eligible. Set of names = allowlist.
    enabled_skills: set[str] | None = None
    # Skills explicitly hidden from discovery (denylist).
    disabled_skills: set[str] | None = None
    skill_max_chars: int = 6000

    # Controlled shell. ``run_shell`` is disabled by default; enable via
    # --allow-shell, config.toml, or BROWSER_AGENT_ENABLE_SHELL=1.
    # ``run_skill_command`` is always exposed regardless of this flag — its
    # cwd is anchored to the selected skill's directory and the same
    # workspace_roots / denylist guard rails apply.
    enable_shell: bool = False
    shell_workspace_roots: list[Path] = field(default_factory=lambda: [
        Path.cwd(),
        Path.home() / ".browser-agent" / "skills",
    ])
    shell_timeout_seconds: int = 60
    shell_max_output_chars: int = 12000

    # 是否在传给 LLM 的 messages 中把"非最近 N 次"的 browser_snapshot
    # tool_result 替换为摘要，用以节省 token。trace.jsonl 不受影响。
    elide_old_snapshots: bool = True
    # 保留最近 N 次 snapshot 完整不压缩的窗口大小。仅当
    # elide_old_snapshots=True 时生效。
    elide_keep_recent: int = 2

    # 用作 UI 上下文进度条分母的"模型上下文窗口"经验值。两家 API 在
    # 响应里都不返回这个数字（`/v1/models` 也没有），所以 LLM client
    # 内置了一份模型→窗口的查表。当查表猜错（比如长上下文档位、
    # 自建网关上的同名模型等）时，可以在 ~/.browser-agent/config.toml
    # 里写 ``context_window_override = 1000000`` 或设
    # ``BROWSER_AGENT_CONTEXT_WINDOW`` 环境变量来覆盖。None = 用查表默认值。
    context_window_override: int | None = None

    # 上下文压缩：当 input_tokens 达到 context_window * compact_trigger_ratio
    # 时自动压缩旧历史。设为 <=0 或 >1 可禁用。
    compact_trigger_ratio: float = 0.75
    # 压缩时保留最近 N 个 action cycle（一次 tool_use + 其结果）不压缩。
    # 注意单位是 cycle 不是 user turn：单条指令常展开成大量工具往返，
    # 按 user turn 切粒度太粗（详见 agent/compact.py 模块注释）。
    compact_keep_recent_cycles: int = 80
    # Tier 1 无损压缩时保留最近 N 次工具调用结果完整不压缩（按 call 顺序）。
    # 覆盖模型下一步决策所需的 working memory —— snapshot 返回的 ref、click
    # 的结果、wait_for 的 evidence 等。其中 browser_snapshot 上限沿用
    # elide_keep_recent（默认 2），避免大 snapshot 独占整个 skip 窗口。
    compact_tier1_keep_recent_tool_results: int = 8

    @classmethod
    def load(cls) -> "Config":
        cfg = cls()
        # 1. file
        if CONFIG_PATH.exists() and tomllib is not None:
            try:
                data = tomllib.loads(CONFIG_PATH.read_text())
                for k, v in data.items():
                    if hasattr(cfg, k):
                        setattr(cfg, k, v)
            except Exception:
                pass
        cfg.skill_roots = _coerce_paths(cfg.skill_roots)
        cfg.shell_workspace_roots = _coerce_paths(cfg.shell_workspace_roots)
        if cfg.enabled_skills is not None:
            cfg.enabled_skills = _coerce_set(cfg.enabled_skills)
        if cfg.disabled_skills is not None:
            cfg.disabled_skills = _coerce_set(cfg.disabled_skills)
        # 2. env overrides
        if os.environ.get("BROWSER_AGENT_HEADLESS", "").lower() in ("1", "true", "yes"):
            cfg.browser_headless = True
        cfg.browser_cdp_url = os.environ.get("BROWSER_AGENT_CDP_URL", cfg.browser_cdp_url)
        # Provider selection. BROWSER_AGENT_LLM_PROVIDER wins over the file
        # default; legacy ANTHROPIC_MODEL still drives llm_model when set.
        cfg.llm_provider = os.environ.get("BROWSER_AGENT_LLM_PROVIDER", cfg.llm_provider)
        cfg.llm_model = os.environ.get("BROWSER_AGENT_LLM_MODEL",
                                       os.environ.get("ANTHROPIC_MODEL", cfg.llm_model))
        cfg.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", cfg.anthropic_api_key)
        cfg.anthropic_base_url = os.environ.get("ANTHROPIC_BASE_URL", cfg.anthropic_base_url)
        cfg.openai_api_key = os.environ.get("OPENAI_API_KEY", cfg.openai_api_key)
        cfg.openai_base_url = os.environ.get("OPENAI_BASE_URL", cfg.openai_base_url)
        cfg.openai_reasoning_effort = os.environ.get(
            "OPENAI_REASONING_EFFORT", cfg.openai_reasoning_effort,
        )
        cfg.openai_reasoning_summary = os.environ.get(
            "OPENAI_REASONING_SUMMARY", cfg.openai_reasoning_summary,
        )
        raw_max_out = os.environ.get("OPENAI_MAX_OUTPUT_TOKENS")
        if raw_max_out:
            try:
                cfg.openai_max_output_tokens = int(raw_max_out)
            except ValueError:
                pass
        raw_skill_paths = os.environ.get("BROWSER_AGENT_SKILL_PATHS")
        if raw_skill_paths:
            cfg.skill_roots = [Path(p).expanduser() for p in raw_skill_paths.split(os.pathsep) if p.strip()]
        raw_enabled = os.environ.get("BROWSER_AGENT_ENABLED_SKILLS")
        if raw_enabled:
            cfg.enabled_skills = {p.strip() for p in raw_enabled.split(",") if p.strip()}
        raw_disabled = os.environ.get("BROWSER_AGENT_DISABLED_SKILLS")
        if raw_disabled:
            cfg.disabled_skills = {p.strip() for p in raw_disabled.split(",") if p.strip()}
        raw_shell_roots = os.environ.get("BROWSER_AGENT_SHELL_WORKSPACE_ROOTS")
        if raw_shell_roots:
            cfg.shell_workspace_roots = [Path(p).expanduser() for p in raw_shell_roots.split(os.pathsep) if p.strip()]
        if os.environ.get("BROWSER_AGENT_ENABLE_SHELL", "").lower() in ("1", "true", "yes"):
            cfg.enable_shell = True
        if os.environ.get("BROWSER_AGENT_DISABLE_SNAPSHOT_ELISION", "").lower() in ("1", "true", "yes"):
            cfg.elide_old_snapshots = False
        raw_elide_keep = os.environ.get("BROWSER_AGENT_ELIDE_KEEP_RECENT")
        if raw_elide_keep:
            try:
                cfg.elide_keep_recent = int(raw_elide_keep)
            except ValueError:
                pass

        raw_ctx_window = os.environ.get("BROWSER_AGENT_CONTEXT_WINDOW")
        if raw_ctx_window:
            try:
                cfg.context_window_override = int(raw_ctx_window)
            except ValueError:
                pass
        # Compact overrides
        raw_compact_ratio = os.environ.get("BROWSER_AGENT_COMPACT_TRIGGER_RATIO")
        if raw_compact_ratio:
            try:
                cfg.compact_trigger_ratio = float(raw_compact_ratio)
            except ValueError:
                pass
        raw_compact_keep = os.environ.get("BROWSER_AGENT_COMPACT_KEEP_CYCLES")
        if raw_compact_keep:
            try:
                cfg.compact_keep_recent_cycles = int(raw_compact_keep)
            except ValueError:
                pass
        raw_tier1_keep = os.environ.get("BROWSER_AGENT_COMPACT_TIER1_KEEP_TOOL_RESULTS")
        if raw_tier1_keep:
            try:
                cfg.compact_tier1_keep_recent_tool_results = int(raw_tier1_keep)
            except ValueError:
                pass
        # Budget overrides
        for env_key, attr in (
            ("BROWSER_AGENT_MAX_INNER_STEPS", "max_inner_steps"),
            ("BROWSER_AGENT_MAX_TAPS", "max_taps"),
            ("BROWSER_AGENT_SKILL_MAX_CHARS", "skill_max_chars"),
            ("BROWSER_AGENT_SHELL_TIMEOUT", "shell_timeout_seconds"),
            ("BROWSER_AGENT_SHELL_MAX_OUTPUT_CHARS", "shell_max_output_chars"),
        ):
            raw = os.environ.get(env_key)
            if raw:
                try:
                    setattr(cfg, attr, int(raw))
                except ValueError:
                    pass
        return cfg


def _coerce_paths(value) -> list[Path]:
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        return [Path(value).expanduser()]
    if isinstance(value, list):
        return [Path(v).expanduser() for v in value]
    if isinstance(value, tuple):
        return [Path(v).expanduser() for v in value]
    return []


def _coerce_set(value) -> set[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return {p.strip() for p in value.split(",") if p.strip()}
    if isinstance(value, (list, tuple, set)):
        return {str(p).strip() for p in value if str(p).strip()}
    return None
