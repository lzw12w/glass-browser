"""CLI entry — interactive REPL + one-shot subcommands.

Usage:
    ba chat                # start REPL (launches a headed Chromium)
    ba chat -m "..."       # one-shot
    ba chat --cdp http://127.0.0.1:9222   # attach to a running Chrome
    ba tools               # list available tools
    ba sessions            # list resumable runs
    ba skills list         # list local skills
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .actions import list_tool_schemas
from .agent import Agent, AgentConfig
from .agent.compact import CompactConfig
from .browser import (
    BrowserDriver, BrowserSession, list_resumable_runs, make_run_workdir,
    resolve_resume_workdir,
)
from .config import Config
from .errors import BrowserAgentError
from .llm import make_llm, read_note_body, resolve_note_path
from .skills import SkillManager
from .trace import Recorder


def _confirm_fn(console: Console):
    """Block on stdin until the user types y / n. Used for high-risk tools."""
    def _ask(tc):
        console.print(
            Panel(
                f"[yellow]Confirm action[/yellow]: [bold]{tc.name}[/bold]\n"
                f"args: {json.dumps(tc.arguments, ensure_ascii=False, indent=2)}",
                border_style="yellow",
            )
        )
        try:
            answer = input("allow? [y/N] ").strip().lower()
        except EOFError:
            return False
        return answer in ("y", "yes")
    return _ask


def _make_llm_for_cfg(cfg: Config):
    """Build an LLMClient using the right kwargs for the chosen provider.

    Each provider takes a different set of kwargs (Anthropic uses
    ``api_key``/``base_url``; OpenAI uses Responses-API-specific reasoning
    knobs). Centralizing this here keeps future entry points in sync so a
    provider switch never accidentally leaves credentials behind.
    """
    if cfg.llm_provider == "anthropic":
        return make_llm(
            "anthropic",
            model=cfg.llm_model,
            api_key=cfg.anthropic_api_key,
            base_url=cfg.anthropic_base_url,
            context_window=cfg.context_window_override,
        )
    if cfg.llm_provider == "openai":
        return make_llm(
            "openai",
            model=cfg.llm_model,
            api_key=cfg.openai_api_key,
            base_url=cfg.openai_base_url,
            max_output_tokens=cfg.openai_max_output_tokens,
            reasoning_effort=cfg.openai_reasoning_effort,
            reasoning_summary=cfg.openai_reasoning_summary,
            context_window=cfg.context_window_override,
        )
    if cfg.llm_provider == "scripted":
        return make_llm("scripted", script=[])
    raise ValueError(f"unknown llm_provider: {cfg.llm_provider}")


def _resume_messages(args, cfg: Config, console: Console) -> tuple[Path, list[dict], str] | None:
    """Resolve ``--resume`` (if present); return (workdir, messages, provider) or None."""
    spec = getattr(args, "resume", None)
    if spec is None:
        return None
    try:
        workdir = resolve_resume_workdir(
            None if spec is True else spec,
            root=cfg.workdir_root,
        )
    except FileNotFoundError as e:
        console.print(f"[red]resume failed:[/red] {e}")
        return None
    try:
        provider, messages = Recorder.load_messages(workdir)
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]resume failed:[/red] {e}")
        return None
    if provider != cfg.llm_provider:
        console.print(
            f"[red]resume failed:[/red] saved provider {provider!r} != "
            f"current provider {cfg.llm_provider!r}; "
            f"set BROWSER_AGENT_LLM_PROVIDER={provider} to continue."
        )
        return None
    console.print(
        f"[green]resumed[/green] from {workdir} ({len(messages)} messages, provider={provider})"
    )
    return workdir, messages, provider


def cmd_tools(args, cfg, console):
    schemas = list_tool_schemas(include_shell=cfg.enable_shell)
    table = Table(title="Available agent tools")
    table.add_column("name", style="cyan")
    table.add_column("description")
    for s in schemas:
        table.add_row(s["name"], (s["description"] or "")[:80])
    console.print(table)
    return 0


def cmd_sessions(args, cfg, console):
    runs = list_resumable_runs(cfg.workdir_root)
    if not runs:
        console.print("[dim]no resumable runs[/dim]")
        return 0
    import datetime as _dt
    table = Table(title="Resumable runs (newest first)")
    table.add_column("name", style="cyan")
    table.add_column("messages", justify="right")
    table.add_column("modified")
    for run in runs[:20]:
        try:
            _, msgs = Recorder.load_messages(run)
            count = str(len(msgs))
        except Exception:
            count = "?"
        mtime = _dt.datetime.fromtimestamp(run.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        table.add_row(run.name, count, mtime)
    console.print(table)
    return 0


def cmd_skills(args, cfg, console):
    manager = SkillManager.from_config(cfg)
    if getattr(args, "skills_cmd", None) == "show":
        result = manager.view_skill(args.name)
        console.print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("ok") else 1
    for meta in manager.metadata_index():
        console.print(f"[cyan]{meta['name']}[/cyan] — {(meta.get('description') or '')[:80]}")
    return 0


def cmd_chat(args, cfg, console):
    if getattr(args, "allow_shell", False):
        cfg.enable_shell = True
    if getattr(args, "headless", False):
        cfg.browser_headless = True
    if getattr(args, "cdp", None):
        cfg.browser_cdp_url = args.cdp
    if cfg.llm_provider == "anthropic" and not cfg.anthropic_api_key:
        console.print("[red]ANTHROPIC_API_KEY not set.[/red] export it or use --provider scripted.")
        return 2
    if cfg.llm_provider == "openai" and not cfg.openai_api_key:
        console.print("[red]OPENAI_API_KEY not set.[/red] export it or switch provider.")
        return 2

    # Resolve --resume up front so failures abort before we launch a browser.
    resume_data = _resume_messages(args, cfg, console)
    workdir = resume_data[0] if resume_data else make_run_workdir(cfg.workdir_root)
    console.print(f"[dim]workdir: {workdir}[/dim]")

    llm = _make_llm_for_cfg(cfg)

    driver = BrowserDriver()
    try:
        if cfg.browser_cdp_url:
            driver.connect_over_cdp(cfg.browser_cdp_url)
            console.print(f"[dim]attached over CDP: {cfg.browser_cdp_url}[/dim]")
        else:
            driver.launch(headless=cfg.browser_headless)
            console.print(f"[dim]chromium launched (headless={cfg.browser_headless})[/dim]")
    except BrowserAgentError as e:
        console.print(f"[red]{e}[/red]")
        return 2

    try:
        with Recorder(workdir) as recorder:
            session = BrowserSession(driver, workdir=workdir)
            session._note_path = resolve_note_path()
            # Snapshot NOTE.md ONCE at session start. Writes made during this
            # session update the file on disk but will NOT be picked up until
            # the next session boot — this is intentional so the agent does
            # not appear to "already know" a rule it just wrote this turn.
            session._note_body = read_note_body(session._note_path)
            agent = Agent(
                llm=llm, session=session, recorder=recorder,
                config=AgentConfig(
                    confirm_for=cfg.confirm_for,
                    max_inner_steps=cfg.max_inner_steps,
                    max_taps=cfg.max_taps,
                    forced_skills=tuple(getattr(args, "skill", None) or ()),
                    enable_shell=cfg.enable_shell,
                    shell_workspace_roots=tuple(cfg.shell_workspace_roots),
                    shell_timeout_seconds=cfg.shell_timeout_seconds,
                    shell_max_output_chars=cfg.shell_max_output_chars,
                    elide_old_snapshots=cfg.elide_old_snapshots,
                    elide_keep_recent=cfg.elide_keep_recent,
                    compact_config=CompactConfig(
                        trigger_ratio=cfg.compact_trigger_ratio,
                        keep_recent_cycles=cfg.compact_keep_recent_cycles,
                        tier1_keep_recent_tool_results=cfg.compact_tier1_keep_recent_tool_results,
                    ),
                ),
                skill_manager=SkillManager.from_config(
                    cfg, forced_enabled=getattr(args, "skill", None) or ()),
                confirm_fn=_confirm_fn(console) if not args.yes else None,
                console=console,
            )

            if resume_data is not None:
                try:
                    agent.load_messages(resume_data[1], provider_id=resume_data[2])
                except ValueError as e:
                    console.print(f"[red]resume failed after agent init:[/red] {e}")
                    return 2

            if args.message:
                agent.chat(args.message)
                return 0

            return _repl(agent, cfg, workdir, recorder, console)
    finally:
        driver.close()


def _repl(agent: Agent, cfg: Config, workdir: Path, recorder: Recorder,
          console: Console) -> int:
    skill_slash_help = ""
    if agent.skill_manager is not None:
        slash_map = agent.skill_manager.slash_command_map()
        if slash_map:
            names = ", ".join(sorted(slash_map.keys()))
            skill_slash_help = f"\nskill slash commands: {names}"

    console.print(Panel.fit(
        "[bold]Browser Agent[/bold]\n"
        "type your request; commands: /reset /trace /workdir /skills /load /sessions /quit"
        + skill_slash_help,
        border_style="cyan",
    ))
    while True:
        try:
            user_text = console.input("[bold cyan]you »[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        if not user_text:
            continue
        if user_text in ("/quit", "/exit", ":q"):
            break
        if user_text == "/reset":
            agent.reset()
            console.print("[dim]conversation reset[/dim]")
            continue
        if user_text == "/trace":
            console.print(f"[dim]trace: {recorder.path}[/dim]")
            console.print(f"[dim]llm context: {recorder.llm_context_path}[/dim]")
            continue
        if user_text == "/workdir":
            console.print(f"[dim]workdir: {workdir}[/dim]")
            continue
        if user_text == "/skills":
            if agent.skill_manager is None:
                console.print("[dim]no skill manager attached[/dim]")
                continue
            for sk in agent.skill_manager.list_skills():
                console.print(f"[cyan]{sk.name}[/cyan] — {(sk.description or '')[:80]}")
            continue
        if user_text == "/help":
            console.print(
                "/reset  /trace  /workdir  /skills  /sessions  /load [run]  "
                "/<skill-name>  /quit"
            )
            continue
        if user_text == "/sessions":
            cmd_sessions(None, cfg, console)
            continue
        if user_text == "/load" or user_text.startswith("/load "):
            _, _, spec = user_text.partition(" ")
            spec = spec.strip() or None
            try:
                target = resolve_resume_workdir(spec, root=cfg.workdir_root)
                provider, messages = Recorder.load_messages(target)
            except (FileNotFoundError, ValueError) as e:
                console.print(f"[red]load failed:[/red] {e}")
                continue
            try:
                agent.load_messages(messages, provider_id=provider)
            except ValueError as e:
                console.print(f"[red]load failed:[/red] {e}")
                continue
            console.print(
                f"[green]loaded[/green] {len(messages)} messages from {target}; "
                f"new turns will be persisted in the current workdir ({workdir})"
            )
            continue

        # Skill activation: ``/<slug> [extra instruction]`` injects the
        # SKILL.md body as a USER message before the next turn.
        if user_text.startswith("/") and agent.skill_manager is not None:
            head, _, rest = user_text.partition(" ")
            slash_map = agent.skill_manager.slash_command_map()
            if head in slash_map:
                skill = slash_map[head]
                instruction = rest.strip()
                agent.inject_skill_message(skill.name, user_instruction=instruction)
                follow_up = instruction or (
                    f"Please follow the {skill.name} skill above and "
                    f"continue from the current state."
                )
                console.print(f"[dim]activated skill: {skill.name}[/dim]")
                try:
                    agent.chat(follow_up)
                except KeyboardInterrupt:
                    console.print("[yellow](interrupted)[/yellow]")
                except Exception as e:
                    console.print(f"[red]agent error:[/red] {e}")
                continue

        try:
            agent.chat(user_text)
        except KeyboardInterrupt:
            console.print("[yellow](interrupted)[/yellow]")
            continue
        except Exception as e:
            console.print(f"[red]agent error:[/red] {e}")
            continue
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ba")
    sub = parser.add_subparsers(dest="cmd")

    pc = sub.add_parser("chat", help="start interactive agent (or -m for one-shot)")
    pc.add_argument("-m", "--message", help="one-shot message instead of REPL")
    pc.add_argument("--provider", help="override llm_provider (anthropic|openai|scripted)")
    pc.add_argument("--model", help="override llm_model")
    pc.add_argument("--resume", nargs="?", const=True, default=None,
                    help="resume a previous run (latest, a run name, or a path)")
    pc.add_argument("--headless", action="store_true", help="run the browser headless")
    pc.add_argument("--cdp", help="attach to a running Chrome over CDP (e.g. http://127.0.0.1:9222)")
    pc.add_argument("--allow-shell", action="store_true", help="enable the run_shell tool")
    pc.add_argument("--skill", action="append",
                    help="force-activate a skill at session start (repeatable)")
    pc.add_argument("-y", "--yes", action="store_true",
                    help="skip confirmation prompts for high-risk tools")

    sub.add_parser("tools", help="list available tools")
    sub.add_parser("sessions", help="list resumable runs (chat --resume targets)")
    psks = sub.add_parser("skills", help="list or inspect local skills")
    psksub = psks.add_subparsers(dest="skills_cmd")
    psksub.add_parser("list", help="list discovered skills")
    pshow = psksub.add_parser("show", help="show one skill")
    pshow.add_argument("name")

    args = parser.parse_args(argv)
    console = Console()
    cfg = Config.load()
    if getattr(args, "provider", None):
        cfg.llm_provider = args.provider
    if getattr(args, "model", None):
        cfg.llm_model = args.model

    if args.cmd == "chat":
        return cmd_chat(args, cfg, console)
    if args.cmd == "tools":
        return cmd_tools(args, cfg, console)
    if args.cmd == "sessions":
        return cmd_sessions(args, cfg, console)
    if args.cmd == "skills":
        return cmd_skills(args, cfg, console)
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
