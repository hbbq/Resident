"""Generate the implementation-wide operator catalog without constructing resources."""
from __future__ import annotations

import ast
import argparse
from pathlib import Path

from .instances import _SUBSCRIPTION_EVENTS, _SUBSCRIPTION_SELECTORS
from .tools import CORE_TOOL_NAMES


ROOT = Path(__file__).resolve().parent.parent

# Setup metadata is kept here; identifiers come from the executable definitions.
# A new static tool, output, or event fails inventory validation until its family
# has a setup description. No instance definitions or secrets are read.
SETUP = {
    "diagnostics": ("Local runtime; no external service.", "Read-only UTC clock."),
    "homeops": ("Set RESIDENT_HOMEOPS_URL to a reachable HomeOps HTTP service.",
                "Read HomeOps measurements and weather forecasts."),
    "agentcontroller": ("Set RESIDENT_AGENTCONTROLLER_SNAPSHOT_PATH to AgentController's dashboard.json.",
                        "Read the versioned workflow snapshot."),
    "camera": ("Configure RESIDENT_CAMERAS with camera IDs and RTSP URLs; frame capture also needs FFmpeg.",
               "List cameras or capture one frame."),
    "realm": ("Add a realm block to the Resident definition with base_url, game_id_env, and actor_id_env; run a compatible Realm service.",
              "Read or mutate the configured Realm game."),
    "messaging": ("Host multiple Residents with a shared durable mailbox; recipients are configured Resident IDs.",
                  "Send an asynchronous message to another Resident."),
    "display": ("Set RESIDENT_DISPLAYS with unique display IDs and RESIDENT_HOMEOPS_URL; use a configured HomeOps display queue.",
                "Enqueue text on a configured display (legacy tool protocol)."),
}

OUTPUT_SETUP = {
    "notify_owner": "Grant outputs: [notify_owner]; use the default terminal Resident or configure an Owner Telegram transport with token_env, owner_user_id_env, and owner_chat_id_env. Requires an available Owner route.",
    "display": "Grant outputs: [display/<display-id>]; configure RESIDENT_DISPLAYS and RESIDENT_HOMEOPS_URL. Each target must exist and have a HomeOps display route.",
}

EVENT_SETUP = {
    "agentcontroller": "Set RESIDENT_AGENTCONTROLLER_SNAPSHOT_PATH; changes follow the first valid snapshot baseline.",
    "camera": "Configure RESIDENT_CAMERAS. cameras_changed requires an explicit camera-set replacement; onvif_property_changed additionally requires per-camera onvif settings and a working ONVIF Event Service.",
    "homeops": "Set RESIDENT_HOMEOPS_URL; changes follow the initial measurements baseline.",
    "messaging": "Use a multi-Resident host and grant the sender messaging or messaging_send; the recipient subscribes to mailbox delivery.",
}


def _tree(name: str) -> ast.Module:
    return ast.parse((ROOT / "resident" / name).read_text(encoding="utf-8"))


def _literal(node: ast.AST) -> str | None:
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError):
        return None
    return value if isinstance(value, str) else None


def _keyword(call: ast.Call, key: str) -> ast.AST | None:
    return next((item.value for item in call.keywords if item.arg == key), None)


def _inventory() -> tuple[dict[str, set[str]], set[str], set[str], set[str]]:
    tools: dict[str, set[str]] = {}
    outputs: set[str] = set()
    core: set[str] = set()
    emitted: set[str] = set()
    files = tuple((ROOT / "resident").glob("*.py"))
    for file in files:
        tree = _tree(file.name)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            if node.func.id == "Capability":
                connector = _literal(_keyword(node, "connector_id") or node.args[0]) if (node.args or _keyword(node, "connector_id")) else None
                name_node = _keyword(node, "name") or (node.args[2] if len(node.args) > 2 else None)
                name = _literal(name_node) if name_node is not None else None
                if connector and name:
                    tools.setdefault(connector, set()).add(name)
                elif file.name not in {"display.py", "external_app.py", "realm.py"}:
                    raise ValueError(f"Uncataloged dynamic capability in {file.name}:{node.lineno}")
            elif node.func.id == "OutputCapability":
                output = _keyword(node, "output_type")
                name = _literal(output) if output else None
                if name is None:
                    raise ValueError(f"Uncataloged output in {file.name}:{node.lineno}")
                outputs.add(name)
            elif node.func.id == "ToolSpec" and file.name == "tools.py":
                name = _literal(node.args[0]) if node.args else None
                if name:
                    core.add(name)
            elif node.func.id == "WakeEvent":
                source_node = _keyword(node, "source") or (node.args[1] if len(node.args) > 1 else None)
                reason_node = _keyword(node, "reason") or (node.args[2] if len(node.args) > 2 else None)
                source = _literal(source_node) if source_node else None
                reason = _literal(reason_node) if reason_node else None
                if source in EVENT_SETUP:
                    if reason is None:
                        raise ValueError(f"Uncataloged event reason in {file.name}:{node.lineno}")
                    emitted.add(f"{source}.{reason}")
    # Realm builds capabilities from its local specs table, not one call per tool.
    realm_tree = _tree("realm.py")
    for node in ast.walk(realm_tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "specs" for t in node.targets):
            if isinstance(node.value, ast.List):
                for item in node.value.elts:
                    if isinstance(item, ast.Tuple) and item.elts:
                        name = _literal(item.elts[0])
                        if name:
                            tools.setdefault("realm", set()).add(name)
    # These are parameterized by a Resident definition or startup configuration.
    tools.setdefault("display", set()).add("<display-id>_show_text")
    tools.setdefault("external_application", set()).add("<provider-id>_<operation-name>")
    return tools, outputs, core, emitted


def render() -> str:
    tools, outputs, core, emitted = _inventory()
    if set(tools) != set(SETUP) | {"external_application"}:
        raise ValueError(f"Capability family metadata mismatch: {set(tools) ^ (set(SETUP) | {'external_application'})}")
    if outputs != set(OUTPUT_SETUP):
        raise ValueError(f"Output metadata mismatch: {outputs ^ set(OUTPUT_SETUP)}")
    if core != set(CORE_TOOL_NAMES):
        raise ValueError(f"Core tool inventory mismatch: {core ^ set(CORE_TOOL_NAMES)}")
    if {item.split(".", 1)[0] for item in _SUBSCRIPTION_EVENTS} != set(EVENT_SETUP):
        raise ValueError("Subscription source metadata mismatch")
    if emitted != _SUBSCRIPTION_EVENTS:
        raise ValueError(f"Emitted event inventory mismatch: {emitted ^ _SUBSCRIPTION_EVENTS}")
    if _SUBSCRIPTION_SELECTORS != (_SUBSCRIPTION_EVENTS |
                                  {item.split(".", 1)[0] for item in _SUBSCRIPTION_EVENTS} | {"*"}):
        raise ValueError("Subscription selector inventory mismatch")

    lines = ["# Available Resident capabilities, outputs, and subscriptions", "",
             "Generated by `python -m resident.catalog`. This describes choices supported by the current implementation across configurations, not grants on any particular Resident. Edit source definitions or setup metadata, regenerate, then run `python -m resident.catalog --check`.", "",
             "A Resident definition grants connector IDs or individual tool names in `capabilities`, grants terminal side effects separately in `outputs`, and selects external events in `subscriptions`. Configuration makes a choice available; a grant authorizes its use. The default terminal Resident has an Owner route; other Residents need an Owner transport for `notify_owner`.", "",
             "## Grantable capabilities", "",
             "| Grant identifier | Available tools | Configuration and dependency | Purpose |", "| --- | --- | --- | --- |"]
    for family in sorted(set(tools) - {"display", "external_application"}):
        setup, purpose = SETUP[family]
        grant = f"`{family}`" if family == "messaging" else f"`{family}` or individual tool name"
        lines.append(f"| {grant} | {', '.join(f'`{name}`' for name in sorted(tools[family]))} | {setup} | {purpose} |")
    lines.extend(["| `<provider-id>` or `<provider-id>_<operation-name>` | `<provider-id>_<operation-name>` | Add `external_applications` to the Resident definition: provider `id`, `description`, `base_url`, optional `bearer_token_env` and `bindings`, and pinned `operations` with names, descriptions, input schemas, and optional mutating flags. Run the matching HTTP service. | Only operations declared on that Resident are available; no remote discovery occurs. |", "",
                  "## Always-present built-in tools", "",
                  "These are runtime tools, not `capabilities` grants. `send_owner_message` is a compatibility tool and is described below.", "",
                  "| Tool | Availability |", "| --- | --- |"])
    for name in sorted(core - {"send_owner_message"}):
        note = ("Requires an authenticated Owner-message context to change guidance." if name in {"set_owner_guidance", "remove_owner_guidance"} else
                "Local runtime state; no extra connector configuration.")
        lines.append(f"| `{name}` | {note} |")
    lines.extend(["", "## Terminal outputs", "", "| Output grant | Configuration and dependency |", "| --- | --- |"])
    for name in sorted(outputs):
        grant = "display/<display-id>" if name == "display" else name
        lines.append(f"| `{grant}` | {OUTPUT_SETUP[name]} |")
    lines.extend(["", "## Non-grantable compatibility tools", "",
                  "These tools serve the Responses fallback or already-active old Managed Agents sessions. New structured-output sessions use terminal outputs.", "",
                  "| Tool | Configuration and dependency |", "| --- | --- |",
                  "| `send_owner_message` | Owner route and `notify_owner` output authorization; not a `capabilities` grant. |",
                  f"| `{next(iter(tools['display']))}` | {SETUP['display'][0]} Use the matching `display/<display-id>` output grant; not an independent capability grant. |", "",
                  "## Subscriptions", "",
                  "Selectors in a Resident definition observe events; they never grant tools. `*` selects every accepted external source event. A source selector selects all accepted reasons from that source. Internal Owner, scheduler, runtime, and output delivery wakes are separate from these declarative selectors.", "",
                  "| Selector | Event source and prerequisite |", "| --- | --- |"])
    for event in sorted(_SUBSCRIPTION_EVENTS):
        source = event.split(".", 1)[0]
        lines.append(f"| `{event}` | {EVENT_SETUP[source]} |")
    for source in sorted(EVENT_SETUP):
        lines.append(f"| `{source}` | All accepted `{source}.*` events; {EVENT_SETUP[source]} |")
    lines.append("| `*` | All accepted external events above; configure the corresponding producers to receive them. |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail when CAPABILITIES.md is stale")
    args = parser.parse_args()
    target = ROOT / "CAPABILITIES.md"
    content = render()
    if args.check:
        if not target.exists() or target.read_text(encoding="utf-8") != content:
            print("CAPABILITIES.md is stale; run python -m resident.catalog")
            return 1
        print("CAPABILITIES.md is current")
    else:
        target.write_text(content, encoding="utf-8", newline="\n")
        print("Wrote CAPABILITIES.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
