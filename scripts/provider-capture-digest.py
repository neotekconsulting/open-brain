#!/usr/bin/env python3
"""Distill chat-provider exports into Open Brain captures.

Supported providers: ChatGPT, Claude, Gemini, Ollama (local).
Approach: fetch/export → summarize with local Ollama → capture via ingestion API.
Token-cheap: each summary is ~50-150 tokens.
Self-hosted/offline-first: no cloud APIs, no keys needed for local providers.
Config-driven via environment variables.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import List, Optional


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

INGESTION_URL: str = os.environ.get("INGESTION_URL", "http://localhost:8080/capture")
OLLAMA_URL: str = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.environ.get("OLLAMA_MODEL", "qwen3.5:9b")
SUMMARIZE: bool = os.environ.get("SUMMARIZE", "1") not in ("0", "false", "no")
DRY_RUN: bool = os.environ.get("DRY_RUN", "0") in ("1", "true", "yes")
MAX_CONVERSATIONS: int = int(os.environ.get("MAX_CONVERSATIONS", "20"))
MAX_MESSAGE_CHARS: int = int(os.environ.get("MAX_MESSAGE_CHARS", "4000"))

# Provider export paths / endpoints (override via env)
CHATGPT_EXPORT_DIR: Optional[str] = os.environ.get("CHATGPT_EXPORT_DIR")
CLAUDE_PROJECTS_DIR: Optional[str] = os.environ.get("CLAUDE_PROJECTS_DIR")
GEMINI_EXPORT_DIR: Optional[str] = os.environ.get("GEMINI_EXPORT_DIR")
OLLAMA_HISTORY_PATH: Optional[str] = os.environ.get("OLLAMA_HISTORY_PATH")


# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------

@dataclass
class ChatMessage:
    role: str
    content: str

@dataclass
class Conversation:
    provider: str
    title: str
    messages: List[ChatMessage]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _truncate(text: str, limit: int = MAX_MESSAGE_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "...[truncated]"


# ---------------------------------------------------------------------------
# Provider loaders
# ---------------------------------------------------------------------------

def load_chatgpt_exports(base_dir: str, limit: int = MAX_CONVERSATIONS) -> List[Conversation]:
    if not base_dir or not os.path.isdir(base_dir):
        return []
    out: List[Conversation] = []
    for name in sorted(os.listdir(base_dir))[:limit]:
        path = os.path.join(base_dir, name)
        if not os.path.isfile(path):
            continue
        try:
            data = _load_json(path)
        except Exception:
            continue
        messages = []
        for msg in data.get("messages", data.get("mapping", {}).values()):
            if isinstance(msg, dict):
                content = msg.get("content", "")
                if isinstance(content, dict):
                    content = content.get("parts", [""])[0]
                role = msg.get("role", "user")
                if content:
                    messages.append(ChatMessage(role=role, content=_truncate(str(content))))
        if messages:
            out.append(Conversation(provider="chatgpt", title=name, messages=messages))
    return out


def load_claude_projects(base_dir: str, limit: int = MAX_CONVERSATIONS) -> List[Conversation]:
    if not base_dir or not os.path.isdir(base_dir):
        return []
    out: List[Conversation] = []
    for proj in sorted(os.listdir(base_dir))[:limit]:
        proj_dir = os.path.join(base_dir, proj)
        chat_path = os.path.join(proj_dir, "chat.json")
        if not os.path.isfile(chat_path):
            continue
        try:
            data = _load_json(chat_path)
        except Exception:
            continue
        messages = []
        for msg in data.get("messages", []):
            content = msg.get("content", "")
            role = msg.get("role", "user")
            if content:
                messages.append(ChatMessage(role=role, content=_truncate(str(content))))
        if messages:
            out.append(Conversation(provider="claude", title=proj, messages=messages))
    return out


def load_gemini_exports(base_dir: str, limit: int = MAX_CONVERSATIONS) -> List[Conversation]:
    if not base_dir or not os.path.isdir(base_dir):
        return []
    out: List[Conversation] = []
    for name in sorted(os.listdir(base_dir))[:limit]:
        path = os.path.join(base_dir, name)
        if not os.path.isfile(path) or not name.endswith(".json"):
            continue
        try:
            data = _load_json(path)
        except Exception:
            continue
        messages: List[ChatMessage] = []
        for msg in data.get("messages", []):
            content = msg.get("content", "")
            role = msg.get("author", {}).get("role", "user")
            if content:
                messages.append(ChatMessage(role=role, content=_truncate(str(content))))
        if messages:
            out.append(Conversation(provider="gemini", title=name, messages=messages))
    return out


def load_ollama_history(path: Optional[str] = None, limit: int = MAX_CONVERSATIONS) -> List[Conversation]:
    if not path:
        path = os.path.expanduser("~/.ollama/history.json")
    if not path or not os.path.isfile(path):
        return []
    try:
        data = _load_json(path)
    except Exception:
        return []
    out: List[Conversation] = []
    for convo in data.get("history", [])[:limit]:
        messages = []
        for msg in convo.get("messages", []):
            content = msg.get("content", "")
            role = msg.get("role", "user")
            if content:
                messages.append(ChatMessage(role=role, content=_truncate(str(content))))
        if messages:
            title = convo.get("title", convo.get("id", "ollama-chat"))
            out.append(Conversation(provider="ollama", title=title, messages=messages))
    return out


# ---------------------------------------------------------------------------
# Summarization (local Ollama)
# ---------------------------------------------------------------------------

def _call_ollama_chat(prompt: str) -> str:
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_URL.rstrip('/')}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return (data.get("message") or {}).get("content", "") or ""
    except Exception as exc:
        print(f"[warn] Ollama summarize failed: {exc}", file=sys.stderr)
        return ""


def summarize_conversation(convo: Conversation) -> str:
    prompt = (
        "Extract only high-signal facts about the user from this conversation: "
        "decisions, preferences, action items, commitments, personal facts. "
        "Bullet style, short. If none, say 'no_signals'.\n\n"
    )
    for msg in convo.messages:
        prompt += f"- {msg.role}: {msg.content}\n"
    raw = _call_ollama_chat(prompt).strip()
    if not raw or "no_signals" in raw.lower():
        return ""
    return raw


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------

def capture_to_brain(text: str, *, provider: str, title: str) -> None:
    if DRY_RUN:
        print(f"[dry-run] capture ({provider}) {title!r}: {text[:120]}...")
        return
    payload = json.dumps({
        "raw_text": f"[{provider}] {title}\n\n{text}",
        "type": "provider_digest",
        "source": f"{provider}:export",
        "topics": [provider, "auto-capture"],
        "people": ["Jeff"],
    }).encode("utf-8")
    req = urllib.request.Request(
        INGESTION_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            _ = resp.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"[error] capture failed {exc.code}: {body}", file=sys.stderr)
    except Exception as exc:
        print(f"[error] capture failed: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    providers = [
        ("ChatGPT", CHATGPT_EXPORT_DIR, load_chatgpt_exports),
        ("Claude", CLAUDE_PROJECTS_DIR, load_claude_projects),
        ("Gemini", GEMINI_EXPORT_DIR, load_gemini_exports),
        ("Ollama", OLLAMA_HISTORY_PATH, load_ollama_history),
    ]

    total_captured = 0
    for provider_name, source, loader in providers:
        if not source:
            print(f"[skip] {provider_name} source not configured")
            continue
        print(f"[info] loading {provider_name} from {source}")
        convos = loader(source)
        print(f"[info] {provider_name}: {len(convos)} conversations")
        for convo in convos:
            summary = summarize_conversation(convo) if SUMMARIZE else "\n".join(
                f"- {m.role}: {m.content}" for m in convo.messages
            )
            if not summary:
                continue
            capture_to_brain(summary, provider=provider_name.lower(), title=convo.title)
            total_captured += 1
            time.sleep(0.2)  # gentle rate limit on local ingestion API
    print(f"[done] captured {total_captured} digests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
