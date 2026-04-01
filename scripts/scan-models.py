#!/usr/bin/env python3
"""
Daily model scanner: fetches provider docs/pricing pages, uses Portkey (LLM) to
extract new models and pricing changes, then merges them into pricing/ and general/.

Usage:
    python scripts/scan-models.py [--output /tmp/scan-report.md] [--provider openai]
"""

import argparse
import copy
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from portkey_ai import Portkey

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
PRICING_DIR = REPO_ROOT / "pricing"
GENERAL_DIR = REPO_ROOT / "general"
CONFIG_FILE = Path(__file__).resolve().parent / "providers-scan-config.json"

# ---------------------------------------------------------------------------
# LLM system prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a precise data extraction assistant. Your only job is to extract
AI model information from provider documentation or pricing pages.

Return a single valid JSON object — no markdown fences, no prose, just JSON — with this shape:
{
  "models": [
    {
      "id": "<exact API model ID as used in API calls>",
      "type": "<one of: chat | text | embedding | image | audio | video | rerank | moderation>",
      "supported": ["<zero or more of: tools, image, json_mode, streaming, cache_control>"],
      "context_window": <integer or null>,
      "input_price_per_million_tokens": <float or null>,
      "output_price_per_million_tokens": <float or null>
    }
  ]
}

Type classification rules:
- "chat"       → model works with /v1/chat/completions (instruction-following, dialogue)
- "text"       → model works with legacy /v1/completions (base/completion models)
- "embedding"  → model produces vector embeddings, used with /v1/embeddings
- "image"      → model generates images (not a vision input capability — that's in supported)
- "audio"      → model does speech-to-text, text-to-speech, or audio understanding
- "video"      → model generates or understands video
- "rerank"     → model reranks a list of documents given a query
- "moderation" → model classifies content for safety/policy

Supported capability rules:
- "tools"         → model supports function calling / tool use
- "image"         → model can accept image inputs (multimodal vision)
- "json_mode"     → model supports structured JSON output mode
- "streaming"     → model supports streaming responses
- "cache_control" → model supports prompt caching

Important:
- Use the canonical API model ID (e.g. "claude-3-5-sonnet-20241022", not "Claude 3.5 Sonnet").
- If price is not listed on the page, set it to null — do not guess.
- Omit deprecated or legacy models that are no longer available via API.
- Do not include models from other providers."""

USER_PROMPT_TEMPLATE = """Provider: {provider}

Below is the text extracted from the provider's documentation and/or pricing page.
Extract all current models and return them as the JSON structure described.

---
{page_text}
---"""

# ---------------------------------------------------------------------------
# HTTP fetching
# ---------------------------------------------------------------------------
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; Portkey-ModelScanner/1.0; "
        "+https://github.com/portkey-ai/models)"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}
FETCH_TIMEOUT = 30
FETCH_RETRIES = 3
FETCH_RETRY_DELAY = 5


def fetch_page_text(url: str) -> str | None:
    """Fetch a URL and return its visible text content (HTML stripped)."""
    for attempt in range(1, FETCH_RETRIES + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=FETCH_TIMEOUT)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            # Remove script / style noise
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
            # Collapse excessive blank lines
            lines = [ln for ln in text.splitlines() if ln.strip()]
            return "\n".join(lines)
        except Exception as exc:
            print(f"  [fetch] attempt {attempt}/{FETCH_RETRIES} failed for {url}: {exc}")
            if attempt < FETCH_RETRIES:
                time.sleep(FETCH_RETRY_DELAY)
    return None


def fetch_provider_text(sources: list[dict]) -> str:
    """Concatenate text from all sources for a provider, labelled by type."""
    parts = []
    for src in sources:
        print(f"  Fetching {src['type']}: {src['url']}")
        text = fetch_page_text(src["url"])
        if text:
            # Limit each source to ~8 000 chars to keep the prompt manageable
            parts.append(f"=== {src['type'].upper()} PAGE ===\n{text[:8000]}")
        else:
            print(f"  [warn] Could not fetch {src['url']}, skipping.")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# LLM call via Portkey
# ---------------------------------------------------------------------------
def call_llm(provider_name: str, page_text: str, client: Portkey) -> list[dict]:
    """Ask the LLM to extract models from page_text. Returns list of model dicts."""
    if not page_text.strip():
        print(f"  [warn] No page text for {provider_name}, skipping LLM call.")
        return []

    user_msg = USER_PROMPT_TEMPLATE.format(
        provider=provider_name,
        page_text=page_text,
    )

    try:
        response = client.chat.completions.create(
            model="",  # Portkey routes via virtual key; model field unused when virtual key is set
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0,
            max_tokens=4096,
        )
        raw = response.choices[0].message.content.strip()

        # Strip optional markdown fences the model may still emit
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        if raw.endswith("```"):
            raw = raw.rsplit("```", 1)[0]

        data = json.loads(raw.strip())
        return data.get("models", [])
    except json.JSONDecodeError as exc:
        print(f"  [error] LLM returned invalid JSON for {provider_name}: {exc}")
        return []
    except Exception as exc:
        print(f"  [error] LLM call failed for {provider_name}: {exc}")
        return []


# ---------------------------------------------------------------------------
# JSON merge helpers
# ---------------------------------------------------------------------------
def load_json(path: Path) -> dict:
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def save_json(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def price_per_token(price_per_million: float | None) -> float | None:
    if price_per_million is None:
        return None
    return round(price_per_million / 1_000_000, 12)


def build_pricing_entry(model: dict) -> dict:
    """Build a pricing JSON entry for a single model."""
    entry: dict = {"pricing_config": {"pay_as_you_go": {}}}
    pay = entry["pricing_config"]["pay_as_you_go"]

    inp = price_per_token(model.get("input_price_per_million_tokens"))
    out = price_per_token(model.get("output_price_per_million_tokens"))

    if inp is not None:
        pay["request_token"] = {"price": inp}
    else:
        pay["request_token"] = {"price": 0}

    if out is not None:
        pay["response_token"] = {"price": out}
    else:
        pay["response_token"] = {"price": 0}

    return entry


def build_general_entry(model: dict) -> dict:
    """Build a general JSON entry for a single model."""
    params = []
    if model.get("context_window"):
        params.append({"key": "max_tokens", "maxValue": model["context_window"]})

    supported = model.get("supported") or []
    entry: dict = {
        "params": params,
        "type": {
            "primary": model.get("type", "chat"),
            "supported": supported,
        },
    }
    # Clean up empty arrays to keep JSON tidy
    if not params:
        del entry["params"]
    if not supported:
        del entry["type"]["supported"]
    return entry


class ChangeTracker:
    def __init__(self, provider: str):
        self.provider = provider
        self.new_models: list[str] = []
        self.price_updates: list[dict] = []

    def record_new(self, model_id: str):
        self.new_models.append(model_id)

    def record_price_update(self, model_id: str, old_inp, new_inp, old_out, new_out):
        self.price_updates.append(
            {
                "model": model_id,
                "input": {"old": old_inp, "new": new_inp},
                "output": {"old": old_out, "new": new_out},
            }
        )

    def has_changes(self) -> bool:
        return bool(self.new_models or self.price_updates)


def merge_pricing(
    provider_key: str,
    models: list[dict],
    tracker: ChangeTracker,
) -> bool:
    """Merge extracted models into pricing/<provider>.json. Returns True if changed."""
    path = PRICING_DIR / f"{provider_key}.json"
    data = load_json(path)
    changed = False

    for model in models:
        mid = model["id"]
        new_entry = build_pricing_entry(model)
        new_inp = new_entry["pricing_config"]["pay_as_you_go"]["request_token"]["price"]
        new_out = new_entry["pricing_config"]["pay_as_you_go"]["response_token"]["price"]

        if mid not in data:
            # Brand-new model
            data[mid] = new_entry
            changed = True
            tracker.record_new(mid)
            print(f"  + new model: {mid}")
        else:
            # Check if price changed
            existing_pay = (
                data[mid].get("pricing_config", {}).get("pay_as_you_go", {})
            )
            old_inp = existing_pay.get("request_token", {}).get("price")
            old_out = existing_pay.get("response_token", {}).get("price")

            price_changed = (
                model.get("input_price_per_million_tokens") is not None
                and old_inp != new_inp
            ) or (
                model.get("output_price_per_million_tokens") is not None
                and old_out != new_out
            )

            if price_changed:
                # Preserve other fields (cache tokens etc.) while updating prices
                if "pricing_config" not in data[mid]:
                    data[mid]["pricing_config"] = {}
                if "pay_as_you_go" not in data[mid]["pricing_config"]:
                    data[mid]["pricing_config"]["pay_as_you_go"] = {}
                pay = data[mid]["pricing_config"]["pay_as_you_go"]
                pay["request_token"] = {"price": new_inp}
                pay["response_token"] = {"price": new_out}
                changed = True
                tracker.record_price_update(mid, old_inp, new_inp, old_out, new_out)
                print(f"  ~ price update: {mid} (in: {old_inp}→{new_inp}, out: {old_out}→{new_out})")

    if changed:
        save_json(path, data)
    return changed


def merge_general(
    provider_key: str,
    models: list[dict],
    tracker: ChangeTracker,
) -> bool:
    """Merge extracted models into general/<provider>.json. Returns True if changed."""
    path = GENERAL_DIR / f"{provider_key}.json"
    data = load_json(path)
    changed = False

    for model in models:
        mid = model["id"]
        if mid in data:
            continue  # Already exists; general capabilities are managed manually

        new_entry = build_general_entry(model)
        data[mid] = new_entry
        changed = True
        # New model already logged by pricing merge; no duplicate log needed

    if changed:
        save_json(path, data)
    return changed


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def generate_report(all_trackers: list[ChangeTracker], date_str: str) -> str:
    lines = [
        f"## Automated Model Scan — {date_str}",
        "",
        "This PR was opened automatically by the daily model scanner.",
        "Please review each change before merging.",
        "",
    ]

    has_any = any(t.has_changes() for t in all_trackers)
    if not has_any:
        lines.append("No changes detected.")
        return "\n".join(lines)

    for tracker in all_trackers:
        if not tracker.has_changes():
            continue
        lines.append(f"### {tracker.provider}")
        lines.append("")

        if tracker.new_models:
            lines.append(f"**New models ({len(tracker.new_models)}):**")
            for m in tracker.new_models:
                lines.append(f"- `{m}`")
            lines.append("")

        if tracker.price_updates:
            lines.append(f"**Price updates ({len(tracker.price_updates)}):**")
            lines.append("")
            lines.append("| Model | Input (old) | Input (new) | Output (old) | Output (new) |")
            lines.append("|-------|-------------|-------------|--------------|--------------|")
            for pu in tracker.price_updates:
                def fmt(v):
                    if v is None:
                        return "—"
                    return f"${v:.10f}".rstrip("0").rstrip(".")
                lines.append(
                    f"| `{pu['model']}` "
                    f"| {fmt(pu['input']['old'])} "
                    f"| {fmt(pu['input']['new'])} "
                    f"| {fmt(pu['output']['old'])} "
                    f"| {fmt(pu['output']['new'])} |"
                )
            lines.append("")

    lines += [
        "---",
        "_Generated by [daily-model-scan.yml](/.github/workflows/daily-model-scan.yml)_",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Scan provider pages for model/pricing updates.")
    parser.add_argument("--output", default=None, help="Write markdown report to this file.")
    parser.add_argument("--provider", default=None, help="Scan a single provider (by key).")
    args = parser.parse_args()

    portkey_api_key = os.environ.get("PORTKEY_API_KEY")
    portkey_virtual_key = os.environ.get("PORTKEY_VIRTUAL_KEY")

    if not portkey_api_key:
        print("ERROR: PORTKEY_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)
    if not portkey_virtual_key:
        print("ERROR: PORTKEY_VIRTUAL_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    client = Portkey(
        api_key=portkey_api_key,
        virtual_key=portkey_virtual_key,
    )

    config: dict = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))

    if args.provider:
        if args.provider not in config:
            print(f"ERROR: provider '{args.provider}' not found in providers-scan-config.json", file=sys.stderr)
            sys.exit(1)
        providers_to_scan = {args.provider: config[args.provider]}
    else:
        providers_to_scan = config

    date_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    all_trackers: list[ChangeTracker] = []

    for provider_key, provider_cfg in providers_to_scan.items():
        print(f"\n{'='*60}")
        print(f"  Scanning: {provider_key}")
        print(f"{'='*60}")

        tracker = ChangeTracker(provider_key)
        all_trackers.append(tracker)

        page_text = fetch_provider_text(provider_cfg["sources"])
        if not page_text.strip():
            print(f"  [warn] No content fetched for {provider_key}, skipping.")
            continue

        print(f"  Calling LLM to extract models...")
        models = call_llm(provider_key, page_text, client)
        print(f"  LLM returned {len(models)} model(s).")

        if not models:
            continue

        # Determine which JSON files to update (may differ, e.g. openai uses open-ai.json for general)
        pricing_key = provider_cfg.get("pricing_file", f"{provider_key}.json").replace(".json", "")
        general_key = provider_cfg.get("general_file", f"{provider_key}.json").replace(".json", "")

        merge_pricing(pricing_key, models, tracker)
        merge_general(general_key, models, tracker)

        if tracker.has_changes():
            print(
                f"  Summary: {len(tracker.new_models)} new, "
                f"{len(tracker.price_updates)} price updates"
            )
        else:
            print("  No changes for this provider.")

    # Report
    report = generate_report(all_trackers, date_str)
    print(f"\n{'='*60}")
    print("SCAN COMPLETE")
    print('='*60)
    print(report)

    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
        print(f"\nReport written to {args.output}")

    total_new = sum(len(t.new_models) for t in all_trackers)
    total_updates = sum(len(t.price_updates) for t in all_trackers)
    print(f"\nTotal: {total_new} new models, {total_updates} price updates across {len(all_trackers)} providers.")


if __name__ == "__main__":
    main()
