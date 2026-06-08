#!/usr/bin/env python3
"""
web-quality-monitor
===================

Periodically fetches a set of web pages, asks Claude to review each one for
editorial quality (English conventions, textual formatting, internal
correctness/consistency), and alerts you only about NEW problems.

Designed to run unattended on a schedule (cron / GitHub Actions).

Key behaviours that make it cheap and quiet for recurring runs:
  * Content hashing  - if a page is byte-for-byte the same as last run, the
                       API call is skipped and the previous findings reused.
  * Finding dedup    - issues already reported are remembered; you only get
                       alerted about findings that are new since last time.

Usage:
    python monitor.py                      # run all pages in config
    python monitor.py --page https://...   # test one URL (ad-hoc, no state)
    python monitor.py --all                # report every finding, not just new
    python monitor.py --dry-run            # analyse + write report, send no alerts
    python monitor.py --config other.yaml  # use a different config file

Environment variables:
    ANTHROPIC_API_KEY   (required)
    SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, ALERT_FROM, ALERT_TO   (email alerts)
    SLACK_WEBHOOK_URL   (Slack alerts)
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import smtplib
import sys
import urllib.request
from email.mime.text import MIMEText
from pathlib import Path

import requests
import yaml
from bs4 import BeautifulSoup

try:
    import anthropic
except ImportError:
    sys.exit("Missing dependency: pip install anthropic")

# --------------------------------------------------------------------------- #
# Config & constants
# --------------------------------------------------------------------------- #

HERE = Path(__file__).resolve().parent
STATE_PATH = HERE / "state.json"
REPORTS_DIR = HERE / "reports"
DEFAULT_MODEL = "claude-sonnet-4-6"      # good quality/cost balance; see README
DEFAULT_MAX_CHARS = 60_000               # cap page text sent to the API (cost control)
SEVERITY_ORDER = {"high": 3, "medium": 2, "low": 1}

SYSTEM_PROMPT = """\
You are a meticulous web content reviewer for a publishing team. You review the
text of a single web page and report concrete quality problems. You are not a
chatbot; you only emit findings.

Review only for the categories the user asks for:
  - "english"     : grammar, spelling, punctuation, awkward or unclear phrasing.
  - "formatting"  : TEXTUAL formatting only - inconsistent capitalization in
                    headings, inconsistent date/number formats, stray double
                    spaces, broken markdown/HTML entities visible as text,
                    non-parallel list items, inconsistent terminology.
  - "correctness" : statements that are internally contradictory, numbers that
                    don't add up, claims that are self-evidently wrong, or dates
                    that contradict each other ELSEWHERE on the same page. You
                    cannot verify external facts unless given tools; when a claim
                    is merely suspicious rather than provably wrong, mark it
                    severity "low" and say so in the issue text.
  - "consistency" : the page contradicting itself, mixed spellings of the same
                    term, mixed voice/tense where it reads as an error.
  - "layout"      : VISUAL problems, judged ONLY from the screenshot when one is
                    provided - overlapping or cut-off elements, text overflowing
                    its container, broken or missing images, unstyled/raw-looking
                    sections, badly misaligned components, content running off
                    screen, or obviously broken spacing/visual hierarchy. Do NOT
                    guess at layout when no screenshot is supplied.

Rules:
  - Report real problems, not stylistic preferences. If something is merely a
    matter of taste, omit it.
  - Quote the smallest exact span of the page text that shows the problem in
    "excerpt" (max ~25 words).
  - Be conservative with "high" severity: reserve it for errors that damage
    credibility or change meaning.
  - If the page is clean, return an empty findings list.

Honor any house-style rules the user supplies; a violation of an explicit
house-style rule is a valid "formatting" or "consistency" finding.

Output ONLY a single JSON object, no prose before or after, in exactly this shape:
{
  "summary": "one short sentence overall assessment",
  "findings": [
    {
      "category": "english|formatting|correctness|consistency|layout",
      "severity": "high|medium|low",
      "excerpt": "exact problematic text, or a short description of the visual spot for layout",
      "issue": "what is wrong, briefly",
      "suggestion": "the concrete fix"
    }
  ]
}
"""

CROSS_PAGE_SYSTEM_PROMPT = """\
You are a brand/style consistency auditor reviewing several pages from the SAME
website together. Your only job is to find places where the pages are
INCONSISTENT WITH EACH OTHER (not problems internal to one page).

Look for:
  - The same product/feature/term spelled or capitalized differently across
    pages (e.g. "Acme Cloud" vs "AcmeCloud", "log in" vs "login" vs "log-in").
  - Inconsistent date, time, number, or currency formats between pages.
  - The same action or link labeled differently on different pages
    (e.g. "Sign up" vs "Get started" vs "Register" for the same thing).
  - Inconsistent heading capitalization style (title case vs sentence case)
    across pages.
  - Differing claims about the same fact (prices, counts, dates) between pages.
  - Inconsistent tone/voice where it reads as an oversight rather than intent.

Honor any house-style rules supplied; flag pages that deviate from the agreed
convention.

Report each inconsistency once. Always name the specific pages that disagree.
Be conservative with "high" severity (reserve for contradictory facts or
brand-name errors). Ignore matters of pure taste.

Output ONLY a single JSON object, no prose before or after:
{
  "summary": "one short sentence on overall cross-page consistency",
  "findings": [
    {
      "category": "consistency",
      "severity": "high|medium|low",
      "excerpt": "the conflicting variants, e.g. \\"login\\" (Pricing) vs \\"log in\\" (Home)",
      "issue": "what is inconsistent and which pages disagree",
      "suggestion": "the convention to standardize on"
    }
  ]
}
"""


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #

def load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except json.JSONDecodeError:
            print("! state.json was corrupt; starting fresh", file=sys.stderr)
    return {"pages": {}}


def save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False))


# --------------------------------------------------------------------------- #
# Fetch & extract
# --------------------------------------------------------------------------- #

def fetch_text(url: str, timeout: int = 30) -> str:
    """Fetch a page and return readable text plus a heading outline.

    Note: this uses a plain HTTP GET, so it does NOT execute JavaScript. For
    JS-rendered pages, see the Playwright note in the README.
    """
    headers = {"User-Agent": "web-quality-monitor/1.0 (+editorial QA bot)"}
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    for tag in soup(["script", "style", "noscript", "template", "svg"]):
        tag.decompose()

    # A light structural outline helps Claude judge heading-case consistency.
    outline = []
    for h in soup.find_all(["h1", "h2", "h3"]):
        txt = " ".join(h.get_text(" ", strip=True).split())
        if txt:
            outline.append(f"[{h.name}] {txt}")

    body = soup.body or soup
    text = body.get_text("\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)

    parts = []
    if outline:
        parts.append("HEADING OUTLINE:\n" + "\n".join(outline))
    parts.append("PAGE TEXT:\n" + text)
    return "\n\n".join(parts)


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fetch_rendered(url: str, viewport_width: int = 1280,
                   timeout: int = 45) -> tuple[str, bytes]:
    """Render the page in a headless browser; return (text, full-page PNG bytes).

    Requires:  pip install playwright  &&  playwright install chromium
    This also handles JavaScript-rendered pages that fetch_text() can't read.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise RuntimeError(
            "render is enabled but Playwright isn't installed. Run:\n"
            "    pip install playwright && playwright install chromium")

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": viewport_width, "height": 900})
        try:
            page.goto(url, wait_until="networkidle", timeout=timeout * 1000)
        except Exception:                            # noqa: BLE001
            # networkidle can time out on chatty pages; fall back to load.
            page.goto(url, wait_until="load", timeout=timeout * 1000)
        page.wait_for_timeout(800)                   # let late layout settle
        text = page.inner_text("body")
        shot = page.screenshot(full_page=True, type="png")
        browser.close()

    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    return "PAGE TEXT:\n" + text, shot


# --------------------------------------------------------------------------- #
# Claude review
# --------------------------------------------------------------------------- #

def build_user_message(page: dict, text: str, categories: list[str],
                       max_chars: int) -> str:
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[...truncated for length...]"
    house_rules = page.get("house_rules") or []
    rules_block = ""
    if house_rules:
        rules_block = "HOUSE-STYLE RULES (treat violations as findings):\n" + \
            "\n".join(f"- {r}" for r in house_rules) + "\n\n"
    return (
        f"Page name: {page.get('name', page['url'])}\n"
        f"URL: {page['url']}\n"
        f"Review for categories: {', '.join(categories)}\n\n"
        f"{rules_block}"
        f"---- BEGIN PAGE CONTENT ----\n{text}\n---- END PAGE CONTENT ----"
    )


def extract_json(raw: str) -> dict:
    """Pull the JSON object out of the model's text, tolerating stray prose."""
    raw = raw.strip()
    # Strip code fences if present.
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE).strip()
    # Find the outermost JSON object.
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in model output:\n{raw[:500]}")
    return json.loads(raw[start:end + 1])


def review_page(client: "anthropic.Anthropic", page: dict, text: str,
                cfg: dict, screenshot: bytes | None = None) -> dict:
    default_cats = ["english", "formatting", "correctness", "consistency"]
    categories = list(page.get("categories") or cfg.get("categories", default_cats))
    if screenshot is not None and "layout" not in categories:
        categories.append("layout")

    tools = []
    if cfg.get("fact_check_with_web_search"):
        # Server-side web search; the API runs the searches and returns the
        # final answer. Costs extra per search. The system prompt still requires
        # the final text block to be the JSON object.
        tools.append({"type": "web_search_20250305", "name": "web_search",
                      "max_uses": cfg.get("web_search_max_uses", 3)})

    user_text = build_user_message(
        page, text, categories, cfg.get("max_chars", DEFAULT_MAX_CHARS))
    if screenshot is not None:
        import base64
        content = [
            {"type": "image", "source": {
                "type": "base64", "media_type": "image/png",
                "data": base64.b64encode(screenshot).decode()}},
            {"type": "text", "text":
                "Above is a full-page screenshot of the page; use it for the "
                "'layout' category.\n\n" + user_text},
        ]
    else:
        content = user_text

    kwargs = dict(
        model=cfg.get("model", DEFAULT_MODEL),
        max_tokens=cfg.get("max_tokens", 8000),
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content}],
    )
    if tools:
        kwargs["tools"] = tools

    msg = client.messages.create(**kwargs)
    raw = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
    data = extract_json(raw)
    data.setdefault("summary", "")
    data.setdefault("findings", [])
    return data


def cross_page_consistency(client: "anthropic.Anthropic",
                           page_texts: list[tuple[str, str]],
                           cfg: dict) -> dict:
    """Compare several pages against each other for site-wide consistency.

    page_texts: list of (page_name, text). One extra API call per run.
    """
    per_page_chars = cfg.get("cross_page_chars_per_page", 4000)
    blocks = []
    for name, text in page_texts:
        snippet = text[:per_page_chars]
        blocks.append(f"===== PAGE: {name} =====\n{snippet}")
    house_rules = cfg.get("house_rules") or []
    rules_block = ""
    if house_rules:
        rules_block = "SITE-WIDE HOUSE-STYLE RULES:\n" + \
            "\n".join(f"- {r}" for r in house_rules) + "\n\n"
    user = (rules_block +
            "Compare the following pages from one website for cross-page "
            "consistency:\n\n" + "\n\n".join(blocks))

    msg = client.messages.create(
        model=cfg.get("model", DEFAULT_MODEL),
        max_tokens=cfg.get("max_tokens", 8000),
        system=CROSS_PAGE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user}],
    )
    raw = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
    data = extract_json(raw)
    data.setdefault("summary", "")
    data.setdefault("findings", [])
    return data


# --------------------------------------------------------------------------- #
# Dedup
# --------------------------------------------------------------------------- #

def finding_fingerprint(url: str, finding: dict) -> str:
    excerpt = " ".join((finding.get("excerpt") or "").lower().split())
    key = f"{url}||{finding.get('category')}||{excerpt}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def split_new_vs_seen(url: str, findings: list[dict],
                      seen_fps: set[str]) -> tuple[list[dict], set[str]]:
    new, all_fps = [], set()
    for f in findings:
        fp = finding_fingerprint(url, f)
        all_fps.add(fp)
        if fp not in seen_fps:
            new.append(f)
    return new, all_fps


# --------------------------------------------------------------------------- #
# Reporting & alerts
# --------------------------------------------------------------------------- #

def severity_at_or_above(findings: list[dict], threshold: str) -> list[dict]:
    floor = SEVERITY_ORDER.get(threshold, 1)
    return [f for f in findings
            if SEVERITY_ORDER.get(f.get("severity", "low"), 1) >= floor]


def render_markdown(results: list[dict], when: str) -> str:
    lines = [f"# Web quality report — {when}", ""]
    total_new = sum(len(r["new_findings"]) for r in results)
    lines.append(f"**{total_new} new finding(s)** across {len(results)} page(s).")
    lines.append("")
    for r in results:
        status = r["status"]
        lines.append(f"## {r['name']}")
        lines.append(f"<{r['url']}>")
        lines.append("")
        if status == "unchanged":
            lines.append("_Page unchanged since last run — not re-reviewed._\n")
            continue
        if status == "error":
            lines.append(f"⚠️ Error: {r['error']}\n")
            continue
        if r.get("summary"):
            lines.append(f"_{r['summary']}_\n")
        findings = r["new_findings"] if not r["show_all"] else r["all_findings"]
        label = "New findings" if not r["show_all"] else "All findings"
        if not findings:
            lines.append("✅ No new issues.\n")
            continue
        lines.append(f"**{label}: {len(findings)}**\n")
        for f in sorted(findings,
                        key=lambda x: -SEVERITY_ORDER.get(x.get("severity", "low"), 1)):
            sev = f.get("severity", "low").upper()
            cat = f.get("category", "?")
            lines.append(f"- **[{sev} · {cat}]** {f.get('issue', '')}")
            if f.get("excerpt"):
                lines.append(f"  - Excerpt: “{f['excerpt']}”")
            if f.get("suggestion"):
                lines.append(f"  - Fix: {f['suggestion']}")
        lines.append("")
    return "\n".join(lines)


def send_email(subject: str, body: str) -> None:
    host = os.environ.get("SMTP_HOST")
    to = os.environ.get("ALERT_TO")
    if not (host and to):
        return
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = os.environ.get("ALERT_FROM", os.environ.get("SMTP_USER", ""))
    msg["To"] = to
    port = int(os.environ.get("SMTP_PORT", "587"))
    with smtplib.SMTP(host, port) as s:
        s.starttls()
        if os.environ.get("SMTP_USER"):
            s.login(os.environ["SMTP_USER"], os.environ.get("SMTP_PASS", ""))
        s.sendmail(msg["From"], [to], msg.as_string())
    print(f"  email alert sent to {to}")


def send_slack(text: str) -> None:
    url = os.environ.get("SLACK_WEBHOOK_URL")
    if not url:
        return
    req = urllib.request.Request(
        url, data=json.dumps({"text": text}).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=15)
    print("  slack alert sent")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def run(cfg: dict, state: dict, show_all: bool, dry_run: bool,
        single_url: str | None) -> int:
    client = anthropic.Anthropic()
    when = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    pages = ([{"url": single_url, "name": single_url}] if single_url
             else cfg.get("pages", []))
    if not pages:
        sys.exit("No pages configured. Add some to config.yaml (or use --page).")

    threshold = cfg.get("alert_on_severity", "medium")
    render_global = bool(cfg.get("render", False))
    results = []
    page_texts: list[tuple[str, str]] = []   # (name, text) for cross-page pass
    changed_any = False

    for page in pages:
        url = page["url"]
        name = page.get("name", url)
        print(f"• {name}")
        page_state = state["pages"].get(url, {})
        do_render = bool(page.get("render", render_global))
        screenshot = None
        try:
            if do_render:
                text, screenshot = fetch_rendered(
                    url, viewport_width=cfg.get("viewport_width", 1280))
            else:
                text = fetch_text(url)
        except Exception as e:                       # noqa: BLE001
            print(f"  fetch error: {e}")
            results.append({"name": name, "url": url, "status": "error",
                            "error": str(e), "new_findings": [],
                            "all_findings": [], "show_all": show_all})
            continue

        page_texts.append((name, text))

        chash = content_hash(text)
        if (not single_url and not show_all
                and page_state.get("content_hash") == chash
                and "findings" in page_state):
            print("  unchanged — skipping API call")
            results.append({"name": name, "url": url, "status": "unchanged",
                            "new_findings": [], "all_findings": [],
                            "show_all": show_all})
            continue

        changed_any = True
        try:
            review = review_page(client, page, text, cfg, screenshot=screenshot)
        except Exception as e:                       # noqa: BLE001
            print(f"  review error: {e}")
            results.append({"name": name, "url": url, "status": "error",
                            "error": str(e), "new_findings": [],
                            "all_findings": [], "show_all": show_all})
            continue

        findings = review["findings"]
        seen_fps = set(page_state.get("finding_fps", []))
        new_findings, all_fps = split_new_vs_seen(url, findings, seen_fps)
        print(f"  {len(findings)} finding(s), {len(new_findings)} new")

        results.append({
            "name": name, "url": url, "status": "ok",
            "summary": review.get("summary", ""),
            "new_findings": new_findings, "all_findings": findings,
            "show_all": show_all,
        })

        if not single_url:
            state["pages"][url] = {
                "content_hash": chash,
                "findings": findings,
                "finding_fps": sorted(all_fps),
                "last_checked": when,
            }

    # ---- Cross-page (site-wide) consistency pass ---------------------------
    cross_result = None
    do_cross = (cfg.get("cross_page_consistency", True) and not single_url
                and len(page_texts) >= 2 and (changed_any or show_all))
    if do_cross:
        print("• cross-page consistency check")
        try:
            cross = cross_page_consistency(client, page_texts, cfg)
            cfindings = cross["findings"]
            seen_fps = set(state["pages"].get("__site__", {}).get("finding_fps", []))
            new_c, all_c = split_new_vs_seen("__site__", cfindings, seen_fps)
            print(f"  {len(cfindings)} finding(s), {len(new_c)} new")
            cross_result = {
                "name": "Cross-page consistency", "url": "(site-wide)",
                "status": "ok", "summary": cross.get("summary", ""),
                "new_findings": new_c, "all_findings": cfindings,
                "show_all": show_all,
            }
            state["pages"]["__site__"] = {
                "findings": cfindings, "finding_fps": sorted(all_c),
                "last_checked": when,
            }
        except Exception as e:                       # noqa: BLE001
            print(f"  cross-page error: {e}")
            cross_result = {"name": "Cross-page consistency", "url": "(site-wide)",
                            "status": "error", "error": str(e),
                            "new_findings": [], "all_findings": [],
                            "show_all": show_all}
    if cross_result:
        results.append(cross_result)

    # Write the full report regardless.
    REPORTS_DIR.mkdir(exist_ok=True)
    report_md = render_markdown(results, when)
    report_path = REPORTS_DIR / f"report-{dt.datetime.now():%Y%m%d-%H%M%S}.md"
    report_path.write_text(report_md, encoding="utf-8")
    print(f"\nReport written: {report_path}")

    # Decide whether to alert: only on NEW findings at/above threshold.
    alertable = []
    for r in results:
        if r["status"] != "ok":
            continue
        hits = severity_at_or_above(r["new_findings"], threshold)
        if hits:
            alertable.append((r, hits))

    if alertable and not dry_run and not single_url:
        total = sum(len(h) for _, h in alertable)
        subject = f"[Web QA] {total} new issue(s) on {len(alertable)} page(s)"
        send_email(subject, report_md)
        send_slack(subject + "\n\n" + report_md[:3500])
    elif alertable and dry_run:
        print("(dry-run) would have alerted on new issues")
    else:
        print("No new alertable issues.")

    if not single_url:
        save_state(state)

    return 1 if alertable else 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Editorial QA monitor powered by Claude")
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--page", help="review a single ad-hoc URL (no state saved)")
    ap.add_argument("--all", action="store_true",
                    help="report every finding, not just new ones")
    ap.add_argument("--dry-run", action="store_true", help="don't send alerts")
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY first.")

    cfg = {}
    if not args.page:
        cfg_path = Path(args.config)
        if not cfg_path.exists():
            sys.exit(f"Config not found: {cfg_path} (copy config.example.yaml)")
        cfg = yaml.safe_load(cfg_path.read_text()) or {}

    state = load_state()
    rc = run(cfg, state, show_all=args.all, dry_run=args.dry_run,
             single_url=args.page)
    sys.exit(rc)


if __name__ == "__main__":
    main()
