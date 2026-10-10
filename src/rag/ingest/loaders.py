"""Load Kubernetes-docs style Markdown (Hugo) into clean ``Document`` records.

Hugo pages carry YAML frontmatter and shortcodes (``{{< note >}}``, ``{{% code_sample %}}``)
that are noise for retrieval. Admonitions keep their content with a bold label; every other
shortcode and HTML comment is dropped. The page title is prepended as an H1 so the chunker's
heading path starts with it.
"""

import hashlib
import html
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import yaml

# Bump whenever cleaning output changes: it is part of the pipeline fingerprint, so every
# document gets re-ingested even though its raw file (and content_hash) did not change.
LOADER_VERSION = 2

DOCS_BASE_URL = "https://kubernetes.io/docs/"
INDEX_FILES = {"_index.md", "index.md"}
ADMONITIONS = ("note", "caution", "warning")

FRONTMATTER_RE = re.compile(r"\A---[ \t]*\n(.*?)\n---[ \t]*(?:\n|\Z)", re.DOTALL)
ADMONITION_RE = re.compile(
    r"{{[<%]\s*(" + "|".join(ADMONITIONS) + r")\s*[>%]}}(.*?){{[<%]\s*/\1\s*[>%]}}",
    re.DOTALL,
)
SHORTCODE_RE = re.compile(r"{{[<%].*?[>%]}}", re.DOTALL)
COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
BLANK_RUN_RE = re.compile(r"\n{3,}")
HEADING_ANCHOR_RE = re.compile(r"^(#{1,6}\s.*?)\s*\{#[^}]*\}\s*$", re.MULTILINE)

# HTML -> text. Only known tag names are touched, so placeholders like <pod-name> survive.
CELL = "\x1f"
HTML_RULES: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(pattern, re.IGNORECASE | re.DOTALL), repl)
    for pattern, repl in (
        (r"<(code|tt)\b[^>]*>(.*?)</\1>", r"`\2`"),
        (r"<(?:br|hr)\s*/?>", "\n"),
        (r"<tr\b[^>]*>", "\n"),
        (r"</tr>", ""),
        (r"</?t[dh]\b[^>]*>", CELL),
        (r"<li\b[^>]*>", "\n- "),
        (r"</li>", ""),
        (r"</?(?:table|thead|tbody|tfoot|p|div|ul|ol|h[1-6]|blockquote)\b[^>]*>", "\n"),
        (r"</?(?:em|i|strong|b|a|span|sup|sub|small|u|caption|colgroup|col|label)\b[^>]*>", ""),
    )
)
GENERATED_DIR = "generated"


@dataclass(frozen=True)
class Document:
    source: str
    url: str
    title: str
    section: str
    text: str
    content_hash: str
    generated: bool


def _admonition(match: re.Match[str]) -> str:
    content = match.group(2).strip()
    return f"**{match.group(1).capitalize()}:** {content}" if content else ""


def _join_cells(line: str) -> str:
    if CELL not in line:
        return line.rstrip()
    return " | ".join(cell.strip() for cell in line.split(CELL) if cell.strip())


def html_to_text(text: str) -> str:
    for pattern, repl in HTML_RULES:
        text = pattern.sub(repl, text)
    text = html.unescape(text)
    return "\n".join(_join_cells(line) for line in text.split("\n"))


def clean_hugo(text: str) -> str:
    text = COMMENT_RE.sub("", text)
    text = ADMONITION_RE.sub(_admonition, text)
    text = SHORTCODE_RE.sub("", text)
    text = html_to_text(text)
    text = HEADING_ANCHOR_RE.sub(r"\1", text)
    return BLANK_RUN_RE.sub("\n\n", text).strip()


def _split_frontmatter(raw: str, path: Path) -> tuple[dict, str]:
    match = FRONTMATTER_RE.match(raw)
    if match is None:
        return {}, raw
    try:
        meta = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"{path}: invalid frontmatter: {exc}") from exc
    if not isinstance(meta, dict):
        raise ValueError(f"{path}: frontmatter must be a mapping, got {type(meta).__name__}")
    return meta, raw[match.end() :]


def _url_for(rel: Path, base_url: str) -> str:
    page = rel.parent if rel.name in INDEX_FILES else rel.with_suffix("")
    slug = page.as_posix()
    return base_url if slug in ("", ".") else f"{base_url}{slug}/"


def load_markdown_file(path: Path, root: Path, base_url: str = DOCS_BASE_URL) -> Document | None:
    """Return a cleaned Document, or None for drafts and pages with no content."""
    raw = path.read_text(encoding="utf-8")
    meta, body = _split_frontmatter(raw, path)
    if meta.get("draft") is True:
        return None
    cleaned = clean_hugo(body)
    if not cleaned:
        return None

    rel = path.relative_to(root)
    title = str(meta.get("title") or path.stem).strip()
    return Document(
        source=rel.as_posix(),
        url=_url_for(rel, base_url),
        title=title,
        section=rel.parts[0] if len(rel.parts) > 1 else "",
        text=f"# {title}\n\n{cleaned}",
        content_hash=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        generated=meta.get("auto_generated") is True or GENERATED_DIR in rel.parts[:-1],
    )


def iter_documents(root: Path, base_url: str = DOCS_BASE_URL) -> Iterator[Document]:
    for path in sorted(root.rglob("*.md")):
        doc = load_markdown_file(path, root, base_url)
        if doc is not None:
            yield doc
