"""Load Kubernetes-docs style Markdown (Hugo) into clean ``Document`` records.

Hugo pages carry YAML frontmatter and shortcodes (``{{< note >}}``, ``{{% code_sample %}}``)
that are noise for retrieval. Admonitions keep their content with a bold label; every other
shortcode and HTML comment is dropped. The page title is prepended as an H1 so the chunker's
heading path starts with it.
"""

import hashlib
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import yaml

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


@dataclass(frozen=True)
class Document:
    source: str
    url: str
    title: str
    section: str
    text: str
    content_hash: str


def _admonition(match: re.Match[str]) -> str:
    content = match.group(2).strip()
    return f"**{match.group(1).capitalize()}:** {content}" if content else ""


def clean_hugo(text: str) -> str:
    text = COMMENT_RE.sub("", text)
    text = ADMONITION_RE.sub(_admonition, text)
    text = SHORTCODE_RE.sub("", text)
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
    )


def iter_documents(root: Path, base_url: str = DOCS_BASE_URL) -> Iterator[Document]:
    for path in sorted(root.rglob("*.md")):
        doc = load_markdown_file(path, root, base_url)
        if doc is not None:
            yield doc
