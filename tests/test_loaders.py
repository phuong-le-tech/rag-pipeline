import hashlib

import pytest

from rag.ingest.loaders import Document, clean_hugo, iter_documents, load_markdown_file

DEPLOYMENT_MD = """---
title: Deployments
content_type: concept
weight: 10
---

<!-- overview -->
A _Deployment_ provides declarative updates for Pods.

{{< glossary_definition term_id="deployment" length="short" >}}

{{< feature-state for_k8s_version="v1.19" state="stable" >}}

## Creating a Deployment

{{% code_sample file="controllers/nginx-deployment.yaml" %}}

{{< note >}}
Do not manage ReplicaSets owned by a Deployment.
{{< /note >}}

{{< caution >}}
Changing the selector is not recommended.
{{< /caution >}}
"""


def write(root, rel, text):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_clean_hugo_drops_comments_and_inline_shortcodes():
    text = (
        "<!-- body -->\nKeep me.\n"
        '{{< feature-state state="beta" >}}\n{{% code_sample file="a.yaml" %}}'
    )

    assert clean_hugo(text) == "Keep me."


def test_clean_hugo_keeps_admonition_content_with_label():
    text = "{{< note >}}\nBe careful.\n{{< /note >}}\n{{< warning >}}\nData loss.\n{{< /warning >}}"

    assert clean_hugo(text) == "**Note:** Be careful.\n**Warning:** Data loss."


def test_clean_hugo_collapses_blank_runs():
    assert clean_hugo("a\n\n\n\n\nb") == "a\n\nb"


FLAG_TABLE = """<table style="width: 100%;">
<tbody>
<tr>
<td colspan="2">--admission-control strings</td>
</tr>
<tr>
<td></td><td style="line-height: 130%;"><p>File with admission control configuration.</p></td>
</tr>
</tbody>
</table>"""


def test_clean_hugo_turns_flag_table_into_lines():
    assert clean_hugo(FLAG_TABLE) == (
        "--admission-control strings\n\nFile with admission control configuration."
    )


def test_clean_hugo_handles_label_hr_and_tt():
    text = "<label>Field</label> uses <tt>spec.replicas</tt>\n<hr>\nNext"

    assert clean_hugo(text) == "Field uses `spec.replicas`\n\nNext"


def test_clean_hugo_joins_table_cells_with_pipes():
    text = (
        "<table><tr><th>Field</th><th>Type</th></tr><tr><td>name</td><td>string</td></tr></table>"
    )

    assert clean_hugo(text) == "Field | Type\nname | string"


def test_clean_hugo_strips_inline_tags_keeps_code_and_decodes_entities():
    text = 'Use <code>kubectl get</code> on <em>all</em> <a href="/x">nodes</a> &amp; pods.'

    assert clean_hugo(text) == "Use `kubectl get` on all nodes & pods."


def test_clean_hugo_leaves_angle_bracket_placeholders_alone():
    text = "Run `kubectl describe pod <pod-name>` to inspect it."

    assert clean_hugo(text) == text


@pytest.mark.parametrize(
    ("rel", "frontmatter", "expected"),
    [
        ("reference/kubernetes-api/pod-v1.md", "auto_generated: true\n", True),
        ("reference/setup-tools/kubeadm/generated/kubeadm_init.md", "", True),
        ("concepts/workloads/pods.md", "", False),
        ("reference/glossary/pod.md", "auto_generated: false\n", False),
    ],
)
def test_generated_flag_from_frontmatter_or_path(tmp_path, rel, frontmatter, expected):
    path = write(tmp_path, rel, f"---\ntitle: X\n{frontmatter}---\nBody.\n")

    assert load_markdown_file(path, root=tmp_path).generated is expected


def test_load_markdown_file_builds_document_with_metadata(tmp_path):
    path = write(tmp_path, "concepts/workloads/deployment.md", DEPLOYMENT_MD)

    doc = load_markdown_file(path, root=tmp_path)

    assert doc == Document(
        source="concepts/workloads/deployment.md",
        url="https://kubernetes.io/docs/concepts/workloads/deployment/",
        title="Deployments",
        section="concepts",
        text=doc.text,
        content_hash=hashlib.sha256(DEPLOYMENT_MD.encode()).hexdigest(),
        generated=False,
    )
    assert doc.text.startswith("# Deployments\n\nA _Deployment_ provides")
    assert "## Creating a Deployment" in doc.text
    assert "**Note:** Do not manage ReplicaSets" in doc.text
    assert "**Caution:** Changing the selector" in doc.text
    assert "{{" not in doc.text and "<!--" not in doc.text


def test_index_file_maps_to_directory_url(tmp_path):
    path = write(tmp_path, "tasks/_index.md", "---\ntitle: Tasks\n---\nHow-to guides.\n")

    doc = load_markdown_file(path, root=tmp_path)

    assert doc.url == "https://kubernetes.io/docs/tasks/"
    assert doc.section == "tasks"


def test_title_falls_back_to_file_stem(tmp_path):
    path = write(tmp_path, "reference/kubectl-cheatsheet.md", "Some body text.\n")

    doc = load_markdown_file(path, root=tmp_path)

    assert doc.title == "kubectl-cheatsheet"


@pytest.mark.parametrize(
    "text",
    [
        "---\ntitle: Draft\ndraft: true\n---\nWork in progress.\n",
        "---\ntitle: Empty\n---\n<!-- overview -->\n{{< note >}}\n{{< /note >}}\n",
    ],
)
def test_skips_drafts_and_empty_pages(tmp_path, text):
    path = write(tmp_path, "concepts/x.md", text)

    assert load_markdown_file(path, root=tmp_path) is None


def test_invalid_frontmatter_raises_with_path(tmp_path):
    path = write(tmp_path, "concepts/bad.md", "---\ntitle: [unclosed\n---\nBody\n")

    with pytest.raises(ValueError, match="bad.md"):
        load_markdown_file(path, root=tmp_path)


def test_iter_documents_walks_markdown_sorted_and_skips_others(tmp_path):
    write(tmp_path, "b/two.md", "---\ntitle: Two\n---\nSecond.\n")
    write(tmp_path, "a/one.md", "---\ntitle: One\n---\nFirst.\n")
    write(tmp_path, "a/draft.md", "---\ndraft: true\n---\nNo.\n")
    write(tmp_path, "a/image.png", "not markdown")

    docs = list(iter_documents(tmp_path))

    assert [d.source for d in docs] == ["a/one.md", "b/two.md"]
