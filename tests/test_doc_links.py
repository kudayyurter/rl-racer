"""Every relative link in the README and the docs points at something that exists.

The lessons link into the code by line range (``racer/sim.py#L192-L218``), so editing a module
can silently break a lesson: the file still exists but the lines moved. This test checks three
things for every relative link in README.md and docs/**/*.md:

- the target file (or directory) exists;
- a ``#Lx`` or ``#Lx-Ly`` range lies inside the file;
- a heading anchor (``#results-three-seeds-on-twisty``) matches a heading of the target file,
  slugged the way GitHub does it.

It cannot check that a line range still shows the code its link text describes; after moving
code, look at the links into that file by hand.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOCS = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]

LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)\)")
LINE_RANGE = re.compile(r"L(\d+)(?:-L(\d+))?")
FENCE = re.compile(r"^\s*(```|~~~)")
INLINE_CODE = re.compile(r"`[^`]*`")


def prose_lines(text: str):
    """(line number, line) outside fenced code blocks, with inline code removed."""
    in_fence = False
    for n, line in enumerate(text.splitlines(), start=1):
        if FENCE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence:
            yield n, INLINE_CODE.sub("", line)


def slug(heading: str) -> str:
    """GitHub's anchor for a heading: lowercase, drop punctuation, spaces become hyphens."""
    text = heading.strip().lower().replace("`", "")
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)   # a link in a heading keeps its text
    text = re.sub(r"[^\w\- ]", "", text)                   # \w keeps letters, digits, _
    return text.replace(" ", "-")


def anchors(path: Path) -> set[str]:
    """All heading anchors of a markdown file, with GitHub's -1, -2 suffixes for repeats."""
    seen: dict[str, int] = {}
    out = set()
    in_fence = False
    for line in path.read_text().splitlines():
        if FENCE.match(line):
            in_fence = not in_fence
            continue
        m = re.match(r"#{1,6}\s+(.*?)\s*#*\s*$", line)
        if in_fence or not m:
            continue
        base = slug(m.group(1))
        k = seen.get(base, 0)
        out.add(base if k == 0 else f"{base}-{k}")
        seen[base] = k + 1
    return out


def links():
    for doc in DOCS:
        for n, line in prose_lines(doc.read_text()):
            for target in LINK.findall(line):
                if re.match(r"[a-z]+:", target):   # http:, https:, mailto: are not checked
                    continue
                yield pytest.param(doc, target, id=f"{doc.relative_to(ROOT)}:{n}:{target}")


def check(doc: Path, target: str) -> str | None:
    """None if the link resolves, else what is wrong with it."""
    path_part, _, fragment = target.partition("#")
    path = (doc.parent / path_part).resolve() if path_part else doc
    if not path.exists():
        return f"no such file: {path_part}"
    if not fragment:
        return None
    if m := LINE_RANGE.fullmatch(fragment):
        first, last = int(m.group(1)), int(m.group(2) or m.group(1))
        n_lines = len(path.read_text().splitlines())
        if not 1 <= first <= last <= n_lines:
            return f"lines {first}-{last} outside {path_part or doc.name} ({n_lines} lines)"
        return None
    if path.suffix != ".md":
        return f"anchor #{fragment} into a non-markdown file"
    if fragment not in anchors(path):
        return f"no heading #{fragment} in {path_part or doc.name}"
    return None


def test_slug_matches_github():
    assert slug("10 · Results: every method side by side") == "10--results-every-method-side-by-side"
    assert (slug("### 1. Mutation size: `mutation_std` 0.02 vs 0.1 vs 0.5".lstrip("# "))
            == "1-mutation-size-mutation_std-002-vs-01-vs-05")


@pytest.mark.parametrize("doc,target", list(links()))
def test_link_resolves(doc, target):
    problem = check(doc, target)
    assert problem is None, problem


def test_check_flags_each_kind_of_broken_link(tmp_path):
    (tmp_path / "code.py").write_text("a = 1\nb = 2\n")
    doc = tmp_path / "doc.md"
    doc.write_text("# Title\n\n## Part `two`\n")
    assert check(doc, "code.py#L1-L2") is None
    assert check(doc, "#part-two") is None
    assert "no such file" in check(doc, "missing.py")
    assert "outside" in check(doc, "code.py#L2-L3")
    assert "no heading" in check(doc, "#part-three")
