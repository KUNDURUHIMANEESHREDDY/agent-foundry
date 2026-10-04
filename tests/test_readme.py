"""Every relative markdown link must resolve.

A dangling anchor is worse than no link: it reads as documentation and points
nowhere. The README is edited often and each edit adds cross-references, so the
check is mechanical.

External URLs are not fetched — that belongs in a link checker, and a test that
depends on the network fails for reasons unrelated to the code.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

README = Path(__file__).resolve().parents[1] / "README.md"


def slugify(heading: str) -> str:
    """Approximate GitHub's heading slugs: lowercase, drop punctuation, hyphenate."""
    s = re.sub(r"[^\w\s-]", "", heading.strip().lower())
    return re.sub(r"\s+", "-", s)


def headings() -> set[str]:
    out: set[str] = set()
    in_fence = False
    for line in README.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            # A `#` inside a fenced block is code, not a heading.
            continue
        m = re.match(r"^#{1,6}\s+(.*?)\s*$", line)
        if m:
            out.add(slugify(m.group(1)))
    return out


def anchors() -> list[str]:
    return re.findall(
        r"\[[^\]]+\]\(#([^)]+)\)", README.read_text(encoding="utf-8")
    )


class TestReadmeLinks:
    def test_readme_exists(self):
        assert README.is_file()

    def test_it_has_internal_links_to_check(self):
        """Otherwise the tests below pass vacuously."""
        assert len(anchors()) >= 5

    @pytest.mark.parametrize("anchor", sorted(set(anchors())))
    def test_anchor_resolves(self, anchor):
        assert anchor in headings(), f"#{anchor} has no heading"

    def test_no_duplicate_heading_slugs(self):
        """A duplicate slug makes GitHub append -1, silently breaking one link."""
        text = README.read_text(encoding="utf-8")
        in_fence = False
        seen: dict[str, int] = {}
        for line in text.splitlines():
            if line.strip().startswith("```"):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            m = re.match(r"^#{1,6}\s+(.*?)\s*$", line)
            if m:
                slug = slugify(m.group(1))
                seen[slug] = seen.get(slug, 0) + 1

        duplicates = {s: n for s, n in seen.items() if n > 1}
        assert not duplicates, f"duplicate heading slugs: {duplicates}"


def relative_file_links() -> list[str]:
    """Markdown links to a sibling file, e.g. `[LICENSE](LICENSE)`.

    Not anchors and not URLs -- a path relative to the README. These break
    silently: the renderer shows the link text and nothing else.
    """
    text = README.read_text(encoding="utf-8")
    out = []
    for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
        if target.startswith(("http://", "https://", "#", "mailto:")):
            continue
        out.append(target)
    return out


class TestReadmeFileLinks:
    def test_it_has_a_relative_file_link_to_check(self):
        """Otherwise the tests below pass vacuously."""
        assert relative_file_links()

    @pytest.mark.parametrize("target", sorted(set(relative_file_links())))
    def test_file_exists(self, target):
        path = (README.parent / target.split("#", 1)[0]).resolve()
        assert path.is_file(), f"{target} does not exist"


class TestLicenseIsDeclaredSomewhere:
    """A licence has to be findable without reading pyproject."""

    def test_license_file_is_tracked(self):
        assert (README.parent / "LICENSE").is_file()

    def test_license_file_is_not_empty(self):
        assert (README.parent / "LICENSE").read_text(encoding="utf-8").strip()

    def test_readme_names_the_license(self):
        text = README.read_text(encoding="utf-8")
        assert re.search(r"(?im)^#+ .*licen[sc]e", text), "README has no licence section"

    def test_pyproject_declares_an_spdx_identifier(self):
        """
        A bare `LICENSE` file is invisible to tooling; PyPI and `pip show` read
        the metadata. Assert the identifier is a real SPDX expression rather than
        free text, so the two cannot drift apart.
        """
        pyproject = (README.parent / "pyproject.toml").read_text(encoding="utf-8")
        declared = re.search(r'(?m)^\s*license\s*=\s*"([^"]+)"', pyproject)
        assert declared, "pyproject declares no license"

        spdx = declared.group(1).strip()
        assert re.fullmatch(r"[A-Za-z0-9.+-]+", spdx), (
            f"{spdx!r} is not a bare SPDX identifier -- PEP 639 form expected"
        )

    def test_the_declared_license_matches_the_file(self):
        """The metadata and the text must not disagree about which one this is."""
        pyproject = (README.parent / "pyproject.toml").read_text(encoding="utf-8")
        spdx = re.search(r'(?m)^\s*license\s*=\s*"([^"]+)"', pyproject).group(1)

        text = (README.parent / "LICENSE").read_text(encoding="utf-8")
        assert spdx in text, (
            f"pyproject declares {spdx!r} but LICENSE does not mention it"
        )