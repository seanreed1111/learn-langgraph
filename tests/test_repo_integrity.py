"""Keep the repo honest: pins match what is installed, and documentation links resolve."""

import re
import tomllib
from importlib import metadata
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]


def _pins(requirements: list[str]) -> list[tuple[str, str]]:
    pins = []
    for req in requirements:
        m = re.fullmatch(r"([A-Za-z0-9_.-]+)(?:\[[^\]]+\])?==([^\s;]+)", req)
        assert m, f"not an exact pin: {req!r}"
        pins.append((str(m.group(1)), str(m.group(2))))
    return pins


def test_every_direct_dependency_is_an_exact_pin():
    groups = [PYPROJECT["dependencies"], *PYPROJECT["optional-dependencies"].values()]
    for reqs in groups:
        _pins(reqs)  # raises with the offending requirement


@pytest.mark.parametrize("name,version", _pins(PYPROJECT["dependencies"] + PYPROJECT["optional-dependencies"]["dev"]))
def test_installed_version_matches_pin(name: str, version: str):
    assert metadata.version(name) == version, f"{name}: pin {version}, installed {metadata.version(name)}"


def test_build_backend_is_hatchling():
    build = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["build-system"]
    assert build["build-backend"] == "hatchling.build"
    assert build["requires"] == ["hatchling"]


def test_type_checker_is_ty_with_every_rule_an_error():
    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert cfg["tool"]["ty"]["rules"] == {"all": "error"}
    assert "mypy" not in (ROOT / "pyproject.toml").read_text(encoding="utf-8").lower().replace("mypy is intentionally not used", "")


LINK = re.compile(r"\]\((?!https?:|mailto:|#)([^)\s#]+)(?:#[^)\s]*)?\)")


def _markdown_files() -> list[Path]:
    files = [*ROOT.glob("*.md"), *(ROOT / "docs").rglob("*.md"), *(ROOT / "lessons").glob("*.md")]
    return sorted(files)


@pytest.mark.parametrize("md", _markdown_files(), ids=lambda p: p.relative_to(ROOT).as_posix())
def test_relative_markdown_links_resolve(md: Path):
    text = re.sub(r"```.*?```", "", md.read_text(encoding="utf-8"), flags=re.S)  # ignore code samples
    text = re.sub(r"`[^`\n]*`", "", text)  # and inline code (illustrative link syntax)
    broken = [t for t in LINK.findall(text) if not (md.parent / t).resolve().exists()]
    assert broken == [], f"{md.name} has dead links: {broken}"


def _slug(heading: str) -> str:
    """GitHub-style heading anchor."""
    return re.sub(r"[^\w\- ]", "", heading.lower().strip()).replace(" ", "-")


def test_readme_toc_links_all_resolve_to_headings():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    prose = re.sub(r"```.*?```", "", text, flags=re.S)
    anchors = {_slug(m.group(1)) for m in re.finditer(r"^#{1,6} (.+)$", prose, flags=re.M)}
    toc = text.split("## Table of contents", 1)[1].split("\n## ", 1)[0]
    links = re.findall(r"\]\(#([^)\s]+)\)", toc)
    assert len(links) >= 30, "the README must keep a table of contents at the top"
    assert [a for a in links if a not in anchors] == []
    assert not re.findall(r"\]\([^)#]*\.md#", toc), "TOC links must be in-page anchors (single-file README)"


def test_every_lesson_plan_is_inline_in_the_readme():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    for n in range(1, 25):
        assert re.search(rf"^### Lesson {n:02d}: ", text, flags=re.M), f"lesson {n:02d} plan missing from README"
    assert re.search(r"^## Capstone: ", text, flags=re.M)
