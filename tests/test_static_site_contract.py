from __future__ import annotations

import json
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit

import pytest


ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "public"

ROUTES = {
    "/": ("index.html", "Overview"),
    "/risk": ("risk.html", "Risk"),
    "/scenarios": ("scenarios.html", "Scenarios"),
    "/models": ("models.html", "Models"),
    "/decision": ("decision.html", "Decision"),
    "/performance": ("performance.html", "Performance"),
    "/research": ("research.html", "Research"),
}

LEGACY_REDIRECTS = {
    "/market-overview": "/",
    "/systemic-bank-network": "/risk#network",
    "/contagion-risk-score": "/risk#composite-score",
    "/stress-testing-lab": "/scenarios",
    "/rl-portfolio-agent": "/models#rl-strategy",
    "/cvar-optimization-lab": "/models#cvar-strategy",
    "/rl-vs-cvar-comparison": "/models#comparison",
    "/model-validation": "/models#validation",
    "/investment-decision-center": "/decision",
    "/performance-tracker": "/performance#strategy-performance",
    "/cvar-paper-fund": "/performance#paper-portfolio",
    "/data-catalog": "/research#data",
}

VOID_ELEMENTS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}


@dataclass
class Node:
    tag: str
    attrs: dict[str, str]
    parent: Node | None = None
    children: list[Node] = field(default_factory=list)
    text_parts: list[str] = field(default_factory=list)

    @property
    def classes(self) -> set[str]:
        return set(self.attrs.get("class", "").split())

    def text(self) -> str:
        parts = [*self.text_parts]
        for child in self.children:
            parts.append(child.text())
        return " ".join(" ".join(parts).split())

    def ancestors(self) -> list[Node]:
        result: list[Node] = []
        current = self.parent
        while current is not None:
            result.append(current)
            current = current.parent
        return result

    def descendants(self) -> list[Node]:
        result: list[Node] = []
        pending = list(reversed(self.children))
        while pending:
            node = pending.pop()
            result.append(node)
            pending.extend(reversed(node.children))
        return result


class StaticPageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = Node("document", {})
        self.stack = [self.root]

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        values = {name: value or "" for name, value in attrs}
        node = Node(tag, values, parent=self.stack[-1])
        self.stack[-1].children.append(node)
        if tag not in VOID_ELEMENTS:
            self.stack.append(node)

    def handle_startendtag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        self.handle_starttag(tag, attrs)
        if tag not in VOID_ELEMENTS:
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        for index in range(len(self.stack) - 1, 0, -1):
            if self.stack[index].tag == tag:
                del self.stack[index:]
                return

    def handle_data(self, data: str) -> None:
        if self.stack[-1].tag not in {"script", "style"}:
            self.stack[-1].text_parts.append(data)

    def nodes(self, tag: str | None = None) -> list[Node]:
        nodes = self.root.descendants()
        return nodes if tag is None else [node for node in nodes if node.tag == tag]


def parse_page(path: Path) -> StaticPageParser:
    parser = StaticPageParser()
    parser.feed(path.read_text(encoding="utf-8"))
    return parser


@pytest.fixture(scope="module")
def pages() -> dict[tuple[Path, str], StaticPageParser]:
    parsed: dict[tuple[Path, str], StaticPageParser] = {}
    for directory in (ROOT, PUBLIC):
        for route, (filename, _) in ROUTES.items():
            parsed[(directory, route)] = parse_page(directory / filename)
    return parsed


def id_map(page: StaticPageParser) -> dict[str, Node]:
    result: dict[str, Node] = {}
    for node in page.nodes():
        node_id = node.attrs.get("id")
        if node_id:
            assert node_id not in result, f"duplicate id: {node_id}"
            result[node_id] = node
    return result


def accessible_name(node: Node, ids: dict[str, Node]) -> str:
    direct = node.attrs.get("aria-label", "").strip()
    if direct:
        return direct

    labelled_by = node.attrs.get("aria-labelledby", "").split()
    if labelled_by:
        missing = [node_id for node_id in labelled_by if node_id not in ids]
        assert not missing, f"aria-labelledby references missing ids: {missing}"
        return " ".join(ids[node_id].text() for node_id in labelled_by).strip()

    captions = [
        descendant.text()
        for descendant in node.descendants()
        if descendant.tag == "caption"
    ]
    return " ".join(caption for caption in captions if caption).strip()


def test_static_export_contains_exactly_seven_matching_routes() -> None:
    expected_files = {filename for filename, _ in ROUTES.values()}

    assert {path.name for path in ROOT.glob("*.html")} == expected_files
    assert {path.name for path in PUBLIC.glob("*.html")} == expected_files

    for filename in expected_files:
        assert (ROOT / filename).read_bytes() == (PUBLIC / filename).read_bytes()


@pytest.mark.parametrize("directory", [ROOT, PUBLIC], ids=["root", "public"])
@pytest.mark.parametrize("route", ROUTES)
def test_page_shell_navigation_and_skip_target(
    directory: Path,
    route: str,
    pages: dict[tuple[Path, str], StaticPageParser],
) -> None:
    page = pages[(directory, route)]
    ids = id_map(page)

    assert len(page.nodes("h1")) == 1

    primary_navs = [
        node
        for node in page.nodes("nav")
        if node.attrs.get("aria-label") == "Primary navigation"
    ]
    assert len(primary_navs) == 1
    nav_links = [
        node
        for node in primary_navs[0].descendants()
        if node.tag == "a" and "nav-link" in node.classes
    ]
    assert len(nav_links) == len(ROUTES)

    current_links = [
        link for link in nav_links if link.attrs.get("aria-current") == "page"
    ]
    assert len(current_links) == 1
    assert "active" in current_links[0].classes
    assert current_links[0].attrs.get("href") == route
    assert current_links[0].text() == ROUTES[route][1]

    skip_links = [
        node
        for node in page.nodes("a")
        if "skip-link" in node.classes
    ]
    assert len(skip_links) == 1
    skip_target = skip_links[0].attrs.get("href", "")
    assert skip_target.startswith("#")
    target = ids.get(skip_target[1:])
    assert target is not None
    assert target.tag == "main"


@pytest.mark.parametrize("directory", [ROOT, PUBLIC], ids=["root", "public"])
@pytest.mark.parametrize("route", ROUTES)
def test_hashes_internal_links_and_table_labels(
    directory: Path,
    route: str,
    pages: dict[tuple[Path, str], StaticPageParser],
) -> None:
    page = pages[(directory, route)]
    ids = id_map(page)

    for link in page.nodes("a"):
        href = link.attrs.get("href", "").strip()
        if not href:
            continue
        parsed = urlsplit(href)
        if parsed.scheme or parsed.netloc:
            continue

        destination_route = parsed.path.rstrip("/") or route
        if parsed.path == "/":
            destination_route = "/"
        assert destination_route in ROUTES, f"{route} has broken link {href}"

        if parsed.fragment:
            destination_page = pages[(directory, destination_route)]
            assert parsed.fragment in id_map(
                destination_page
            ), f"{route} has broken hash link {href}"

    table_names: list[str] = []
    for table in [
        node for node in page.nodes("table") if "data-table" in node.classes
    ]:
        regions = [
            ancestor
            for ancestor in table.ancestors()
            if ancestor.attrs.get("role") == "region"
            and "table-scroll" in ancestor.classes
        ]
        assert regions, f"{route} has an unwrapped data table"
        name = accessible_name(regions[0], ids) or accessible_name(table, ids)
        assert name, f"{route} has an unlabeled data table region"
        table_names.append(name)

    assert len(table_names) == len(
        set(table_names)
    ), f"{route} repeats data table region labels: {table_names}"


def test_vercel_legacy_redirects_resolve_to_real_routes_and_fragments(
    pages: dict[tuple[Path, str], StaticPageParser],
) -> None:
    config = json.loads((ROOT / "vercel.json").read_text(encoding="utf-8"))
    assert config.get("cleanUrls") is True
    assert config.get("trailingSlash") is False

    redirects = config.get("redirects", [])
    assert {entry["source"]: entry["destination"] for entry in redirects} == (
        LEGACY_REDIRECTS
    )
    assert all(entry.get("permanent") is True for entry in redirects)

    for destination in LEGACY_REDIRECTS.values():
        parsed = urlsplit(destination)
        route = parsed.path.rstrip("/") or "/"
        assert route in ROUTES
        if parsed.fragment:
            assert parsed.fragment in id_map(pages[(ROOT, route)])
