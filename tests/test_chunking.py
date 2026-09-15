from rag_insight.chunking import chunk_sections
from rag_insight.config import Settings
from rag_insight.models import Section


class CharacterTokenizer:
    def encode(self, text, **kwargs):
        return list(text)


def test_fenced_heading_is_not_a_section_and_budget_is_bounded():
    section = Section("d", "v", "a.md", "# Main\n\n```python\n# code comment\nprint(1)\n```\n\n" + "x" * 180, line_start=1)
    chunks = chunk_sections([section], Settings(chunk_tokens=80, overlap_tokens=10), CharacterTokenizer())
    assert all(len(chunk.text) <= 80 for chunk in chunks)
    assert all(chunk.heading_path == "Main" for chunk in chunks)
    assert any("# code comment" in chunk.text for chunk in chunks)
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)


def test_structure_ancestry_resets_for_siblings_and_recursive_has_none():
    text = "# Root\n\n## First\n\none\n\n### Child\n\ntwo\n\n## Second\n\nthree"
    section = Section("d", "v", "a.md", text, line_start=1)
    structure = chunk_sections([section], Settings(chunk_tokens=200), CharacterTokenizer())
    paths = [chunk.heading_path for chunk in structure]
    assert "Root > First > Child" in paths
    assert "Root > Second" in paths
    assert all("First" not in path for path in paths if path.endswith("Second"))
    recursive = chunk_sections([section], Settings(chunk_strategy="recursive", chunk_tokens=200), CharacterTokenizer())
    assert all(not chunk.heading_path for chunk in recursive)


def test_line_locations_overlap_and_ids_are_deterministic():
    section = Section("d", "v", "a.txt", "alpha beta gamma delta epsilon zeta", line_start=10)
    settings = Settings(chunk_strategy="recursive", chunk_tokens=12, overlap_tokens=3)
    first = chunk_sections([section], settings, CharacterTokenizer())
    second = chunk_sections([section], settings, CharacterTokenizer())
    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert all(chunk.line_start == 10 and chunk.line_end == 10 for chunk in first)
    assert all(len(chunk.text) <= settings.chunk_tokens for chunk in first)
    assert first[0].text[-3:] in first[1].text
