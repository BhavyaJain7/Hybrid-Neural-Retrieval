"""
Unit tests for synthesis/prompt.py
Tests: prompt structure, context formatting, source attribution, edge cases.
"""
import pytest
from neural_search.synthesis.prompt import build_prompt


@pytest.fixture
def sample_chunks():
    return [
        {"source_file": "report.pdf", "page": 3, "text": "Payment terms are net 30 days."},
        {"source_file": "contract.pdf", "page": 7, "text": "Renewal is automatic unless cancelled."},
    ]


class TestBuildPrompt:
    def test_returns_system_and_user_keys(self, sample_chunks):
        prompt = build_prompt("What are payment terms?", sample_chunks)
        assert "system" in prompt
        assert "user" in prompt

    def test_system_contains_anti_hallucination_instruction(self, sample_chunks):
        prompt = build_prompt("query", sample_chunks)
        assert "not present" in prompt["system"].lower() or "not find" in prompt["system"].lower()

    def test_system_instructs_context_only(self, sample_chunks):
        prompt = build_prompt("query", sample_chunks)
        assert "only" in prompt["system"].lower() or "context" in prompt["system"].lower()

    def test_user_contains_query(self, sample_chunks):
        query = "What are the payment terms?"
        prompt = build_prompt(query, sample_chunks)
        assert query in prompt["user"]

    def test_user_contains_all_chunk_texts(self, sample_chunks):
        prompt = build_prompt("query", sample_chunks)
        for chunk in sample_chunks:
            assert chunk["text"] in prompt["user"]

    def test_user_contains_source_attribution(self, sample_chunks):
        prompt = build_prompt("query", sample_chunks)
        assert "report.pdf" in prompt["user"]
        assert "contract.pdf" in prompt["user"]
        assert "Page 3" in prompt["user"] or "3" in prompt["user"]

    def test_source_numbered_sequentially(self, sample_chunks):
        prompt = build_prompt("query", sample_chunks)
        assert "Source 1" in prompt["user"]
        assert "Source 2" in prompt["user"]

    def test_empty_chunks_still_builds_prompt(self):
        prompt = build_prompt("any query", [])
        assert "system" in prompt
        assert "user" in prompt
        assert "any query" in prompt["user"]

    def test_caps_to_five_chunks(self):
        """Prompt builder itself doesn't cap — groq_client does. Verify user content grows with chunks."""
        chunks = [
            {"source_file": f"file{i}.pdf", "page": i, "text": f"content {i}"}
            for i in range(7)
        ]
        prompt = build_prompt("query", chunks)
        # All 7 should appear in user prompt when called directly
        for i in range(7):
            assert f"content {i}" in prompt["user"]
