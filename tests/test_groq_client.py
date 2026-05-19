"""
Unit tests for synthesis/groq_client.py
"""
import pytest
from unittest.mock import MagicMock, patch


# Current active model — update when Groq changes recommendation
CURRENT_MODEL = "llama-3.1-8b-instant"


def _mock_response(answer: str) -> MagicMock:
    response = MagicMock()
    response.choices[0].message.content = answer
    return response


class TestGroqSynthesizer:
    def test_successful_synthesis_returns_answer(self, sample_chunks):
        with patch("neural_search.synthesis.groq_client.Groq") as MockGroq:
            MockGroq.return_value.chat.completions.create.return_value = (
                _mock_response("The answer is 42.")
            )
            from neural_search.synthesis.groq_client import GroqSynthesizer
            result = GroqSynthesizer().synthesize("What is the answer?", sample_chunks)

        assert result["answer"] == "The answer is 42."
        assert result["model"] == CURRENT_MODEL
        assert "sources_used" in result

    def test_sources_capped_at_five_even_with_more_chunks(self, sample_chunks):
        assert len(sample_chunks) > 5
        with patch("neural_search.synthesis.groq_client.Groq") as MockGroq:
            MockGroq.return_value.chat.completions.create.return_value = (
                _mock_response("answer")
            )
            from neural_search.synthesis.groq_client import GroqSynthesizer
            result = GroqSynthesizer().synthesize("query", sample_chunks)

        assert len(result["sources_used"]) == 5

    def test_sources_contain_required_fields(self, sample_chunks):
        with patch("neural_search.synthesis.groq_client.Groq") as MockGroq:
            MockGroq.return_value.chat.completions.create.return_value = (
                _mock_response("answer")
            )
            from neural_search.synthesis.groq_client import GroqSynthesizer
            result = GroqSynthesizer().synthesize("query", sample_chunks)

        for source in result["sources_used"]:
            assert "source_file" in source
            assert "page" in source

    def test_retries_once_on_rate_limit_then_succeeds(self, sample_chunks):
        from groq import RateLimitError
        with patch("neural_search.synthesis.groq_client.Groq") as MockGroq, \
             patch("neural_search.synthesis.groq_client.time.sleep") as mock_sleep:

            MockGroq.return_value.chat.completions.create.side_effect = [
                RateLimitError("rate limit", response=MagicMock(), body={}),
                _mock_response("retry worked"),
            ]
            from neural_search.synthesis.groq_client import GroqSynthesizer
            result = GroqSynthesizer().synthesize("query", sample_chunks[:2])

        assert result["answer"] == "retry worked"
        mock_sleep.assert_called_once()

    def test_exhausted_retries_returns_fallback_message(self, sample_chunks):
        from groq import RateLimitError
        with patch("neural_search.synthesis.groq_client.Groq") as MockGroq, \
             patch("neural_search.synthesis.groq_client.time.sleep"):

            MockGroq.return_value.chat.completions.create.side_effect = RateLimitError(
                "rate limit", response=MagicMock(), body={}
            )
            from neural_search.synthesis.groq_client import GroqSynthesizer
            result = GroqSynthesizer().synthesize("query", sample_chunks[:2], retries=3)

        assert result["sources_used"] == []
        assert result["model"] == CURRENT_MODEL
        assert any(
            word in result["answer"].lower()
            for word in ("unable", "try again", "failed", "error")
        )

    def test_api_error_returns_fallback(self, sample_chunks):
        from groq import APIError
        with patch("neural_search.synthesis.groq_client.Groq") as MockGroq, \
             patch("neural_search.synthesis.groq_client.time.sleep"):

            # Properly mock APIError without relying on constructor
            error = APIError.__new__(APIError)
            error.args = ("server error",)

            MockGroq.return_value.chat.completions.create.side_effect = error

            from neural_search.synthesis.groq_client import GroqSynthesizer
            result = GroqSynthesizer().synthesize("query", sample_chunks[:2])

        assert result["sources_used"] == []
        assert result["model"] == CURRENT_MODEL

    def test_model_name_comes_from_settings(self, sample_chunks, patch_settings):
        patch_settings.groq_model = "custom-model"

        with patch("neural_search.synthesis.groq_client.Groq") as MockGroq:
            MockGroq.return_value.chat.completions.create.return_value = (
                _mock_response("ok")
            )
            from neural_search.synthesis.groq_client import GroqSynthesizer
            result = GroqSynthesizer(
                settings_obj=patch_settings
            ).synthesize("query", sample_chunks[:1])

        assert result["model"] == "custom-model"

    def test_decommissioned_model_triggers_fallback_to_current(self, patch_settings):
        """Settings with old model name should be silently upgraded."""
        patch_settings.groq_model = "llama3-8b-8192"

        with patch("neural_search.synthesis.groq_client.Groq") as MockGroq:
            MockGroq.return_value.chat.completions.create.return_value = (
                _mock_response("ok")
            )
            from neural_search.synthesis.groq_client import GroqSynthesizer
            synth = GroqSynthesizer(settings_obj=patch_settings)

        assert synth._model == CURRENT_MODEL
