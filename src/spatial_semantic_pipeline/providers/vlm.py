"""Built-in OpenAI-compatible/Gemini semantic label provider."""

from ..context import retry_call_vlm_api, vlm_api_key, vlm_input_mode
from .base import SemanticLabelRequest, SemanticLabelResult


class OpenAICompatibleSemanticLabeler:
    def label(self, request: SemanticLabelRequest) -> SemanticLabelResult:
        text, raw_response = retry_call_vlm_api(request.config, request.prompt, request.clip)
        return SemanticLabelResult(text=text, raw_response=raw_response)


__all__ = [
    "OpenAICompatibleSemanticLabeler",
    "retry_call_vlm_api",
    "vlm_api_key",
    "vlm_input_mode",
]
