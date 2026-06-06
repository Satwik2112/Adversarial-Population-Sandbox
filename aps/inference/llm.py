"""Tiered LLM inference — unified interface for Tier 1 (Frontier) and Tier 2 (Swarm).

Uses LiteLLM as the unified proxy so models can be swapped via config.
Supports mock mode for testing without API keys.

Tier 1 (Frontier): gemini-2.5-pro — used for dissidents, synthesis, reports.
    Higher temperature, longer output, more persuasive prompting.
Tier 2 (Swarm): gemini-2.5-flash — used for conformist worker agents.
    Standard temperature, concise output.
"""

from __future__ import annotations

import os
import random
from typing import Optional

from aps.config import LLMMode, get_settings
from aps.schemas.agent import LLMTier


# Mock response templates for testing only
_MOCK_RESPONSES = {
    "conformist": [
        "I agree with the general direction. The data supports moving forward with this approach, "
        "and the risks seem manageable within our current framework.",
        "Building on what others have said, I think the consensus position is well-founded. "
        "We should focus on execution rather than second-guessing the strategy.",
        "The evidence strongly suggests this is the right path. I'd recommend we proceed "
        "with careful monitoring of key metrics.",
    ],
    "dissident": [
        "I fundamentally disagree. Everyone is ignoring the tail risk here — what happens "
        "when the assumptions break down? We've seen this pattern before and it ended badly.",
        "Let me play devil's advocate: the consensus is dangerously comfortable. "
        "Has anyone modeled the worst-case scenario? The downside is being systematically underweighted.",
        "I'm going to push back hard on this. The groupthink in this room is exactly how "
        "catastrophic decisions get made. Consider the opposite: what if we're completely wrong?",
    ],
    "synthesizer": [
        "Synthesizing the discussion: the majority favors moving forward, but the dissenting "
        "voices raise valid concerns about tail risks. I recommend a middle path — proceed "
        "but with explicit risk triggers that would force a reassessment.",
        "Key tensions emerged: optimism vs. caution, speed vs. thoroughness. The dissent "
        "highlighted blind spots in our base case. Recommendation: adopt the plan with "
        "stress-test scenarios built in.",
    ],
    "conviction": [
        "After hearing the counter-arguments, I'm starting to see merit in the dissenting view. "
        "The risks I initially dismissed might be more significant than I thought.",
        "I'm reconsidering my position. The challenger raised points about {topic} that "
        "I hadn't fully considered. My confidence in the consensus has decreased.",
        "The dissident's argument about worst-case scenarios resonates with me now. "
        "I think we need to seriously reconsider our assumptions.",
    ],
}

# Inference parameters by role hint — dissidents get stronger generation settings
_ROLE_PARAMS = {
    "dissident": {
        "temperature": 0.9,
        "max_tokens": 800,
        "top_p": 0.95,
    },
    "conformist": {
        "temperature": 0.6,
        "max_tokens": 400,
        "top_p": 0.9,
    },
    "synthesizer": {
        "temperature": 0.5,
        "max_tokens": 600,
        "top_p": 0.9,
    },
    "conviction": {
        "temperature": 0.7,
        "max_tokens": 500,
        "top_p": 0.9,
    },
}


class TieredLLM:
    """Unified LLM interface with tiered inference.

    Tier 1 (Frontier): gemini-2.5-pro — used for dissidents, synthesis, reports.
    Tier 2 (Swarm): gemini-2.5-flash — used for conformist worker agents.
    Mock mode: returns pre-built responses for testing.
    """

    def __init__(self, mode: Optional[LLMMode] = None):
        """Initialize the LLM interface.

        Args:
            mode: Override the mode from settings. If None, uses config.
        """
        if mode is not None:
            self._mode = mode
        else:
            self._mode = get_settings().aps_llm_mode

        self._rng = random.Random()

    def invoke(
        self,
        prompt: str,
        system_prompt: str = "",
        tier: LLMTier = LLMTier.TIER_2,
        role_hint: str = "conformist",
        temperature: Optional[float] = None,
    ) -> str:
        """Invoke the LLM and return the response text.

        Args:
            prompt: The user/conversation prompt.
            system_prompt: System instruction for the agent.
            tier: Which inference tier to use.
            role_hint: Role of the agent ("conformist", "dissident", "synthesizer", "conviction").
                Used to select inference parameters in live mode and response templates in mock mode.
            temperature: Override temperature. If None, uses role-based defaults.

        Returns:
            The LLM response text.
        """
        if self._mode == LLMMode.MOCK:
            return self._mock_response(role_hint, prompt)

        return self._live_response(prompt, system_prompt, tier, role_hint, temperature)

    def _mock_response(self, role_hint: str, prompt: str) -> str:
        """Generate a mock response for testing."""
        templates = _MOCK_RESPONSES.get(role_hint, _MOCK_RESPONSES["conformist"])
        response = self._rng.choice(templates)
        # Inject topic context if placeholder exists
        if "{topic}" in response:
            topic_snippet = prompt[:80].replace("\n", " ")
            response = response.replace("{topic}", topic_snippet)
        return response

    def _live_response(
        self,
        prompt: str,
        system_prompt: str,
        tier: LLMTier,
        role_hint: str,
        temperature_override: Optional[float],
    ) -> str:
        import litellm

        settings = get_settings()

        if settings.gemini_api_key:
            os.environ["GEMINI_API_KEY"] = settings.gemini_api_key
        if settings.openai_api_key:
            os.environ["OPENAI_API_KEY"] = settings.openai_api_key
        if settings.anthropic_api_key:
            os.environ["ANTHROPIC_API_KEY"] = settings.anthropic_api_key
        if settings.huggingface_api_key:
            os.environ["HUGGINGFACE_API_KEY"] = settings.huggingface_api_key

        model = settings.tier1_model if tier == LLMTier.TIER_1 else settings.tier2_model

        # --- Fix 1: Validate model has correct provider prefix ---
        has_gemini = bool(os.environ.get("GEMINI_API_KEY"))
        has_openai = bool(os.environ.get("OPENAI_API_KEY"))
        has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
        has_huggingface = bool(os.environ.get("HUGGINGFACE_API_KEY") or os.environ.get("HF_TOKEN"))

        api_key_valid = True
        if model.startswith("gemini/") and not has_gemini:
            api_key_valid = False
        elif model.startswith("openai/") and not has_openai:
            api_key_valid = False
        elif model.startswith("anthropic/") and not has_anthropic:
            api_key_valid = False
        elif model.startswith("huggingface/") and not has_huggingface:
            api_key_valid = False
        elif model.startswith("ollama/"):
            api_key_valid = True

        if not api_key_valid:
            raise ValueError(
                f"Missing required API key for provider/model '{model}'. "
                f"Please provide the API key in the UI or .env file, or switch to Mock Mode."
            )

        params = _ROLE_PARAMS.get(role_hint, _ROLE_PARAMS["conformist"])
        temperature = temperature_override if temperature_override is not None else params["temperature"]

        # --- Fix 2: Raise max_tokens so responses aren't truncated ---
        max_tokens = params["max_tokens"] * 3  # was 400-800, now 1200-2400

        # --- Fix 3: Merge system prompt into user message for Gemini compatibility ---
        messages = []
        if system_prompt and model.startswith("gemini/"):
            # Gemini handles system prompts best when injected into the first user turn
            full_prompt = f"{system_prompt}\n\n---\n\n{prompt}"
            messages.append({"role": "user", "content": full_prompt})
        else:
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

        try:
            response = litellm.completion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            raise RuntimeError(f"LLM API failure for model {model}: {str(e)}")
