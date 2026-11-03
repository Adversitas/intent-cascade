"""LLM fallback classifier.

Called only for messages the trained model isn't confident about. The prompt is
deliberately narrow:

- **Candidate labels** come from the trained model's top-k. Top-10 accuracy is
  far higher than top-1, so the right label is almost always in the list, and a
  short list is cheaper and easier to choose from than all 77 intents.
- **Few-shot examples** are the nearest labelled messages from the vector store.
- **Output is constrained** with a JSON schema whose ``label`` field is an enum
  of the candidates plus ``none_of_these``. The model cannot return a label that
  doesn't exist, and ``none_of_these`` gives it a sanctioned way to say the
  message doesn't fit, which the cascade turns into a hand-off to a human.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from .data import humanize
from .vectorstore import Neighbour

DEFAULT_MODEL = os.environ.get("INTENT_CASCADE_LLM", "claude-opus-5-5")
NONE_LABEL = "none_of_these"

SYSTEM_PROMPT = """\
You classify customer-support messages sent to an online bank's in-app chat. \
Pick the single intent from <candidates> that best describes what the customer \
needs, so the message can be routed to the right workflow.

- Choose by what the customer is asking for, not by keywords they mention. \
"My card hasn't arrived, when will it come?" is about delivery timing even \
though it mentions the card.
- The <examples> are real, correctly labelled messages similar to this one. \
Use them to learn where the boundaries between similar intents lie.
- If no candidate fits, answer "none_of_these". A wrong label sends the \
customer down the wrong workflow, which is worse than handing the message to \
a human."""


@dataclass
class LLMDecision:
    label: str
    model: str
    input_tokens: int
    output_tokens: int


class LLMError(RuntimeError):
    pass


def build_schema(candidates: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {"label": {"type": "string", "enum": [*candidates, NONE_LABEL]}},
        "required": ["label"],
        "additionalProperties": False,
    }


def build_prompt(text: str, candidates: list[str], examples: list[Neighbour]) -> str:
    cand = "\n".join(f"- {c}: {humanize(c)}" for c in candidates)
    shots = "\n".join(f'<example intent="{ex.label}">{ex.text}</example>' for ex in examples)
    return (
        f"<candidates>\n{cand}\n</candidates>\n\n"
        f"<examples>\n{shots}\n</examples>\n\n"
        f"<message>{text}</message>"
    )


class ClaudeIntentClassifier:
    def __init__(self, model: str = DEFAULT_MODEL, effort: str = "low", client=None) -> None:
        import anthropic

        self.model = model
        self.effort = effort
        self.client = client or anthropic.Anthropic()

    def classify(self, text: str, candidates: list[str], examples: list[Neighbour]) -> LLMDecision:
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_prompt(text, candidates, examples)}],
            output_config={
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": build_schema(candidates)},
            },
            # On a safety-classifier decline, retry server-side on the fallback model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if response.stop_reason == "refusal":
            raise LLMError("model declined to classify")
        if response.stop_reason == "max_tokens":
            raise LLMError("response truncated at max_tokens")
        text_block = next(b.text for b in response.content if b.type == "text")
        label = json.loads(text_block)["label"]
        if label not in candidates and label != NONE_LABEL:
            # The schema should make this impossible; treat it as a hand-off if it happens.
            label = NONE_LABEL
        return LLMDecision(
            label=label,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
