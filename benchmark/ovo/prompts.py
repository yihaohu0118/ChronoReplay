"""
Pluggable prompt templates for OVO-S evaluation.

Usage:
    from prompts import build_prompt

    prompt = build_prompt(question, options)                  # default
    prompt = build_prompt(question, options, style="cot")     # chain-of-thought

Register new templates with the @register decorator:

    @register("my_style")
    def my_prompt(question, options):
        ...
        return prompt_string
"""

from typing import Dict

from synthesis.direct_vlm import build_direct_mcq_prompt


PROMPT_REGISTRY: Dict[str, callable] = {}


def register(name: str):
    """Decorator to register a prompt template function."""
    def decorator(fn):
        PROMPT_REGISTRY[name] = fn
        return fn
    return decorator


def build_prompt(question: str, options: Dict[str, str], style: str = "default") -> str:
    """Build a prompt using the named template.

    Falls back to 'default' if *style* is not found in the registry.
    """
    fn = PROMPT_REGISTRY.get(style, PROMPT_REGISTRY["default"])
    return fn(question, options)


# ---------------------------------------------------------------------------
# Built-in templates
# ---------------------------------------------------------------------------

@register("default")
def _default_prompt(question: str, options: Dict[str, str]) -> str:
    return build_direct_mcq_prompt(question, options)


@register("verbose")
def _verbose_prompt(question: str, options: Dict[str, str]) -> str:
    """Original prompt with extra analysis instructions."""
    options_text = "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))
    return f"""You are evaluating a video understanding task. Based on the video frames provided, answer the following multiple choice question.

Question: {question}

Options:
{options_text}

Instructions:
- Carefully analyze the video frames to understand the spatial relationships and scene content.
- Select the most appropriate answer from the given options.
- Respond with ONLY the letter of your answer (e.g., "A" or "B").
- Do not include any explanation or additional text.

Your answer:"""


@register("cot")
def _cot_prompt(question: str, options: Dict[str, str]) -> str:
    """Chain-of-thought: model reasons step-by-step then gives the answer."""
    options_text = "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))
    return f"""You are evaluating a video understanding task. Based on the video frames provided, answer the following multiple choice question.

Question: {question}

Options:
{options_text}

Instructions:
- Think step by step about the spatial relationships and scene content shown in the frames.
- After your reasoning, output your final answer on a new line in the format: Answer: X

Your reasoning and answer:"""


@register("spatial_state")
def _spatial_state_prompt(question: str, options: Dict[str, str]) -> str:
    """Answer-only spatial-state reasoning without exposing a long rationale.

    The checklist is intentionally question-visible and task-blind.  It names
    reusable operations needed by streaming spatial questions rather than
    benchmark levels, categories, expected answers, or option labels.
    """
    options_text = "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))
    return f"""You are evaluating a video understanding task. Based on the video frames provided, answer the following multiple choice question.

Question: {question}

Options:
{options_text}

Instructions:
- Reason internally from the observations in chronological order.
- Maintain a compact spatial state: persistent landmarks, camera heading, ordered turns, and revisited places.
- If a place is revisited, compare the two visits rather than merely adjacent frames.
- If options contain maps or trajectories, match path topology and the ordered left/right turns; do not rely on option-letter frequency.
- Respond with ONLY the letter of your answer (e.g., "A" or "B").
- Do not include any explanation or additional text.

Your answer:"""


def _options_block(options: Dict[str, str]) -> str:
    return "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))


@register("l1_direct")
def _l1_direct_prompt(question: str, options: Dict[str, str]) -> str:
    return f"""You are viewing an egocentric video stream. Answer from the CURRENT view only.

Question: {question}

Options:
{_options_block(options)}

Instructions:
- Focus on the most recent frames near the question time.
- Respond with ONLY the letter of your answer.

Your answer:"""


@register("l2_memory")
def _l2_memory_prompt(question: str, options: Dict[str, str]) -> str:
    return f"""You are viewing an egocentric walk-through video (streaming prefix only).

Question: {question}

Options:
{_options_block(options)}

Instructions:
- First recall places/objects seen EARLIER in the video that are no longer visible.
- Then pick the answer that matches that memory.
- Respond with ONLY the letter of your answer.

Your answer:"""


@register("l2_recent_count")
def _l2_recent_count_prompt(question: str, options: Dict[str, str]) -> str:
    return f"""You are viewing the recent part of an egocentric walk-through video.

Question: {question}

Options:
{_options_block(options)}

Instructions:
- Count the requested objects in the scene or area that was just shown.
- Use only the recent frames near the question time.
- Respond with ONLY the letter of your answer.

Your answer:"""


@register("l3_simulate")
def _l3_simulate_prompt(question: str, options: Dict[str, str]) -> str:
    return f"""You are viewing an egocentric video stream. Reason about spatial structure step by step.

Question: {question}

Options:
{_options_block(options)}

Instructions:
- Mentally simulate the spatial change (rotation, removal, route) using the video.
- Keep reasoning brief (max 3 steps), then give ONLY the answer letter on the last line.

Your answer:"""


@register("l4_map")
def _l4_map_prompt(question: str, options: Dict[str, str]) -> str:
    return f"""You are viewing a long egocentric walk-through. Build a mental map of rooms and paths.

Question: {question}

Options:
{_options_block(options)}

Instructions:
- Integrate the full video prefix into an allocentric layout before answering.
- For map/trajectory image options, compare each image to the path you walked.
- Respond with ONLY the letter of your answer.

Your answer:"""
