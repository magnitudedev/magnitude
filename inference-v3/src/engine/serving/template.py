"""One native preparation owns prompt, parser, constraints and template identity."""

import json
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns, time
from typing import Literal

from pydantic import TypeAdapter

from engine.data import TokenId
from engine.generation.constraints import ConstraintPlan
from engine.inputs.formats.gguf_tokenizer import TokenizerArtifact
from engine.inputs.media import PreparedMedia
from engine.inputs.tokenizer import ByteBPETokenizer
from engine.serving.metrics import PreparationMetrics
from engine.serving.requests import Message, NamedChoice, Tool
from engine.serving.tool_choice import select_tools
from templates import Template
from templates.bundle import Variant
from templates.native import PreparedRequest
from templates.reasoning import REASONING_CONTROLS, ReasoningProfile, inspect_reasoning


@dataclass(frozen=True)
class PreparedChat:
    text: str
    tokens: tuple[TokenId, ...]
    native: PreparedRequest
    constraint: ConstraintPlan | None
    variant: Variant
    profile: ReasoningProfile
    metrics: PreparationMetrics
    media: PreparedMedia | None = None

    def close(self) -> None:
        self.native.close()


class ChatTemplate:
    def __init__(
        self,
        artifact: TokenizerArtifact,
        *,
        variant: str | None = None,
        override: Variant | None = None,
        image_directory: Path | None = None,
    ):
        self.artifact = artifact
        self.tokenizer = ByteBPETokenizer(artifact.config)
        self.variant, self.override = variant, override
        self._image_directory = image_directory
        self._images = None
        self._lock = threading.RLock()
        self._closed = False
        self._templates: dict[str, Template] = {}
        self._profiles: OrderedDict[tuple[str, str], tuple[ReasoningProfile, int]] = OrderedDict()
        self._profile_bytes = 0
        # Validate configured selection even before the first request.
        artifact.templates.select(tools_offered=False, variant=variant, override=override)

    def describe(self) -> dict:
        """Report profiles per effective selection, never a union of capabilities."""
        with self._lock:
            if self._closed:
                raise RuntimeError("chat template is closed")
            profiles = {}
            for offered in (False, True):
                variant, template, profile, _ = self._selected(offered)
                profiles[variant.name] = {
                    "provenance": variant.provenance,
                    "identity": template.identity,
                    "reasoning": profile.model_dump(mode="json"),
                    "reasoning_fingerprint": profile.fingerprint,
                    "native_capabilities": template.capabilities(),
                }
            return {
                "bundle_fingerprint": self.artifact.templates.fingerprint,
                "available_variants": [
                    variant.name for variant in self.artifact.templates.variants
                ],
                "profiles": profiles,
            }

    def _selected(self, tools_offered, arguments=None):
        selected = self.artifact.templates.select(
            tools_offered=tools_offered, variant=self.variant, override=self.override
        )
        create_ns = probe_ns = 0
        if selected.name not in self._templates:
            started = perf_counter_ns()
            self._templates[selected.name] = Template(
                selected.source,
                special_tokens={
                    token.name: token.text for token in self.artifact.templates.special_tokens
                },
            )
            create_ns = perf_counter_ns() - started
        template = self._templates[selected.name]
        context = {
            name: value
            for name, value in (arguments or {}).items()
            if name not in REASONING_CONTROLS
        }
        context_json = json.dumps(context, sort_keys=True, separators=(",", ":"), allow_nan=False)
        key = template.identity, context_json
        hit = key in self._profiles
        if hit:
            self._profiles.move_to_end(key)
            profile = self._profiles[key][0]
        else:
            started = perf_counter_ns()
            profile = inspect_reasoning(template, template_arguments=context)
            probe_ns = perf_counter_ns() - started
            size = len(context_json.encode()) + len(profile.model_dump_json().encode())
            if size <= 1024 * 1024:
                while self._profiles and (
                    len(self._profiles) >= 16 or self._profile_bytes + size > 1024 * 1024
                ):
                    _, (_, removed) = self._profiles.popitem(last=False)
                    self._profile_bytes -= removed
                self._profiles[key] = profile, size
                self._profile_bytes += size
        return (
            selected,
            template,
            profile,
            PreparationMetrics(
                template_create_ns=create_ns, effort_probe_ns=probe_ns, profile_cache_hit=hit
            ),
        )

    def render(
        self,
        messages,
        *,
        tools=None,
        tool_choice: Literal["auto", "required", "none"] | NamedChoice = "auto",
        parallel_tool_calls=True,
        chat_template_kwargs=None,
        reasoning_effort=None,
        json_schema=None,
        now: int | None = None,
    ) -> PreparedChat:
        messages = TypeAdapter(list[Message]).validate_python(messages)
        tools = TypeAdapter(list[Tool]).validate_python([] if tools is None else tools)
        normalized = [message.model_dump(mode="json", exclude_none=True) for message in messages]
        has_images = any(
            isinstance(message.get("content"), list)
            and any(part["type"] == "image_url" for part in message["content"])
            for message in normalized
        )
        images = ()
        image_preparation = None
        image_prepare_ns = 0
        if has_images:
            image_started = perf_counter_ns()
            from engine.models.qwen35.preparation import ImagePreparation
            from engine.serving.images import resolve_images

            if self._image_directory is None:
                raise ValueError("the served artifact does not provide an image encoder")
            with self._lock:
                if self._images is None:
                    self._images = ImagePreparation(
                        self._image_directory, self.artifact.config.pieces
                    )
                image_preparation = self._images
            normalized, images = resolve_images(normalized)
            image_prepare_ns = perf_counter_ns() - image_started
        for message in normalized:
            content = message.get("content", "")
            message["content"] = (
                "".join(
                    image_preparation.marker
                    if image_preparation is not None and part["type"] == "image"
                    else part["text"]
                    for part in content
                )
                if isinstance(content, list)
                else content
            )
            for call in message.get("tool_calls", []):
                arguments = call["function"]["arguments"]
                if isinstance(arguments, str):
                    arguments = json.loads(arguments) if arguments.strip() else {}
                if not isinstance(arguments, dict):
                    raise ValueError("historical tool arguments must be a JSON object")
                call["function"]["arguments"] = arguments
        if not normalized:
            raise ValueError("chat requires at least one message")
        offered = [
            tool.model_dump(mode="json", exclude_none=True, exclude_unset=True) for tool in tools
        ]
        names = [tool["function"]["name"] for tool in offered]
        if len(names) != len(set(names)):
            raise ValueError("tools must have unique names")
        choice = (
            tool_choice.model_dump(mode="json")
            if isinstance(tool_choice, NamedChoice)
            else tool_choice
        )
        selection = select_tools(offered, choice)
        with self._lock:
            if self._closed:
                raise RuntimeError("chat template is closed")
            arguments = dict(chat_template_kwargs or {})
            variant, template, profile, metrics = self._selected(bool(selection.tools), arguments)
            reasoning_controls = {
                control.name for mapping in profile.mappings for control in mapping.controls
            }
            if reasoning_effort is not None and reasoning_controls.intersection(arguments):
                raise ValueError("reasoning_effort conflicts with raw template reasoning controls")
            arguments.update(profile.resolve(reasoning_effort))
            started = perf_counter_ns()
            native = template.prepare(
                normalized,
                tools=list(selection.tools),
                tool_choice="required" if selection.required else "auto",
                parallel_tool_calls=parallel_tool_calls,
                template_arguments=arguments,
                json_schema=json_schema,
                now=int(time()) if now is None else now,
            )
            rendered = perf_counter_ns()
            try:
                description = native.description
                text, media = description.prompt, None
                if images:
                    assert image_preparation is not None
                    image_started = perf_counter_ns()
                    text, media = image_preparation.prepare(text, images)
                    image_prepare_ns += perf_counter_ns() - image_started
                tokenize_started = perf_counter_ns() if images else rendered
                tokens = self.tokenizer.encode(text)
                tokenized = perf_counter_ns()
                if not tokens:
                    raise ValueError("chat template produced no input tokens")
                constraint = (
                    ConstraintPlan(
                        artifact_identity=self.artifact.config.artifact_identity,
                        template_identity=template.identity,
                        grammar=description.grammar,
                        initial_prefix=description.grammar_initial_prefix,
                    )
                    if description.grammar
                    else None
                )
                if (selection.required or json_schema is not None) and constraint is None:
                    raise ValueError(
                        "selected template did not produce required output constraints"
                    )
                return PreparedChat(
                    text,
                    tokens,
                    native,
                    constraint,
                    variant,
                    profile,
                    metrics.model_copy(
                        update={
                            "render_ns": rendered - started,
                            "tokenize_ns": tokenized - tokenize_started,
                            "image_prepare_ns": image_prepare_ns,
                        }
                    ),
                    media,
                )
            except BaseException:
                native.close()
                raise

    def close(self) -> None:
        with self._lock:
            self._closed = True
            for template in self._templates.values():
                template.close()
            self._templates.clear()
            self._profiles.clear()
            self._profile_bytes = 0
