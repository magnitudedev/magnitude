"""Artifact-declared template variants and deterministic source selection."""

import hashlib

from pydantic import Field, model_validator

from templates.events import Record


class Variant(Record):
    name: str = Field(min_length=1)
    source: str = Field(min_length=1)
    provenance: str = Field(min_length=1)


class SpecialToken(Record):
    name: str = Field(min_length=1)
    text: str


class TemplateBundle(Record):
    variants: tuple[Variant, ...]
    default: str
    special_tokens: tuple[SpecialToken, ...] = ()

    @model_validator(mode="after")
    def declared_names(self):
        names = [variant.name for variant in self.variants]
        tokens = [token.name for token in self.special_tokens]
        if len(set(names)) != len(names) or len(set(tokens)) != len(tokens):
            raise ValueError("template variants and special-token names must be unique")
        if self.default not in names:
            raise ValueError("template bundle requires a usable declared default")
        return self

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()

    def select(
        self, *, tools_offered: bool, variant: str | None = None, override: Variant | None = None
    ) -> Variant:
        if override is not None:
            if variant is not None:
                raise ValueError("configure a source override or a variant, not both")
            return override
        names = {item.name: item for item in self.variants}
        selected = variant or (
            "tool_use" if tools_offered and "tool_use" in names else self.default
        )
        if selected not in names:
            raise ValueError(f"Unknown template variant: {selected}")
        return names[selected]
