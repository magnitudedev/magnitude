"""Validated Chat Completions inputs; unsupported options fail at admission."""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from magnitude_engine.generation.sampling_policy import SamplingPolicy


class StreamOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    include_usage: bool = False


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    model: str = Field(min_length=1)
    messages: list[dict] = Field(min_length=1, max_length=4096)
    tools: list[dict] = Field(default_factory=list, max_length=512)
    tool_choice: str | dict = "auto"
    parallel_tool_calls: bool = True
    response_format: dict | None = None
    chat_template_kwargs: dict = Field(default_factory=dict)
    max_tokens: int | None = Field(default=None, ge=0)
    max_completion_tokens: int | None = Field(default=None, ge=0)
    temperature: float = Field(default=1, ge=0, allow_inf_nan=False)
    top_p: float = Field(default=1, gt=0, le=1)
    top_k: int = Field(default=0, ge=0)
    min_p: float = Field(default=0, ge=0, le=1)
    repetition_penalty: float = Field(default=1, gt=0, allow_inf_nan=False)
    presence_penalty: float = Field(default=0, allow_inf_nan=False)
    frequency_penalty: float = Field(default=0, allow_inf_nan=False)
    seed: int | None = None
    stop: str | list[str] | None = None
    stream: bool = False
    stream_options: StreamOptions = Field(default_factory=StreamOptions)
    n: int = Field(default=1, ge=1, le=1)

    @model_validator(mode="after")
    def validate_options(self):
        if (
            self.max_tokens is not None
            and self.max_completion_tokens is not None
            and self.max_tokens != self.max_completion_tokens
        ):
            raise ValueError("max_tokens and max_completion_tokens disagree")
        if len(self.stops) > 4 or any(not stop or len(stop) > 1024 for stop in self.stops):
            raise ValueError(
                "provide at most four nonempty stop strings of at most 1024 characters"
            )
        return self

    @property
    def output_limit(self) -> int:
        value = (
            self.max_completion_tokens
            if self.max_completion_tokens is not None
            else self.max_tokens
        )
        return 512 if value is None else value

    @property
    def stops(self) -> tuple[str, ...]:
        if self.stop is None:
            return ()
        return (self.stop,) if isinstance(self.stop, str) else tuple(self.stop)

    def sampling(self) -> SamplingPolicy:
        return SamplingPolicy(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            repetition_penalty=self.repetition_penalty,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            seed=self.seed,
        )
