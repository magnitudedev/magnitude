"""Capacity-only rendering, executed in the selected engine's frozen environment."""

import json
import sys
from copy import deepcopy
from pathlib import Path


def main():
    kind, artifact, source, output = sys.argv[1:]
    requests = [json.loads(line) for line in Path(source).read_text().splitlines()]
    if kind == "magnitude":
        from magnitude_engine.inputs.formats.gguf_tokenizer import TokenizerArtifact
        from magnitude_engine.serving.template import ChatTemplate

        template = ChatTemplate(TokenizerArtifact.load(Path(artifact)))

        def count(request):
            return len(
                template.render(
                    request["messages"],
                    tools=request["tools"],
                    tool_choice="required" if request["tools"] else "auto",
                    chat_template_kwargs={"enable_thinking": False},
                ).tokens
            )
    else:
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.server.openai import _prepare_chat_tool_choice
        from mlx_vlm.utils import load_config, load_processor, prepare_inputs

        processor = load_processor(Path(artifact), local_files_only=True, trust_remote_code=False)
        config = load_config(Path(artifact))

        def count(request):
            messages, tools, choice = _prepare_chat_tool_choice(
                deepcopy(request["messages"]),
                request["tools"] or None,
                "required" if request["tools"] else None,
            )
            # The stock HTTP route decodes tool arguments before rendering history.
            for message in messages:
                for call in message.get("tool_calls", []):
                    value = call["function"]["arguments"]
                    if isinstance(value, str):
                        call["function"]["arguments"] = json.loads(value)
            prompt = apply_chat_template(
                processor,
                config,
                messages,
                num_images=0,
                num_audios=0,
                tools=tools,
                tool_choice=choice,
                enable_thinking=False,
            )
            special = (
                getattr(processor, "chat_template", None) is None
                if config.get("model_type") in ("gemma3", "gemma3n", "gemma4", "gemma4_unified")
                else True
            )
            inputs = prepare_inputs(
                processor,
                prompts=prompt,
                add_special_tokens=special,
                image_token_index=config.get("image_token_index"),
            )
            return int(inputs["input_ids"].size)

    Path(output).write_text(json.dumps({r["id"]: count(r) for r in requests}))


if __name__ == "__main__":
    main()
