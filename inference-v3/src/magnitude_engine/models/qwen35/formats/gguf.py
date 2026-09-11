"""Read a GGUF directory into Qwen's architecture-owned weight roles.

This is the one place the Qwen architecture and the GGUF container meet. A new
container for this model is a new file beside this one.
"""

from pydantic import TypeAdapter

from magnitude_engine.models.qwen35.description import (
    AttentionWeights,
    BlockWeights,
    DenseDescription,
    Geometry,
    MixerKind,
    RecurrentWeights,
)
from magnitude_engine.weights.descriptor import WeightDescriptor
from magnitude_engine.weights.formats.gguf import Directory, GGUFFormat
from magnitude_engine.weights.identity import ArtifactIdentity


def describe(format: GGUFFormat) -> DenseDescription:
    return inspect_dense(format.directory, format.identity)


def inspect_dense(directory: Directory, identity: ArtifactIdentity) -> DenseDescription:
    if directory.value("general.architecture") != "qwen35":
        raise ValueError("dense Qwen binding requires the qwen35 GGUF architecture")
    metadata = {item.name: item.value for item in directory.metadata}
    if metadata.get("qwen35.rope.scaling.type", "none") != "none":
        raise ValueError("scaled Qwen rotary definitions are not yet qualified")

    def integer(key: str) -> int:
        value = directory.value("qwen35." + key)
        if type(value) is not int or value <= 0:
            raise ValueError(f"qwen35.{key} must be a positive integer")
        return value

    def number(key: str) -> float:
        value = directory.value("qwen35." + key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"qwen35.{key} must be numeric")
        return float(value)

    # A container may bundle the speculative multi-token head as trailing
    # blocks. Those are not model layers: this program does not run them, and
    # counting them would shift every layer's mixer kind.
    speculative = metadata.get("qwen35.nextn_predict_layers", 0)
    if type(speculative) is not int or not 0 <= speculative < integer("block_count"):
        raise ValueError("qwen35.nextn_predict_layers must be a count below the block count")
    layer_count = integer("block_count") - speculative
    flags = metadata.get("qwen35.attention.recurrent_layers")
    if flags is None:
        interval = integer("full_attention_interval")
        kinds = tuple(
            MixerKind.ATTENTION if (i + 1) % interval == 0 else MixerKind.RECURRENT
            for i in range(layer_count)
        )
    else:
        if (
            not isinstance(flags, tuple)
            or len(flags) != layer_count
            or any(type(v) is not bool for v in flags)
        ):
            raise ValueError("invalid per-layer recurrent flags")
        kinds = tuple(MixerKind.RECURRENT if flag else MixerKind.ATTENTION for flag in flags)
    embedding = directory.tensor("token_embd.weight")
    if len(embedding.shape) != 2:
        raise ValueError("Qwen embedding must be a matrix")
    vocabulary = metadata.get("tokenizer.ggml.tokens")
    if vocabulary is not None and (
        not isinstance(vocabulary, tuple) or len(vocabulary) != embedding.shape[0]
    ):
        raise ValueError("tokenizer vocabulary differs from the embedding table")
    sections = directory.value("qwen35.rope.dimension_sections")
    if (
        not isinstance(sections, tuple)
        or len(sections) != 4
        or any(type(n) is not int for n in sections)
    ):
        raise ValueError("Qwen rotary sections must contain four integers")
    geometry = Geometry(
        hidden=integer("embedding_length"),
        intermediate=integer("feed_forward_length"),
        vocabulary=embedding.shape[0],
        context_limit=integer("context_length"),
        layers=kinds,
        attention_heads=integer("attention.head_count"),
        kv_heads=integer("attention.head_count_kv"),
        attention_width=integer("attention.key_length"),
        rotary_width=integer("rope.dimension_count"),
        rotary_base=number("rope.freq_base"),
        rotary_sections=TypeAdapter(tuple[int, int, int, int]).validate_python(
            sections, strict=True
        ),
        epsilon=number("attention.layer_norm_rms_epsilon"),
        convolution_width=integer("ssm.conv_kernel"),
        recurrent_key_heads=integer("ssm.group_count"),
        recurrent_value_heads=integer("ssm.time_step_rank"),
        recurrent_width=integer("ssm.state_size"),
    )
    if integer("attention.value_length") != geometry.attention_width:
        raise ValueError("Qwen attention requires equal key/value head widths")
    if integer("ssm.inner_size") != geometry.recurrent_value_heads * geometry.recurrent_width:
        raise ValueError("Qwen recurrent inner size differs from head geometry")
    consumed: set[str] = set()

    def tensor(name: str, shape: tuple[int, ...]) -> WeightDescriptor:
        value = directory.tensor(name)
        if value.shape != shape:
            raise ValueError(f"Qwen weight {name}: expected {shape}, received {value.shape}")
        consumed.add(name)
        return WeightDescriptor(name=name, shape=shape)

    g = geometry
    embedding = tensor(embedding.name, (g.vocabulary, g.hidden))
    output_norm = tensor("output_norm.weight", (g.hidden,))
    output_name = (
        "output.weight"
        if any(t.name == "output.weight" for t in directory.tensors)
        else embedding.name
    )
    output = tensor(output_name, (g.vocabulary, g.hidden))
    blocks = []
    for i, kind in enumerate(kinds):
        prefix = f"blk.{i}."
        if kind == MixerKind.ATTENTION:
            mixer = AttentionWeights(
                query_gate=tensor(
                    prefix + "attn_q.weight", (2 * g.attention_heads * g.attention_width, g.hidden)
                ),
                key=tensor(prefix + "attn_k.weight", (g.kv_heads * g.attention_width, g.hidden)),
                value=tensor(prefix + "attn_v.weight", (g.kv_heads * g.attention_width, g.hidden)),
                query_norm=tensor(prefix + "attn_q_norm.weight", (g.attention_width,)),
                key_norm=tensor(prefix + "attn_k_norm.weight", (g.attention_width,)),
                output=tensor(
                    prefix + "attn_output.weight", (g.hidden, g.attention_heads * g.attention_width)
                ),
            )
        else:
            mixer = RecurrentWeights(
                query_key_value=tensor(
                    prefix + "attn_qkv.weight", (g.recurrent_channels, g.hidden)
                ),
                gate=tensor(
                    prefix + "attn_gate.weight",
                    (g.recurrent_value_heads * g.recurrent_width, g.hidden),
                ),
                alpha=tensor(prefix + "ssm_alpha.weight", (g.recurrent_value_heads, g.hidden)),
                beta=tensor(prefix + "ssm_beta.weight", (g.recurrent_value_heads, g.hidden)),
                convolution=tensor(
                    prefix + "ssm_conv1d.weight", (g.recurrent_channels, g.convolution_width)
                ),
                decay=tensor(prefix + "ssm_a", (g.recurrent_value_heads,)),
                time_bias=tensor(prefix + "ssm_dt.bias", (g.recurrent_value_heads,)),
                norm=tensor(prefix + "ssm_norm.weight", (g.recurrent_width,)),
                output=tensor(
                    prefix + "ssm_out.weight",
                    (g.hidden, g.recurrent_value_heads * g.recurrent_width),
                ),
            )
        blocks.append(
            BlockWeights(
                input_norm=tensor(prefix + "attn_norm.weight", (g.hidden,)),
                mixer=mixer,
                feedforward_norm=tensor(prefix + "post_attention_norm.weight", (g.hidden,)),
                feedforward_gate=tensor(prefix + "ffn_gate.weight", (g.intermediate, g.hidden)),
                feedforward_up=tensor(prefix + "ffn_up.weight", (g.intermediate, g.hidden)),
                feedforward_down=tensor(prefix + "ffn_down.weight", (g.hidden, g.intermediate)),
            )
        )
    speculative_blocks = tuple(f"blk.{layer_count + i}." for i in range(speculative))
    unused = {
        tensor.name
        for tensor in directory.tensors
        if tensor.name not in consumed
        and not tensor.name.startswith(speculative_blocks or ("\0",))
    }
    if unused:
        raise ValueError(f"dense Qwen artifact contains unbound weight roles: {sorted(unused)}")
    return DenseDescription(
        artifact_identity=identity,
        geometry=g,
        embedding=embedding,
        output_norm=output_norm,
        output=output,
        blocks=tuple(blocks),
    )
