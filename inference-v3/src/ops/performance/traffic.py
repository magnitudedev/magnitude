"""Ideal boundary traffic, not emitted load counts or observed DRAM traffic.

Only reads crossing the selected formula boundary and escaping writes count.
Internal intermediates can stay on chip. Ranges are unioned by alias backing;
selection and mutable state use concrete controls, never allocated capacity.
"""

import math


def boundary_accesses(graph, values):
    from ..tensor.primitive import primitives

    external = {graph.alias_root(value) for value in (*graph.inputs, *graph.constants, *graph.resources)}
    reads, writes = {}, {}

    def record(table, identity, regions=None):
        spec = graph.value(identity).spec
        root = graph.alias_root(identity)
        if regions is None:
            regions = ((0, spec.elements),)
        # Fractional byte addressing is an optimistic bit-density convention for
        # packed planes. Alignment, group rereads and decode are amplification.
        density = spec.storage_nbytes / spec.elements if spec.elements else 0
        target = table.setdefault(root, [])
        for start, end in regions:
            if not 0 <= start <= end <= spec.elements:
                raise ValueError("semantic memory access lies outside its tensor")
            parts = [(start * density, end * density)]
            if table is reads:
                # State produced earlier inside this boundary need not be read
                # back from external memory: the fused schedule may retain it.
                for written_start, written_end in writes.get(root, ()):
                    remaining = []
                    for first, last in parts:
                        if written_end <= first or written_start >= last:
                            remaining.append((first, last))
                        else:
                            if first < written_start:
                                remaining.append((first, written_start))
                            if last > written_end:
                                remaining.append((written_end, last))
                    parts = remaining
            target.extend(parts)

    def concrete(node, index):
        value = values.get(node.inputs[index])
        if value is None:
            raise ValueError(f"{node.operation} memory model requires input {index}")
        return value

    def rows(spec, indices, axis=0):
        stride = math.prod(spec.shape[axis + 1:])
        block = spec.shape[axis] * stride
        return tuple((prefix * block + int(row) * stride, prefix * block + (int(row) + 1) * stride)
                     for prefix in range(math.prod(spec.shape[:axis])) for row in set(map(int, indices)))

    for node in graph.nodes:
        primitive = primitives.get(node.operation)
        if node.operation in {"reshape", "scalar"}:
            continue
        selected, mutated, omitted = {}, {}, set()
        specs = tuple(graph.value(value).spec for value in node.inputs)
        if node.operation in {"embedding", "take_rows"}:
            table, indices = (1, 0) if node.operation == "embedding" else (0, 1)
            selected[table] = rows(specs[table], concrete(node, indices).flat)
        elif node.operation == "routed_experts":
            for index in (3, 4, 5):
                selected[index] = rows(specs[index], concrete(node, 1).flat)
        elif node.operation == "causal_attention":
            capacity = specs[1].shape[1]
            if len(node.inputs) == 3:
                visible = concrete(node, 2)
                ranges = visible if visible.ndim == 2 else tuple((0, int(count)) for count in visible)
            else:
                ranges = ((0, capacity),)
            stride = math.prod(specs[1].shape[2:])
            selected[1] = tuple(((plane * capacity + int(start)) * stride,
                                  (plane * capacity + int(start) + int(count)) * stride)
                                 for plane in range(2) for start, count in ranges)
        elif node.operation == "kv_append":
            destinations = concrete(node, 3)
            mutated[0] = rows(specs[0], destinations[destinations >= 0], axis=1)
            active = tuple(index for index, destination in enumerate(destinations) if destination >= 0)
            selected[1], selected[2] = rows(specs[1], active), rows(specs[2], active)
            omitted.add(0)
        elif node.operation == "kv_copy":
            ranges = concrete(node, 1)
            stride = math.prod(specs[0].shape[2:])
            capacity = specs[0].shape[1]
            selected[0] = tuple(((plane * capacity + int(source)) * stride,
                                  (plane * capacity + int(source) + int(count)) * stride)
                                 for plane in range(2) for source, _, count in ranges if count > 0)
            mutated[0] = tuple(((plane * capacity + int(destination)) * stride,
                                 (plane * capacity + int(destination) + int(count)) * stride)
                                for plane in range(2) for _, destination, count in ranges if count > 0)
        elif node.operation == "byte_copy":
            offset, count = map(int, concrete(node, 2))
            selected[0] = ((0, count),)
            mutated[1] = ((offset, offset + count),)
            omitted.add(1)
        elif node.operation == "quantized_import":
            raise ValueError("quantized_import needs a codec-specific accessed-region contract")
        for index, identity in enumerate(node.inputs):
            if index not in omitted and graph.alias_root(identity) in external:
                record(reads, identity, selected.get(index))
        for index in primitive.resource_writes:
            record(writes, node.inputs[index], mutated.get(index))
    for identity in graph.outputs:
        if graph.alias_root(identity) not in external:
            record(writes, identity)

    return reads, writes


def unique_bytes(regions):
    total, previous = 0, 0
    for start, end in sorted(regions):
        total += max(0, end - max(previous, start))
        previous = max(previous, end)
    return total


def boundary_traffic(graph, values):
    reads, writes = boundary_accesses(graph, values)
    return sum(unique_bytes(regions) for table in (reads, writes) for regions in table.values())
