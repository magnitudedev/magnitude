"""Ideal boundary traffic, not emitted load counts or observed DRAM traffic.

Only reads crossing the selected formula boundary and escaping writes count.
Internal intermediates can stay on chip. Ranges are unioned by alias backing;
selection and mutable state use concrete controls, never allocated capacity.
"""

import math


def boundary_accesses(graph, values, *, unresolved=None):
    from ..tensor.primitive import primitives
    from ..kv import KVRepresentation

    external = {graph.alias_root(value) for value in (*graph.inputs, *graph.constants, *graph.resources)}
    reads, writes = {}, {}
    unknown_writes = set()

    def record(table, identity, regions=None):
        spec = graph.value(identity).spec
        root = graph.alias_root(identity)
        if table is reads and root in unknown_writes:
            return
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
        # A symbolic binding retains the proven subset of unavoidable traffic.
        # Unresolved indexed accesses are explicit nonnegative obligations; they
        # cannot erase known reads elsewhere or be replaced by allocated capacity.
        controls = {"embedding": (0,), "take_rows": (1,), "routed_experts": (1,),
                    "causal_attention": (2,) if len(node.inputs) == 3 else (),
                    "persistent_attention": (4,), "kv_append": (3,), "kv_copy": (1,),
                    "byte_copy": (2,)}.get(node.operation, ())
        missing = [index for index in controls if node.inputs[index] not in values]
        if missing and unresolved is not None:
            accessed = {"embedding": (1,), "take_rows": (0,), "routed_experts": (3, 4, 5),
                        "causal_attention": (1,), "persistent_attention": (1, 2, 3),
                        "kv_append": (0, 1, 2), "kv_copy": (0,), "byte_copy": (0, 1)}[node.operation]
            selected.update((index, ()) for index in accessed)
            for index in primitive.resource_writes:
                mutated[index] = ()
                unknown_writes.add(graph.alias_root(node.inputs[index]))
            unresolved.extend(f"{node.operation} indexed traffic requires value {node.inputs[index]}" for index in missing)
        elif node.operation in {"embedding", "take_rows"}:
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
        elif node.operation == "persistent_attention":
            visible = concrete(node, 4)
            stride = math.prod(specs[1].shape[1:])
            selected[1] = tuple((int(start) * stride, (int(start) + int(count)) * stride)
                                for row in visible for start, count in zip(row[:-2:2], row[1:-2:2], strict=True))
            for index in (2, 3):
                stride = math.prod(specs[index].shape[1:])
                selected[index] = tuple((int(start) * stride, (int(start) + int(count)) * stride)
                                        for row in visible for start, count in (row[-2:],))
        elif node.operation == "kv_append":
            destinations = concrete(node, 3)
            axis = 0 if isinstance(specs[0].representation, KVRepresentation) else 1
            mutated[0] = rows(specs[0], destinations[destinations >= 0], axis=axis)
            active = tuple(index for index, destination in enumerate(destinations) if destination >= 0)
            selected[1], selected[2] = rows(specs[1], active), rows(specs[2], active)
            omitted.add(0)
        elif node.operation == "kv_copy":
            ranges = concrete(node, 1)
            typed = isinstance(specs[0].representation, KVRepresentation)
            axis = 0 if typed else 1
            stride = math.prod(specs[0].shape[axis + 1:])
            capacity = specs[0].shape[axis]
            selected[0] = tuple(((plane * capacity + int(source)) * stride,
                                  (plane * capacity + int(source) + int(count)) * stride)
                                 for plane in range(1 if typed else 2) for source, _, count in ranges if count > 0)
            mutated[0] = tuple(((plane * capacity + int(destination)) * stride,
                                 (plane * capacity + int(destination) + int(count)) * stride)
                                for plane in range(1 if typed else 2) for _, destination, count in ranges if count > 0)
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


def boundary_traffic_bound(graph, values):
    """Certified lower demand plus the still-unbound indexed demand parameters."""
    unresolved = []
    reads, writes = boundary_accesses(graph, values, unresolved=unresolved)
    amount = sum(unique_bytes(regions) for table in (reads, writes) for regions in table.values())
    return amount, tuple(unresolved)
