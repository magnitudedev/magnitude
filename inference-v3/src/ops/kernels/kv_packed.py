"""Packet-owned KV persistence and tile-local reads of atomic cache planes."""
from __future__ import annotations

import tilelang.language as T

from ..kv import AffineKVCodec, DenseKVCodec, RotatedLloydMax
from ..kv_codecs import lloyd_max_centroids, rotation_signs


def validate_vector_codec(spec, subgroup_width):
    representation = spec.representation
    if (subgroup_width != 32 or representation.key_width % 8
            or representation.value_width != representation.key_width):
        raise ValueError("packed KV realization requires equal widths divisible by eight")
    for codec in (representation.key, representation.value):
        if isinstance(codec, AffineKVCodec) and codec.group_size:
            raise ValueError("packed KV realization requires whole-vector affine groups")
        if isinstance(codec, RotatedLloydMax) and representation.key_width < subgroup_width:
            raise ValueError("rotated KV realization requires at least one subgroup of coordinates")


def plane_view(storage, spec, name):
    plane = next(p for p in spec.representation.planes(spec.shape[0] * spec.shape[1])
                 if p.name == name)
    # Resolve the borrowed buffer's byte offset once at the pointer boundary.
    # Relative plane indexing then retains its natural integer width, including
    # vectorized staging. The full-width address still supports sliced storage.
    view = T.make_tensor(T.access_ptr(storage, "rw"),
                         (spec.storage_nbytes // plane.dtype.itemsize,), plane.dtype.value)
    return view, plane.offset // plane.dtype.itemsize, plane.row_elements


def word_offset(vector, word, row_words, capacity, heads, blocked):
    if blocked:
        return ((vector % heads) * (row_words // 4) + word // 4) * capacity * 4 + (vector // heads) * 4 + word % 4
    return vector * row_words + word


@T.macro
def _wht_local(values, scratch, count, step):
    for item in T.unroll(count):
        scratch[item] = T.if_then_else(item & step == 0,
                                       values[item] + values[item ^ step],
                                       values[item ^ step] - values[item])
    for item in T.unroll(count):
        values[item] = scratch[item]


@T.macro
def _wht_cross(values, scratch, count, lane, step, subgroup_width):
    for item in T.unroll(count):
        other = T.tvm_warp_shuffle(0xFFFFFFFF, values[item], lane ^ step,
                                   subgroup_width, subgroup_width)
        scratch[item] = T.if_then_else(lane & step == 0,
                                       values[item] + other, other - values[item])
    for item in T.unroll(count):
        values[item] = scratch[item]


def signed_wht(values, scratch, width, subgroup_width, lane, seed):
    count = width // subgroup_width
    signs = rotation_signs(width, seed)
    masks = tuple(sum((1 << owner) for owner in range(subgroup_width)
                      if signs[owner * count + item] > 0) for item in range(count))
    _apply_signs(values, count, lane, masks)
    step = 1
    while step < count:
        _wht_local(values, scratch, count, step)
        step *= 2
    step = 1
    while step < subgroup_width:
        _wht_cross(values, scratch, count, lane, step, subgroup_width)
        step *= 2


def _apply_signs(values, count, lane, masks):
    for item in range(count):
        _apply_sign_item(values, item, lane, masks[item])


@T.macro
def _apply_sign_item(values, item, lane, mask):
    values[item] *= T.cast((T.cast(mask, "uint32") >> lane) & 1, "float32") * 2.0 - 1.0


def _quantizer_table(centers):
    table = T.alloc_local((1,), "float32")
    # One threshold per lane. The binary search performs four subgroup
    # exchanges/comparisons, independent of the sixteen-entry codebook size.
    for boundary in range(15):
        _codebook_entry(table, boundary, (centers[boundary] + centers[boundary + 1]) * 0.5)
    _codebook_entry(table, 15, 0.0)
    return table


@T.macro
def _quantize_centroid(projected, code, table):
    code[0] = 0
    for step in T.unroll(4):
        candidate = code[0] + (8 >> step)
        threshold = T.tvm_warp_shuffle(0xFFFFFFFF, table[0], T.cast(candidate - 1, "int32"), 32, 32)
        code[0] = T.if_then_else(projected > threshold, candidate, code[0])


@T.macro
def _publish_word(packed, words, word_base, destination, row_words, count,
                  bits, packet, lane, capacity, heads, blocked):
    per_word = 32 // bits
    if count >= per_word:
        words[word_base + word_offset(destination, lane * (count // per_word) + packet,
                                      row_words, capacity, heads, blocked)] = packed[0]
    else:
        # Narrow heads share one word across adjacent lanes. All lanes take
        # part in the exchange; exactly one owns the final aligned word store.
        owners = per_word // count
        first_lane = lane // owners * owners
        joined = T.alloc_local((1,), "uint32")
        joined[0] = 0
        for owner in T.unroll(owners):
            contribution = T.tvm_warp_shuffle(0xFFFFFFFF, packed[0], first_lane + owner, 32, 32)
            joined[0] |= contribution << (owner * count * bits)
        if lane % owners == 0:
            words[word_base + word_offset(destination, lane // owners,
                                          row_words, capacity, heads, blocked)] = joined[0]


@T.macro
def _affine_store(vector, words, scales, zeros, word_base, scale_base, zero_base,
                  destination, row_words, count, bits, lane, metadata_dtype, capacity, heads, blocked, width):
    low = T.alloc_local((1,), "float32")
    high = T.alloc_local((1,), "float32")
    low[0] = float("inf")
    high[0] = -float("inf")
    for item in T.unroll(count):
        if lane * count + item < width:
            low[0] = T.min(low[0], vector[item])
            high[0] = T.max(high[0], vector[item])
    minimum = T.warp_reduce_min(low[0])
    zero = T.cast(minimum, metadata_dtype)
    scale = T.cast((T.warp_reduce_max(high[0]) - minimum) / ((1 << bits) - 1), metadata_dtype)
    inverse = T.if_then_else(scale > 0, 1.0 / T.cast(scale, "float32"), 0.0)
    if lane == 0:
        scales[scale_base + destination] = scale
        zeros[zero_base + destination] = zero
    per_word = 32 // bits
    if width == count * 32 and (count % per_word == 0 or per_word % count == 0):
        for packet in T.unroll(max(1, count // per_word)):
            packed = T.alloc_local((1,), "uint32")
            packed[0] = 0
            for item in T.unroll(min(count, per_word)):
                code = T.cast(T.min((1 << bits) - 1,
                                    T.max(0, T.round((vector[packet * per_word + item]
                                                     - T.cast(zero, "float32")) * inverse))), "uint32")
                packed[0] |= code << (item * bits)
            _publish_word(packed, words, word_base, destination, row_words, count,
                          bits, packet, lane, capacity, heads, blocked)
    else:
        # Quantization ownership is independent of packed-word ownership.
        # Gather coordinates across lane boundaries only for irregular groups.
        codes = T.alloc_local((count,), "uint32")
        for item in T.unroll(count):
            codes[item] = T.if_then_else(lane * count + item < width,
                T.cast(T.min((1 << bits) - 1, T.max(0, T.round(
                    (vector[item] - T.cast(zero, "float32")) * inverse))), "uint32"), T.uint32(0))
        for packet in T.unroll(T.ceildiv(row_words, 32)):
            word = packet * 32 + lane
            packed = T.alloc_local((1,), "uint32")
            packed[0] = 0
            for slot in T.unroll(per_word):
                coordinate = word * per_word + slot
                owner = T.min(coordinate // count, 31)
                for item in T.unroll(count):
                    code = T.tvm_warp_shuffle(0xFFFFFFFF, codes[item], owner, 32, 32)
                    packed[0] |= T.if_then_else(coordinate < width and coordinate % count == item,
                                               code << (slot * bits), T.uint32(0))
            if word < row_words:
                words[word_base + word_offset(destination, word, row_words, capacity, heads, blocked)] = packed[0]


@T.macro
def _rotated_store(vector, scratch, words, norms, word_base, norm_base,
                   destination, row_words, width, count, lane, seed, norm_dtype,
                   centers, subgroup_width, capacity, heads, blocked):
    table = _quantizer_table(centers)
    square = T.alloc_local((1,), "float32")
    square[0] = 0
    for item in T.unroll(count):
        square[0] += vector[item] * vector[item]
    norm = T.sqrt(T.warp_reduce_sum(square[0]))
    inverse = T.if_then_else(norm > 0, 1.0 / norm, 0.0)
    signed_wht(vector, scratch, width, subgroup_width, lane, seed)
    if lane == 0:
        norms[norm_base + destination] = T.cast(norm, norm_dtype)
    for packet in T.unroll(max(1, count // 8)):
        packed = T.alloc_local((1,), "uint32")
        packed[0] = 0
        for item in T.unroll(min(count, 8)):
            projected = vector[packet * 8 + item] * inverse
            code = T.alloc_local((1,), "uint32")
            code[0] = 0
            _quantize_centroid(projected, code, table)
            packed[0] |= code[0] << (item * 4)
        _publish_word(packed, words, word_base, destination, row_words, count,
                      4, packet, lane, capacity, heads, blocked)


def store_vector(vector, scratch, storage, spec, prefix, destination, lane, subgroup_width):
    representation = spec.representation
    codec = representation.key if prefix == "key" else representation.value
    width = representation.key_width if prefix == "key" else representation.value_width
    count = (width + subgroup_width - 1) // subgroup_width
    if isinstance(codec, DenseKVCodec):
        dense, base, row = plane_view(storage, spec, prefix + ".dense")
        _dense_store(vector, dense, base, destination, row, count, lane, width)
        return
    words, word_base, row_words = plane_view(storage, spec, prefix + ".codes")
    if isinstance(codec, AffineKVCodec):
        if codec.group_size:
            raise ValueError("subgroup KV persistence requires whole-vector affine groups")
        scales, scale_base, _ = plane_view(storage, spec, prefix + ".scale")
        zeros, zero_base, _ = plane_view(storage, spec, prefix + ".zero")
        _affine_store(vector, words, scales, zeros, word_base, scale_base, zero_base,
                      destination, row_words, count, codec.bits, lane, codec.scale_dtype.value,
                      spec.shape[0], spec.shape[1], representation.packing_version == 2, width)
    elif isinstance(codec, RotatedLloydMax):
        norms, norm_base, _ = plane_view(storage, spec, prefix + ".norm")
        _rotated_store(vector, scratch, words, norms, word_base, norm_base, destination,
                       row_words, width, count, lane, codec.sign_seed, codec.norm_dtype.value,
                       lloyd_max_centroids(width), subgroup_width,
                       spec.shape[0], spec.shape[1], representation.packing_version == 2)


@T.macro
def _dense_store(vector, dense, base, destination, row, count, lane, width):
    for item in T.unroll(count):
        if lane * count + item < width:
            dense[base + destination * row + lane * count + item] = vector[item]


@T.macro
def _copy_plane(storage, ranges, nbytes, offset, row_bytes, heads, copies, max_count, threads):
    raw = T.decl_buffer((nbytes,), "uint8", data=storage.data)
    elements = copies * max_count * heads * row_bytes
    with T.Kernel(T.ceildiv(elements, threads), threads=threads) as block:
        for lane in T.Parallel(threads):
            flat = block * threads + lane
            if flat < elements:
                copy = flat // (max_count * heads * row_bytes)
                position = flat // (heads * row_bytes) % max_count
                within = flat % (heads * row_bytes)
                if position < ranges[copy, 2]:
                    raw[offset + (ranges[copy, 1] + position) * heads * row_bytes + within] = raw[
                        offset + (ranges[copy, 0] + position) * heads * row_bytes + within]


def copy_bundle(storage, ranges, spec, range_spec, max_count, threads):
    for plane in spec.representation.planes(spec.shape[0] * spec.shape[1]):
        if spec.representation.packing_version == 2 and plane.name.endswith(".codes"):
            _copy_blocked_codes(storage, ranges, plane.offset // 4, plane.row_elements,
                                spec.shape[0], spec.shape[1], range_spec.shape[0], max_count, threads)
        else:
            _copy_plane(storage, ranges, spec.storage_nbytes, plane.offset,
                        plane.row_elements * plane.dtype.itemsize, spec.shape[1],
                        range_spec.shape[0], max_count, threads)


@T.macro
def _copy_blocked_codes(storage, ranges, offset, row_words, capacity, heads, copies, max_count, threads):
    elements = copies * max_count * heads * row_words
    with T.Kernel(T.ceildiv(elements, threads), threads=threads) as block:
        for lane in T.Parallel(threads):
            flat = block * threads + lane
            if flat < elements:
                copy = flat // (max_count * heads * row_words)
                word = flat % 4 + (flat // (max_count * 4) % (row_words // 4)) * 4
                head = flat // (max_count * row_words) % heads
                position = flat // 4 % max_count
                if position < ranges[copy, 2]:
                    source = (ranges[copy, 0] + position) * heads + head
                    target = (ranges[copy, 1] + position) * heads + head
                    storage[offset + word_offset(target, word, row_words, capacity, heads, True)] = storage[
                        offset + word_offset(source, word, row_words, capacity, heads, True)]


def _packet_layout(groups, tile, threads):
    return T.Fragment((groups, tile, 4),
                       forward_thread_fn=lambda group, item, word: ((group * tile + item) * 4 + word) % threads,
                       forward_index_fn=lambda group, item, word: ((group * tile + item) * 4 + word) // threads)


@T.macro
def _stage_blocked(words, coefficient, zero, output, word_base, coefficient_base, zero_base,
                    row_words, capacity, heads, head, base, first, count, tile, width,
                    bits, transpose, rotated, table, codes_only, first_channel=0, logical_width=0):
    packet_width = 32 // bits
    groups = (width + packet_width * 4 - 1) // (packet_width * 4)
    packets = T.alloc_fragment((groups, tile, 4), "uint32")
    T.annotate_layout({packets: _packet_layout(groups, tile, T.get_thread_extent())})
    for group, item, word in T.Parallel(groups, tile, 4):
        vector = (base + first + item) * heads + head
        packets[group, item, word] = T.if_then_else(
            first + item < count and (width % (packet_width * 4) == 0
                                     or first_channel // packet_width + group * 4 + word < row_words),
            words[word_base + word_offset(vector, first_channel // packet_width + group * 4 + word, row_words, capacity, heads, True)], T.uint32(0))
    for group, item, word in T.Parallel(groups, tile, 4):
        vector = (base + first + item) * heads + head
        factor = 1.0 if codes_only else T.if_then_else(first + item < count,
                                T.cast(coefficient[coefficient_base + vector], "float32"), 0.0)
        bias = 0.0 if codes_only else T.if_then_else(first + item < count and not rotated,
                              T.cast(zero[zero_base + vector], "float32"), 0.0)
        packed = packets[group, item, word]
        for element in T.unroll(packet_width):
            code = (packed >> (element * bits)) & ((1 << bits) - 1)
            decoded = T.alloc_local((1,), "float32")
            if rotated:
                centroid = lookup_centroid(code, table)
                decoded[0] = centroid if codes_only else centroid * factor * (logical_width if logical_width else width) ** -0.5
            else:
                decoded[0] = T.cast(code, "float32") * factor + bias
            channel = (group * 4 + word) * packet_width + element
            if channel < width:
                if transpose:
                    output[0, channel, item] = decoded[0]
                else:
                    output[0, item, channel] = decoded[0]


@T.macro
def _stage_affine(words, scales, zeros, output, word_base, scale_base, zero_base,
                   row_words, heads, head, base, first, count, tile, width, bits, transpose, codes_only, first_channel=0):
    packet_width = 32 // bits
    for item, packet in T.Parallel(tile, width // packet_width):
        vector = (base + first + item) * heads + head
        packed = T.if_then_else(first + item < count,
                                words[word_base + vector * row_words + first_channel // packet_width + packet], T.uint32(0))
        scale = 1.0 if codes_only else T.if_then_else(first + item < count,
                               T.cast(scales[scale_base + vector], "float32"), 0.0)
        zero = 0.0 if codes_only else T.if_then_else(first + item < count,
                              T.cast(zeros[zero_base + vector], "float32"), 0.0)
        for element in T.unroll(packet_width):
            value = T.cast((packed >> (element * bits)) & ((1 << bits) - 1), "float32") * scale + zero
            if transpose:
                output[0, packet * packet_width + element, item] = value
            else:
                output[0, item, packet * packet_width + element] = value


@T.macro
def _stage_rotated(words, norms, output, word_base, norm_base, row_words, heads,
                    head, base, first, count, tile, width, table, codes_only, transpose, first_channel=0, logical_width=0):
    for item, packet in T.Parallel(tile, width // 8):
        vector = (base + first + item) * heads + head
        packed = T.if_then_else(first + item < count,
                                words[word_base + vector * row_words + first_channel // 8 + packet], T.uint32(0))
        norm = 1.0 if codes_only else T.if_then_else(first + item < count,
                              T.cast(norms[norm_base + vector], "float32") * (logical_width if logical_width else width) ** -0.5, 0.0)
        for element in T.unroll(8):
            code = (packed >> (element * 4)) & 15
            centroid = lookup_centroid(code, table)
            if transpose:
                output[0, packet * 8 + element, item] = centroid * norm
            else:
                output[0, item, packet * 8 + element] = centroid * norm


def lookup_centroid(code, centers):
    """Select the exact symmetric scalar codebook without cross-lane gathers."""
    if not isinstance(centers, tuple):
        return centers[T.cast(code, 'int32')]
    assert len(centers) == 16 and all(centers[i] == -centers[15 - i] for i in range(8))
    code = T.cast(code, 'uint32')
    magnitude = (code ^ ((code >> 3) - T.uint32(1))) & T.uint32(7)
    values = tuple(T.float32(value) for value in centers[8:])
    for bit in range(3):
        values = tuple(T.Select((magnitude & T.uint32(1 << bit)) != 0,
                                values[index + 1], values[index])
                       for index in range(0, len(values), 2))
    sign = ((code & T.uint32(8)) ^ T.uint32(8)) << 28
    return T.reinterpret(T.reinterpret(values[0], 'uint32') ^ sign, 'float32')


@T.macro
def _codebook_entry(table, code, center):
    lane = T.get_thread_binding() % 16
    if lane == code:
        table[0] = center


@T.macro
def _store_centroid(table, index, value):
    if T.get_thread_binding() == index:
        table[index] = value


def _immutable_centroid_table(centers):
    # Static immutable storage has a distinct lifetime from the reusable
    # dynamic KV arena. Never merge its reads with that arena's write epochs.
    table = T.alloc_shared((16,), 'float32', scope='shared')
    for index, value in enumerate(centers):
        _store_centroid(table, index, value)
    T.sync_threads()
    return table


def prepare_codebook(spec, from_history):
    if from_history and isinstance(spec.representation.key, RotatedLloydMax):
        return _immutable_centroid_table(lloyd_max_centroids(spec.representation.key_width))
    return None


@T.macro
def _stage_dense_plane(dense, output, plane_base, heads, head, base, first, count,
                        tile, width, transpose, first_channel=0, logical_width=0):
    for item, channel in T.Parallel(tile, width):
        value = T.if_then_else(first + item < count,
                               dense[plane_base + ((base + first + item) * heads + head) * (logical_width if logical_width else width) + first_channel + channel], 0)
        if transpose:
            output[0, channel, item] = value
        else:
            output[0, item, channel] = value


def stage_history(storage, output, spec, prefix, head, base, first, count, tile, table=None, codes_only=False,
                  key_major=False, first_channel=0, tile_width=None):
    rep = spec.representation
    codec = rep.key if prefix == "key" else rep.value
    logical_width = rep.key_width if prefix == "key" else rep.value_width
    width = logical_width if tile_width is None else tile_width
    transpose = prefix == "key" and not key_major
    if isinstance(codec, DenseKVCodec):
        dense, offset, _ = plane_view(storage, spec, prefix + ".dense")
        _stage_dense_plane(dense, output, offset, spec.shape[1], head, base, first,
                            count, tile, width, transpose, first_channel, logical_width)
        return
    words, word_base, row_words = plane_view(storage, spec, prefix + ".codes")
    if rep.packing_version == 2:
        rotated = isinstance(codec, RotatedLloydMax)
        coefficient, coefficient_base, _ = plane_view(storage, spec, prefix + (".norm" if rotated else ".scale"))
        zero, zero_base, _ = (coefficient, coefficient_base, 1) if rotated else plane_view(storage, spec, prefix + ".zero")
        _stage_blocked(words, coefficient, zero, output, word_base, coefficient_base, zero_base,
                        row_words, spec.shape[0], spec.shape[1], head, base, first, count, tile, width,
                        codec.bits, transpose, rotated, table, codes_only, first_channel, logical_width)
        return
    if isinstance(codec, AffineKVCodec):
        scales, scale_base, _ = plane_view(storage, spec, prefix + ".scale")
        zeros, zero_base, _ = plane_view(storage, spec, prefix + ".zero")
        _stage_affine(words, scales, zeros, output, word_base, scale_base, zero_base,
                       row_words, spec.shape[1], head, base, first, count, tile, width,
                       codec.bits, transpose, codes_only, first_channel)
    else:
        norms, norm_base, _ = plane_view(storage, spec, prefix + ".norm")
        _stage_rotated(words, norms, output, word_base, norm_base, row_words, spec.shape[1],
                        head, base, first, count, tile, width, table, codes_only, transpose, first_channel, logical_width)


@T.macro
def stage_current(current, output, head, base, first, count, tile, width, transpose, first_channel=0):
    for item, channel in T.Parallel(tile, width):
        value = T.if_then_else(first + item < count, current[base + first + item, head, first_channel + channel], 0)
        if transpose:
            output[0, channel, item] = value
        else:
            output[0, item, channel] = value
