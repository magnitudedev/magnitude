//! Exact packet decoding expressed in the existing execution IR. Each owned
//! subpacket reads its code words and coefficients once, then publishes
//! decoded elements using the representation's FMA semantics. No contraction
//! algebra or accumulation order changes here.
use super::*;
use crate::{ast::BinaryOp, repr};

pub(crate) fn supported(ty: &Ty) -> bool {
    let Some(shape) = ty.shaped() else {
        return false;
    };
    let Elem::Repr(name) = &shape.elem else {
        return false;
    };
    let Some(r) = repr::lookup(name) else {
        return false;
    };
    shape.packed_axis == shape.shape.len().checked_sub(1)
        && !shape.shape.is_empty()
        && shape
            .shape
            .iter()
            .all(|n| n.as_constant().is_none_or(|n| n >= 0))
        && shape
            .shape
            .last()
            .unwrap()
            .as_constant()
            .is_none_or(|n| n >= i64::from(r.group))
        && (r.group * r.bits).is_multiple_of(32)
}

/// Alignment follows captured value/view provenance, not the selected storage
/// mode. Materialized packed snapshots retain their original packet prefix.
pub(crate) fn aligned(e: &Expr, vars: &[Var], known: &HashSet<VarId>) -> bool {
    let Some(shape) = e.ty.shaped() else {
        return false;
    };
    let Elem::Repr(name) = &shape.elem else {
        return false;
    };
    let Some(r) = repr::lookup(name) else {
        return false;
    };
    if shape.packed_axis != shape.shape.len().checked_sub(1) || shape.shape.is_empty() {
        return false;
    }
    let group = i64::from(r.group);
    match &e.kind {
        ExprKind::Var(id) => {
            known.contains(id)
                || (matches!(vars[*id].kind, VarKind::Param(_))
                    && shape.shape.last().unwrap().div_exact(group).is_some())
        }
        ExprKind::Load { view, .. } => aligned(view, vars, known),
        ExprKind::Builtin {
            name: Builtin::Load,
            args,
        } => args.first().is_some_and(|view| aligned(view, vars, known)),
        ExprKind::Index { base, indices } => {
            if !aligned(base, vars, known) {
                return false;
            }
            let Some(base_shape) = base.ty.shaped() else {
                return false;
            };
            match indices.get(base_shape.shape.len() - 1) {
                None => true,
                Some(Index::Slice { start, .. }) => start.as_ref().is_none_or(|x| {
                    let Some(start) = &x.sym else { return false };
                    let width = base_shape.shape.last().unwrap();
                    // Clamping an aligned endpoint to an unaligned row end can
                    // destroy alignment. Prove either the clamp boundary or the
                    // in-bounds endpoint; the slice end need not be aligned.
                    start.div_exact(group).is_some()
                        && (width.div_exact(group).is_some()
                            || start
                                .as_constant()
                                .zip(width.as_constant())
                                .is_some_and(|(s, w)| s >= 0 && s <= w))
                }),
                Some(Index::Point(_)) => false,
            }
        }
        _ => false,
    }
}

pub(super) fn stream_aligned(
    view: &Expr,
    axis: usize,
    capacity: Option<i64>,
    vars: &[Var],
    known: &HashSet<VarId>,
) -> bool {
    if !aligned(view, vars, known) {
        return false;
    }
    let shape = view.ty.shaped().unwrap();
    if Some(axis) != shape.packed_axis {
        return true;
    }
    let Elem::Repr(name) = &shape.elem else {
        return false;
    };
    let group = i64::from(repr::lookup(name).unwrap().group);
    // No capacity means one unsplit invocation of the captured view.
    capacity.is_none_or(|n| n > 0 && n % group == 0)
}

pub(crate) fn decode(
    source: &Expr,
    codes_per_owner: u32,
    decoder: repr::PacketDecoder,
    vars: &mut Vec<Var>,
) -> Result<(Expr, Vec<Stmt>), String> {
    let length = source
        .ty
        .shaped()
        .ok_or("packet input must be shaped")?
        .shape
        .last()
        .ok_or("packet input needs an axis")?
        .clone();
    decode_region(
        source,
        Sym::constant(0),
        length,
        codes_per_owner,
        decoder,
        None,
        None,
        vars,
    )
}

pub(crate) fn decode_segment(
    source: &Expr,
    start: Sym,
    length: i64,
    width: u32,
    decoder: repr::PacketDecoder,
    coefficients: Option<&Coefficients>,
    words: Option<&Words>,
    vars: &mut Vec<Var>,
) -> Result<(Expr, Vec<Stmt>), String> {
    decode_region(
        source,
        start,
        Sym::constant(length),
        width,
        decoder,
        coefficients,
        words,
        vars,
    )
}

/// Parameterized packet decode used by retained source families. The owner
/// width remains a symbolic source numeric parameter; ownership is expressed
/// through ordinary owner/local arithmetic over the dense output tile. This
/// keeps specialized and indexed decoder arms distinct without enumerating
/// numeric assignments.
pub(crate) fn decode_segment_parameterized(
    source: &Expr,
    start: Sym,
    length: Sym,
    width: Sym,
    decoder: repr::PacketDecoder,
    coefficients: Option<&Coefficients>,
    words: Option<&Words>,
    vars: &mut Vec<Var>,
) -> Result<(Expr, Vec<Stmt>), String> {
    if let Some(width) = width.as_constant().and_then(|n| u32::try_from(n).ok()) {
        return decode_region(source, start, length, width, decoder, coefficients, words, vars);
    }
    decode_region_parameterized(source, start, length, width, decoder, coefficients, words, vars)
}

/// Encoded words are retained without decoding or changing their bit pattern.
/// Word-aligned segment boundaries make the captured payload exact: no extra
/// word is read before or after the source segment.
pub(crate) struct Words {
    values: Expr,
    origin: Sym,
}

pub(super) fn retain_words(ty: &Ty, length: i64) -> bool {
    if !supported(ty) || length <= 0 {
        return false;
    }
    let Elem::Repr(name) = &ty.shaped().unwrap().elem else {
        return false;
    };
    let r = repr::lookup(name).unwrap();
    length
        .checked_mul(i64::from(r.bits))
        .is_some_and(|bits| bits % 32 == 0)
}

pub(crate) fn prepare_words(
    source: &Expr,
    start: Sym,
    length: i64,
    vars: &mut Vec<Var>,
) -> Result<(Words, Vec<Stmt>), String> {
    if !retain_words(&source.ty, length) {
        return Err("retained words require complete word-aligned segments".into());
    }
    let source_shape = source.ty.shaped().unwrap();
    let Elem::Repr(name) = &source_shape.elem else {
        unreachable!()
    };
    let r = repr::lookup(name).unwrap();
    let origin = start
        .scale(i64::from(r.bits))
        .div_exact(32)
        .ok_or("retained words require a word-aligned origin")?;
    let count = length * i64::from(r.bits) / 32;
    let mut shape = source_shape.shape.clone();
    *shape.last_mut().unwrap() = Sym::constant(count);
    let ty = Ty::Tile(Shaped::new(shape.clone(), Elem::Dtype(DType::U32)));
    let mut b = Builder {
        vars,
        span: source.span,
    };
    let values = b.local("retained_words", ty.clone(), false);
    let coordinates = shape
        .iter()
        .map(|_| b.local("word_row", Ty::Scalar(DType::I32), true))
        .collect::<Vec<_>>();
    let word = b.local("word_index", Ty::Scalar(DType::I32), true);
    let mut source_at = coordinates.clone();
    *source_at.last_mut().unwrap() = b.binary(
        BinaryOp::Add,
        word.clone(),
        b.symbol(origin.clone()),
        DType::I32,
    );
    let mut cache_at = coordinates.clone();
    *cache_at.last_mut().unwrap() = word.clone();
    let read = b.accessor(source, r, "words", source_at)?;
    let copy = b.assign(b.element(values.clone(), cache_at, DType::U32), read);
    let ExprKind::Var(word_id) = word.kind else {
        unreachable!()
    };
    let row = b.statement(StmtKind::Range {
        var: word_id,
        lo: Sym::constant(0),
        hi: Sym::constant(count),
        body: vec![copy],
    });
    // One owner per row acquires its complete contiguous word span. This is
    // an ordinary view of the actual allocation, not a synthetic ownership tile.
    let mut domain_shape = shape.clone();
    *domain_shape.last_mut().unwrap() = Sym::constant(1);
    let mut indices = shape
        .iter()
        .map(|_| Index::Slice {
            start: None,
            end: None,
        })
        .collect::<Vec<_>>();
    *indices.last_mut().unwrap() = Index::Slice {
        start: None,
        end: Some(b.int(1)),
    };
    let domain = b.expr(
        ExprKind::Index {
            base: Box::new(values.clone()),
            indices,
        },
        Ty::Tile(Shaped::new(domain_shape, Elem::Dtype(DType::U32))),
        None,
    );
    let allocation = b.expr(
        ExprKind::TileAlloc {
            shape,
            dtype: Elem::Dtype(DType::U32),
        },
        ty,
        None,
    );
    let producer = vec![
        b.assign(values.clone(), allocation),
        b.statement(StmtKind::Owned {
            vars: coordinates
                .iter()
                .map(|coordinate| match coordinate.kind {
                    ExprKind::Var(id) => id,
                    _ => unreachable!(),
                })
                .collect(),
            tile: domain,
            body: vec![row],
        }),
    ];
    Ok((Words { values, origin }, producer))
}

/// Exact F32 coefficient values retained independently of decoded code windows.
/// These are ordinary private/shared tiles whose ownership is decided downstream.
pub(crate) struct Coefficients {
    scale: Expr,
    bias: Option<Expr>,
    origin: Sym,
}
pub(crate) fn prepare_coefficients(
    source: &Expr,
    start: Sym,
    length: i64,
    vars: &mut Vec<Var>,
) -> Result<(Coefficients, Vec<Stmt>), String> {
    if !supported(&source.ty) {
        return Err("unsupported retained coefficient input".into());
    }
    let shape = source.ty.shaped().unwrap();
    let Elem::Repr(name) = &shape.elem else {
        unreachable!()
    };
    let r = repr::lookup(name).unwrap();
    let group = i64::from(r.group);
    let alignment = length.min(group);
    if length <= 0
        || (length % group != 0 && group % length != 0)
        || start.div_exact(alignment).is_none()
    {
        return Err("retained coefficients require complete aligned segments".into());
    }
    let origin = start.quot(&Sym::constant(group));
    let mut shape = shape.shape.clone();
    *shape.last_mut().unwrap() = Sym::constant(length / group + i64::from(length % group != 0));
    let ty = Ty::Tile(Shaped::new(shape.clone(), Elem::Dtype(DType::F32)));
    let mut b = Builder {
        vars,
        span: source.span,
    };
    let scale = b.local("retained_scale", ty.clone(), false);
    let bias = r
        .has_bias()
        .then(|| b.local("retained_bias", ty.clone(), false));
    let coordinates = shape
        .iter()
        .map(|_| b.local("coefficient_index", Ty::Scalar(DType::I32), true))
        .collect::<Vec<_>>();
    let mut source_at = coordinates.clone();
    *source_at.last_mut().unwrap() = b.binary(
        BinaryOp::Add,
        source_at.last().unwrap().clone(),
        b.symbol(origin.clone()),
        DType::I32,
    );
    let mut producer = Vec::new();
    let mut body = Vec::new();
    for (is_bias, cache) in std::iter::once((false, &scale))
        .chain(bias.as_ref().map(|bias| (true, bias)))
    {
        let allocation = b.expr(
            ExprKind::TileAlloc {
                shape: shape.clone(),
                dtype: Elem::Dtype(DType::F32),
            },
            ty.clone(),
            None,
        );
        producer.push(b.assign(cache.clone(), allocation));
        let read = b.coefficient(source, r, is_bias, source_at.clone(), &mut body)?;
        body.push(b.assign(
            b.element(cache.clone(), coordinates.clone(), DType::F32),
            read,
        ));
    }
    producer.push(
        b.statement(StmtKind::Owned {
            vars: coordinates
                .iter()
                .map(|e| match e.kind {
                    ExprKind::Var(id) => id,
                    _ => unreachable!(),
                })
                .collect(),
            tile: scale.clone(),
            body,
        }),
    );
    Ok((
        Coefficients {
            scale,
            bias,
            origin,
        },
        producer,
    ))
}

/// The input keeps its captured physical origin. Destination coordinates are
/// relative to the prepared region, while packet accessors address the original
/// snapshot directly; no sliced physical accessor or copied packed alias exists.
fn decode_region(
    source: &Expr,
    offset: Sym,
    length: Sym,
    codes_per_owner: u32,
    decoder: repr::PacketDecoder,
    coefficients: Option<&Coefficients>,
    words: Option<&Words>,
    vars: &mut Vec<Var>,
) -> Result<(Expr, Vec<Stmt>), String> {
    if !supported(&source.ty) {
        return Err("packet decode requires an innermost word-aligned coefficient group".into());
    }
    let shape = source.ty.shaped().unwrap();
    let Elem::Repr(name) = &shape.elem else {
        unreachable!()
    };
    let r = repr::lookup(name).unwrap();
    if codes_per_owner == 0 || codes_per_owner > r.group {
        return Err("invalid packet owner width".into());
    }
    let group = i64::from(r.group);
    let subsegment = length.as_constant().filter(|&n| n > 0 && n < group);
    let subwidth = subsegment.unwrap_or(group);
    if group % subwidth != 0 || offset.div_exact(subwidth).is_none() {
        return Err("packet region does not preserve selected subpacket alignment".into());
    }
    let codes_per_owner = codes_per_owner.min(subwidth as u32);
    let parts = (subwidth as u32).div_ceil(codes_per_owner);
    // A word-aligned segment has the same extraction pattern at every group
    // position. Move that position into its word address instead of cloning
    // the decoder behind a branch for every possible segment origin.
    let word_aligned_segment = subsegment.is_some() && (subwidth * i64::from(r.bits)) % 32 == 0;
    let span = source.span;
    let mut b = Builder { vars, span };
    let mut cache_type = dense_type(&source.ty);
    let Ty::Tile(cache_shape) = &mut cache_type else {
        unreachable!()
    };
    *cache_shape.shape.last_mut().unwrap() = length.clone();
    let cache_shape = cache_shape.shape.clone();
    let cache = b.local("decoded_packets", cache_type, false);
    let mut packet_shape = cache_shape.clone();
    let width = length;
    let groups = width.quot(&Sym::constant(group));
    let owners = if subsegment.is_some() {
        Sym::constant(i64::from(parts))
    } else {
        groups.scale(i64::from(parts))
    };
    *packet_shape.last_mut().unwrap() = owners.clone();
    let coordinates = packet_shape
        .iter()
        .map(|_| b.local("packet_index", Ty::Scalar(DType::I32), true))
        .collect::<Vec<_>>();
    let owner = coordinates.last().unwrap().clone();
    let local_packet = if parts == 1 {
        owner.clone()
    } else {
        b.binary(
            BinaryOp::Div,
            owner.clone(),
            b.int(i64::from(parts)),
            DType::I32,
        )
    };
    let packet_offset = offset.quot(&Sym::constant(group));
    let packet = if subsegment.is_some() {
        b.symbol(packet_offset)
    } else {
        b.binary(
            BinaryOp::Add,
            local_packet.clone(),
            b.symbol(packet_offset),
            DType::I32,
        )
    };
    let part = if word_aligned_segment {
        owner
    } else if subsegment.is_some() {
        b.binary(
            BinaryOp::Add,
            b.symbol(
                offset
                    .rem(&Sym::constant(group))
                    .quot(&Sym::constant(subwidth))
                    .scale(i64::from(parts)),
            ),
            owner,
            DType::I32,
        )
    } else {
        b.binary(BinaryOp::Rem, owner, b.int(i64::from(parts)), DType::I32)
    };
    let mut body = Vec::new();

    let mut coefficient_at = coordinates.clone();
    *coefficient_at.last_mut().unwrap() = packet.clone();
    let scale_read = if let Some(coefficients) = coefficients {
        let mut at = coefficient_at.clone();
        *at.last_mut().unwrap() = b.binary(
            BinaryOp::Sub,
            packet.clone(),
            b.symbol(coefficients.origin.clone()),
            DType::I32,
        );
        b.element(coefficients.scale.clone(), at, DType::F32)
    } else {
        b.coefficient(source, r, false, coefficient_at.clone(), &mut body)?
    };
    let scale = b.bind("packet_scale", scale_read, &mut body);
    let bias = if r.has_bias() {
        let read = if let Some(coefficients) = coefficients {
            let mut at = coefficient_at.clone();
            *at.last_mut().unwrap() = b.binary(
                BinaryOp::Sub,
                packet.clone(),
                b.symbol(coefficients.origin.clone()),
                DType::I32,
            );
            b.element(
                coefficients.bias.as_ref().ok_or("missing retained packet bias")?.clone(),
                at,
                DType::F32,
            )
        } else {
            b.coefficient(source, r, true, coefficient_at.clone(), &mut body)?
        };
        b.bind("packet_bias", read, &mut body)
    } else {
        b.float(0.0)
    };
    let word_count = r.group * r.bits / 32;
    let word_offset = if word_aligned_segment {
        offset
            .rem(&Sym::constant(group))
            .scale(i64::from(r.bits))
            .quot(&Sym::constant(32))
    } else {
        Sym::constant(0)
    };
    if decoder == repr::PacketDecoder::Specialized {
        let mut chunks = Vec::new();
        let covered = if word_aligned_segment {
            subwidth as u32
        } else {
            r.group
        };
        let chunks_at = (0..covered).step_by(subwidth as usize).flat_map(|base| {
            (base..base + subwidth as u32)
                .step_by(codes_per_owner as usize)
                .map(move |first| (first, (first + codes_per_owner).min(base + subwidth as u32)))
        });
        for (first, last) in chunks_at {
            let first_word = first * r.bits / 32;
            let last_word = (last * r.bits).div_ceil(32);
            let mut body = Vec::new();
            let mut decoded_words = Vec::new();
            for word in first_word..last_word {
                let mut at = coordinates.clone();
                *at.last_mut().unwrap() = b.binary(
                    BinaryOp::Add,
                    b.binary(
                        BinaryOp::Mul,
                        packet.clone(),
                        b.int(i64::from(word_count)),
                        DType::I32,
                    ),
                    b.symbol(word_offset.add(&Sym::constant(i64::from(word)))),
                    DType::I32,
                );
                let read = if let Some(words) = words {
                    *at.last_mut().unwrap() = b.binary(
                        BinaryOp::Sub,
                        at.last().unwrap().clone(),
                        b.symbol(words.origin.clone()),
                        DType::I32,
                    );
                    b.element(words.values.clone(), at, DType::U32)
                } else {
                    b.accessor(source, r, "words", at)?
                };
                decoded_words.push(b.bind("packet_word", read, &mut body));
            }
            for code in first..last {
                let bit = code * r.bits;
                let word = (bit / 32 - first_word) as usize;
                let shift = bit % 32;
                let mut raw = b.binary(
                    BinaryOp::Shr,
                    decoded_words[word].clone(),
                    b.uint(i64::from(shift)),
                    DType::U32,
                );
                if shift + r.bits > 32 {
                    raw = b.binary(
                        BinaryOp::BitOr,
                        raw,
                        b.binary(
                            BinaryOp::Shl,
                            decoded_words[word + 1].clone(),
                            b.uint(i64::from(32 - shift)),
                            DType::U32,
                        ),
                        DType::U32,
                    );
                }
                raw = b.binary(
                    BinaryOp::BitAnd,
                    raw,
                    b.uint(i64::from(u32::MAX >> (32 - r.bits))),
                    DType::U32,
                );
                let decoded = b.decode_code(raw, r, &mut body)?;
                let value = b.expr(
                    ExprKind::Builtin {
                        name: Builtin::Fma,
                        args: vec![decoded, scale.clone(), bias.clone()],
                    },
                    Ty::Scalar(DType::F32),
                    None,
                );
                let mut at = coordinates.clone();
                *at.last_mut().unwrap() = b.binary(
                    BinaryOp::Add,
                    if subsegment.is_some() {
                        b.int(0)
                    } else {
                        b.binary(
                            BinaryOp::Mul,
                            local_packet.clone(),
                            b.int(group),
                            DType::I32,
                        )
                    },
                    b.int(i64::from(code) % subwidth),
                    DType::I32,
                );
                body.push(b.assign(b.element(cache.clone(), at, DType::F32), value));
            }
            chunks.push(body);
        }
        body.extend(b.dispatch(&part, 0, chunks));
    } else {
        // One bounded element loop replaces the specialized owner's branch
        // family. Only a code that crosses a word boundary reads the next word.
        let code = b.local("packet_code", Ty::Scalar(DType::I32), true);
        let ExprKind::Var(code_id) = code.kind else {
            unreachable!()
        };
        let first = b.binary(
            BinaryOp::Mul,
            b.binary(
                BinaryOp::Rem,
                coordinates.last().unwrap().clone(),
                b.int(i64::from(parts)),
                DType::I32,
            ),
            b.int(i64::from(codes_per_owner)),
            DType::I32,
        );
        let local_code = b.binary(BinaryOp::Add, first, code, DType::I32);
        let group_code = if subsegment.is_some() {
            b.binary(
                BinaryOp::Add,
                b.symbol(offset.rem(&Sym::constant(group))),
                local_code.clone(),
                DType::I32,
            )
        } else {
            local_code.clone()
        };
        let mut visit = Vec::new();
        let bit = b.bind(
            "packet_bit",
            b.binary(
                BinaryOp::Mul,
                group_code,
                b.int(i64::from(r.bits)),
                DType::I32,
            ),
            &mut visit,
        );
        let shift = b.bind(
            "packet_shift",
            b.cast(
                b.binary(BinaryOp::Rem, bit.clone(), b.int(32), DType::I32),
                DType::U32,
            ),
            &mut visit,
        );
        let word = b.bind(
            "packet_word_index",
            b.binary(
                BinaryOp::Add,
                b.binary(
                    BinaryOp::Mul,
                    packet.clone(),
                    b.int(i64::from(word_count)),
                    DType::I32,
                ),
                b.binary(BinaryOp::Div, bit, b.int(32), DType::I32),
                DType::I32,
            ),
            &mut visit,
        );
        let read_word = |b: &Builder<'_>, word: Expr| -> Result<Expr, String> {
            let mut at = coordinates.clone();
            *at.last_mut().unwrap() = word;
            if let Some(words) = words {
                *at.last_mut().unwrap() = b.binary(
                    BinaryOp::Sub,
                    at.last().unwrap().clone(),
                    b.symbol(words.origin.clone()),
                    DType::I32,
                );
                Ok(b.element(words.values.clone(), at, DType::U32))
            } else {
                b.accessor(source, r, "words", at)
            }
        };
        let first_read = read_word(&b, word.clone())?;
        let raw = b.bind(
            "packet_raw",
            b.binary(BinaryOp::Shr, first_read, shift.clone(), DType::U32),
            &mut visit,
        );
        let next_read = read_word(&b, b.binary(BinaryOp::Add, word, b.int(1), DType::I32))?;
        let carried = b.binary(
            BinaryOp::BitOr,
            raw.clone(),
            b.binary(
                BinaryOp::Shl,
                next_read,
                b.binary(BinaryOp::Sub, b.uint(32), shift.clone(), DType::U32),
                DType::U32,
            ),
            DType::U32,
        );
        visit.push(b.statement(StmtKind::If {
            cond: b.binary(
                BinaryOp::Gt,
                shift,
                b.uint(i64::from(32 - r.bits)),
                DType::Bool,
            ),
            then: vec![b.assign(raw.clone(), carried)],
            els: vec![],
        }));
        let raw = b.binary(
            BinaryOp::BitAnd,
            raw,
            b.uint(i64::from(u32::MAX >> (32 - r.bits))),
            DType::U32,
        );
        let decoded = b.decode_code(raw, r, &mut visit)?;
        let value = b.expr(
            ExprKind::Builtin {
                name: Builtin::Fma,
                args: vec![decoded, scale.clone(), bias.clone()],
            },
            Ty::Scalar(DType::F32),
            None,
        );
        let mut at = coordinates.clone();
        *at.last_mut().unwrap() = if subsegment.is_some() {
            local_code.clone()
        } else {
            b.binary(
                BinaryOp::Add,
                b.binary(
                    BinaryOp::Mul,
                    local_packet.clone(),
                    b.int(group),
                    DType::I32,
                ),
                local_code.clone(),
                DType::I32,
            )
        };
        visit.push(b.assign(b.element(cache.clone(), at, DType::F32), value));
        body.push(b.statement(StmtKind::Range {
            var: code_id,
            lo: Sym::constant(0),
            hi: Sym::constant(i64::from(codes_per_owner)),
            body: vec![b.statement(StmtKind::If {
                cond: b.binary(BinaryOp::Lt, local_code, b.int(subwidth), DType::Bool),
                then: visit,
                els: vec![],
            })],
        }));
    }

    // One coordinate per selected subpacket; writes cover disjoint code ranges.
    // The domain remains a view of the destination allocation, preserving its
    // placement constraints and ordinary publication barrier. No dummy tile or
    // reshape of a potentially noncontiguous prefix is needed.
    let mut indices = packet_shape
        .iter()
        .map(|_| Index::Slice {
            start: None,
            end: None,
        })
        .collect::<Vec<_>>();
    *indices.last_mut().unwrap() = Index::Slice {
        start: None,
        end: Some(b.symbol(owners)),
    };
    let domain = b.expr(
        ExprKind::Index {
            base: Box::new(cache.clone()),
            indices,
        },
        Ty::Tile(Shaped::new(packet_shape, Elem::Dtype(DType::F32))),
        None,
    );
    let allocation = b.expr(
        ExprKind::TileAlloc {
            shape: cache_shape.clone(),
            dtype: Elem::Dtype(DType::F32),
        },
        cache.ty.clone(),
        None,
    );
    let mut producer = vec![
        b.assign(cache.clone(), allocation),
        b.statement(StmtKind::Owned {
            vars: coordinates
                .iter()
                .map(|e| match e.kind {
                    ExprKind::Var(id) => id,
                    _ => unreachable!(),
                })
                .collect(),
            tile: domain,
            body,
        }),
    ];
    let remainder = width.rem(&Sym::constant(group));
    if subsegment.is_none() && !remainder.is_zero() {
        let start = groups.scale(group);
        let mut tail_shape = cache_shape;
        *tail_shape.last_mut().unwrap() = remainder;
        let tail_indices = tail_shape
            .iter()
            .map(|_| b.local("packet_tail", Ty::Scalar(DType::I32), true))
            .collect::<Vec<_>>();
        let mut at = tail_indices.clone();
        *at.last_mut().unwrap() = b.binary(
            BinaryOp::Add,
            at.last().unwrap().clone(),
            b.symbol(start.clone()),
            DType::I32,
        );
        let mut slices = tail_shape
            .iter()
            .map(|_| Index::Slice {
                start: None,
                end: None,
            })
            .collect::<Vec<_>>();
        *slices.last_mut().unwrap() = Index::Slice {
            start: Some(b.symbol(start)),
            end: None,
        };
        let tail = b.expr(
            ExprKind::Index {
                base: Box::new(cache.clone()),
                indices: slices,
            },
            Ty::Tile(Shaped::new(tail_shape, Elem::Dtype(DType::F32))),
            None,
        );
        producer.push(
            b.statement(StmtKind::Owned {
                vars: tail_indices
                    .iter()
                    .map(|e| match e.kind {
                        ExprKind::Var(id) => id,
                        _ => unreachable!(),
                    })
                    .collect(),
                tile: tail,
                body: vec![b.assign(
                    b.element(cache.clone(), at.clone(), DType::F32),
                    b.element(
                        source.clone(),
                        {
                            let mut source_at = at;
                            let last = source_at.last_mut().unwrap();
                            *last = b.binary(
                                BinaryOp::Add,
                                last.clone(),
                                b.symbol(offset.clone()),
                                DType::I32,
                            );
                            source_at
                        },
                        DType::F32,
                    ),
                )],
            }),
        );
    }
    Ok((cache, producer))
}

fn decode_region_parameterized(
    source: &Expr,
    offset: Sym,
    length: Sym,
    width: Sym,
    decoder: repr::PacketDecoder,
    coefficients: Option<&Coefficients>,
    words: Option<&Words>,
    vars: &mut Vec<Var>,
) -> Result<(Expr, Vec<Stmt>), String> {
    if !supported(&source.ty) {
        return Err("packet decode requires an innermost word-aligned coefficient group".into());
    }
    let shape = source.ty.shaped().unwrap();
    let Elem::Repr(name) = &shape.elem else { unreachable!() };
    let packed_axis = shape
        .packed_axis
        .ok_or("packet decode requires a packed axis")?;
    if packed_axis != shape.shape.len().saturating_sub(1) {
        return Err("parameterized packet decode requires an innermost packed axis".into());
    }
    let r = repr::lookup(name).unwrap();
    let group = Sym::constant(i64::from(r.group));
    let span = source.span;
    let mut b = Builder { vars, span };

    // The cache is dense but keeps every non-packed source extent. Its packed
    // axis is replaced with the retained segment length, which may remain a
    // source numeric atom until materialization.
    let mut cache_type = dense_type(&source.ty);
    let Ty::Tile(cache_shape) = &mut cache_type else { unreachable!() };
    cache_shape.shape[packed_axis] = length.clone();
    let cache_shape = cache_shape.shape.clone();
    let cache = b.local("decoded_packets_parameterized", cache_type, false);
    let coordinates = cache_shape
        .iter()
        .map(|_| b.local("packet_index_parameterized", Ty::Scalar(DType::I32), true))
        .collect::<Vec<_>>();
    let output = coordinates[packed_axis].clone();

    // Width controls the retained owner partition. Reconstructing the output
    // coordinate from owner/local coordinates keeps that source parameter in
    // the emitted IR without enumerating its numeric assignments.
    let owner = b.binary(
        BinaryOp::Div,
        output.clone(),
        b.symbol(width.clone()),
        DType::I32,
    );
    let local = b.binary(
        BinaryOp::Rem,
        output.clone(),
        b.symbol(width.clone()),
        DType::I32,
    );
    let owner_base = b.binary(
        BinaryOp::Mul,
        owner,
        b.symbol(width.clone()),
        DType::I32,
    );
    let local_output = b.binary(BinaryOp::Add, owner_base, local, DType::I32);
    let global = b.binary(
        BinaryOp::Add,
        b.symbol(offset.clone()),
        local_output,
        DType::I32,
    );
    let packet = b.binary(
        BinaryOp::Div,
        global.clone(),
        b.symbol(group.clone()),
        DType::I32,
    );
    let group_code = b.binary(
        BinaryOp::Rem,
        global.clone(),
        b.symbol(group.clone()),
        DType::I32,
    );

    let coefficient_at = {
        let mut at = coordinates.clone();
        *at.last_mut().unwrap() = packet.clone();
        at
    };
    let mut body = Vec::new();
    let scale_read = if let Some(coefficients) = coefficients {
        let mut at = coefficient_at.clone();
        *at.last_mut().unwrap() = b.binary(
            BinaryOp::Sub,
            packet.clone(),
            b.symbol(coefficients.origin.clone()),
            DType::I32,
        );
        b.element(coefficients.scale.clone(), at, DType::F32)
    } else {
        b.coefficient(source, r, false, coefficient_at.clone(), &mut body)?
    };
    let scale = b.bind(
        "packet_scale_parameterized",
        scale_read,
        &mut body,
    );
    let bias = if r.has_bias() {
        let bias_read = if let Some(coefficients) = coefficients {
            let mut at = coefficient_at.clone();
            *at.last_mut().unwrap() = b.binary(
                BinaryOp::Sub,
                packet.clone(),
                b.symbol(coefficients.origin.clone()),
                DType::I32,
            );
            b.element(
                coefficients.bias.as_ref().ok_or("missing retained packet bias")?.clone(),
                at,
                DType::F32,
            )
        } else {
            b.coefficient(source, r, true, coefficient_at.clone(), &mut body)?
        };
        b.bind(
            "packet_bias_parameterized",
            bias_read,
            &mut body,
        )
    } else {
        b.float(0.0)
    };

    let bit = b.binary(
        BinaryOp::Mul,
        group_code,
        b.int(i64::from(r.bits)),
        DType::I32,
    );
    let word_index = b.binary(BinaryOp::Div, bit.clone(), b.int(32), DType::I32);
    let shift = b.cast(
        b.binary(BinaryOp::Rem, bit, b.int(32), DType::I32),
        DType::U32,
    );
    let read_word = |b: &Builder<'_>, index: Expr| -> Result<Expr, String> {
        let mut at = coordinates.clone();
        *at.last_mut().unwrap() = index;
        if let Some(words) = words {
            *at.last_mut().unwrap() = b.binary(
                BinaryOp::Sub,
                at.last().unwrap().clone(),
                b.symbol(words.origin.clone()),
                DType::I32,
            );
            Ok(b.element(words.values.clone(), at, DType::U32))
        } else {
            b.accessor(source, r, "words", at)
        }
    };
    let first = read_word(&b, word_index.clone())?;
    let shifted = b.bind(
        "packet_raw_parameterized",
        b.binary(BinaryOp::Shr, first, shift.clone(), DType::U32),
        &mut body,
    );
    let raw = if r.bits == 32 {
        shifted
    } else {
        let raw = b.local("packet_raw_value_parameterized", Ty::Scalar(DType::U32), false);
        body.push(b.assign(raw.clone(), shifted));
        let next = read_word(
            &b,
            b.binary(BinaryOp::Add, word_index, b.int(1), DType::I32),
        )?;
        let carried = b.binary(
            BinaryOp::BitOr,
            raw.clone(),
            b.binary(
                BinaryOp::Shl,
                next,
                b.binary(
                    BinaryOp::Sub,
                    b.uint(32),
                    shift.clone(),
                    DType::U32,
                ),
                DType::U32,
            ),
            DType::U32,
        );
        body.push(b.statement(StmtKind::If {
            cond: b.binary(
                BinaryOp::Gt,
                shift,
                b.uint(i64::from(32 - r.bits)),
                DType::Bool,
            ),
            then: vec![b.assign(raw.clone(), carried)],
            els: Vec::new(),
        }));
        raw
    };
    let raw = b.binary(
        BinaryOp::BitAnd,
        raw,
        b.uint(i64::from(u32::MAX >> (32 - r.bits))),
        DType::U32,
    );
    let decoded = b.decode_code(raw, r, &mut body)?;
    let value = b.expr(
        ExprKind::Builtin {
            name: Builtin::Fma,
            args: vec![decoded, scale, bias],
        },
        Ty::Scalar(DType::F32),
        None,
    );
    let mut at = coordinates.clone();
    at[packed_axis] = output;
    body.push(b.assign(b.element(cache.clone(), at, DType::F32), value));

    let domain = b.expr(
        ExprKind::Index {
            base: Box::new(cache.clone()),
            indices: cache_shape
                .iter()
                .map(|_| Index::Slice {
                    start: None,
                    end: None,
                })
                .collect(),
        },
        Ty::Tile(Shaped::new(cache_shape.clone(), Elem::Dtype(DType::F32))),
        None,
    );
    let allocation = b.expr(
        ExprKind::TileAlloc {
            shape: cache_shape.clone(),
            dtype: Elem::Dtype(DType::F32),
        },
        cache.ty.clone(),
        None,
    );
    let producer = vec![
        b.assign(cache.clone(), allocation),
        b.statement(StmtKind::Owned {
            vars: coordinates
                .iter()
                .filter_map(|e| match e.kind {
                    ExprKind::Var(id) => Some(id),
                    _ => None,
                })
                .collect(),
            tile: domain,
            body,
        }),
    ];
    // The categorical decoder arm is retained by the family branch. Both
    // decoder forms use this exact bit extraction so they share semantics.
    let _ = decoder;
    Ok((cache, producer))
}

struct Builder<'a> {
    vars: &'a mut Vec<Var>,
    span: Span,
}
impl Builder<'_> {
    /// Read a logical scale or bias value from the representation-owned
    /// coefficient planes. Direct representations expose one dense plane;
    /// hierarchical representations multiply a packed per-group field by its
    /// larger-group factor plane. The extraction is ordinary IR so both
    /// retained and direct packet paths account for the same operations.
    fn coefficient(
        &mut self,
        source: &Expr,
        r: &repr::Repr,
        bias: bool,
        at: Vec<Expr>,
        body: &mut Vec<Stmt>,
    ) -> Result<Expr, String> {
        let Some(coefficient) = r.coefficient(bias) else {
            return Err("representation has no requested packet coefficient".into());
        };
        match coefficient {
            repr::Coefficient::Direct { plane } => {
                Ok(self.cast(self.accessor(source, r, plane.name, at)?, DType::F32))
            }
            repr::Coefficient::Product {
                factor,
                coefficients,
                field,
                sign,
            } => {
                let packet = at.last().cloned().ok_or("packet coefficient has no packed axis")?;
                let factor_stride = i64::from(factor.group / r.group.max(1));
                let factor_index = self.binary(
                    BinaryOp::Div,
                    packet,
                    self.int(factor_stride.max(1)),
                    DType::I32,
                );
                let mut factor_at = at.clone();
                *factor_at.last_mut().unwrap() = factor_index;
                let factor = self.cast(
                    self.accessor(source, r, factor.name, factor_at)?,
                    DType::F32,
                );
                let fields = i64::from(coefficients.fields.max(1));
                let entry = self.binary(
                    BinaryOp::Add,
                    self.binary(
                        BinaryOp::Mul,
                        at.last().unwrap().clone(),
                        self.int(fields),
                        DType::I32,
                    ),
                    self.int(i64::from(field)),
                    DType::I32,
                );
                let value = self.packed_plane_entry(source, r, &coefficients, entry, at, body)?;
                let value = self.cast(value, DType::F32);
                let value = self.binary(BinaryOp::Mul, factor, value, DType::F32);
                Ok(if sign == 1 {
                    value
                } else {
                    self.binary(BinaryOp::Mul, value, self.float(f64::from(sign)), DType::F32)
                })
            }
        }
    }

    /// Read and decode one packed coefficient-plane entry. A dynamic entry
    /// may straddle two words; the second word is only accessed in the
    /// boundary branch so a final entry never performs an invalid read.
    fn packed_plane_entry(
        &mut self,
        source: &Expr,
        r: &repr::Repr,
        plane: &repr::Plane,
        entry: Expr,
        coordinates: Vec<Expr>,
        body: &mut Vec<Stmt>,
    ) -> Result<Expr, String> {
        let repr::PlaneEncoding::Packed { bits, interpretation } = &plane.encoding else {
            return Err("packed coefficient helper requires a packed plane".into());
        };
        let bit = self.binary(
            BinaryOp::Mul,
            entry,
            self.int(i64::from(*bits)),
            DType::I32,
        );
        let word = self.binary(BinaryOp::Div, bit.clone(), self.int(32), DType::I32);
        let shift = self.cast(
            self.binary(BinaryOp::Rem, bit, self.int(32), DType::I32),
            DType::U32,
        );
        let read = |builder: &Builder<'_>, index: Expr| -> Result<Expr, String> {
            let mut at = coordinates.clone();
            *at.last_mut().unwrap() = index;
            builder.accessor(source, r, plane.name, at)
        };
        let first = read(self, word.clone())?;
        let shifted = self.bind(
            "packet_coefficient_raw",
            self.binary(BinaryOp::Shr, first, shift.clone(), DType::U32),
            body,
        );
        let raw = if *bits == 32 {
            shifted
        } else {
            let raw = self.local("packet_coefficient_value", Ty::Scalar(DType::U32), false);
            body.push(self.assign(raw.clone(), shifted));
            let next = read(
                self,
                self.binary(BinaryOp::Add, word, self.int(1), DType::I32),
            )?;
            let carried = self.binary(
                BinaryOp::BitOr,
                raw.clone(),
                self.binary(
                    BinaryOp::Shl,
                    next,
                    self.binary(BinaryOp::Sub, self.uint(32), shift.clone(), DType::U32),
                    DType::U32,
                ),
                DType::U32,
            );
            body.push(self.statement(StmtKind::If {
                cond: self.binary(
                    BinaryOp::Gt,
                    shift,
                    self.uint(i64::from(32 - *bits)),
                    DType::Bool,
                ),
                then: vec![self.assign(raw.clone(), carried)],
                els: Vec::new(),
            }));
            raw
        };
        let raw = self.binary(
            BinaryOp::BitAnd,
            raw,
            self.uint(i64::from(u32::MAX >> (32 - *bits))),
            DType::U32,
        );
        self.decode_interpretation(raw, interpretation, *bits, body)
    }

    /// Evaluate the representation's integer table with bit selections. All
    /// shift amounts are constants and unknown code data never controls a
    /// branch, so accounting follows the complete emitted instruction family.
    /// Every entry remains its exact I32 bit pattern until the final F32 cast.
    fn table_code(
        &mut self,
        raw: Expr,
        table: &[i32],
        body: &mut Vec<Stmt>,
    ) -> Result<Expr, String> {
        let last = *table.last().ok_or("empty representation code table")?;
        let raw = self.bind("packet_code", raw, body);
        let mut values = table
            .iter()
            .map(|&entry| self.uint(i64::from(entry as u32)))
            .collect::<Vec<_>>();
        values.resize(
            values.len().next_power_of_two(),
            self.uint(i64::from(last as u32)),
        );
        let mut bit = 0;
        while values.len() > 1 {
            let bit_value = self.binary(
                BinaryOp::BitAnd,
                self.binary(BinaryOp::Shr, raw.clone(), self.uint(bit), DType::U32),
                self.uint(1),
                DType::U32,
            );
            let mask = self.bind(
                "packet_lookup_mask",
                self.binary(BinaryOp::Sub, self.uint(0), bit_value, DType::U32),
                body,
            );
            values = values
                .chunks_exact(2)
                .map(|pair| {
                    let difference = self.binary(
                        BinaryOp::BitXor,
                        pair[0].clone(),
                        pair[1].clone(),
                        DType::U32,
                    );
                    let selected = self.binary(
                        BinaryOp::BitXor,
                        pair[0].clone(),
                        self.binary(BinaryOp::BitAnd, difference, mask.clone(), DType::U32),
                        DType::U32,
                    );
                    self.bind("packet_lookup_value", selected, body)
                })
                .collect();
            bit += 1;
        }
        Ok(self.cast(self.cast(values.pop().unwrap(), DType::I32), DType::F32))
    }

    fn decode_interpretation(
        &mut self,
        raw: Expr,
        interpretation: &repr::CodeInterpretation,
        bits: u32,
        body: &mut Vec<Stmt>,
    ) -> Result<Expr, String> {
        Ok(match interpretation {
            repr::CodeInterpretation::Unsigned => self.cast(raw, DType::F32),
            repr::CodeInterpretation::Offset(zero) => self.cast(
                self.binary(
                    BinaryOp::Sub,
                    self.cast(raw, DType::I32),
                    self.int(i64::from(*zero)),
                    DType::I32,
                ),
                DType::F32,
            ),
            repr::CodeInterpretation::TwosComplement => self.cast(
                self.binary(
                    BinaryOp::Shr,
                    self.cast(
                        self.binary(
                            BinaryOp::Shl,
                            raw,
                            self.uint(i64::from(32 - bits)),
                            DType::U32,
                        ),
                        DType::I32,
                    ),
                    self.int(i64::from(32 - bits)),
                    DType::I32,
                ),
                DType::F32,
            ),
            repr::CodeInterpretation::Table(table) => self.table_code(raw, table, body)?,
        })
    }

    fn decode_code(
        &mut self,
        raw: Expr,
        r: &repr::Repr,
        body: &mut Vec<Stmt>,
    ) -> Result<Expr, String> {
        self.decode_interpretation(raw, &r.code, r.bits, body)
    }

    // The finite code ranges partition a coefficient group. A balanced branch
    // tree selects the owner's range without dynamic word-array indexing or
    // speculative reads beyond the last word. It remains ordinary accounted IR.
    fn dispatch(&self, part: &Expr, first: i64, mut chunks: Vec<Vec<Stmt>>) -> Vec<Stmt> {
        if chunks.len() == 1 {
            return chunks.pop().unwrap();
        }
        let middle = chunks.len() / 2;
        let right = chunks.split_off(middle);
        let split = first + middle as i64;
        vec![self.statement(StmtKind::If {
            cond: self.binary(BinaryOp::Lt, part.clone(), self.int(split), DType::Bool),
            then: self.dispatch(part, first, chunks),
            els: self.dispatch(part, split, right),
        })]
    }
    fn expr(&self, kind: ExprKind, ty: Ty, sym: Option<Sym>) -> Expr {
        Expr {
            kind,
            ty,
            sym,
            span: self.span,
        }
    }
    fn int(&self, n: i64) -> Expr {
        self.expr(
            ExprKind::Int(n),
            Ty::Scalar(DType::I32),
            Some(Sym::constant(n)),
        )
    }
    fn symbol(&self, value: Sym) -> Expr {
        match value.as_constant() {
            Some(n) => self.int(n),
            None => self.expr(
                ExprKind::ShapeParam(value.to_string()),
                Ty::Scalar(DType::I32),
                Some(value),
            ),
        }
    }
    fn uint(&self, n: i64) -> Expr {
        self.cast(self.int(n), DType::U32)
    }
    fn float(&self, n: f64) -> Expr {
        self.expr(ExprKind::Float(n), Ty::Scalar(DType::F32), None)
    }
    fn cast(&self, e: Expr, dtype: DType) -> Expr {
        if e.ty == Ty::Scalar(dtype) {
            e
        } else {
            self.expr(
                ExprKind::Cast {
                    dtype,
                    expr: Box::new(e),
                },
                Ty::Scalar(dtype),
                None,
            )
        }
    }
    fn binary(&self, op: BinaryOp, lhs: Expr, rhs: Expr, dtype: DType) -> Expr {
        let sym = lhs
            .sym
            .as_ref()
            .zip(rhs.sym.as_ref())
            .and_then(|(a, b)| match op {
                BinaryOp::Add => Some(a.add(b)),
                BinaryOp::Mul => Some(a.mul(b)),
                BinaryOp::Div if b.as_constant().is_some_and(|n| n > 0) => Some(a.quot(b)),
                BinaryOp::Rem if b.as_constant().is_some_and(|n| n > 0) => Some(a.rem(b)),
                _ => None,
            });
        self.expr(
            ExprKind::Binary {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            },
            Ty::Scalar(dtype),
            sym,
        )
    }
    fn statement(&self, kind: StmtKind) -> Stmt {
        Stmt {
            id: None,
            kind,
            span: self.span,
        }
    }
    fn assign(&self, target: Expr, value: Expr) -> Stmt {
        self.statement(StmtKind::Assign {
            target,
            op: AssignOp::Assign,
            value,
        })
    }
    fn local(&mut self, prefix: &str, ty: Ty, index: bool) -> Expr {
        let id = self.vars.len();
        let name = format!("{prefix}_{id}");
        self.vars.push(Var {
            kind: if index {
                VarKind::Index(Atom::Param(name.clone()))
            } else {
                VarKind::Local
            },
            name,
            ty,
            span: self.span,
        });
        super::super::variable(id, self.vars)
    }
    fn bind(&mut self, name: &str, value: Expr, body: &mut Vec<Stmt>) -> Expr {
        let local = self.local(name, value.ty.clone(), false);
        body.push(self.assign(local.clone(), value));
        local
    }
    fn element(&self, base: Expr, at: Vec<Expr>, dtype: DType) -> Expr {
        self.expr(
            ExprKind::Index {
                base: Box::new(base),
                indices: at.into_iter().map(Index::Point).collect(),
            },
            Ty::Scalar(dtype),
            None,
        )
    }
    fn accessor(
        &self,
        source: &Expr,
        r: &repr::Repr,
        name: &str,
        at: Vec<Expr>,
    ) -> Result<Expr, String> {
        let mut shape = source.ty.shaped().unwrap().shape.clone();
        let width = shape.last_mut().unwrap();
        let dtype = if name == "scale" || name == "bias" {
            *width = r.groups_extent(width);
            r.coefficient_dtype()
        } else {
            let plane = r.plane(name).ok_or("missing packed plane")?;
            *width = plane.extent(width);
            plane.dtype()
        };
        let accessor = self.expr(
            ExprKind::Accessor {
                base: Box::new(source.clone()),
                name: name.into(),
            },
            Ty::Tile(Shaped::new(shape, Elem::Dtype(dtype))),
            None,
        );
        Ok(self.element(accessor, at, dtype))
    }
}
