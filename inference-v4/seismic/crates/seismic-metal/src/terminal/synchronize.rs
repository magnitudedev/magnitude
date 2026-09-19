//! Coalesce publication barriers only across nonconflicting shared accesses.
//! This is a transformation of the terminal program, before transfer/traversal
//! domains are discovered. Both emission and accounting consume the result.
use super::{Expression as E, Site, Space, Statement as S};
use std::collections::{HashMap, HashSet};

/// Known low address bits are sufficient to distinguish disjoint columns of a
/// strided tile without claiming that their full bounding intervals are disjoint.
/// All arithmetic is modulo the actual integer width, including signed casts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
struct LowBits {
    bits: u8,
    value: u64,
}
impl LowBits {
    fn new(bits: u8, value: u64) -> Self {
        Self {
            bits,
            value: value
                & if bits == 64 {
                    u64::MAX
                } else {
                    (1u64 << bits) - 1
                },
        }
    }
    fn exact(value: u64) -> Self {
        Self { bits: 64, value }
    }
    fn add(self, other: Self) -> Self {
        Self::new(
            self.bits.min(other.bits),
            self.value.wrapping_add(other.value),
        )
    }
    fn scale(self, constant: u64) -> Self {
        if constant == 0 {
            return Self::exact(0);
        }
        Self::new(
            (u32::from(self.bits) + constant.trailing_zeros()).min(64) as u8,
            self.value.wrapping_mul(constant),
        )
    }
    fn typed(self, ty: super::Type) -> Self {
        use super::Type::*;
        match ty {
            I32 if self.bits >= 32 => Self::exact(self.value as i32 as i64 as u64),
            U32 if self.bits >= 32 => Self::exact(u64::from(self.value as u32)),
            I32 | U32 | I64 | U64 => self,
            _ => Self::default(),
        }
    }
    fn of(e: &E) -> Self {
        use seismic_lang::syntax::ast::BinaryOp as B;
        match e {
            E::Integer(n, ty) => Self::exact(*n as u64).typed(*ty),
            E::Cast(ty, e) | E::Bitcast(ty, e) => Self::of(e).typed(*ty),
            E::Binary(op, a, b, ty) => {
                let (a, b) = (Self::of(a), Self::of(b));
                let value = match op {
                    B::Add => a.add(b),
                    B::Sub => a.add(b.scale(u64::MAX)),
                    B::Mul if a.bits == 64 => b.scale(a.value),
                    B::Mul if b.bits == 64 => a.scale(b.value),
                    _ => Self::default(),
                };
                value.typed(*ty)
            }
            _ => Self::default(),
        }
    }
    fn overlaps(self, other: Self) -> bool {
        Self::new(self.bits.min(other.bits), self.value ^ other.value).value == 0
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Location {
    root: String,
    address: LowBits,
}
impl Location {
    fn overlaps(&self, other: &Self) -> bool {
        self.root == other.root && self.address.overlaps(other.address)
    }
}
#[derive(Clone, Default)]
struct Accesses {
    reads: HashSet<Location>,
    writes: HashSet<Location>,
    opaque: bool,
}
impl Accesses {
    fn extend(&mut self, other: &Self) {
        self.reads.extend(other.reads.iter().cloned());
        self.writes.extend(other.writes.iter().cloned());
        self.opaque |= other.opaque;
    }
    fn conflicts(&self, other: &Self) -> bool {
        self.opaque
            || other.opaque
            || self.writes.iter().any(|v| {
                other
                    .reads
                    .iter()
                    .chain(&other.writes)
                    .any(|w| v.overlaps(w))
            })
            || self
                .reads
                .iter()
                .any(|v| other.writes.iter().any(|w| v.overlaps(w)))
    }
}
struct Node {
    site: usize,
    accesses: Accesses,
    barrier: bool,
}
#[derive(Clone, PartialEq, Eq)]
struct Pointer {
    base: String,
    offset: LowBits,
}
struct Aliases(HashMap<String, Option<Pointer>>);
impl Aliases {
    fn root(&self, name: &str) -> Option<(String, LowBits)> {
        if name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            return None;
        }
        let mut root = name;
        let mut offset = LowBits::exact(0);
        let mut visited = HashSet::new();
        while let Some(base) = self.0.get(root) {
            if !visited.insert(root) {
                return None;
            }
            let pointer = base.as_ref()?;
            offset = offset.add(pointer.offset);
            root = &pointer.base;
        }
        Some((root.to_owned(), offset))
    }
    fn add(&mut self, name: &str, base: &str, index: &E, ty: super::Type) {
        let pointer = Pointer {
            base: base.into(),
            offset: LowBits::of(index).scale(ty.bytes()),
        };
        match self.0.get(name) {
            Some(Some(old)) if old == &pointer => {}
            Some(_) => {
                self.0.insert(name.into(), None);
            }
            None => {
                self.0.insert(name.into(), Some(pointer));
            }
        }
    }
}
fn access(
    out: &mut Accesses,
    aliases: &Aliases,
    name: &str,
    write: bool,
    element: Option<(&E, super::Type)>,
) {
    let Some((root, base)) = aliases.root(name) else {
        out.opaque = true;
        return;
    };
    let locations = if let Some((index, ty)) = element {
        let address = base.add(LowBits::of(index).scale(ty.bytes()));
        (0..ty.bytes())
            .map(|byte| Location {
                root: root.clone(),
                address: address.add(LowBits::exact(byte)),
            })
            .collect::<Vec<_>>()
    } else {
        vec![Location {
            root,
            address: LowBits::default(),
        }]
    };
    if write {
        out.writes.extend(locations);
    } else {
        out.reads.extend(locations);
    }
}
fn expression(e: &E, aliases: &Aliases, out: &mut Accesses) {
    match e {
        E::Read {
            name,
            index,
            space,
            ty,
        } => {
            if *space == Space::Threadgroup {
                access(out, aliases, name, false, Some((index, *ty)));
            }
            expression(index, aliases, out);
        }
        E::Binary(_, a, b, _)
        | E::ShortCircuit {
            left: a, right: b, ..
        } => {
            expression(a, aliases, out);
            expression(b, aliases, out);
        }
        E::Unary(_, e, _) | E::Cast(_, e) | E::Bitcast(_, e) => expression(e, aliases, out),
        E::Select(c, a, b) | E::EagerSelect(c, a, b) => {
            expression(c, aliases, out);
            expression(a, aliases, out);
            expression(b, aliases, out);
        }
        // Support helpers use only device/status pointers. Their argument
        // expressions can still read shared values and must be included.
        E::Helper(_, args, _) => {
            for e in args {
                expression(e, aliases, out);
            }
        }
        E::Builtin(name, args, _) => {
            // Unknown target calls may synchronize or access memory. Recognized
            // scalar/subgroup builtins have no shared-memory effect themselves.
            if !matches!(
                name.as_str(),
                "fma"
                    | "exp"
                    | "log"
                    | "sqrt"
                    | "rsqrt"
                    | "sin"
                    | "cos"
                    | "tanh"
                    | "abs"
                    | "min"
                    | "max"
                    | "pow"
                    | "simd_shuffle"
                    | "simd_sum"
                    | "simd_min"
                    | "simd_max"
            ) {
                out.opaque = true;
            }
            for e in args {
                expression(e, aliases, out);
            }
        }
        E::Unmapped(..) => out.opaque = true,
        _ => {}
    }
}
fn statement(s: &S, aliases: &Aliases) -> Accesses {
    let mut out = Accesses::default();
    match s {
        S::Let { value, .. } | S::Assign { value, .. } | S::Evaluate(value) | S::If(value) => {
            expression(value, aliases, &mut out)
        }
        S::Write {
            name,
            index,
            space,
            value,
            ty,
        } => {
            if *space == Space::Threadgroup {
                access(&mut out, aliases, name, true, Some((index, *ty)));
            }
            expression(index, aliases, &mut out);
            expression(value, aliases, &mut out);
        }
        S::MatrixLoad {
            base,
            offset,
            leading,
            space,
            ..
        }
        | S::MatrixStore {
            base,
            offset,
            leading,
            space,
            ..
        } => {
            if *space == Space::Threadgroup {
                access(
                    &mut out,
                    aliases,
                    base,
                    matches!(s, S::MatrixStore { .. }),
                    None,
                );
            }
            expression(offset, aliases, &mut out);
            expression(leading, aliases, &mut out);
        }
        S::For { start, end, .. } => {
            expression(start, aliases, &mut out);
            expression(end, aliases, &mut out);
        }
        S::Pointer { index, .. } | S::VectorRead { index, .. } => {
            expression(index, aliases, &mut out)
        }
        S::ReturnIf(_) | S::Return(_) | S::Unmapped(_) => out.opaque = true,
        _ => {}
    }
    out
}

/// Return whether any sites were removed. No barrier crosses a control boundary;
/// complete barrier-free nested loops contribute their shared access summaries.
pub(crate) fn coalesce(body: &mut Vec<Site>) -> Result<bool, String> {
    let mut aliases = Aliases(HashMap::new());
    for site in body.iter() {
        if let S::Pointer {
            name,
            base,
            space: Space::Threadgroup,
            index,
            ty,
        } = &site.statement
        {
            aliases.add(name, base, index, *ty);
        }
    }
    fn region(
        body: &[Site],
        at: &mut usize,
        aliases: &Aliases,
        remove: &mut HashSet<usize>,
    ) -> Result<Accesses, String> {
        let mut nodes = Vec::new();
        while *at < body.len() && !matches!(body[*at].statement, S::End | S::Else) {
            let site = *at;
            let s = &body[site].statement;
            let mut accesses = statement(s, aliases);
            *at += 1;
            if matches!(s, S::For { .. } | S::If(_) | S::Scope) {
                accesses.extend(&region(body, at, aliases, remove)?);
                if matches!(body.get(*at).map(|s| &s.statement), Some(S::Else)) {
                    if !matches!(s, S::If(_)) {
                        return Err("terminal else does not belong to an if".into());
                    }
                    *at += 1;
                    accesses.extend(&region(body, at, aliases, remove)?);
                }
                if !matches!(body.get(*at).map(|s| &s.statement), Some(S::End)) {
                    return Err("unclosed synchronization scope".into());
                }
                *at += 1;
            }
            nodes.push(Node {
                site,
                accesses,
                // A threadgroup barrier also orders the lanes of each SIMD group. It is never
                // removed: it orders accesses of other SIMD groups, which this pass cannot see.
                barrier: matches!(s, S::Barrier | S::GroupBarrier),
            });
        }
        let mut pending: Option<Accesses> = None;
        for (i, node) in nodes.iter().enumerate() {
            if matches!(body[node.site].statement, S::GroupBarrier) {
                pending = Some(Accesses::default());
            } else if node.barrier {
                let mut next = Accesses::default();
                let mut later = false;
                for following in &nodes[i + 1..] {
                    if following.barrier {
                        later = true;
                        break;
                    }
                    next.extend(&following.accesses);
                    if next.opaque {
                        break;
                    }
                }
                if later && pending.as_ref().is_some_and(|p| !p.conflicts(&next)) {
                    remove.insert(node.site);
                } else {
                    pending = Some(Accesses::default());
                }
            } else if node.accesses.opaque {
                pending = None;
            } else if let Some(pending) = &mut pending {
                pending.extend(&node.accesses);
            }
        }
        let mut summary = Accesses::default();
        for node in &nodes {
            summary.extend(&node.accesses);
            if node.barrier && !remove.contains(&node.site) {
                summary.opaque = true;
            }
        }
        Ok(summary)
    }
    let mut remove = HashSet::new();
    let mut at = 0;
    region(body, &mut at, &aliases, &mut remove)?;
    if at != body.len() {
        return Err("unexpected terminal synchronization scope boundary".into());
    }
    if remove.is_empty() {
        return Ok(false);
    }
    let mut position = 0;
    body.retain(|_| {
        let retain = !remove.contains(&position);
        position += 1;
        retain
    });
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terminal::Type as T;
    fn write(name: &str) -> S {
        S::Write {
            name: name.into(),
            index: E::integer(0),
            space: Space::Threadgroup,
            ty: T::F32,
            value: E::Float(1f64.to_bits(), T::F32),
        }
    }
    fn read(name: &str) -> S {
        S::Evaluate(E::Read {
            name: name.into(),
            index: Box::new(E::integer(0)),
            space: Space::Threadgroup,
            ty: T::F32,
        })
    }
    fn barriers(statements: Vec<S>) -> usize {
        let mut body = statements
            .into_iter()
            .map(|statement| Site {
                operation: None,
                statement,
            })
            .collect();
        coalesce(&mut body).unwrap();
        body.iter()
            .filter(|s| matches!(s.statement, S::Barrier))
            .count()
    }
    #[test]
    fn shared_hazards_keep_publication_and_completion_barriers() {
        assert_eq!(
            barriers(vec![
                S::Barrier,
                write("a"),
                S::Barrier,
                write("b"),
                S::Barrier
            ]),
            2
        );
        for (before, after) in [
            (write("a"), read("a")),
            (read("a"), write("a")),
            (write("a"), write("a")),
        ] {
            assert_eq!(
                barriers(vec![S::Barrier, before, S::Barrier, after, S::Barrier]),
                3
            );
        }
        assert_eq!(
            barriers(vec![
                S::Barrier,
                read("a"),
                S::Barrier,
                read("a"),
                S::Barrier
            ]),
            2
        );
        assert_eq!(
            barriers(vec![
                S::Barrier,
                write("a"),
                S::Barrier,
                S::ReturnIf(E::Integer(0, T::Bool)),
                write("b"),
                S::Barrier
            ]),
            3
        );
        let pointer = |name: &str| S::Pointer {
            name: name.into(),
            base: "slot".into(),
            index: E::integer(0),
            space: Space::Threadgroup,
            ty: T::F32,
        };
        assert_eq!(
            barriers(vec![
                pointer("a"),
                pointer("b"),
                S::Barrier,
                write("a"),
                S::Barrier,
                read("b"),
                S::Barrier
            ]),
            3
        );
        let loop_start = S::For {
            name: "i".into(),
            start: E::integer(0),
            end: E::integer(3),
            step: 1,
        };
        assert_eq!(
            barriers(vec![
                S::Barrier,
                write("a"),
                S::Barrier,
                loop_start,
                write("b"),
                S::End,
                S::Barrier
            ]),
            2
        );
    }
    #[test]
    fn strided_columns_coalesce_but_overlapping_bytes_and_wrapping_indices_do_not() {
        use seismic_lang::syntax::ast::BinaryOp as B;
        let coordinate = |column| {
            E::Binary(
                B::Add,
                Box::new(E::Binary(
                    B::Mul,
                    Box::new(E::variable("row", T::I64)),
                    Box::new(E::Integer(16, T::I64)),
                    T::I64,
                )),
                Box::new(E::Integer(column, T::I64)),
                T::I64,
            )
        };
        let column = |c| S::Write {
            name: "tile".into(),
            index: coordinate(c),
            space: Space::Threadgroup,
            ty: T::F32,
            value: E::Float(0, T::F32),
        };
        assert_eq!(
            barriers(vec![
                S::Barrier,
                column(0),
                S::Barrier,
                column(1),
                S::Barrier
            ]),
            2
        );
        assert_eq!(
            barriers(vec![
                S::Barrier,
                column(0),
                S::Barrier,
                column(16),
                S::Barrier
            ]),
            3
        );
        let access = |index, ty| S::Write {
            name: "tile".into(),
            index: E::Integer(index, T::I64),
            space: Space::Threadgroup,
            ty,
            value: E::Integer(0, T::I32),
        };
        assert_eq!(
            barriers(vec![
                S::Barrier,
                access(2, T::F32),
                S::Barrier,
                access(4, T::F16),
                S::Barrier
            ]),
            3
        );
        for stride in [0i32, 1, 4, 16, 64, 256, -1] {
            for offset in [-1i32, 0, 1, 255] {
                let e = E::Binary(
                    B::Add,
                    Box::new(E::Binary(
                        B::Mul,
                        Box::new(E::variable("x", T::I32)),
                        Box::new(E::Integer(i64::from(stride), T::I32)),
                        T::I32,
                    )),
                    Box::new(E::Integer(i64::from(offset), T::I32)),
                    T::I32,
                );
                let known = LowBits::of(&e);
                for x in [i32::MIN, -31, -1, 0, 1, 31, i32::MAX] {
                    let actual = x.wrapping_mul(stride).wrapping_add(offset) as i64 as u64;
                    assert_eq!(
                        LowBits::new(known.bits, actual).value,
                        known.value,
                        "{x} * {stride} + {offset}"
                    );
                }
            }
        }
    }
}
