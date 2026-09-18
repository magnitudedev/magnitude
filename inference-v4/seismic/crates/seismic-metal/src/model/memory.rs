//! Binding and known-byte state for the existing terminal derivation walker.
//! External bytes are used only for backings with no possible publication in
//! this invocation. Local arrays retain lane ownership and byte aliasing.
use super::{access::AccessPattern, execution::{Values, affine}};
use crate::{msl::Emitted, terminal::{Expression as E, Space, Statement as S, Type}};
use seismic_accounting::workload::{ScalarWorkload, IntegerDomain, IntegerInput};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Backing { Device(u64), Scratch(usize), Private(String), Shared(String) }
impl Backing { fn device(&self) -> bool { matches!(self, Self::Device(_) | Self::Scratch(_)) } }
#[derive(Clone, Debug)]
pub(super) struct Pointer { backing: Backing, offsets: Values, symbolic: affine::Values, bounds: [(u64, u64); 32] }
impl Pointer {
    pub fn offset_symbolic(&self, indices: Values, index_type: Type, width: u64, affine: affine::Values) -> Self {
        let mut result = self.offset(indices, index_type, width);
        result.symbolic = std::array::from_fn(|i| {
            let base = self.symbolic[i].clone().or_else(|| self.offsets[i].map(|n| affine::Value::constant(i128::from(n))))?;
            base.add(affine[i].as_ref()?.scale(i128::from(width))?)
        });
        result
    }
    pub fn offset(&self, indices: Values, index_type: Type, width: u64) -> Self {
        Self { backing: self.backing.clone(), bounds: self.bounds, symbolic: std::array::from_fn(|_| None), offsets: std::array::from_fn(|i| {
            self.offsets[i].zip(indices[i]).and_then(|(base, index)| {
                let index = match index_type {
                    Type::I32 => i128::from(index as i32), Type::I64 => i128::from(index as i64), _ => i128::from(index),
                };
                u64::try_from(i128::from(base).checked_add(index.checked_mul(i128::from(width))?)?).ok()
            })
        }) }
    }
}
struct Allocation {
    bytes: u64,
    alignment: u64,
    known: BTreeMap<u64, u8>,
    domains: BTreeMap<u64, (IntegerDomain, affine::Value)>,
    /// Typed symbolic values written into local storage. Partial or overlapping
    /// writes invalidate the entire value, just as they invalidate known bytes.
    symbolic: BTreeMap<u64, (Type, affine::Value)>,
}
#[derive(Default)]
pub(super) struct Memory {
    pub pointers: BTreeMap<String, Pointer>,
    allocations: BTreeMap<Backing, Allocation>,
    bindings: BTreeMap<String, Pointer>,
    reads: Vec<(Backing, affine::Value, Type, affine::Value)>,
    launch_writes: Vec<BTreeSet<Backing>>,
    current_writes: BTreeSet<Backing>,
    serial_publication: bool,
}
impl Memory {
    /// A repeated region can retain operation counts without interpreting its
    /// loop-carried data. Forget values rather than treating one visit as the
    /// value produced by the last visit.
    pub fn forget_values(&mut self) {
        for (backing, allocation) in &mut self.allocations {
            if !backing.device() || self.current_writes.contains(backing) { allocation.known.clear(); allocation.symbolic.clear(); }
        }
        self.reads.retain(|(backing, ..)| backing.device() && !self.current_writes.contains(backing));
    }
    pub fn initialize(&mut self, emitted: &Emitted, workload: &ScalarWorkload) -> Result<(), String> {
        if emitted.scratch.len() != emitted.scratch_bindings.len() || emitted.scratch.iter().zip(&emitted.scratch_bindings).any(|(bytes, binding)| *bytes != binding.bytes || binding.alignment == 0) {
            return Err("Metal compiler-owned storage ABI disagrees with its typed bindings".into());
        }
        self.launch_writes = written_backings(emitted, workload);
        let written = self.launch_writes.iter().flatten().cloned().collect::<BTreeSet<_>>();
        self.allocations = workload.allocations.iter().map(|a| (Backing::Device(a.id), Allocation {
            bytes: a.bytes, alignment: a.alignment,
            known: if written.contains(&Backing::Device(a.id)) { BTreeMap::new() } else { a.known_bytes.clone() },
            domains: if written.contains(&Backing::Device(a.id)) { BTreeMap::new() } else {
                workload.integer_domains.iter().enumerate().filter_map(|(index, domain)| match domain.input {
                    IntegerInput::Allocation { allocation, offset } if allocation == a.id => Some((offset, (domain.clone(), affine::Value::domain(index as u64 + 1, domain).expect("validated integer domain")))),
                    _ => None,
                }).collect()
            },
            symbolic: BTreeMap::new(),
        })).collect();
        self.bindings = emitted.buffers.iter().zip(&workload.buffers).map(|(spec, binding)| {
            (buffer_name(spec), Pointer { backing: Backing::Device(binding.allocation), offsets: [Some(binding.offset); 32], symbolic: std::array::from_fn(|_| None), bounds: [(0, workload.allocations.iter().find(|a| a.id == binding.allocation).expect("validated allocation").bytes); 32] })
        }).collect();
        for (slot, spec) in emitted.scratch_bindings.iter().enumerate() {
            let backing = Backing::Scratch(slot);
            let bytes = u64::try_from(spec.bytes).map_err(|_| "Metal scratch capacity overflow")?;
            self.allocations.insert(backing.clone(), Allocation { bytes, alignment: spec.alignment as u64,
                known: BTreeMap::new(), domains: BTreeMap::new(), symbolic: BTreeMap::new() });
            self.bindings.insert(buffer_name(spec), Pointer { backing, offsets: [Some(0); 32],
                symbolic: std::array::from_fn(|_| None), bounds: [(0, bytes); 32] });
        }
        Ok(())
    }
    /// A singleton work item has a source-ordered publication. Other launches
    /// cannot establish device values from the walk order of parallel groups.
    pub fn launch(&mut self, launch: usize, single_item: bool) {
        self.current_writes = self.launch_writes[launch].clone();
        self.serial_publication = single_item;
        if !single_item { self.unfinished_publication(); }
    }
    /// An unvisited suffix may overwrite any declared output. Forget those
    /// facts before a subsequent launch can use them as control or addresses.
    pub fn unfinished_publication(&mut self) {
        for backing in &self.current_writes {
            if let Some(allocation) = self.allocations.get_mut(backing) {
                allocation.known.clear();
                allocation.symbolic.clear();
            }
        }
        self.reads.retain(|(backing, ..)| !self.current_writes.contains(backing));
    }
    pub fn subgroup(&mut self) {
        self.allocations.retain(|backing, _| backing.device());
        self.pointers = self.bindings.clone();
        self.reads.clear();
    }
    pub fn array(&mut self, name: &str, width: u64, elements: u64, space: Space) -> Result<(), String> {
        let bytes = width.checked_mul(elements).ok_or("Metal array byte overflow")?;
        let private = space == Space::Private;
        let backing = if private { Backing::Private(name.into()) } else { Backing::Shared(name.into()) };
        let capacity = if private { bytes.checked_mul(32).ok_or("Metal private array byte overflow")? } else { bytes };
        self.reads.retain(|(previous, ..)| previous != &backing);
        self.allocations.insert(backing.clone(), Allocation { bytes: capacity, alignment: width, known: BTreeMap::new(), domains: BTreeMap::new(), symbolic: BTreeMap::new() });
        self.pointers.insert(name.into(), Pointer { backing, offsets: std::array::from_fn(|i| Some(if private { i as u64 * bytes } else { 0 })), symbolic: std::array::from_fn(|_| None), bounds: std::array::from_fn(|i| if private { (i as u64 * bytes, (i as u64 + 1) * bytes) } else { (0, bytes) }) });
        Ok(())
    }
    pub fn vector_access(&self, name: &str, indices: Values, index_type: Type, element_bytes: u64, width: u64, active: u32) -> Option<AccessPattern> {
        let pointer = self.pointers.get(name)?.offset(indices, index_type, element_bytes);
        if !pointer.backing.device() { return None; }
        let allocation = self.allocations.get(&pointer.backing)?;
        let mut intervals = Vec::with_capacity(active.count_ones() as usize);
        for (i, offset) in pointer.offsets.iter().enumerate() {
            if active & (1 << i) != 0 {
                let start = (*offset)?;
                let end = start.checked_add(width)?;
                if end > allocation.bytes { return None; }
                intervals.push((start, end));
            }
        }
        AccessPattern::new(allocation.alignment, intervals)
    }
    /// Derive one exact translated pattern for the entire dispatch domain.
    /// Different lane strides would change coalescing and remain unresolved.
    pub fn symbolic_access(&self, name: &str, indices: affine::Values, element_bytes: u64, width: u64, active: u32) -> Option<AccessPattern> {
        let base = self.pointers.get(name)?;
        if !base.backing.device() { return None; }
        let allocation = self.allocations.get(&base.backing)?;
        let pointer = base.offset_symbolic([None; 32], Type::I64, element_bytes, indices);
        let mut stride = None;
        let mut intervals = Vec::new();
        for lane in 0..32 {
            if active & (1 << lane) == 0 { continue; }
            let value = pointer.symbolic[lane].as_ref()?;
            if stride.as_ref().is_some_and(|previous| previous != &value.terms) { return None; }
            stride = Some(value.terms.clone());
            let (lo, hi) = value.bounds()?;
            if lo < i128::from(pointer.bounds[lane].0) || hi.checked_add(i128::from(width))? > i128::from(pointer.bounds[lane].1.min(allocation.bytes)) { return None; }
            let start = u64::try_from(value.origin()?).ok()?;
            intervals.push((start, start.checked_add(width)?));
        }
        let alignment = affine::Value { base: 0, terms: stride? }.alignment(allocation.alignment);
        AccessPattern::new(alignment, intervals)
    }
    /// Exact or bounded integer reads from immutable invocation controls. A
    /// finite strided superset of addresses must be completely covered by known
    /// bytes or typed input domains; otherwise no fact is returned.
    pub fn read_symbolic(&mut self, name: &str, indices: Values, affine: affine::Values,
        index_type: Type, ty: Type, active: u32, next_coordinate: &mut u64, maximum_values: usize)
        -> Result<affine::Values, String> {
        let mut result = std::array::from_fn(|_| None);
        let signed = matches!(ty, Type::I32 | Type::I64);
        if !matches!(ty, Type::I32 | Type::I64 | Type::U32 | Type::U64) { return Ok(result); }
        let Some(base) = self.pointers.get(name) else { return Ok(result); };
        let pointer = base.offset_symbolic(indices, index_type, ty.bytes(), affine);
        let Some(allocation) = self.allocations.get(&pointer.backing) else { return Ok(result); };
        for lane in 0..32 {
            if active & (1 << lane) == 0 { continue; }
            let Some(address) = pointer.symbolic[lane].clone().or_else(|| pointer.offsets[lane].map(|v| affine::Value::constant(i128::from(v)))) else { continue; };
            if let Some((_, _, _, value)) = self.reads.iter().find(|(backing, offset, element, _)| *backing == pointer.backing && offset == &address && *element == ty) {
                result[lane] = Some(value.clone()); continue;
            }
            let Some((lo, hi)) = address.bounds() else { continue; };
            if lo < i128::from(pointer.bounds[lane].0) || hi.checked_add(i128::from(ty.bytes())).is_none_or(|end| end > i128::from(pointer.bounds[lane].1.min(allocation.bytes))) { continue; }
            let gcd = |mut a: u128, mut b: u128| { while b != 0 { let r = a % b; a = b; b = r; } a };
            let stride = address.terms.values().fold(0u128, |a, (coefficient, _)| gcd(a, coefficient.unsigned_abs())).max(1);
            let Some(count) = (hi - lo).checked_div(stride as i128).and_then(|n| n.checked_add(1)).and_then(|n| usize::try_from(n).ok()) else { continue; };
            if count > maximum_values { continue; }
            let (mut min, mut max) = (i128::MAX, i128::MIN);
            let mut direct = None;
            let mut complete = true;
            for visit in 0..count {
                let offset = (lo + visit as i128 * stride as i128) as u64;
                if let Some((_, value)) = allocation.symbolic.get(&offset).filter(|(stored, _)| *stored == ty) {
                    let Some((lo, hi)) = value.bounds() else { complete = false; break; };
                    min = min.min(lo); max = max.max(hi);
                    direct = Some(value.clone());
                } else if let Some((domain, value)) = allocation.domains.get(&offset).filter(|(d, _)| u64::from(d.bytes) == ty.bytes() && d.signed == signed) {
                    min = min.min(domain.range.min); max = max.max(domain.range.max);
                    direct = Some(value.clone());
                } else {
                    let bytes = (offset..offset + ty.bytes()).map(|offset| allocation.known.get(&offset).copied()).collect::<Option<Vec<_>>>();
                    let Some(bytes) = bytes else { complete = false; break; };
                    let mut bits = [0u8; 8]; bits[..bytes.len()].copy_from_slice(&bytes);
                    let raw = u64::from_le_bytes(bits);
                    let value = match ty { Type::I32 => i128::from(raw as i32), Type::I64 => i128::from(raw as i64), _ => i128::from(raw) };
                    min = min.min(value); max = max.max(value);
                    direct = Some(affine::Value::constant(value));
                }
            }
            if !complete { continue; }
            let value = if count == 1 { direct.expect("one admitted read") }
                else if min == max { affine::Value::constant(min) }
                else {
                    let id = *next_coordinate;
                    *next_coordinate = id.checked_add(1).ok_or("indirect read coordinate overflow")?;
                    affine::Value::coordinate(id, min, max)
                };
            self.reads.push((pointer.backing.clone(), address, ty, value.clone()));
            result[lane] = Some(value);
        }
        Ok(result)
    }
    pub fn read(&self, name: &str, indices: Values, index_type: Type, width: u64, active: u32) -> Values {
        let Some(base) = self.pointers.get(name) else { return [None; 32]; };
        let Some(allocation) = self.allocations.get(&base.backing).filter(|a| !a.known.is_empty()) else { return [None; 32]; };
        let pointer = base.offset(indices, index_type, width);
        std::array::from_fn(|i| {
            if active & (1 << i) == 0 { return None; }
            let start = pointer.offsets[i]?;
            let end = start.checked_add(width)?;
            if start < pointer.bounds[i].0 || end > pointer.bounds[i].1 || end > allocation.bytes || width > 8 { return None; }
            let mut value = 0u64;
            for byte in 0..width { value |= u64::from(*allocation.known.get(&(start + byte))?) << (byte * 8); }
            Some(value)
        })
    }
    pub fn invalidate(&mut self, name: &str, space: Space) {
        if let Some(pointer) = self.pointers.get(name) {
            self.reads.retain(|(backing, ..)| backing != &pointer.backing);
            if let Some(allocation) = self.allocations.get_mut(&pointer.backing) { allocation.known.clear(); allocation.symbolic.clear(); }
        } else {
            self.reads.clear();
            for (backing, allocation) in &mut self.allocations {
                if matches!((space, backing), (Space::Device, Backing::Device(_) | Backing::Scratch(_)) | (Space::Private, Backing::Private(_)) | (Space::Threadgroup, Backing::Shared(_))) { allocation.known.clear(); allocation.symbolic.clear(); }
            }
        }
    }
    pub fn write(&mut self, name: &str, indices: Values, index_type: Type, ty: Type, values: Values, symbolic: affine::Values, active: u32, space: Space) {
        // External publications are not made known by an arbitrary traversal of
        // parallel groups. Their initial bytes were excluded before the walk.
        let Some(base) = self.pointers.get(name) else { self.invalidate(name, space); return; };
        if space == Space::Device && (!self.serial_publication || !matches!(base.backing, Backing::Scratch(_))) { return; }
        self.reads.retain(|(backing, ..)| backing != &base.backing);
        let Some(allocation) = self.allocations.get_mut(&base.backing) else { return; };
        if allocation.known.is_empty() && allocation.symbolic.is_empty() && (0..32).all(|i| active & (1 << i) == 0 || (values[i].is_none() && symbolic[i].is_none())) { return; }
        let width = ty.bytes();
        let pointer = base.offset(indices, index_type, width);
        let mut writes: BTreeMap<u64, Option<u8>> = BTreeMap::new();
        let mut typed_writes = Vec::new();
        for i in 0..32 {
            if active & (1 << i) == 0 { continue; }
            let Some(start) = pointer.offsets[i].filter(|s| *s >= pointer.bounds[i].0 && s.checked_add(width).is_some_and(|end| end <= pointer.bounds[i].1 && end <= allocation.bytes)) else { allocation.known.clear(); allocation.symbolic.clear(); return; };
            typed_writes.push((start, symbolic[i].clone()));
            for byte in 0..width {
                let value = values[i].map(|value| (value >> (byte * 8)) as u8);
                writes.entry(start + byte).and_modify(|previous| { if *previous != value { *previous = None; } }).or_insert(value);
            }
        }
        for (offset, value) in writes {
            if let Some(value) = value { allocation.known.insert(offset, value); }
            else { allocation.known.remove(&offset); }
        }
        allocation.symbolic.retain(|offset, (stored, _)| !typed_writes.iter().any(|(start, _)|
            *offset < start + width && *start < offset + stored.bytes()));
        for (start, value) in &typed_writes {
            let Some(value) = value else { continue; };
            if typed_writes.iter().any(|(other_start, other_value)|
                *start < other_start + width && *other_start < start + width
                && (other_start != start || other_value.as_ref() != Some(value))) { continue; }
            allocation.symbolic.insert(*start, (ty, value.clone()));
        }
    }
}

fn buffer_name(spec: &seismic_realization::BufferSpec) -> String {
    if spec.plane.is_empty() { spec.parameter.clone() } else { format!("{}_{}", spec.parameter, spec.plane) }
}
/// A conservative write-effect scan of the retained typed implementation, using
/// actual binding aliases. It does not execute control or estimate work. Unknown
/// pointer effects suppress all external known bytes, never invent independence.
fn written_backings(emitted: &Emitted, workload: &ScalarWorkload) -> Vec<BTreeSet<Backing>> {
    type Roots = BTreeMap<String, BTreeSet<Backing>>;
    fn publish(name: Option<&str>, roots: &Roots, all: &BTreeSet<Backing>, written: &mut BTreeSet<Backing>) {
        written.extend(name.and_then(|n| roots.get(n)).unwrap_or(all).iter().cloned());
    }
    fn expr(e: &E, roots: &Roots, all: &BTreeSet<Backing>, written: &mut BTreeSet<Backing>) {
        match e {
            E::Helper(helper, args, _) => {
                if *helper == crate::support::Helper::Write { publish(args.first().and_then(|e| if let E::Variable(name, _) = e { Some(name.as_str()) } else { None }), roots, all, written); }
                for arg in args { expr(arg, roots, all, written); }
            }
            E::Builtin(_, args, _) => { for arg in args { expr(arg, roots, all, written); } }
            E::Binary(_, a, b, _) | E::ShortCircuit { left: a, right: b, .. } => { expr(a, roots, all, written); expr(b, roots, all, written); }
            E::Select(c, a, b) => { expr(c, roots, all, written); expr(a, roots, all, written); expr(b, roots, all, written); }
            E::Cast(_, a) | E::Bitcast(_, a) | E::Unary(_, a, _) | E::Read { index: a, .. } => expr(a, roots, all, written),
            E::Unmapped(..) => written.extend(all.iter().cloned()),
            E::Integer(..) | E::Float(..) | E::Variable(..) | E::Parameter { .. } | E::VectorElement { .. } => {}
        }
    }
    let mut bindings: Roots = emitted.buffers.iter().zip(&workload.buffers).map(|(spec, binding)| (buffer_name(spec), [Backing::Device(binding.allocation)].into())).collect();
    for (slot, spec) in emitted.scratch_bindings.iter().enumerate() { bindings.insert(buffer_name(spec), [Backing::Scratch(slot)].into()); }
    let all = bindings.values().flatten().cloned().collect::<BTreeSet<_>>();
    emitted.terminal.launches().into_iter().map(|launch| {
        let mut written = BTreeSet::new();
        let mut roots = bindings.clone();
        for site in launch {
            match &site.statement {
                S::Pointer { name, base, index, space, .. } => {
                    expr(index, &roots, &all, &mut written);
                    if *space == Space::Device {
                        let source = roots.get(base).unwrap_or(&all).clone();
                        roots.entry(name.clone()).or_default().extend(source);
                    }
                }
                S::Write { name, index, value, space, .. } => {
                    expr(index, &roots, &all, &mut written); expr(value, &roots, &all, &mut written);
                    if *space == Space::Device { publish(Some(name), &roots, &all, &mut written); }
                }
                S::MatrixStore { base, offset, leading, space, .. } => {
                    expr(offset, &roots, &all, &mut written); expr(leading, &roots, &all, &mut written);
                    if *space == Space::Device { publish(Some(base), &roots, &all, &mut written); }
                }
                S::MatrixLoad { offset, leading, .. } | S::For { start: offset, end: leading, .. } => { expr(offset, &roots, &all, &mut written); expr(leading, &roots, &all, &mut written); }
                S::Let { value, .. } | S::Assign { value, .. } | S::Evaluate(value) | S::If(value) | S::ReturnIf(value) | S::Return(Some(value)) => expr(value, &roots, &all, &mut written),
                S::Unmapped(_) => written.extend(all.iter().cloned()),
                S::VectorRead { index, .. } => expr(index, &roots, &all, &mut written),
                S::Array { .. } | S::Fragment { .. } | S::MatrixMultiplyAccumulate { .. } | S::Barrier | S::End | S::Else | S::Scope | S::FailureStatus | S::Return(None) => {}
            }
        }
        written
    }).collect()
}

#[cfg(test)]
mod symbolic_tests {
    use super::*;
    #[test]
    fn local_symbolic_values_follow_aliases_and_overlapping_writes() {
        let mut memory = Memory::default();
        memory.array("local", 4, 2, Space::Private).unwrap();
        let coordinate = affine::Value::coordinate(4, 0, 1023);
        let values = std::array::from_fn(|_| Some(coordinate.clone()));
        memory.write("local", [Some(1); 32], Type::I32, Type::I32, [None; 32], values.clone(), u32::MAX, Space::Private);
        let base = memory.pointers["local"].clone();
        memory.pointers.insert("alias".into(), base.offset([Some(1); 32], Type::I32, 4));
        let mut next = 5;
        let read = |memory: &mut Memory, next: &mut u64| memory.read_symbolic("alias", [Some(0); 32], std::array::from_fn(|_| None), Type::I32, Type::I32, u32::MAX, next, 64).unwrap();
        assert_eq!(read(&mut memory, &mut next), values);
        // Overwrite one half of lane 0's typed value. Both the retained value
        // and a previous symbolic read of its alias must be invalidated.
        memory.write("alias", [Some(0); 32], Type::I32, Type::F16, [None; 32], std::array::from_fn(|_| None), 1, Space::Private);
        let after = read(&mut memory, &mut next);
        assert!(after[0].is_none());
        assert_eq!(after[1], Some(coordinate));
        memory.forget_values();
        assert!(read(&mut memory, &mut next).iter().all(Option::is_none));
    }

    #[test]
    fn conflicting_shared_symbolic_writes_do_not_invent_a_value() {
        let mut memory = Memory::default();
        memory.array("shared", 4, 1, Space::Threadgroup).unwrap();
        let mut values = std::array::from_fn(|_| None);
        values[0] = Some(affine::Value::coordinate(0, 0, 7));
        values[1] = Some(affine::Value::coordinate(1, 0, 7));
        memory.write("shared", [Some(0); 32], Type::I32, Type::I32, [None; 32], values, 3, Space::Threadgroup);
        let read = memory.read_symbolic("shared", [Some(0); 32], std::array::from_fn(|_| None), Type::I32, Type::I32, 3, &mut 2, 64).unwrap();
        assert!(read.iter().all(Option::is_none));
    }

    #[test]
    fn translated_pointer_geometry_matches_every_concrete_group() {
        let mut memory = Memory::default();
        memory.allocations.insert(Backing::Device(0), Allocation { bytes: 4096, alignment: 256, known: BTreeMap::new(), domains: BTreeMap::new(), symbolic: BTreeMap::new() });
        let base = Pointer { backing: Backing::Device(0), offsets: [Some(0); 32], symbolic: std::array::from_fn(|_| None), bounds: [(0, 4096); 32] };
        let groups = std::array::from_fn(|_| affine::Value::coordinate(0, 0, 15).scale(32));
        memory.pointers.insert("shifted".into(), base.offset_symbolic([None; 32], Type::I64, 4, groups));
        let lanes = std::array::from_fn(|i| Some(affine::Value::constant(i as i128)));
        for mask in [1, 0x55555555, u32::MAX] {
            let compact = memory.symbolic_access("shifted", lanes.clone(), 4, 4, mask).unwrap();
            for group in 0..16 {
                memory.pointers.insert("concrete".into(), base.offset([Some(group * 32); 32], Type::I64, 4));
                let concrete = memory.vector_access("concrete", std::array::from_fn(|i| Some(i as u64)), Type::I64, 4, 4, mask).unwrap();
                for width in [4, 16, 64, 128] { assert_eq!(compact.transactions(width), concrete.transactions(width)); }
            }
            assert_eq!(compact.transactions(256), None, "unaligned translations do not establish larger transactions");
        }
        let mut varying = lanes.clone();
        varying[1] = affine::Value::coordinate(0, 0, 15).add(affine::Value::constant(1));
        assert!(memory.symbolic_access("shifted", varying, 4, 4, 3).is_none());
        let outside = std::array::from_fn(|i| affine::Value::coordinate(1, 0, 4096).add(affine::Value::constant(i as i128)));
        assert!(memory.symbolic_access("shifted", outside, 4, 4, u32::MAX).is_none());
    }
}
