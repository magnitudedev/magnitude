//! Which tensor values of one semantic function share contents.
//!
//! A content class is the set of tensor values that name the same storage
//! contents: a view and its base, a written place and the place version the
//! write produces, a region capture and the region parameter it binds, and the
//! states of one loop carry. Operations that produce new contents (`Copy`,
//! `RepresentationConvert`, `Alloc`, `Fill`, computed tensors and call results)
//! start a new class. A caller's `&mut` argument keeps its caller class; what a
//! callee writes into it is the callee's own summary, not a class union.
//!
//! Classes are formed over value leaves, so tuple components that carry
//! tensors are tracked component-wise. The analysis is policy-independent.

use seismic_lang::entry::{SemanticFunction, SemanticNodeView, SemanticType};
use seismic_lang::ids::{RegionId, SemanticValueId};
use std::ops::Range;

/// One content class of one function, dense from zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct ContentClassId(u32);

impl ContentClassId {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// Union-find result over the tensor leaves of one `SemanticFunction`.
///
/// This is the single owner of the function's leaf layout: every per-leaf fact
/// of the function (its type, its class) is indexed by the leaves it assigns.
pub(crate) struct ContentClasses<'a> {
    function: &'a SemanticFunction,
    /// Value index → first leaf; one trailing entry closes the last range.
    leaves: Vec<u32>,
    /// Leaf → its type.
    types: Vec<&'a SemanticType>,
    /// Leaf → its class, for tensor leaves only.
    classes: Vec<Option<ContentClassId>>,
    len: usize,
}

impl<'a> ContentClasses<'a> {
    pub fn analyze(function: &'a SemanticFunction) -> Self {
        let mut leaves = Vec::new();
        let mut types = Vec::new();
        for (id, info) in function.values() {
            assert_eq!(id.index(), leaves.len(), "semantic values are not dense");
            leaves.push(leaf_count(types.len()));
            flatten(&info.ty, &mut types);
        }
        leaves.push(leaf_count(types.len()));
        let mut unions = Unions {
            function,
            leaves: &leaves,
            parent: (0..leaf_count(types.len())).collect(),
        };
        unions.region(function.root());
        let mut dense = vec![None; types.len()];
        let mut len = 0;
        let classes = types
            .iter()
            .enumerate()
            .map(|(leaf, ty)| {
                matches!(ty, SemanticType::Tensor(_)).then(|| {
                    let root = unions.find(leaf as u32) as usize;
                    *dense[root].get_or_insert_with(|| {
                        len += 1;
                        ContentClassId(len - 1)
                    })
                })
            })
            .collect();
        Self {
            function,
            leaves,
            types,
            classes,
            len: len as usize,
        }
    }

    /// The class of a tensor-typed value.
    pub fn class(&self, value: SemanticValueId) -> ContentClassId {
        match self.classes[self.leaf_range(value)] {
            [Some(class)] => class,
            _ => panic!("content class requested for non-tensor value {value:?}"),
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    /// The leaves of one value, in component order: tuples are flattened and
    /// `Void` has none.
    pub fn leaf_range(&self, value: SemanticValueId) -> Range<usize> {
        self.leaves[value.index()] as usize..self.leaves[value.index() + 1] as usize
    }

    /// The leaves of tuple component `index` of `tuple`.
    pub fn component_range(&self, tuple: SemanticValueId, index: u32) -> Range<usize> {
        let base = self.leaves[tuple.index()];
        let component = component_offsets(&self.function.value(tuple).ty, index);
        (base + component.start) as usize..(base + component.end) as usize
    }

    /// The class of one leaf; `None` for a non-tensor leaf.
    pub fn leaf_class(&self, leaf: usize) -> Option<ContentClassId> {
        self.classes[leaf]
    }

    /// The type of one leaf; never a tuple or `Void`.
    pub fn leaf_type(&self, leaf: usize) -> &'a SemanticType {
        self.types[leaf]
    }

    pub fn leaf_total(&self) -> usize {
        self.types.len()
    }
}

fn leaf_count(count: usize) -> u32 {
    u32::try_from(count).expect("semantic function has more than u32::MAX value leaves")
}

/// Appends the leaf types of `ty` in component order.
fn flatten<'a>(ty: &'a SemanticType, leaves: &mut Vec<&'a SemanticType>) {
    match ty {
        SemanticType::Tuple(items) => items.iter().for_each(|item| flatten(item, leaves)),
        SemanticType::Void => {}
        SemanticType::Tensor(_)
        | SemanticType::Scalar(_)
        | SemanticType::Integer
        | SemanticType::Index { .. }
        | SemanticType::Range { .. }
        | SemanticType::Opaque { .. } => leaves.push(ty),
    }
}

/// Leaf offsets of tuple component `index` within a value of type `ty`.
fn component_offsets(ty: &SemanticType, index: u32) -> Range<u32> {
    let SemanticType::Tuple(items) = ty else {
        panic!("checked tuple operation over a non-tuple value")
    };
    let mut leaves = Vec::new();
    items[..index as usize]
        .iter()
        .for_each(|item| flatten(item, &mut leaves));
    let start = leaf_count(leaves.len());
    leaves.clear();
    flatten(&items[index as usize], &mut leaves);
    start..start + leaf_count(leaves.len())
}

struct Unions<'a> {
    function: &'a SemanticFunction,
    leaves: &'a [u32],
    parent: Vec<u32>,
}

impl Unions<'_> {
    fn find(&mut self, mut leaf: u32) -> u32 {
        while self.parent[leaf as usize] != leaf {
            let grandparent = self.parent[self.parent[leaf as usize] as usize];
            self.parent[leaf as usize] = grandparent;
            leaf = grandparent;
        }
        leaf
    }

    fn union_leaves(&mut self, a: u32, b: u32) {
        let (a, b) = (self.find(a), self.find(b));
        if a != b {
            self.parent[a.max(b) as usize] = a.min(b);
        }
    }

    fn range(&self, value: SemanticValueId) -> Range<u32> {
        self.leaves[value.index()]..self.leaves[value.index() + 1]
    }

    /// Unions two values of one type leaf by leaf.
    fn union_values(&mut self, a: SemanticValueId, b: SemanticValueId) {
        let (a, b) = (self.range(a), self.range(b));
        assert_eq!(
            a.len(),
            b.len(),
            "content union over values of different types"
        );
        for (a, b) in a.zip(b) {
            self.union_leaves(a, b);
        }
    }

    /// Unions `component` of `tuple` with `value`.
    fn union_component(&mut self, tuple: SemanticValueId, index: u32, value: SemanticValueId) {
        let component = component_offsets(&self.function.value(tuple).ty, index);
        let base = self.leaves[tuple.index()];
        let value = self.range(value);
        assert_eq!(
            component.len(),
            value.len(),
            "tuple component type differs from its value"
        );
        for (a, b) in component.zip(value) {
            self.union_leaves(base + a, b);
        }
    }

    fn region(&mut self, region: RegionId) {
        let function = self.function;
        for (_, node) in function.nodes(region) {
            match node.view() {
                SemanticNodeView::View { base, output, .. } => self.union_values(base, output),
                SemanticNodeView::ElementWrite { place, output, .. }
                | SemanticNodeView::Atomic { place, output, .. } => {
                    self.union_values(place, output)
                }
                SemanticNodeView::Store {
                    destination,
                    output,
                    ..
                } => self.union_values(destination, output),
                SemanticNodeView::If {
                    captures,
                    outputs,
                    then,
                    otherwise,
                    ..
                } => {
                    for arm in [then, otherwise] {
                        let arm_region = function.region(arm);
                        assert_eq!(
                            arm_region.parameters().len(),
                            captures.len(),
                            "checked if arm parameters differ from its captures"
                        );
                        assert_eq!(
                            arm_region.results().len(),
                            outputs.len(),
                            "checked if arm results differ from its outputs"
                        );
                        for (capture, parameter) in captures.iter().zip(arm_region.parameters()) {
                            self.union_values(*capture, *parameter);
                        }
                        for (output, result) in outputs.iter().zip(arm_region.results()) {
                            self.union_values(*output, *result);
                        }
                        self.region(arm);
                    }
                }
                SemanticNodeView::Loop {
                    captures,
                    body,
                    carries,
                    ..
                } => {
                    let parameters = function.region(body).parameters();
                    assert_eq!(
                        parameters.len(),
                        captures.len() + 1,
                        "checked loop parameters differ from binder and captures"
                    );
                    for (capture, parameter) in captures.iter().zip(&parameters[1..]) {
                        self.union_values(*capture, *parameter);
                    }
                    for carry in carries {
                        self.union_values(carry.initial, carry.parameter);
                        self.union_values(carry.parameter, carry.yielded);
                        self.union_values(carry.yielded, carry.result);
                    }
                    self.region(body);
                }
                SemanticNodeView::TuplePack { inputs, output } => {
                    for (index, input) in inputs.iter().enumerate() {
                        self.union_component(output, index as u32, *input);
                    }
                }
                SemanticNodeView::TupleGet {
                    tuple,
                    index,
                    output,
                } => self.union_component(tuple, index, output),
                SemanticNodeView::Primitive { .. }
                | SemanticNodeView::Intrinsic { .. }
                | SemanticNodeView::Elementwise { .. }
                | SemanticNodeView::Reduce { .. }
                | SemanticNodeView::Call { .. }
                | SemanticNodeView::Alloc { .. }
                | SemanticNodeView::Fill { .. }
                | SemanticNodeView::Copy { .. }
                | SemanticNodeView::RepresentationConvert { .. }
                | SemanticNodeView::ElementRead { .. }
                | SemanticNodeView::Check { .. }
                | SemanticNodeView::Extent { .. } => {}
            }
        }
    }
}
