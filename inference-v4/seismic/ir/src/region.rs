//! Physical products transported by structured schedule regions. These contain
//! actual operands and destinations, never another semantic producer program.
use crate::repr::ScalarKind;
use crate::schedule::{AnyScalarSlot, HostQuantityKind, HostQuantitySlot};
use crate::storage::AnyBufferView;
use seismic_lang::expr::{IntExpr, NatExpr, SymbolId};
use seismic_lang::types::DType;

#[derive(Clone, Copy, Debug)]
pub enum ScalarOperand {
    Natural(NatExpr),
    Word { symbol: SymbolId, dtype: DType },
}

/// An exact host value, with no fixed-width kernel ABI interpretation.
#[derive(Clone, Copy, Debug)]
pub enum QuantityOperand {
    Integer(IntExpr),
    Natural(NatExpr),
}
impl QuantityOperand {
    pub fn kind(self) -> HostQuantityKind {
        match self {
            Self::Integer(_) => HostQuantityKind::Integer,
            Self::Natural(_) => HostQuantityKind::Natural,
        }
    }
}
impl ScalarOperand {
    pub fn kind(self) -> ScalarKind {
        match self {
            Self::Natural(_) => ScalarKind::Nat64,
            Self::Word { dtype, .. } => ScalarKind::Scalar(dtype),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum ValueOperand {
    Scalar(ScalarOperand),
    Quantity(QuantityOperand),
    Tensor(AnyBufferView),
}

#[derive(Clone, Copy, Debug)]
pub enum ValueDestination {
    Scalar(AnyScalarSlot),
    Quantity(HostQuantitySlot),
    Tensor(AnyBufferView),
}

/// Tuple/range structure is preserved from the actual region result product.
#[derive(Clone, Debug, Default)]
pub enum Product<T> {
    #[default]
    Unit,
    Leaf(T),
    Range(Box<Product<T>>, Box<Product<T>>),
    Tuple(Vec<Product<T>>),
}
impl<T> Product<T> {
    /// Heap storage owned by this product, excluding its enclosing inline value.
    pub fn retained_heap_bytes(&self, leaf: &impl Fn(&T) -> usize) -> usize {
        match self {
            Self::Unit => 0,
            Self::Leaf(value) => leaf(value),
            Self::Range(a,b) => 2 * std::mem::size_of::<Self>() + a.retained_heap_bytes(leaf) + b.retained_heap_bytes(leaf),
            Self::Tuple(values) => values.capacity() * std::mem::size_of::<Self>() + values.iter().map(|value|value.retained_heap_bytes(leaf)).sum::<usize>(),
        }
    }

    pub fn try_map<U, E>(&self, f: &mut impl FnMut(&T) -> Result<U, E>) -> Result<Product<U>, E> {
        Ok(match self {
            Self::Unit => Product::Unit,
            Self::Leaf(value) => Product::Leaf(f(value)?),
            Self::Range(start, end) => Product::Range(Box::new(start.try_map(f)?), Box::new(end.try_map(f)?)),
            Self::Tuple(values) => Product::Tuple(values.iter().map(|value| value.try_map(f)).collect::<Result<_, _>>()?),
        })
    }
    pub fn map<U>(&self, f: &mut impl FnMut(&T) -> U) -> Product<U> {
        match self {
            Self::Unit => Product::Unit,
            Self::Leaf(value) => Product::Leaf(f(value)),
            Self::Range(start, end) => Product::Range(Box::new(start.map(f)), Box::new(end.map(f))),
            Self::Tuple(values) => {
                Product::Tuple(values.iter().map(|value| value.map(f)).collect())
            }
        }
    }
    pub fn visit(&self, f: &mut impl FnMut(&T)) {
        match self {
            Self::Unit => (),
            Self::Leaf(value) => f(value),
            Self::Range(start, end) => {
                start.visit(f);
                end.visit(f);
            }
            Self::Tuple(values) => values.iter().for_each(|value| value.visit(f)),
        }
    }
}

/// A selected arm forwards its complete value into one parent-owned location.
#[derive(Clone, Debug)]
pub struct BranchResult {
    pub(crate) then_value: ValueOperand,
    pub(crate) else_value: ValueOperand,
    pub(crate) result: ValueDestination,
}
impl BranchResult {
    pub fn then_value(&self) -> ValueOperand { self.then_value }
    pub fn else_value(&self) -> ValueOperand { self.else_value }
    pub fn result(&self) -> ValueDestination { self.result }
    pub(crate) fn remap(&self, view: impl Fn(AnyBufferView) -> AnyBufferView, slot: impl Fn(AnyScalarSlot) -> AnyScalarSlot, quantity: impl Fn(HostQuantitySlot) -> HostQuantitySlot) -> Self {
        let operand = |value| match value {
            ValueOperand::Tensor(value) => ValueOperand::Tensor(view(value)),
            other => other,
        };
        let result = match self.result {
            ValueDestination::Tensor(value) => ValueDestination::Tensor(view(value)),
            ValueDestination::Scalar(value) => ValueDestination::Scalar(slot(value)),
            ValueDestination::Quantity(value) => ValueDestination::Quantity(quantity(value)),
        };
        Self { then_value: operand(self.then_value), else_value: operand(self.else_value), result }
    }
}

/// Each row is made by the repeat constructor, which creates both destinations
/// and obtains the backedge operand by consuming its actual body result.
#[derive(Clone, Debug)]
pub struct RepeatCarry {
    pub(crate) initial: ValueOperand,
    pub(crate) header: ValueDestination,
    pub(crate) backedge: ValueOperand,
    pub(crate) result: ValueDestination,
}
impl RepeatCarry {
    pub fn initial(&self) -> ValueOperand {
        self.initial
    }
    pub fn header(&self) -> ValueDestination {
        self.header
    }
    pub fn backedge(&self) -> ValueOperand {
        self.backedge
    }
    pub fn result(&self) -> ValueDestination {
        self.result
    }
}

impl RepeatCarry {
    pub(crate) fn remap(
        &self,
        view: impl Fn(AnyBufferView) -> AnyBufferView,
        slot: impl Fn(AnyScalarSlot) -> AnyScalarSlot,
        quantity: impl Fn(HostQuantitySlot) -> HostQuantitySlot,
    ) -> Self {
        let operand = |value| match value {
            ValueOperand::Scalar(value) => ValueOperand::Scalar(value),
            ValueOperand::Quantity(value) => ValueOperand::Quantity(value),
            ValueOperand::Tensor(value) => ValueOperand::Tensor(view(value)),
        };
        let destination = |value| match value {
            ValueDestination::Scalar(value) => ValueDestination::Scalar(slot(value)),
            ValueDestination::Quantity(value) => ValueDestination::Quantity(quantity(value)),
            ValueDestination::Tensor(value) => ValueDestination::Tensor(view(value)),
        };
        Self {
            initial: operand(self.initial),
            header: destination(self.header),
            backedge: operand(self.backedge),
            result: destination(self.result),
        }
    }
}

pub(crate) fn carry_leaves<'a>(
    product: &'a Product<RepeatCarry>,
    leaves: &mut Vec<&'a RepeatCarry>,
) {
    match product {
        Product::Unit => (),
        Product::Leaf(value) => leaves.push(value),
        Product::Range(a, b) => {
            carry_leaves(a, leaves);
            carry_leaves(b, leaves);
        }
        Product::Tuple(values) => values.iter().for_each(|value| carry_leaves(value, leaves)),
    }
}

pub(crate) fn collect_carries<'a>(
    steps: &'a [crate::schedule::ScheduleStep],
    leaves: &mut Vec<&'a RepeatCarry>,
) {
    use crate::schedule::ScheduleStep;
    for step in steps {
        match step {
            ScheduleStep::Imported { body, .. } => collect_carries(body, leaves),
            ScheduleStep::Repeat { carries, body, .. } => {
                carry_leaves(carries, leaves);
                collect_carries(body, leaves);
            }
            ScheduleStep::If {
                then_steps,
                else_steps,
                ..
            } => {
                collect_carries(then_steps, leaves);
                collect_carries(else_steps, leaves);
            }
            _ => (),
        }
    }
}

pub(crate) fn collect_tensor_definitions(steps: &[crate::schedule::ScheduleStep], definitions: &mut Vec<(AnyBufferView, AnyBufferView)>) {
    use crate::schedule::ScheduleStep;
    for step in steps {
        match step {
            ScheduleStep::BeginAllocationInstance { source, result } | ScheduleStep::BindArgumentTensor { source, result } => definitions.push((*source, *result)),
            ScheduleStep::Imported { body, .. } | ScheduleStep::Repeat { body, .. } => collect_tensor_definitions(body, definitions),
            ScheduleStep::If { then_steps, else_steps, results, .. } => {
                collect_tensor_definitions(then_steps, definitions); collect_tensor_definitions(else_steps, definitions);
                results.visit(&mut |result| if let ValueDestination::Tensor(destination) = result.result() {
                    for operand in [result.then_value(), result.else_value()] {
                        let ValueOperand::Tensor(source) = operand else { unreachable!("closed branch product kind") };
                        definitions.push((source, destination));
                    }
                });
            }
            _ => {},
        }
    }
}

pub(crate) fn possible_backings(
    views: &[crate::storage::BufferViewLayout],
    view: AnyBufferView,
    products: &[&RepeatCarry],
    definitions: &[(AnyBufferView, AnyBufferView)],
) -> Option<Vec<crate::storage::GlobalAllocationId>> {
    use crate::storage::{GlobalAllocationId, TensorValueId, ViewBase};
    use std::collections::BTreeSet;
    fn visit(
        view: AnyBufferView,
        views: &[crate::storage::BufferViewLayout],
        products: &[&RepeatCarry],
        definitions: &[(AnyBufferView, AnyBufferView)],
        visiting: &mut BTreeSet<TensorValueId>,
        roots: &mut BTreeSet<GlobalAllocationId>,
    ) -> Option<()> {
        match views[view.index() as usize].base {
            ViewBase::Allocation(root) => {
                roots.insert(root);
            }
            ViewBase::TensorValue(id) => {
                if !visiting.insert(id) {
                    return Some(());
                }
                let mut defined = false;
                for (source, _) in definitions.iter().filter(|(_, result)| views[result.index() as usize].base == ViewBase::TensorValue(id)) {
                    defined = true;
                    visit(*source, views, products, definitions, visiting, roots)?;
                }
                if defined { return Some(()); }
                let product = products.iter().find(|carry| [carry.header(),carry.result()].into_iter().any(|destination| {
                    matches!(destination,ValueDestination::Tensor(value) if views[value.index() as usize].base == ViewBase::TensorValue(id))
                }))?;
                for operand in [product.initial(), product.backedge()] {
                    let ValueOperand::Tensor(value) = operand else {
                        panic!("tensor region operand changed kind")
                    };
                    visit(value, views, products, definitions, visiting, roots)?;
                }
            }
        }
        Some(())
    }
    let mut roots = BTreeSet::new();
    visit(view, views, products, definitions, &mut BTreeSet::new(), &mut roots)?;
    Some(roots.into_iter().collect())
}
