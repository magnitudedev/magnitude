//! Backend-neutral physical scalar instruction lowering.
//!
//! Schedule construction supplies all value and storage mappings. This module
//! lowers only a verified scheduling-normal `LogicalTask::body`; calls,
//! iteration, reductions, placement, participation and linking are absent.

use seismic_lang::{
    intrinsics::Operation,
    logical::{
        LocalStorageId, LocalValueId, LocalViewId, LogicalBlock, LogicalConditionalCase,
        LogicalExpr, LogicalExprKind, LogicalIndex, LogicalOperationKind, LogicalPattern,
        LogicalTask, LogicalTaskGraph, OperandId, StorageRef, Type, ValueRef,
    },
    precision::NumericalEffect,
    sir,
    span::Span,
    sym::Sym,
    syntax::ast::{AssignOp, BinaryOp, UnaryOp},
    types::DType,
};
use seismic_realization::executable::{
    AccessMode, ExecutableDialect, InstructionConsequences, InstructionResources, PhysicalAccess,
    ResolvedStorageId, StorageId,
};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ScalarValueId(pub u32);

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ValueBindingKey {
    Input(u32),
    Local(LocalValueId),
    Result(u32),
}

impl From<&ValueRef> for ValueBindingKey {
    fn from(value: &ValueRef) -> Self {
        match value {
            ValueRef::Input(port) => Self::Input(*port),
            ValueRef::Local(value) => Self::Local(*value),
            ValueRef::Result(port) => Self::Result(*port),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScalarStorageBinding {
    pub storage: StorageId,
    pub plane: Vec<String>,
}

/// Schedule-owned physical bindings for one task. The lowering pass neither
/// allocates storage nor chooses an addressing/participation strategy.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ScalarBindings {
    pub values: BTreeMap<ValueBindingKey, ScalarValueId>,
    pub operands: BTreeMap<OperandId, ScalarValueId>,
    pub storage: BTreeMap<StorageRef, Vec<ScalarStorageBinding>>,
    pub views: BTreeMap<LocalViewId, Vec<ScalarStorageBinding>>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct ScalarCapabilitySet(pub BTreeSet<String>);

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ScalarLayout {
    pub name: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ScalarInstruction {
    pub result: Option<ScalarValueId>,
    pub kind: ScalarInstructionKind,
    consequences: InstructionConsequences<ScalarCapabilitySet>,
    pub span: Span,
}

impl ScalarInstruction {
    pub fn consequences(&self) -> &InstructionConsequences<ScalarCapabilitySet> {
        &self.consequences
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ScalarLiteral {
    Int(i64),
    Float(u64),
    Bool(bool),
}

#[derive(Clone, Debug, PartialEq)]
pub enum ScalarIndex {
    Point(ScalarValueId),
    Coordinate(ScalarValueId),
    Slice(u32),
    Range {
        start: Option<ScalarValueId>,
        end: Option<ScalarValueId>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StorageOperation {
    Construct,
    Fill,
    Snapshot,
    Decode,
    Materialize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ConditionalCase {
    pub predicates: Vec<(ScalarValueId, bool)>,
    pub value: ScalarValueId,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ScalarInstructionKind {
    NoOp,
    Literal(ScalarLiteral),
    Shape(Sym),
    Tuple(Vec<ScalarValueId>),
    Range(ScalarValueId, ScalarValueId),
    Field {
        value: ScalarValueId,
        index: usize,
    },
    Storage {
        operation: StorageOperation,
        bindings: Vec<ScalarStorageBinding>,
        inputs: Vec<ScalarValueId>,
        fill_bits: Option<u64>,
    },
    View {
        view: LocalViewId,
        base: ScalarValueId,
    },
    Index {
        base: ScalarValueId,
        indices: Vec<ScalarIndex>,
    },
    Cast {
        dtype: DType,
        value: ScalarValueId,
    },
    Unary {
        op: UnaryOp,
        value: ScalarValueId,
    },
    Binary {
        op: BinaryOp,
        lhs: ScalarValueId,
        rhs: ScalarValueId,
    },
    Math {
        op: sir::Math,
        arguments: Vec<ScalarValueId>,
    },
    Select {
        condition: ScalarValueId,
        then_value: ScalarValueId,
        else_value: ScalarValueId,
    },
    Extent {
        base: ScalarValueId,
        axis: usize,
    },
    Intrinsic {
        operation: Operation,
        arguments: Vec<ScalarValueId>,
    },
    Accessor {
        base: ScalarValueId,
        name: String,
    },
    Geometry {
        base: ScalarValueId,
        axis: usize,
        valid: bool,
    },
    Atomic {
        op: BinaryOp,
        place: ScalarValueId,
        value: ScalarValueId,
    },
    Assign {
        target: ScalarValueId,
        op: AssignOp,
        value: ScalarValueId,
    },
    Publish {
        value: ScalarValueId,
        destination: ScalarValueId,
    },
    Conditional {
        condition: ScalarValueId,
        then_body: Vec<ScalarInstruction>,
        else_body: Vec<ScalarInstruction>,
    },
    ConditionalMerge {
        cases: Vec<ConditionalCase>,
    },
    Yield(Vec<ScalarValueId>),
    Return {
        port: u32,
        path: Vec<u32>,
        value: ScalarValueId,
        transfer: bool,
    },
}

/// Fully concrete scalar instruction consumed by native encoders. Symbolic
/// shapes and template storage identities cannot cross this boundary.
#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedScalarInstruction {
    pub result: Option<ScalarValueId>,
    pub kind: ResolvedScalarInstructionKind,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedScalarStorageBinding {
    pub storage: ResolvedStorageId,
    pub plane: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ResolvedScalarInstructionKind {
    NoOp,
    Literal(ScalarLiteral),
    Shape(i64),
    Tuple(Vec<ScalarValueId>),
    Range(ScalarValueId, ScalarValueId),
    Field {
        value: ScalarValueId,
        index: usize,
    },
    Storage {
        operation: StorageOperation,
        bindings: Vec<ResolvedScalarStorageBinding>,
        inputs: Vec<ScalarValueId>,
        fill_bits: Option<u64>,
    },
    View {
        view: LocalViewId,
        base: ScalarValueId,
    },
    Index {
        base: ScalarValueId,
        indices: Vec<ScalarIndex>,
    },
    Cast {
        dtype: DType,
        value: ScalarValueId,
    },
    Unary {
        op: UnaryOp,
        value: ScalarValueId,
    },
    Binary {
        op: BinaryOp,
        lhs: ScalarValueId,
        rhs: ScalarValueId,
    },
    Math {
        op: sir::Math,
        arguments: Vec<ScalarValueId>,
    },
    Select {
        condition: ScalarValueId,
        then_value: ScalarValueId,
        else_value: ScalarValueId,
    },
    Extent {
        base: ScalarValueId,
        axis: usize,
    },
    Intrinsic {
        operation: Operation,
        arguments: Vec<ScalarValueId>,
    },
    Accessor {
        base: ScalarValueId,
        name: String,
    },
    Geometry {
        base: ScalarValueId,
        axis: usize,
        valid: bool,
    },
    Atomic {
        op: BinaryOp,
        place: ScalarValueId,
        value: ScalarValueId,
    },
    Assign {
        target: ScalarValueId,
        op: AssignOp,
        value: ScalarValueId,
    },
    Publish {
        value: ScalarValueId,
        destination: ScalarValueId,
    },
    Conditional {
        condition: ScalarValueId,
        then_body: Vec<ResolvedScalarInstruction>,
        else_body: Vec<ResolvedScalarInstruction>,
    },
    ConditionalMerge {
        cases: Vec<ConditionalCase>,
    },
    Yield(Vec<ScalarValueId>),
    Return {
        port: u32,
        path: Vec<u32>,
        value: ScalarValueId,
        transfer: bool,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScalarDialect;

impl ExecutableDialect for ScalarDialect {
    type TemplateInstruction = ScalarInstruction;
    type ResolvedInstruction = ResolvedScalarInstruction;
    type TemplateLayout = ScalarLayout;
    type ResolvedLayout = ScalarLayout;
    type Capability = ScalarCapabilitySet;

    fn consequences(
        instruction: &Self::TemplateInstruction,
    ) -> InstructionConsequences<Self::Capability> {
        instruction.consequences.clone()
    }

    fn resolve_instruction(
        instruction: &Self::TemplateInstruction,
        symbols: &BTreeMap<String, i64>,
        storage: &BTreeMap<StorageId, ResolvedStorageId>,
    ) -> Result<Self::ResolvedInstruction, String> {
        resolve_scalar_instruction(instruction, symbols, storage)
    }

    fn resolve_layout(
        layout: &Self::TemplateLayout,
        _symbols: &BTreeMap<String, i64>,
    ) -> Result<Self::ResolvedLayout, String> {
        Ok(layout.clone())
    }
}

pub fn resolve_scalar_instruction(
    instruction: &ScalarInstruction,
    symbols: &BTreeMap<String, i64>,
    storage: &BTreeMap<StorageId, ResolvedStorageId>,
) -> Result<ResolvedScalarInstruction, String> {
    use ResolvedScalarInstructionKind as R;
    use ScalarInstructionKind as T;
    let resolve_nested = |values: &[ScalarInstruction]| {
        values
            .iter()
            .map(|value| resolve_scalar_instruction(value, symbols, storage))
            .collect::<Result<Vec<_>, _>>()
    };
    let kind = match &instruction.kind {
        T::NoOp => R::NoOp,
        T::Literal(value) => R::Literal(value.clone()),
        T::Shape(value) => R::Shape(
            value
                .eval(&|name| symbols.get(name).copied())
                .ok_or_else(|| format!("unresolved scalar shape `{value}`"))?,
        ),
        T::Tuple(values) => R::Tuple(values.clone()),
        T::Range(start, end) => R::Range(*start, *end),
        T::Field { value, index } => R::Field {
            value: *value,
            index: *index,
        },
        T::Storage {
            operation,
            bindings,
            inputs,
            fill_bits,
        } => R::Storage {
            operation: *operation,
            bindings: bindings
                .iter()
                .map(|binding| {
                    Ok(ResolvedScalarStorageBinding {
                        storage: storage.get(&binding.storage).copied().ok_or_else(|| {
                            format!(
                                "scalar instruction names absent storage#{}",
                                binding.storage.0
                            )
                        })?,
                        plane: binding.plane.clone(),
                    })
                })
                .collect::<Result<Vec<_>, String>>()?,
            inputs: inputs.clone(),
            fill_bits: *fill_bits,
        },
        T::View { view, base } => R::View {
            view: *view,
            base: *base,
        },
        T::Index { base, indices } => R::Index {
            base: *base,
            indices: indices.clone(),
        },
        T::Cast { dtype, value } => R::Cast {
            dtype: *dtype,
            value: *value,
        },
        T::Unary { op, value } => R::Unary {
            op: *op,
            value: *value,
        },
        T::Binary { op, lhs, rhs } => R::Binary {
            op: *op,
            lhs: *lhs,
            rhs: *rhs,
        },
        T::Math { op, arguments } => R::Math {
            op: *op,
            arguments: arguments.clone(),
        },
        T::Select {
            condition,
            then_value,
            else_value,
        } => R::Select {
            condition: *condition,
            then_value: *then_value,
            else_value: *else_value,
        },
        T::Extent { base, axis } => R::Extent {
            base: *base,
            axis: *axis,
        },
        T::Intrinsic {
            operation,
            arguments,
        } => R::Intrinsic {
            operation: *operation,
            arguments: arguments.clone(),
        },
        T::Accessor { base, name } => R::Accessor {
            base: *base,
            name: name.clone(),
        },
        T::Geometry { base, axis, valid } => R::Geometry {
            base: *base,
            axis: *axis,
            valid: *valid,
        },
        T::Atomic { op, place, value } => R::Atomic {
            op: *op,
            place: *place,
            value: *value,
        },
        T::Assign { target, op, value } => R::Assign {
            target: *target,
            op: *op,
            value: *value,
        },
        T::Publish { value, destination } => R::Publish {
            value: *value,
            destination: *destination,
        },
        T::Conditional {
            condition,
            then_body,
            else_body,
        } => R::Conditional {
            condition: *condition,
            then_body: resolve_nested(then_body)?,
            else_body: resolve_nested(else_body)?,
        },
        T::ConditionalMerge { cases } => R::ConditionalMerge {
            cases: cases.clone(),
        },
        T::Yield(values) => R::Yield(values.clone()),
        T::Return {
            port,
            path,
            value,
            transfer,
        } => R::Return {
            port: *port,
            path: path.clone(),
            value: *value,
            transfer: *transfer,
        },
    };
    Ok(ResolvedScalarInstruction {
        result: instruction.result,
        kind,
        span: instruction.span,
    })
}

#[derive(Clone, Debug, PartialEq)]
pub struct ScalarProgram {
    task: seismic_lang::logical::TaskId,
    instructions: Vec<ScalarInstruction>,
    operand_values: BTreeMap<OperandId, ScalarValueId>,
    value_types: BTreeMap<ScalarValueId, Type>,
}

impl ScalarProgram {
    pub fn task(&self) -> seismic_lang::logical::TaskId {
        self.task
    }

    pub fn instructions(&self) -> &[ScalarInstruction] {
        &self.instructions
    }

    pub fn operand_value(&self, operand: OperandId) -> Option<ScalarValueId> {
        self.operand_values.get(&operand).copied()
    }

    pub fn value_type(&self, value: ScalarValueId) -> Option<&Type> {
        self.value_types.get(&value)
    }
}

pub fn lower_task(
    graph: &LogicalTaskGraph,
    task: &LogicalTask,
    bindings: &ScalarBindings,
) -> Result<ScalarProgram, String> {
    if graph.task(task.id) != Some(task) {
        return Err(format!(
            "task#{} is not owned by graph#{}",
            task.id.0, graph.id.0
        ));
    }
    for input in &task.inputs {
        if !bindings.operands.contains_key(input) {
            return Err(format!(
                "task#{} input operand#{} has no physical value",
                task.id.0, input.0
            ));
        }
    }
    let mut value_types = BTreeMap::new();
    for (operand, value) in &bindings.operands {
        let ty = &graph
            .operand(*operand)
            .ok_or_else(|| format!("binding names absent operand#{}", operand.0))?
            .ty;
        value_types.insert(*value, ty.clone());
    }
    for (key, value) in &bindings.values {
        let ty = match key {
            ValueBindingKey::Input(port) => graph.inputs.get(*port as usize).map(|port| &port.ty),
            ValueBindingKey::Result(port) => graph.results.get(*port as usize).map(|port| &port.ty),
            ValueBindingKey::Local(local) => graph.values.get(local.0 as usize),
        }
        .ok_or_else(|| format!("value binding {key:?} has no logical type"))?;
        value_types.insert(*value, ty.clone());
    }
    let mut lowerer = Lowerer {
        graph,
        task,
        bindings,
        values: bindings.values.clone(),
        operand_values: bindings.operands.clone(),
        instructions: Vec::new(),
        next_value: bindings
            .values
            .values()
            .chain(bindings.operands.values())
            .map(|value| value.0)
            .max()
            .map_or(0, |value| value + 1),
        pending_numerical: task.numerical_semantics.clone(),
        value_types,
    };
    lowerer.block(&task.body.operations)?;
    if lowerer.instructions.is_empty() {
        lowerer.emit(
            None,
            ScalarInstructionKind::NoOp,
            vec![],
            None,
            Span::default(),
        );
    }
    if !lowerer.pending_numerical.is_empty() {
        return Err(format!(
            "task#{} has unattached numerical semantics",
            task.id.0
        ));
    }
    let mut returned =
        lowerer
            .instructions
            .iter()
            .filter_map(|instruction| match instruction.kind {
                ScalarInstructionKind::Return { value, .. } => Some(value),
                _ => None,
            });
    for operand in &task.outputs {
        let logical = graph.operand(*operand).ok_or_else(|| {
            format!(
                "task#{} names absent output operand#{}",
                task.id.0, operand.0
            )
        })?;
        let value = lowerer
            .values
            .get(&ValueBindingKey::from(&logical.value))
            .copied()
            .or_else(|| returned.next())
            .ok_or_else(|| format!("task#{} did not produce operand#{}", task.id.0, operand.0))?;
        lowerer.operand_values.insert(*operand, value);
    }
    Ok(ScalarProgram {
        task: task.id,
        instructions: lowerer.instructions,
        operand_values: lowerer.operand_values,
        value_types: lowerer.value_types,
    })
}

struct Lowerer<'a> {
    graph: &'a LogicalTaskGraph,
    task: &'a LogicalTask,
    bindings: &'a ScalarBindings,
    values: BTreeMap<ValueBindingKey, ScalarValueId>,
    operand_values: BTreeMap<OperandId, ScalarValueId>,
    instructions: Vec<ScalarInstruction>,
    next_value: u32,
    pending_numerical: Vec<NumericalEffect>,
    value_types: BTreeMap<ScalarValueId, Type>,
}

impl Lowerer<'_> {
    fn fresh(&mut self, ty: Type) -> ScalarValueId {
        let value = ScalarValueId(self.next_value);
        self.next_value += 1;
        self.value_types.insert(value, ty);
        value
    }

    fn emit(
        &mut self,
        result: Option<ScalarValueId>,
        kind: ScalarInstructionKind,
        accesses: Vec<PhysicalAccess>,
        capability: Option<ScalarCapabilitySet>,
        span: Span,
    ) {
        let numerical = std::mem::take(&mut self.pending_numerical);
        self.instructions.push(ScalarInstruction {
            result,
            kind,
            consequences: InstructionConsequences {
                accesses,
                capability,
                numerical,
                resources: InstructionResources {
                    registers: u32::from(result.is_some()),
                    private_bytes: 0,
                    workgroup_bytes: 0,
                },
            },
            span,
        });
    }

    fn block(&mut self, block: &LogicalBlock) -> Result<(), String> {
        for operation in block {
            match &operation.kind {
                LogicalOperationKind::Bind { pattern, value } => {
                    let value = self.expr(value)?;
                    self.bind(pattern, value)?;
                }
                LogicalOperationKind::Assign { target, op, value } => {
                    let accesses = self
                        .expr_storage(target)
                        .map(|storage| self.storage_access(&storage, AccessMode::Write))
                        .transpose()?
                        .unwrap_or_default();
                    let target = self.expr(target)?;
                    let value = self.expr(value)?;
                    self.emit(
                        None,
                        ScalarInstructionKind::Assign {
                            target,
                            op: *op,
                            value,
                        },
                        accesses,
                        None,
                        operation.span,
                    );
                }
                LogicalOperationKind::If {
                    condition,
                    then,
                    els,
                } => {
                    let condition = self.expr(condition)?;
                    let then_body = self.nested(then)?;
                    let else_body = self.nested(els)?;
                    let accesses = nested_accesses(&then_body, &else_body);
                    self.emit(
                        None,
                        ScalarInstructionKind::Conditional {
                            condition,
                            then_body,
                            else_body,
                        },
                        accesses,
                        None,
                        operation.span,
                    );
                }
                LogicalOperationKind::Publish { value, destination } => {
                    let accesses = self
                        .expr_storage(destination)
                        .map(|storage| self.storage_access(&storage, AccessMode::Write))
                        .transpose()?
                        .unwrap_or_default();
                    let value = self.expr(value)?;
                    let destination = self.expr(destination)?;
                    self.emit(
                        None,
                        ScalarInstructionKind::Publish { value, destination },
                        accesses,
                        None,
                        operation.span,
                    );
                }
                LogicalOperationKind::Yield(values) => {
                    let values = self.exprs(values)?;
                    self.emit(
                        None,
                        ScalarInstructionKind::Yield(values),
                        vec![],
                        None,
                        operation.span,
                    );
                }
                LogicalOperationKind::Return(writes) => {
                    for write in writes {
                        let value = self.expr(&write.value)?;
                        let accesses = if matches!(write.value.ty, Type::Tensor(_)) {
                            self.storage_access(
                                &StorageRef::Result {
                                    port: write.port,
                                    path: Vec::new(),
                                },
                                AccessMode::Write,
                            )?
                        } else {
                            vec![]
                        };
                        self.emit(
                            None,
                            ScalarInstructionKind::Return {
                                port: write.port,
                                path: write.path.clone(),
                                value,
                                transfer: write.transfer,
                            },
                            accesses,
                            None,
                            operation.span,
                        );
                    }
                }
                LogicalOperationKind::ConditionalMerge { binder, cases } => {
                    let cases = cases
                        .iter()
                        .map(|case| self.case(case))
                        .collect::<Result<Vec<_>, _>>()?;
                    let ty = self
                        .graph
                        .values
                        .get(binder.0 as usize)
                        .cloned()
                        .ok_or_else(|| {
                            format!("conditional merge binder#{} has no type", binder.0)
                        })?;
                    let result = self.fresh(ty);
                    self.values.insert(ValueBindingKey::Local(*binder), result);
                    self.emit(
                        Some(result),
                        ScalarInstructionKind::ConditionalMerge { cases },
                        vec![],
                        None,
                        operation.span,
                    );
                }
                LogicalOperationKind::Expr(value) => {
                    self.expr(value)?;
                }
                LogicalOperationKind::Reduction { .. } => {
                    return Err("logical reduction requires an explicit physical reduction".into())
                }
                LogicalOperationKind::Region(_)
                | LogicalOperationKind::Stages(_)
                | LogicalOperationKind::For { .. }
                | LogicalOperationKind::Coordinates { .. }
                | LogicalOperationKind::Members { .. } => {
                    return Err("scalar lowering received a scheduling construct".into())
                }
            }
        }
        Ok(())
    }

    fn nested(&mut self, block: &LogicalBlock) -> Result<Vec<ScalarInstruction>, String> {
        let outer = std::mem::take(&mut self.instructions);
        let values = self.values.clone();
        self.block(block)?;
        self.values = values;
        Ok(std::mem::replace(&mut self.instructions, outer))
    }

    fn bind(&mut self, pattern: &LogicalPattern, value: ScalarValueId) -> Result<(), String> {
        match pattern {
            LogicalPattern::Value(local) => {
                self.values.insert(ValueBindingKey::Local(*local), value);
            }
            LogicalPattern::Tuple(items) => {
                for (index, item) in items.iter().enumerate() {
                    let ty = pattern_type(self.graph, item)?;
                    let field = self.fresh(ty);
                    self.emit(
                        Some(field),
                        ScalarInstructionKind::Field { value, index },
                        vec![],
                        None,
                        Span::default(),
                    );
                    self.bind(item, field)?;
                }
            }
        }
        Ok(())
    }

    fn case(&mut self, case: &LogicalConditionalCase) -> Result<ConditionalCase, String> {
        let value = *self
            .operand_values
            .get(&case.value)
            .ok_or_else(|| format!("operand#{} has no physical value", case.value.0))?;
        let predicates = case
            .predicates
            .iter()
            .map(|predicate| Ok((self.expr(&predicate.condition)?, predicate.when_true)))
            .collect::<Result<_, String>>()?;
        Ok(ConditionalCase { predicates, value })
    }

    fn expr(&mut self, expression: &LogicalExpr) -> Result<ScalarValueId, String> {
        let span = expression.span;
        let (kind, accesses, capability) = match &expression.kind {
            LogicalExprKind::Value(value) => {
                return self
                    .values
                    .get(&ValueBindingKey::from(value))
                    .copied()
                    .ok_or_else(|| format!("missing physical value for {value:?}"))
            }
            LogicalExprKind::Coordinate(value) => {
                return self
                    .values
                    .get(&ValueBindingKey::Local(*value))
                    .copied()
                    .ok_or_else(|| format!("missing coordinate local#{}", value.0))
            }
            LogicalExprKind::Int(value) => (
                ScalarInstructionKind::Literal(ScalarLiteral::Int(*value)),
                vec![],
                None,
            ),
            LogicalExprKind::Float(value) => (
                ScalarInstructionKind::Literal(ScalarLiteral::Float(*value)),
                vec![],
                None,
            ),
            LogicalExprKind::Bool(value) => (
                ScalarInstructionKind::Literal(ScalarLiteral::Bool(*value)),
                vec![],
                None,
            ),
            LogicalExprKind::Shape(value) => {
                (ScalarInstructionKind::Shape(value.clone()), vec![], None)
            }
            LogicalExprKind::Tuple(values) => (
                ScalarInstructionKind::Tuple(self.exprs(values)?),
                vec![],
                None,
            ),
            LogicalExprKind::Range(lo, hi) => (
                ScalarInstructionKind::Range(self.expr(lo)?, self.expr(hi)?),
                vec![],
                None,
            ),
            LogicalExprKind::Field(value, index) => (
                ScalarInstructionKind::Field {
                    value: self.expr(value)?,
                    index: *index,
                },
                vec![],
                None,
            ),
            LogicalExprKind::Construct { storage } => {
                self.storage_op(StorageOperation::Construct, *storage, vec![], None)?
            }
            LogicalExprKind::Filled {
                storage,
                like,
                value,
            } => {
                let like = self.expr(like)?;
                self.storage_op(StorageOperation::Fill, *storage, vec![like], Some(*value))?
            }
            LogicalExprKind::View { view, base } => {
                let base = self.expr(base)?;
                (
                    ScalarInstructionKind::View { view: *view, base },
                    self.view_access(*view, AccessMode::Read)?,
                    None,
                )
            }
            LogicalExprKind::Index { base, indices } => {
                let accesses = self
                    .expr_storage(base)
                    .map(|storage| self.storage_access(&storage, AccessMode::Read))
                    .transpose()?
                    .unwrap_or_default();
                let base = self.expr(base)?;
                let indices = indices
                    .iter()
                    .map(|index| self.index(index))
                    .collect::<Result<_, _>>()?;
                (
                    ScalarInstructionKind::Index { base, indices },
                    accesses,
                    None,
                )
            }
            LogicalExprKind::Snapshot { storage, source } => {
                let source = self.expr(source)?;
                self.storage_op(StorageOperation::Snapshot, *storage, vec![source], None)?
            }
            LogicalExprKind::Decode { storage, source } => {
                let source = self.expr(source)?;
                self.storage_op(StorageOperation::Decode, *storage, vec![source], None)?
            }
            LogicalExprKind::Materialize { storage, value } => {
                let value = self.expr(value)?;
                self.storage_op(StorageOperation::Materialize, *storage, vec![value], None)?
            }
            LogicalExprKind::Cast { dtype, expr } => {
                let value = self.expr(expr)?;
                (
                    ScalarInstructionKind::Cast {
                        dtype: *dtype,
                        value,
                    },
                    vec![],
                    None,
                )
            }
            LogicalExprKind::Unary { op, expr } => {
                let value = self.expr(expr)?;
                (
                    ScalarInstructionKind::Unary { op: *op, value },
                    vec![],
                    None,
                )
            }
            LogicalExprKind::Binary { op, lhs, rhs } => {
                let lhs = self.expr(lhs)?;
                let rhs = self.expr(rhs)?;
                (
                    ScalarInstructionKind::Binary { op: *op, lhs, rhs },
                    vec![],
                    None,
                )
            }
            LogicalExprKind::Math { op, args } => (
                ScalarInstructionKind::Math {
                    op: *op,
                    arguments: self.exprs(args)?,
                },
                vec![],
                None,
            ),
            LogicalExprKind::Select { cond, then, els } => {
                let condition = self.expr(cond)?;
                let then_value = self.expr(then)?;
                let else_value = self.expr(els)?;
                (
                    ScalarInstructionKind::Select {
                        condition,
                        then_value,
                        else_value,
                    },
                    vec![],
                    None,
                )
            }
            LogicalExprKind::Extent { base, axis } => {
                let base = self.expr(base)?;
                (
                    ScalarInstructionKind::Extent { base, axis: *axis },
                    vec![],
                    None,
                )
            }
            LogicalExprKind::Intrinsic { operation, args } => {
                let mut accesses = Vec::new();
                for (ordinal, argument) in args.iter().enumerate() {
                    if let Some(storage) = self.expr_storage(argument) {
                        let mode = if operation.writes_arguments().contains(&ordinal) {
                            AccessMode::ReadWrite
                        } else {
                            AccessMode::Read
                        };
                        accesses.extend(self.storage_access(&storage, mode)?);
                    }
                }
                (
                    ScalarInstructionKind::Intrinsic {
                        operation: *operation,
                        arguments: self.exprs(args)?,
                    },
                    accesses,
                    Some(ScalarCapabilitySet(self.task.capabilities.clone())),
                )
            }
            LogicalExprKind::Accessor { base, name } => {
                let base = self.expr(base)?;
                (
                    ScalarInstructionKind::Accessor {
                        base,
                        name: name.clone(),
                    },
                    vec![],
                    None,
                )
            }
            LogicalExprKind::Geometry { base, axis, valid } => {
                let base = self.expr(base)?;
                (
                    ScalarInstructionKind::Geometry {
                        base,
                        axis: *axis,
                        valid: *valid,
                    },
                    vec![],
                    None,
                )
            }
            LogicalExprKind::Atomic { op, place, value } => {
                let accesses = self
                    .expr_storage(place)
                    .map(|storage| self.storage_access(&storage, AccessMode::Atomic))
                    .transpose()?
                    .unwrap_or_default();
                let place = self.expr(place)?;
                let value = self.expr(value)?;
                (
                    ScalarInstructionKind::Atomic {
                        op: *op,
                        place,
                        value,
                    },
                    accesses,
                    None,
                )
            }
            LogicalExprKind::Call { .. } => return Err("scalar lowering received a call".into()),
            LogicalExprKind::Reduce { .. } => {
                return Err("scalar lowering received an unresolved reduction".into())
            }
            LogicalExprKind::Region(_) => return Err("scalar lowering received a region".into()),
        };
        let result = self.fresh(expression.ty.clone());
        self.emit(Some(result), kind, accesses, capability, span);
        Ok(result)
    }

    fn exprs(&mut self, values: &[LogicalExpr]) -> Result<Vec<ScalarValueId>, String> {
        values.iter().map(|value| self.expr(value)).collect()
    }

    fn index(&mut self, index: &LogicalIndex) -> Result<ScalarIndex, String> {
        Ok(match index {
            LogicalIndex::Point(value) => ScalarIndex::Point(self.expr(value)?),
            LogicalIndex::Coordinate(value) => ScalarIndex::Coordinate(
                *self
                    .values
                    .get(&ValueBindingKey::Local(*value))
                    .ok_or_else(|| format!("missing coordinate local#{}", value.0))?,
            ),
            LogicalIndex::Slice(slice) => ScalarIndex::Slice(*slice),
            LogicalIndex::Range { start, end } => ScalarIndex::Range {
                start: start.as_ref().map(|value| self.expr(value)).transpose()?,
                end: end.as_ref().map(|value| self.expr(value)).transpose()?,
            },
        })
    }

    fn storage_op(
        &self,
        operation: StorageOperation,
        storage: LocalStorageId,
        inputs: Vec<ScalarValueId>,
        fill_bits: Option<u64>,
    ) -> Result<
        (
            ScalarInstructionKind,
            Vec<PhysicalAccess>,
            Option<ScalarCapabilitySet>,
        ),
        String,
    > {
        let key = StorageRef::Local(storage);
        let bindings = self
            .bindings
            .storage
            .get(&key)
            .cloned()
            .ok_or_else(|| format!("missing physical storage for local#{}", storage.0))?;
        let mode = if matches!(
            operation,
            StorageOperation::Construct | StorageOperation::Fill | StorageOperation::Materialize
        ) {
            AccessMode::Write
        } else {
            AccessMode::ReadWrite
        };
        Ok((
            ScalarInstructionKind::Storage {
                operation,
                bindings: bindings.clone(),
                inputs,
                fill_bits,
            },
            accesses(&bindings, mode),
            None,
        ))
    }

    fn storage_access(
        &self,
        storage: &StorageRef,
        mode: AccessMode,
    ) -> Result<Vec<PhysicalAccess>, String> {
        self.bindings
            .storage
            .get(storage)
            .map(|bindings| accesses(bindings, mode))
            .ok_or_else(|| format!("missing physical storage mapping for {storage:?}"))
    }

    fn view_access(
        &self,
        view: LocalViewId,
        mode: AccessMode,
    ) -> Result<Vec<PhysicalAccess>, String> {
        self.bindings
            .views
            .get(&view)
            .map(|bindings| accesses(bindings, mode))
            .ok_or_else(|| format!("missing physical view mapping for view#{}", view.0))
    }

    fn expr_storage(&self, expression: &LogicalExpr) -> Option<StorageRef> {
        match &expression.kind {
            LogicalExprKind::Construct { storage }
            | LogicalExprKind::Filled { storage, .. }
            | LogicalExprKind::Snapshot { storage, .. }
            | LogicalExprKind::Decode { storage, .. }
            | LogicalExprKind::Materialize { storage, .. } => Some(StorageRef::Local(*storage)),
            LogicalExprKind::View { view, .. } => self
                .graph
                .views
                .get(view.0 as usize)
                .map(|view| view.storage.clone()),
            LogicalExprKind::Value(ValueRef::Input(port))
                if matches!(expression.ty, Type::Tensor(_)) =>
            {
                Some(StorageRef::Input {
                    port: *port,
                    path: vec![],
                })
            }
            LogicalExprKind::Value(ValueRef::Result(port))
                if matches!(expression.ty, Type::Tensor(_)) =>
            {
                Some(StorageRef::Result {
                    port: *port,
                    path: vec![],
                })
            }
            LogicalExprKind::Field(base, _)
            | LogicalExprKind::Index { base, .. }
            | LogicalExprKind::Accessor { base, .. }
            | LogicalExprKind::Geometry { base, .. } => self.expr_storage(base),
            _ => None,
        }
    }
}

fn accesses(bindings: &[ScalarStorageBinding], mode: AccessMode) -> Vec<PhysicalAccess> {
    bindings
        .iter()
        .map(|binding| PhysicalAccess {
            storage: binding.storage,
            mode,
        })
        .collect()
}

fn pattern_type(graph: &LogicalTaskGraph, pattern: &LogicalPattern) -> Result<Type, String> {
    match pattern {
        LogicalPattern::Value(local) => graph
            .values
            .get(local.0 as usize)
            .cloned()
            .ok_or_else(|| format!("pattern local#{} has no type", local.0)),
        LogicalPattern::Tuple(items) => items
            .iter()
            .map(|item| pattern_type(graph, item))
            .collect::<Result<Vec<_>, _>>()
            .map(Type::Tuple),
    }
}

fn nested_accesses(
    then_body: &[ScalarInstruction],
    else_body: &[ScalarInstruction],
) -> Vec<PhysicalAccess> {
    let mut values = then_body
        .iter()
        .chain(else_body)
        .flat_map(|instruction| instruction.consequences.accesses.iter().cloned())
        .collect::<Vec<_>>();
    values.sort_by_key(|access| (access.storage, mode_order(access.mode)));
    values.dedup_by(|a, b| a.storage == b.storage && a.mode == b.mode);
    values
}

fn mode_order(mode: AccessMode) -> u8 {
    match mode {
        AccessMode::Read => 0,
        AccessMode::Write => 1,
        AccessMode::ReadWrite => 2,
        AccessMode::Atomic => 3,
    }
}
