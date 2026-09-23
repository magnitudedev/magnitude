//! Recurrence inference over the actual checked region and its bound operands.
//! The kernel loop constructor checks the resulting inductive carry schema.
use super::*;
use registry::IntrinsicUniformity as U;

pub(super) fn join(a: U, b: U) -> U {
    match (a, b) {
        (U::Varying, _) | (_, U::Varying) => U::Varying,
        (U::Subgroup, _) | (_, U::Subgroup) => U::Subgroup,
        _ => U::Workgroup,
    }
}
pub(super) fn all(values: impl IntoIterator<Item = U>) -> U {
    values.into_iter().fold(U::Workgroup, join)
}

fn scalar_failure_scope(recipe: &seismic_lang::reference_math::ReferenceRecipe, inputs: &[U]) -> U {
    use seismic_lang::reference_math::ReferenceNode as N;
    if recipe.failures().is_empty() {
        return U::Workgroup;
    }
    let mut scopes = Vec::with_capacity(recipe.nodes().len());
    for node in recipe.nodes() {
        let get = |value: seismic_lang::reference_math::ReferenceValue| scopes[value.ordinal()];
        scopes.push(match *node {
            N::Input { operand, .. } => inputs[operand as usize],
            N::Constant(_) => U::Workgroup,
            N::Word { a, b, .. } | N::Compare { a, b, .. } | N::And { a, b } => {
                join(get(a), get(b))
            }
            N::Not { value } | N::Bits { value } | N::FromBits { value, .. } => get(value),
            N::Select { condition, yes, no } => all([get(condition), get(yes), get(no)]),
        });
    }
    all(recipe
        .failures()
        .iter()
        .map(|(predicate, _)| scopes[predicate.ordinal()]))
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Value {
    scalar: U,
    axes: Vec<U>,
    members: Vec<Value>,
}
impl Value {
    pub(super) fn scalar(scalar: U) -> Self {
        Self {
            scalar,
            axes: vec![],
            members: vec![],
        }
    }
    pub(super) fn scalar_scope(&self) -> U {
        self.scalar
    }
    fn under(mut self, control: U) -> Self {
        self.scalar = join(self.scalar, control);
        self.axes
            .iter_mut()
            .for_each(|axis| *axis = join(*axis, control));
        self.members = self.members.into_iter().map(|v| v.under(control)).collect();
        self
    }
}
pub(super) type Values = BTreeMap<SemanticValueId, Value>;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Meaning {
    /// Scope of a physical value, including the control that defined it.
    Physical,
    /// Equality of the actual successful source result. Failed executions
    /// have no result; they do not contribute an invented placeholder value.
    Successful,
}

struct Recurrence {
    carries: Vec<Value>,
    safe: U,
    body: Values,
}
fn value(values: &Values, id: SemanticValueId) -> Value {
    values
        .get(&id)
        .cloned()
        .expect("uniformity operand is not bound")
}
fn scalar(values: &Values, id: SemanticValueId) -> U {
    value(values, id).scalar
}

pub(super) fn bound<B: seismic_target::TargetFamily>(
    kernel: &PortableBuilder<'_, B>,
    bound: &SegmentBound,
) -> Value {
    match bound {
        SegmentBound::Scalar(v) => Value::scalar(kernel.uniformity(*v)),
        SegmentBound::Tensor(t) => Value {
            // A tensor address is not a stability argument for its contents.
            scalar: U::Varying,
            axes: t.axes.iter().map(|v| kernel.uniformity(*v)).collect(),
            members: vec![],
        },
        SegmentBound::Tuple(v) => Value {
            scalar: U::Workgroup,
            axes: vec![],
            members: v.iter().map(|v| self::bound(kernel, v)).collect(),
        },
        SegmentBound::Range { start, end } => Value {
            scalar: U::Workgroup,
            axes: vec![],
            members: vec![self::bound(kernel, start), self::bound(kernel, end)],
        },
        SegmentBound::Opaque(_) => Value::scalar(U::Varying),
        SegmentBound::Unit => Value::scalar(U::Workgroup),
    }
}

pub(super) struct Inference<'a> {
    pub arena: &'a ExprArena,
    pub helpers: &'a BTreeMap<FamilyId, &'a SemanticFunction>,
}
impl Inference<'_> {
    /// Derived from this selected function and the actual captured operands.
    /// The result is for source-value equality only; it cannot change an IR
    /// value's scope or establish that every cohort member successfully ran.
    pub(super) fn successful_values(
        &self,
        function: &SemanticFunction,
        region: RegionId,
        inputs: &Values,
        control: U,
    ) -> Values {
        let mut values = inputs.clone();
        self.region_with(function, region, &mut values, control, Meaning::Successful);
        values
    }
    fn expression(&self, values: &Values, expression: AnyExpr, control: U) -> U {
        all(self.arena.free_symbols(expression).into_iter().map(|s| {
            match self.arena.symbol_kind(s) {
                SymbolKind::RuntimeValue(v) => scalar(values, v),
                SymbolKind::LoopBinder(_) => control,
                SymbolKind::CallDimension(_)
                | SymbolKind::CallScalar(_)
                | SymbolKind::TargetConstant(_)
                | SymbolKind::Decision(_)
                | SymbolKind::ScheduleSlot(_) => U::Workgroup,
                SymbolKind::TemplateDimension(_) => unreachable!("uninstantiated template"),
            }
        }))
    }
    fn shaped(
        &self,
        function: &SemanticFunction,
        values: &Values,
        id: SemanticValueId,
        control: U,
    ) -> Value {
        let axes = match &function.value(id).ty {
            SemanticType::Tensor(t) => t
                .axes
                .iter()
                .map(|a| self.expression(values, (*a).into(), control))
                .collect(),
            _ => vec![],
        };
        Value {
            scalar: U::Varying,
            axes,
            members: vec![],
        }
    }
    fn region(
        &self,
        function: &SemanticFunction,
        region: RegionId,
        values: &mut Values,
        control: U,
    ) -> U {
        self.region_with(function, region, values, control, Meaning::Physical)
    }
    fn region_with(
        &self,
        function: &SemanticFunction,
        region: RegionId,
        values: &mut Values,
        mut control: U,
        meaning: Meaning,
    ) -> U {
        let mut safe = U::Workgroup;
        for (_, node) in function.nodes(region) {
            match node.view() {
                SemanticNodeView::Primitive {
                    primitive,
                    inputs,
                    output,
                } => {
                    if meaning == Meaning::Physical {
                        if let Some(recipe) = source_scalar::recipe(function, primitive, inputs) {
                            let inputs = inputs
                                .iter()
                                .map(|id| scalar(values, *id))
                                .collect::<Vec<_>>();
                            safe = join(safe, scalar_failure_scope(&recipe, &inputs));
                            control = join(control, safe);
                        }
                    }
                    let v = match primitive {
                        PrimitiveId::RangeMake => Value {
                            scalar: U::Workgroup,
                            axes: vec![],
                            members: inputs.iter().map(|v| value(values, *v)).collect(),
                        },
                        PrimitiveId::RangeStart | PrimitiveId::RangeEnd => value(values, inputs[0])
                            .members[usize::from(matches!(primitive, PrimitiveId::RangeEnd))]
                        .clone(),
                        PrimitiveId::Symbolic(e) => {
                            Value::scalar(self.expression(values, (*e).into(), control))
                        }
                        _ => Value::scalar(all(inputs.iter().map(|v| scalar(values, *v)))),
                    };
                    values.insert(output, v.under(control));
                }
                SemanticNodeView::Intrinsic {
                    intrinsic, output, ..
                } => {
                    let result = match &function.value(output).ty {
                        SemanticType::Tensor(_) => self.shaped(function, values, output, control),
                        _ => Value::scalar(
                            registry::intrinsic_signature(intrinsic)
                                .effects
                                .result_uniformity,
                        ),
                    };
                    values.insert(output, result.under(control));
                }
                SemanticNodeView::Elementwise {
                    primitive,
                    inputs,
                    output,
                } => {
                    if meaning == Meaning::Physical {
                        if let Some(recipe) = source_scalar::recipe(function, primitive, inputs) {
                            let inputs = inputs
                                .iter()
                                .map(|id| scalar(values, *id))
                                .collect::<Vec<_>>();
                            safe = join(safe, scalar_failure_scope(&recipe, &inputs));
                            control = join(control, safe);
                        }
                    }
                    let v = self.shaped(function, values, output, control);
                    values.insert(output, v.under(control));
                }
                SemanticNodeView::Alloc { extents, output } => {
                    let v = Value { scalar: U::Varying, axes: extents.iter().map(|id| scalar(values,*id)).collect(), members: vec![] };
                    values.insert(output,v.under(control));
                }
                SemanticNodeView::Fill { like: input, output, .. }
                | SemanticNodeView::Copy { input, output }
                | SemanticNodeView::RepresentationConvert { input, output, .. } => {
                    let mut v = value(values,input);
                    v.scalar = U::Varying;
                    values.insert(output,v.under(control));
                }
                SemanticNodeView::Reduce { output, .. } => {
                    let v = self.shaped(function, values, output, control);
                    values.insert(output, v.under(control));
                }
                SemanticNodeView::View {
                    base,
                    transform,
                    extents,
                    output,
                } => {
                    let mut v = value(values, base);
                    match transform {
                        ViewTransform::Identity => {}
                        ViewTransform::Transpose { permutation } => {
                            v.axes = permutation.iter().map(|i| v.axes[*i as usize]).collect();
                        }
                        ViewTransform::Reshape { .. } => {
                            v.axes = extents.iter().map(|id| scalar(values,*id)).collect();
                        }
                        ViewTransform::Plane { .. } => {
                            v = self.shaped(function, values, output, control);
                        }
                        ViewTransform::Slice { axes } => {
                            let mut out = vec![];
                            let reference = |r: &ScalarRef| match r {
                                ScalarRef::Value(v) => scalar(values, *v),
                                ScalarRef::Static(e) => {
                                    self.expression(values, (*e).into(), control)
                                }
                            };
                            for (i, axis) in axes.iter().enumerate() {
                                match axis {
                                    SliceAxis::Point { .. } => {}
                                    SliceAxis::Full => out.push(v.axes[i]),
                                    SliceAxis::Range { start, end, .. } => out.push(all([
                                        start.as_ref().map(&reference).unwrap_or(U::Workgroup),
                                        end.as_ref().map(&reference).unwrap_or(v.axes[i]),
                                    ])),
                                }
                            }
                            out.extend_from_slice(&v.axes[axes.len()..]);
                            v.axes = out;
                        }
                    }
                    values.insert(output, v.under(control));
                }
                SemanticNodeView::ElementRead { output, .. } => {
                    // Inference cannot grant stability to mutable storage merely
                    // because every participant computes the same address.
                    values.insert(output, Value::scalar(U::Varying));
                }
                SemanticNodeView::ElementWrite { place, output, .. }
                | SemanticNodeView::Atomic { place, output, .. }
                | SemanticNodeView::Store {
                    destination: place,
                    output,
                    ..
                } => {
                    let v = value(values, place).under(control);
                    values.insert(output, v);
                }
                SemanticNodeView::Extent {
                    tensor,
                    axis,
                    output,
                } => {
                    let scope = value(values, tensor).axes[axis as usize];
                    values.insert(output, Value::scalar(scope).under(control));
                }
                SemanticNodeView::TuplePack { inputs, output } => {
                    let members = inputs.iter().map(|v| value(values, *v)).collect();
                    values.insert(
                        output,
                        Value {
                            scalar: U::Workgroup,
                            axes: vec![],
                            members,
                        }
                        .under(control),
                    );
                }
                SemanticNodeView::TupleGet {
                    tuple,
                    index,
                    output,
                } => {
                    let v = value(values, tuple).members[index as usize]
                        .clone()
                        .under(control);
                    values.insert(output, v);
                }
                SemanticNodeView::Check { condition, .. } => {
                    if meaning == Meaning::Physical {
                        safe = join(safe, scalar(values, condition));
                        control = join(control, safe);
                    }
                }
                SemanticNodeView::Call {
                    family,
                    inputs,
                    outputs,
                } => {
                    let callee = self.helpers[&family];
                    // Instantiated shape expressions may retain the caller's
                    // scoped RuntimeValue IDs. Preserve that actual environment
                    // while binding callee formals; only declared results leave
                    // this child traversal.
                    let mut child = values.clone();
                    for (p, input) in callee.parameters().iter().zip(inputs) {
                        child.insert(p.value, value(values, *input));
                    }
                    let child_safe =
                        self.region_with(callee, callee.root(), &mut child, control, meaning);
                    if meaning == Meaning::Physical {
                        safe = join(safe, child_safe);
                        control = join(control, safe);
                    }
                    for (output, result) in outputs.iter().zip(callee.results()) {
                        values.insert(*output, value(&child, *result).under(control));
                    }
                }
                SemanticNodeView::If {
                    condition,
                    captures,
                    outputs,
                    then,
                    otherwise,
                } => {
                    let branch_control = join(control, scalar(values, condition));
                    let mut arms = vec![];
                    for arm in [then, otherwise] {
                        let mut child = values.clone();
                        for (p, input) in function.region(arm).parameters().iter().zip(captures) {
                            child.insert(*p, value(values, *input));
                        }
                        let arm_safe =
                            self.region_with(function, arm, &mut child, branch_control, meaning);
                        if meaning == Meaning::Physical {
                            safe = join(safe, join(branch_control, arm_safe));
                        }
                        arms.push(
                            function
                                .region(arm)
                                .results()
                                .iter()
                                .map(|v| value(&child, *v))
                                .collect::<Vec<_>>(),
                        );
                        // Same-function lexical definitions have distinct IDs.
                        // Callee-local definitions are deliberately not exported:
                        // a later call can substitute different actual operands.
                        if meaning == Meaning::Successful {
                            values.extend(child);
                        }
                    }
                    for ((out, left), right) in outputs.iter().zip(&arms[0]).zip(&arms[1]) {
                        values.insert(*out, merge(left, right).under(branch_control));
                    }
                    if meaning == Meaning::Physical {
                        control = join(control, safe);
                    }
                }
                SemanticNodeView::Loop {
                    start,
                    end,
                    captures,
                    body,
                    carries,
                    ..
                } => {
                    let range = all([control, scalar(values, start), scalar(values, end)]);
                    let recurrence = self
                        .recurrence_with(function, body, captures, carries, values, range, meaning);
                    if meaning == Meaning::Successful {
                        values.extend(recurrence.body);
                    }
                    for (carry, scope) in carries.iter().zip(recurrence.carries) {
                        values.insert(carry.result, scope);
                    }
                    if meaning == Meaning::Physical {
                        safe = join(safe, recurrence.safe);
                        control = join(control, safe);
                    }
                }
            }
        }
        join(safe, control)
    }
    pub(super) fn recurrence(
        &self,
        function: &SemanticFunction,
        body: RegionId,
        captures: &[SemanticValueId],
        carries: &[seismic_lang::entry::Carry],
        parent: &Values,
        range: U,
    ) -> Vec<U> {
        let recurrence = self.recurrence_with(
            function,
            body,
            captures,
            carries,
            parent,
            range,
            Meaning::Physical,
        );
        recurrence
            .carries
            .into_iter()
            .map(|v| v.scalar)
            .chain(std::iter::once(recurrence.safe))
            .collect()
    }
    fn recurrence_with(
        &self,
        function: &SemanticFunction,
        body: RegionId,
        captures: &[SemanticValueId],
        carries: &[seismic_lang::entry::Carry],
        parent: &Values,
        range: U,
        meaning: Meaning,
    ) -> Recurrence {
        let mut current = carries
            .iter()
            .map(|c| value(parent, c.initial).under(range))
            .collect::<Vec<_>>();
        let mut current_safe = range;
        loop {
            let mut values = parent.clone();
            let parameters = function.region(body).parameters();
            values.insert(parameters[0], Value::scalar(range));
            for (p, capture) in parameters[1..].iter().zip(captures) {
                values.insert(*p, value(parent, *capture));
            }
            for (carry, scope) in carries.iter().zip(&current) {
                values.insert(carry.parameter, scope.clone());
            }
            let safe = self.region_with(
                function,
                body,
                &mut values,
                if meaning == Meaning::Physical {
                    join(range, current_safe)
                } else {
                    range
                },
                meaning,
            );
            let next = carries
                .iter()
                .zip(&current)
                .map(|(c, before)| merge(before, &value(&values, c.yielded)))
                .collect::<Vec<_>>();
            let next_safe = join(current_safe, safe);
            if next == current && next_safe == current_safe {
                return Recurrence {
                    carries: current,
                    safe: current_safe,
                    body: values,
                };
            }
            // Each component can rise twice in this finite lattice. There is
            // no heuristic iteration cutoff and no optimistic partial result.
            current = next;
            current_safe = next_safe;
        }
    }
}
fn merge(a: &Value, b: &Value) -> Value {
    assert_eq!(a.axes.len(), b.axes.len());
    assert_eq!(a.members.len(), b.members.len());
    Value {
        scalar: join(a.scalar, b.scalar),
        axes: a
            .axes
            .iter()
            .zip(&b.axes)
            .map(|(a, b)| join(*a, *b))
            .collect(),
        members: a
            .members
            .iter()
            .zip(&b.members)
            .map(|(a, b)| merge(a, b))
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use seismic_lang::checked::{check_source, SourceFile, SourceSet};
    use seismic_lang::entry::ElementBindings;

    #[test]
    fn shift_failure_scope_comes_from_count_not_shifted_payload() {
        use seismic_lang::reference_math::{scalar_recipe, ScalarOp};
        let recipe = scalar_recipe(
            ScalarOp::Binary(ast::BinaryOp::Shl),
            &[DType::I32, DType::I32],
        );
        assert_eq!(
            scalar_failure_scope(&recipe, &[U::Varying, U::Workgroup]),
            U::Workgroup
        );
        assert_eq!(
            scalar_failure_scope(&recipe, &[U::Workgroup, U::Varying]),
            U::Varying
        );
        assert_eq!(
            scalar_failure_scope(&recipe, &[U::Varying, U::Subgroup]),
            U::Subgroup
        );
    }

    #[test]
    fn carry_scope_reaches_a_fixed_point_through_selected_helper() {
        let module = check_source(SourceSet::new(vec![SourceFile {
            path: "inductive-carry.seismic".into(),
            text: r#"fn identity(x: i32) -> i32:
    return x

fn probe(x: i32, times: range[4]) -> i32:
    let mut early = 0
    let mut late = 0
    for i in times:
        late = early
        early = identity(x)
    return late
"#
            .into(),
        }]))
        .unwrap();
        let entry = module
            .entry(
                module.entry_named("probe").unwrap(),
                &ElementBindings::default(),
            )
            .unwrap();
        let program = entry.program();
        let helpers = program
            .families()
            .map(|(id, family)| (id, program.function(family.reference().function())))
            .collect();
        let function = program.function(program.family(program.root()).reference().function());
        let inference = Inference {
            arena: entry.arena(),
            helpers: &helpers,
        };
        for input in [U::Workgroup, U::Subgroup, U::Varying] {
            let mut values = Values::new();
            values.insert(function.parameters()[0].value, Value::scalar(input));
            values.insert(
                function.parameters()[1].value,
                Value {
                    scalar: U::Workgroup,
                    axes: vec![],
                    members: vec![Value::scalar(U::Workgroup), Value::scalar(U::Workgroup)],
                },
            );
            let successful =
                inference.successful_values(function, function.root(), &values, U::Workgroup);
            assert_eq!(scalar(&successful, function.results()[0]), input);
            inference.region(function, function.root(), &mut values, U::Workgroup);
            assert_eq!(scalar(&values, function.results()[0]), input,
                "the second carry must inherit the first carry's later scope, including through a call");
        }
    }

    fn result_scopes(source: &str, inputs: &[U], meaning: Meaning) -> Vec<U> {
        let module = check_source(SourceSet::new(vec![SourceFile {
            path: "successful-equality.seismic".into(),
            text: source.into(),
        }]))
        .unwrap();
        let entry = module
            .entry(
                module.entry_named("probe").unwrap(),
                &ElementBindings::default(),
            )
            .unwrap();
        let program = entry.program();
        let helpers = program
            .families()
            .map(|(id, family)| (id, program.function(family.reference().function())))
            .collect();
        let function = program.function(program.family(program.root()).reference().function());
        assert_eq!(function.parameters().len(), inputs.len());
        let mut values = function
            .parameters()
            .iter()
            .zip(inputs)
            .map(|(parameter, scope)| (parameter.value, Value::scalar(*scope)))
            .collect();
        Inference {
            arena: entry.arena(),
            helpers: &helpers,
        }
        .region_with(
            function,
            function.root(),
            &mut values,
            U::Workgroup,
            meaning,
        );
        function
            .results()
            .iter()
            .map(|id| scalar(&values, *id))
            .collect()
    }

    #[test]
    fn successful_constant_is_not_a_failed_lanes_placeholder() {
        let source = r#"fn checked(x: i32) -> i32:
    let ignored = 1 / x
    return 7

fn probe(x: i32) -> i32:
    return checked(x)
"#;
        assert_eq!(
            result_scopes(source, &[U::Varying], Meaning::Physical),
            vec![U::Varying]
        );
        assert_eq!(
            result_scopes(source, &[U::Varying], Meaning::Successful),
            vec![U::Workgroup]
        );
    }

    #[test]
    fn successful_parent_continuation_recovers_after_varying_branch() {
        let source = r#"fn probe(condition: bool) -> i32:
    let mut ignored = 0
    if condition:
        ignored = 1
    else:
        ignored = 2
    return 7
"#;
        assert_eq!(
            result_scopes(source, &[U::Varying], Meaning::Successful),
            vec![U::Workgroup]
        );
        let selected = source.replace("return 7", "return ignored");
        assert_eq!(
            result_scopes(&selected, &[U::Varying], Meaning::Successful),
            vec![U::Varying]
        );
    }

    #[test]
    fn successful_helper_results_use_each_calls_actual_operands() {
        let source = r#"fn identity(x: i32) -> i32:
    return x

fn probe(varying: i32, uniform: i32) -> i32:
    let ignored = identity(varying)
    return identity(uniform)
"#;
        assert_eq!(
            result_scopes(source, &[U::Varying, U::Workgroup], Meaning::Successful),
            vec![U::Workgroup]
        );
        let reversed = source
            .replace("identity(varying)", "identity(uniform)")
            .replace("return identity(uniform)", "return identity(varying)");
        assert_eq!(
            result_scopes(&reversed, &[U::Varying, U::Workgroup], Meaning::Successful),
            vec![U::Varying]
        );
    }
}
