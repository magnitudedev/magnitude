//! Source-ordered continuation inside one native parallel segment.
use super::*;

use seismic_ir::kernel::dynamic::{PortableBranch, PortableRepeat};

pub(super) enum SegmentFrame {
    Nodes {
        region: RegionId,
        next: usize,
    },
    Bind {
        outputs: Vec<SemanticValueId>,
        results: Vec<SemanticValueId>,
    },
    Then(SegmentIf),
    Otherwise {
        branch: SegmentIf,
        alive: PortableValue,
        products: Vec<SegmentBound>,
    },
    Repeat(SegmentRepeat),
}
impl SegmentFrame {
    pub(super) fn region(region: RegionId) -> Self {
        Self::Nodes { region, next: 0 }
    }
}

#[derive(Clone)]
pub(super) struct SegmentEnvironment {
    pub(super) function: FunctionId,
    pub(super) values: BTreeMap<SemanticValueId, SegmentBound>,
    pub(super) alive: PortableValue,
    pub(super) successful: uniformity::Values,
    pub(super) lexical: registry::IntrinsicUniformity,
}

pub(super) struct SegmentIf {
    token: PortableBranch,
    condition: PortableValue,
    captures: Vec<SegmentBound>,
    outputs: Vec<SemanticValueId>,
    then: RegionId,
    otherwise: RegionId,
    parent: SegmentEnvironment,
    lexical: registry::IntrinsicUniformity,
}

pub(super) struct SegmentRepeat {
    token: PortableRepeat,
    parent: SegmentEnvironment,
    body: RegionId,
    carries: Vec<seismic_lang::entry::Carry>,
    before: Vec<PortableValue>,
    scope: Option<cohort::Scope>,
}

impl<'s, 'k, 'f, 'r, B: seismic_target::TargetFamily> SegmentLowerer<'s, 'k, 'f, 'r, B> {
    pub(super) fn initialize_successful_values(&mut self, region: RegionId) {
        let inputs = self
            .values
            .iter()
            .map(|(id, value)| {
                (
                    *id,
                    self.successful
                        .get(id)
                        .cloned()
                        .unwrap_or_else(|| uniformity::bound(self.kernel, value)),
                )
            })
            .collect();
        let inference = uniformity::Inference {
            arena: self.kernel.expression_arena(),
            helpers: self.helpers,
        };
        self.successful = inference.successful_values(self.function, region, &inputs, self.lexical);
    }

    fn collective_gate(&mut self, scope: cohort::Scope) {
        let cohort = self
            .cohort
            .expect("collective segment owns rendezvous storage");
        let promoted = self
            .values
            .iter()
            .filter_map(|(id, bound)| {
                let SegmentBound::Scalar(value) = bound else {
                    return None;
                };
                let successful = self.successful.get(id)?.scalar_scope();
                let required = scope.uniformity();
                (uniformity::join(successful, required) == required
                    && uniformity::join(self.kernel.uniformity(*value), required) != required)
                    .then_some(*value)
            })
            .collect::<Vec<_>>();
        let (alive, values) = cohort.gate(self.kernel, scope, self.alive, &promoted);
        self.alive = alive;
        if !promoted.is_empty() {
            for bound in self.values.values_mut() {
                let fields = bound
                    .scalar_fields()
                    .into_iter()
                    .map(|field| {
                        promoted
                            .iter()
                            .position(|before| *before == field)
                            .map(|i| values[i])
                            .unwrap_or(field)
                    })
                    .collect::<Vec<_>>();
                *bound = bound.with_scalar_fields(self.kernel, &mut fields.into_iter());
            }
        }
    }

    pub(super) fn environment(&self) -> SegmentEnvironment {
        SegmentEnvironment {
            function: self.function.id(),
            values: self.values.clone(),
            alive: self.alive,
            successful: self.successful.clone(),
            lexical: self.lexical,
        }
    }

    pub(super) fn into_environment(self) -> SegmentEnvironment {
        SegmentEnvironment {
            function: self.function.id(),
            values: self.values,
            alive: self.alive,
            successful: self.successful,
            lexical: self.lexical,
        }
    }

    fn restore_environment(&mut self, environment: SegmentEnvironment) {
        self.function = self.program.function(environment.function);
        self.values = environment.values;
        self.alive = environment.alive;
        self.successful = environment.successful;
        self.lexical = environment.lexical;
    }

    /// Advances one actual source/control frame. The frame stack contains no
    /// borrowed source bodies or kernel builders and can be retained on pause.
    pub(super) fn lower_frame(
        &mut self,
        frames: &mut Vec<SegmentFrame>,
    ) -> Result<(), capacity::CapacityPending> {
        let frame = frames.pop().expect("segment step has a current frame");
        let (region, next) = match frame {
            SegmentFrame::Bind { outputs, results } => {
                for (output, result) in outputs.into_iter().zip(results) {
                    self.values.insert(output, self.bound(result));
                }
                return Ok(());
            }
            SegmentFrame::Then(branch) => {
                let alive = self.alive;
                let products = self
                    .program
                    .function(branch.then.function())
                    .region(branch.then)
                    .results()
                    .iter()
                    .map(|value| self.bound(*value))
                    .collect();
                self.kernel.begin_otherwise(&branch.token);
                self.restore_environment(branch.parent.clone());
                self.lexical = branch.lexical;
                bind_segment_region_parameters(
                    &mut self.values,
                    self.function,
                    branch.otherwise,
                    &branch.captures,
                );
                self.initialize_successful_values(branch.otherwise);
                let body = SegmentFrame::region(branch.otherwise);
                frames.push(SegmentFrame::Otherwise {
                    branch,
                    alive,
                    products,
                });
                frames.push(body);
                return Ok(());
            }
            SegmentFrame::Otherwise {
                branch,
                alive: then_alive,
                products: then_products,
            } => {
                let else_products = self
                    .program
                    .function(branch.otherwise.function())
                    .region(branch.otherwise)
                    .results()
                    .iter()
                    .map(|value| self.bound(*value))
                    .collect::<Vec<_>>();
                let mut fields = vec![(Some(then_alive), Some(self.alive))];
                for (left, right) in then_products.iter().zip(&else_products) {
                    products::join_fields(left, right, &mut fields);
                }
                let mut fields = self
                    .kernel
                    .finish_branch_products(branch.token, fields)
                    .into_iter();
                let alive = fields.next().unwrap();
                self.restore_environment(branch.parent);
                self.alive = alive;
                for ((output, left), right) in branch
                    .outputs
                    .iter()
                    .zip(&then_products)
                    .zip(&else_products)
                {
                    let joined =
                        products::joined(self.kernel, branch.condition, left, right, &mut fields);
                    self.values.insert(*output, joined);
                }
                assert!(fields.next().is_none());
                return Ok(());
            }
            SegmentFrame::Repeat(repeat) => {
                self.function = self.program.function(repeat.body.function());
                if let Some(scope) = repeat.scope {
                    self.collective_gate(scope);
                }
                let mut next = repeat
                    .carries
                    .iter()
                    .zip(&repeat.before)
                    .map(|(carry, before)| {
                        self.kernel
                            .select(self.alive, self.scalar(carry.yielded), *before)
                    })
                    .collect::<Vec<_>>();
                next.push(self.alive);
                let results = self.kernel.finish_repeat(repeat.token, next);
                self.restore_environment(repeat.parent);
                let (&alive, values) = results.split_last().unwrap();
                self.alive = alive;
                for (carry, value) in repeat.carries.iter().zip(values) {
                    self.values
                        .insert(carry.result, SegmentBound::Scalar(*value));
                }
                return Ok(());
            }
            SegmentFrame::Nodes { region, next } => (region, next),
        };
        self.function = self.program.function(region.function());
        let function = self.function;
        let Some((id, node)) = function.nodes(region).nth(next) else {
            return Ok(());
        };
        let result = match self.node_tensor_result(id) {
            Ok(result) => result,
            Err(reason) => {
                frames.push(SegmentFrame::Nodes { region, next });
                return Err(reason);
            }
        };
        if let Err(reason) = self.preserve_captures_before(id) {
            frames.push(SegmentFrame::Nodes { region, next });
            return Err(reason);
        }
        frames.push(SegmentFrame::Nodes {
            region,
            next: next + 1,
        });
        match node.view() {
            SemanticNodeView::Check { condition, reason } => {
                let condition = self.scalar(condition);
                let stopped = self.kernel.not(self.alive);
                let passed = self.kernel.logic(LogicOp::Or, stopped, condition);
                let status = self.checks
                    [&SourceFailure::at(function, id, SourceFailureCause::Check(reason.clone()))]
                    .clone();
                self.kernel.record_source_check(&status.0, status.1, passed);
                self.alive = self.kernel.logic(LogicOp::And, self.alive, condition);
            }
            SemanticNodeView::Call {
                family,
                inputs,
                outputs,
            } => {
                let helper = self.helpers[&family];
                let mut equality = self.successful.clone();
                for (parameter, input) in helper.parameters().iter().zip(inputs) {
                    self.values.insert(parameter.value, self.bound(*input));
                    equality.insert(parameter.value, self.successful[input].clone());
                }
                let inference = uniformity::Inference {
                    arena: self.kernel.expression_arena(),
                    helpers: self.helpers,
                };
                let equality =
                    inference.successful_values(helper, helper.root(), &equality, self.lexical);
                self.successful.extend(equality);
                frames.push(SegmentFrame::Bind {
                    outputs: outputs.to_vec(),
                    results: helper.results().to_vec(),
                });
                frames.push(SegmentFrame::region(helper.root()));
            }
            SemanticNodeView::If {
                condition,
                captures,
                outputs,
                then,
                otherwise,
            } => {
                if let Some(scope) = cohort::Scope::node(function, id, self.helpers) {
                    self.collective_gate(scope);
                }
                let lexical =
                    uniformity::join(self.lexical, self.successful[&condition].scalar_scope());
                let condition = self.scalar(condition);
                let captures = captures
                    .iter()
                    .map(|value| self.bound(*value))
                    .collect::<Vec<_>>();
                let parent = self.environment();
                let token = self.kernel.begin_branch(condition);
                self.lexical = lexical;
                bind_segment_region_parameters(&mut self.values, function, then, &captures);
                self.initialize_successful_values(then);
                frames.push(SegmentFrame::Then(SegmentIf {
                    token,
                    condition,
                    captures,
                    outputs: outputs.to_vec(),
                    then,
                    otherwise,
                    parent,
                    lexical,
                }));
                frames.push(SegmentFrame::region(then));
            }
            SemanticNodeView::Loop {
                start,
                end,
                captures,
                body,
                carries,
                ..
            } => {
                let scope = cohort::Scope::region(function, body, self.helpers);
                if let Some(scope) = scope {
                    self.collective_gate(scope);
                }
                let lexical = uniformity::all([
                    self.lexical,
                    self.successful[&start].scalar_scope(),
                    self.successful[&end].scalar_scope(),
                ]);
                let start = portable_index(self.kernel, self.scalar(start));
                let end = portable_index(self.kernel, self.scalar(end));
                let mut initial = carries
                    .iter()
                    .map(|carry| self.scalar(carry.initial))
                    .collect::<Vec<_>>();
                initial.push(self.alive);
                let physical = self
                    .values
                    .iter()
                    .map(|(value, bound)| (*value, uniformity::bound(self.kernel, bound)))
                    .collect();
                let range = uniformity::all([
                    self.kernel.uniformity(start),
                    self.kernel.uniformity(end),
                    self.kernel.uniformity(self.alive),
                ]);
                let inference = uniformity::Inference {
                    arena: self.kernel.expression_arena(),
                    helpers: self.helpers,
                };
                let recurrence =
                    inference.recurrence(function, body, captures, carries, &physical, range);
                let captures = captures
                    .iter()
                    .map(|value| self.bound(*value))
                    .collect::<Vec<_>>();
                let parent = self.environment();
                let (token, binder, carried) =
                    self.kernel.begin_repeat(start, end, initial, &recurrence);
                let (&alive, carried) = carried.split_last().unwrap();
                self.alive = alive;
                self.lexical = lexical;
                let parameters = function.region(body).parameters();
                self.values
                    .insert(parameters[0], SegmentBound::Scalar(binder));
                for (parameter, capture) in parameters[1..].iter().zip(captures) {
                    self.values.insert(*parameter, capture);
                }
                for (carry, value) in carries.iter().zip(carried) {
                    self.values
                        .insert(carry.parameter, SegmentBound::Scalar(*value));
                }
                self.initialize_successful_values(body);
                frames.push(SegmentFrame::Repeat(SegmentRepeat {
                    token,
                    parent,
                    body,
                    carries: carries.to_vec(),
                    before: carried.to_vec(),
                    scope,
                }));
                frames.push(SegmentFrame::region(body));
            }
            SemanticNodeView::Intrinsic { .. } => {
                if let Some(scope) = cohort::Scope::node(function, id, self.helpers) {
                    self.collective_gate(scope);
                }
                self.lower_guarded_node(id, result);
            }
            _ if source_total_constructor(function, id) => self.lower_node(id, result),
            _ => self.lower_guarded_node(id, result),
        }
        Ok(())
    }

    fn lower_guarded_node(
        &mut self,
        id: NodeId,
        result: Option<producer::TensorResult<PortableValue>>,
    ) {
        let outputs = self.function.node(id).results();
        let incoming = self.alive;
        let branch = self.kernel.begin_branch(incoming);
        self.lower_node(id, result);
        let products = outputs.iter().map(|id| self.bound(*id)).collect::<Vec<_>>();
        let mut fields = vec![(Some(self.alive), Some(incoming))];
        fields.extend(
            products
                .iter()
                .flat_map(SegmentBound::scalar_fields)
                .map(|v| (Some(v), None)),
        );
        self.kernel.begin_otherwise(&branch);
        let mut fields = self
            .kernel
            .finish_branch_products(branch, fields)
            .into_iter();
        self.alive = fields.next().unwrap();
        for (id, product) in outputs.into_iter().zip(products) {
            self.values
                .insert(id, product.with_scalar_fields(self.kernel, &mut fields));
        }
        assert!(fields.next().is_none());
    }
}

/// Only a genuinely total constructor can run as internal work for stopped
/// participants. Reads, producer realization, checks and intrinsics are never
/// included here merely because their result is unused.
fn source_total_constructor(function: &SemanticFunction, id: NodeId) -> bool {
    match function.node(id).view() {
        SemanticNodeView::TuplePack { .. }
        | SemanticNodeView::TupleGet { .. }
        | SemanticNodeView::Extent { .. } => true,
        SemanticNodeView::Primitive {
            primitive, inputs, ..
        } => match primitive {
            PrimitiveId::Constant(_)
            | PrimitiveId::Select
            | PrimitiveId::RangeMake
            | PrimitiveId::RangeStart
            | PrimitiveId::RangeEnd => true,
            _ => source_scalar::recipe(function, primitive, inputs)
                .is_some_and(|recipe| recipe.failures().is_empty()),
        },
        _ => false,
    }
}
