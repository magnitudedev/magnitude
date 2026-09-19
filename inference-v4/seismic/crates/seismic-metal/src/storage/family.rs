//! Joint placement and participant ownership, without a placement traversal.
use super::{StorageFamily, StoragePlan};
use magnitude_solver::model::{Constraint, Domain, LinearTerm, Literal, ModelBuilder, VarId};
use seismic_accounting::algebra::{Algebra, Error, Symbolic, Value};
use seismic_lang::ir;
use seismic_realization::dispatch::{TilePlacement, geometry};
use std::collections::BTreeMap;
use std::sync::Arc;

pub struct Binding {
    family: Arc<StorageFamily>,
    pub placements: BTreeMap<ir::VarId, Placement>,
    pub layouts: BTreeMap<ir::VarId, Physical>,
    presences: BTreeMap<ir::VarId, VarId>,
    owners: BTreeMap<ir::VarId, VarId>,
}
pub struct Physical {
    pub capacity: Value,
    pub shape: Vec<Value>,
    pub strides: Vec<Value>,
    pub packets: Option<Packets>,
}
pub struct Packets {
    pub physical_width: Value,
    pub strides: Vec<Value>,
    pub planes: Vec<Plane>,
}
pub struct Plane {
    pub plane: seismic_lang::repr::Plane,
    pub elements_per_row: Value,
    pub elements: Value,
}
pub struct Placement {
    /// Original local alternative ordinal; no placement preference is encoded.
    pub ordinal: VarId,
    pub private_elements_per_lane: Value,
    pub shared_elements_per_item: Value,
    pub private_bytes_per_lane: Value,
    pub shared_bytes_per_group: Value,
    /// The same participant requirements used by model export and local
    /// emission. An alternative applies these facts only when it is present.
    pub(crate) ownership: Vec<Vec<(VarId, bool)>>,
}
fn invalid(error: impl ToString) -> Error {
    Error::Invalid(error.to_string())
}
impl StorageFamily {
    pub fn append(
        self: &Arc<Self>,
        builder: &mut ModelBuilder,
        name: &str,
        lanes: Value,
        items: Value,
    ) -> Result<Binding, Error> {
        self.append_bound(builder, name, lanes, items, &BTreeMap::new())
    }
    /// Reuse the implementation registry's original placement variables. The
    /// geometry and participant equations must not introduce a second choice
    /// for the same allocation request.
    pub(crate) fn append_bound(
        self: &Arc<Self>, builder: &mut ModelBuilder, name: &str, lanes: Value, items: Value,
        ordinals: &BTreeMap<ir::VarId, VarId>,
    ) -> Result<Binding, Error> {
        self.append_guarded_bound(builder, name, lanes, items, ordinals, &BTreeMap::new())
    }
    /// Append storage equations under the source arm that defines each value.
    /// Inactive arms receive canonical zero geometry, so an unresolved shape
    /// from an arm that is not present cannot constrain the retained family.
    pub(crate) fn append_guarded_bound(
        self: &Arc<Self>, builder: &mut ModelBuilder, name: &str, lanes: Value, items: Value,
        ordinals: &BTreeMap<ir::VarId, VarId>,
        guards: &BTreeMap<ir::VarId, Vec<Literal>>,
    ) -> Result<Binding, Error> {
        fn conjunction(builder: &mut ModelBuilder, name: &str, guards: &[Literal]) -> Result<VarId, Error> {
            let mut inputs = Vec::new();
            for (index, guard) in guards.iter().enumerate() {
                if guard.value == 1 { inputs.push(guard.variable); continue; }
                let variable = builder.local_variable(format!("{name}.predicate{index}"), Domain::boolean()).map_err(invalid)?;
                let expected = builder.variable(format!("{name}.literal{index}"), Domain::singleton(guard.value));
                builder.guarded_constraint(vec![Literal::new(variable, 1)], Constraint::Equal { left: guard.variable, right: expected });
                builder.guarded_constraint(vec![Literal::new(variable, 0)], Constraint::NotEqual { left: guard.variable, right: expected });
                inputs.push(variable);
            }
            let output = builder.local_variable(name.to_owned(), Domain::boolean()).map_err(invalid)?;
            if inputs.is_empty() { builder.constraint(Constraint::InDomain { variable: output, domain: Domain::singleton(1) }); }
            else { builder.constraint(Constraint::BoolAnd { output, inputs }); }
            Ok(output)
        }
        let ownership = &self.analysis.ownership;
        let groups = ownership
            .groups()
            .into_iter()
            .map(|group| {
                Ok((
                    group,
                    builder.local_variable(format!("{name}.ownership{group}"), Domain::boolean()).map_err(invalid)?,
                ))
            })
            .collect::<Result<BTreeMap<_, _>, Error>>()?;
        let mut supports = groups
            .keys()
            .map(|&group| (group, Vec::<VarId>::new()))
            .collect::<BTreeMap<_, _>>();
        let forced = ownership
            .forced_groups()
            .collect::<std::collections::BTreeSet<_>>();
        for &group in &forced {
            builder.constraint(Constraint::InDomain {
                variable: groups[&group],
                domain: Domain::singleton(1),
            });
        }
        let mut placements = BTreeMap::new();
        let mut layouts = BTreeMap::new();
        let mut presences = BTreeMap::new();
        for decision in &self.decisions {
            let label = format!("{name}.tile{}", decision.variable);
            let mut expressions = seismic_compiler::tuner::expressions::Expressions::new(self.parameters.clone());
            let physical = self.physical(decision.variable).ok_or_else(|| invalid("storage value has no physical geometry"))?;
            let presence = conjunction(builder, &format!("{label}.present"), guards.get(&decision.variable).map(Vec::as_slice).unwrap_or(&[]))?;
            presences.insert(decision.variable, presence);
            let mut geometry_index = 0usize;
            let mut guarded_nonnegative = |expression: &seismic_lang::sym::Sym| -> Result<Value, Error> {
                // `Expressions::nonnegative` may have a strictly positive
                // lower bound (for example a packet row count). Its raw
                // coordinate is therefore not legal to force to zero when
                // the source arm is inactive. Reify a fresh 0..max envelope
                // and constrain it to the raw expression only under the
                // active arm; the inactive arm gets the canonical zero.
                let raw = builder.when(Literal::new(presence, 1), |builder| expressions.nonnegative(builder, &label, expression))?;
                let maximum = i64::try_from(raw.bounds().1).map_err(invalid)?;
                let domain = Domain::interval(0, maximum).map_err(invalid)?;
                let value = builder
                    .local_variable(format!("{label}.geometry{geometry_index}"), domain.clone())
                    .map_err(invalid)?;
                geometry_index += 1;
                builder.guarded_constraint(
                    vec![Literal::new(presence, 1)],
                    Constraint::Equal { left: value, right: raw.id() },
                );
                builder.guarded_constraint(
                    vec![Literal::new(presence, 0)],
                    Constraint::InDomain { variable: value, domain: Domain::singleton(0) },
                );
                Value::binding(value, &domain)
            };
            let capacity = guarded_nonnegative(&physical.capacity)?;
            let shape = physical.shape.iter().map(&mut guarded_nonnegative).collect::<Result<Vec<_>, _>>()?;
            let strides = physical.strides.iter().map(&mut guarded_nonnegative).collect::<Result<Vec<_>, _>>()?;
            let packets = physical.packets.as_ref().map(|packet| {
                let physical_width = guarded_nonnegative(&packet.physical_width)?;
                let strides = packet.strides.iter().map(&mut guarded_nonnegative).collect::<Result<Vec<_>, _>>()?;
                let planes = packet.planes.iter().map(|part| Ok(Plane {
                    plane: part.plane.clone(),
                    elements_per_row: guarded_nonnegative(&part.elements_per_row)?,
                    elements: guarded_nonnegative(&part.elements)?,
                })).collect::<Result<Vec<_>, Error>>()?;
                Ok::<Packets, Error>(Packets { physical_width, strides, planes })
            }).transpose()?;
            drop(guarded_nonnegative);
            let count = i64::try_from(decision.alternatives.len()).map_err(invalid)?;
            if count == 0 {
                return Err(invalid("empty placement domain"));
            }
            let ordinal = match ordinals.get(&decision.variable) {
                Some(&ordinal) => ordinal,
                None => builder.local_variable(
                    format!("{label}.placement"),
                    Domain::interval(0, count - 1).map_err(invalid)?,
                )
                .map_err(invalid)?,
            };
            // A source-inactive value has no selected allocation. Keep its
            // retained placement ordinal deterministic so later guarded
            // accounting cannot observe an arbitrary local assignment.
            builder.guarded_constraint(vec![Literal::new(presence, 0)], Constraint::InDomain {
                variable: ordinal, domain: Domain::singleton(0),
            });
            let mut cases = Vec::new();
            let mut requirements = Vec::new();
            for (index, placement) in decision.alternatives.iter().enumerate() {
                let selected = builder
                    .local_variable(format!("{label}.alternative{index}"), Domain::boolean())
                    .map_err(invalid)?;
                let value = builder.variable(
                    format!("{label}.ordinal{index}"),
                    Domain::singleton(index as i64),
                );
                builder.guarded_constraint(
                    vec![Literal::new(selected, 1)],
                    Constraint::Equal {
                        left: ordinal,
                        right: value,
                    },
                );
                builder.guarded_constraint(
                    vec![Literal::new(selected, 0)],
                    Constraint::NotEqual {
                        left: ordinal,
                        right: value,
                    },
                );
                let cooperative = *placement != TilePlacement::Replicated;
                let group = ownership.group(decision.variable);
                let mut required = vec![(groups[&group], cooperative)];
                if *placement == TilePlacement::Distributed {
                    required.extend(ownership.read_owner_groups(decision.variable).into_iter()
                        .map(|owner| (groups[&owner], true)));
                }
                let effective = conjunction(
                    builder,
                    &format!("{label}.active_alternative{index}"),
                    &[Literal::new(presence, 1), Literal::new(selected, 1)],
                )?;
                for &(variable, cooperative) in &required {
                    builder.guarded_constraint(vec![Literal::new(effective, 1)], Constraint::InDomain {
                        variable, domain: Domain::singleton(i64::from(cooperative)),
                    });
                }
                requirements.push(required);
                if cooperative {
                    supports.get_mut(&group).unwrap().push(effective);
                }
                if *placement == TilePlacement::Distributed {
                    for owner in ownership.read_owner_groups(decision.variable) {
                        supports.get_mut(&owner).unwrap().push(effective);
                    }
                }
                let quantity = builder.when(Literal::new(presence, 1), |builder| builder.when(Literal::new(selected, 1), |builder| {
                    let mut algebra = Symbolic::new(builder, &label);
                    if let Some(packet) = &packets {
                        let zero = algebra.constant(0)?;
                        let mut total = geometry::Storage { private_elements_per_lane: zero, shared_elements_per_item: zero,
                            private_bytes_per_lane: zero, shared_bytes_per_group: zero };
                        for part in &packet.planes {
                            let plane = geometry::storage(&mut algebra, part.elements, u64::from(part.plane.dtype().bytes()), placement, lanes, items)?;
                            total.private_elements_per_lane = algebra.sum(total.private_elements_per_lane, plane.private_elements_per_lane)?;
                            total.shared_elements_per_item = algebra.sum(total.shared_elements_per_item, plane.shared_elements_per_item)?;
                            total.private_bytes_per_lane = algebra.sum(total.private_bytes_per_lane, plane.private_bytes_per_lane)?;
                            total.shared_bytes_per_group = algebra.sum(total.shared_bytes_per_group, plane.shared_bytes_per_group)?;
                        }
                        Ok(total)
                    } else {
                        geometry::storage(&mut algebra, capacity, u64::from(decision.dtype.bytes()), placement, lanes, items)
                    }
                }))?;
                cases.push((selected, quantity));
            }
            let select_quantity = |builder: &mut ModelBuilder,
                                   field: fn(&geometry::Storage<Value>) -> Value|
             -> Result<Value, Error> {
                let maximum = cases
                    .iter()
                    .map(|(_, case)| field(case).bounds().1)
                    .max()
                    .unwrap();
                let result = Symbolic::new(builder, &label).variable(
                    "selected_quantity",
                    Domain::interval(0, i64::try_from(maximum).map_err(invalid)?)
                        .map_err(invalid)?,
                )?;
                for (selected, case) in &cases {
                    builder.guarded_constraint(
                        vec![Literal::new(presence, 1), Literal::new(*selected, 1)],
                        Constraint::Equal {
                            left: result.id(),
                            right: field(case).id(),
                        },
                    );
                }
                builder.guarded_constraint(vec![Literal::new(presence, 0)], Constraint::InDomain {
                    variable: result.id(), domain: Domain::singleton(0),
                });
                Ok(result)
            };
            placements.insert(
                decision.variable,
                Placement {
                    ordinal,
                    ownership: requirements,
                    private_elements_per_lane: select_quantity(builder, |s| {
                        s.private_elements_per_lane
                    })?,
                    shared_elements_per_item: select_quantity(builder, |s| {
                        s.shared_elements_per_item
                    })?,
                    private_bytes_per_lane: select_quantity(builder, |s| s.private_bytes_per_lane)?,
                    shared_bytes_per_group: select_quantity(builder, |s| s.shared_bytes_per_group)?,
                },
            );
            layouts.insert(decision.variable, Physical { capacity, shape, strides, packets });
        }
        // An otherwise unconstrained owner remains serial, as in concrete
        // ownership selection. This prevents the solver inventing a new form.
        for (group, support) in supports {
            if forced.contains(&group) {
                continue;
            }
            let mut terms = vec![LinearTerm::new(groups[&group], 1)];
            terms.extend(
                support
                    .into_iter()
                    .map(|variable| LinearTerm::new(variable, -1)),
            );
            builder.constraint(Constraint::LinearLe { terms, rhs: 0 });
        }
        let owners = ownership
            .owner_variables()
            .map(|variable| (variable, groups[&ownership.group(variable)]))
            .collect();
        Ok(Binding {
            family: self.clone(),
            placements,
            layouts,
            presences,
            owners,
        })
    }
}
impl Binding {
    pub(crate) fn owners(&self) -> &BTreeMap<ir::VarId, VarId> { &self.owners }
    pub fn family(&self) -> &Arc<StorageFamily> {
        &self.family
    }
    pub fn reconstruct(
        &self,
        values: &[i64],
        dispatch: &seismic_realization::dispatch::GroupDispatch,
    ) -> Result<StoragePlan, Error> {
        let read = |id: VarId| {
            values
                .get(id.0)
                .copied()
                .ok_or_else(|| Error::Reconstruction("missing placement family value".into()))
        };
        let parameters = self.family.parameters.iter().map(|(name, value)| Ok((name.clone(), read(value.id())?)))
            .collect::<Result<BTreeMap<_, _>, Error>>()?;
        // A retained family contains geometry for every source arm. Inactive
        // declarations can carry expressions that are intentionally
        // unresolved or negative for the selected invocation, so remove them
        // from placement reconstruction before evaluating any physical
        // expression or asking the selector for an ordinal.
        let mut active = std::collections::BTreeSet::new();
        for (&variable, &presence) in &self.presences {
            match read(presence)? {
                0 => {}
                1 => {
                    active.insert(variable);
                }
                _ => return Err(Error::Reconstruction("storage presence is outside its boolean domain".into())),
            }
        }
        let mut family = (*self.family).clone();
        family.decisions.retain(|decision| active.contains(&decision.variable));
        let plan = family.select_parameters(&parameters, &mut |restricted| {
                let binding = self
                    .placements
                    .get(&restricted.variable)
                    .ok_or("placement missing from its retained family")?;
                let ordinal = usize::try_from(read(binding.ordinal).map_err(|e| e.to_string())?)
                    .map_err(|_| "negative placement ordinal")?;
                let original = family
                    .decisions
                    .iter()
                    .find(|decision| decision.variable == restricted.variable)
                    .ok_or("placement has no source definition")?;
                original
                    .alternatives
                    .get(ordinal)
                    .cloned()
                    .ok_or_else(|| "placement outside its original domain".into())
            })
            .map_err(Error::Reconstruction)?;
        for (&variable, &owner) in &self.owners {
            if read(owner)? != i64::from(plan.owned_cooperative(variable)) {
                return Err(Error::Reconstruction(
                    "participant ownership differs from the retained storage family".into(),
                ));
            }
        }
        for (&variable, binding) in &self.placements {
            let presence = self.presences.get(&variable).copied().ok_or_else(|| Error::Reconstruction("storage presence is missing".into()))?;
            if read(presence)? == 0 {
                let layout = &self.layouts[&variable];
                let mut values = vec![layout.capacity];
                values.extend(layout.shape.iter().copied());
                values.extend(layout.strides.iter().copied());
                if let Some(packet) = &layout.packets {
                    values.push(packet.physical_width);
                    values.extend(packet.strides.iter().copied());
                    for plane in &packet.planes { values.push(plane.elements_per_row); values.push(plane.elements); }
                }
                values.extend([
                    binding.private_elements_per_lane,
                    binding.shared_elements_per_item,
                    binding.private_bytes_per_lane,
                    binding.shared_bytes_per_group,
                ]);
                if values.iter().any(|value| read(value.id()).ok() != Some(0)) {
                    return Err(Error::Reconstruction("inactive storage arm was not canonicalized".into()));
                }
                continue;
            }
            let declaration = plan.declaration(variable).map_err(Error::Reconstruction)?;
            let physical = &self.layouts[&variable];
            let source = plan.physical(variable).ok_or_else(|| Error::Reconstruction("missing selected physical storage".into()))?;
            let evaluate = |expression: &seismic_lang::sym::Sym| expression.eval(&|name| parameters.get(name).copied())
                .and_then(|value| u64::try_from(value).ok()).ok_or_else(|| Error::Reconstruction("invalid selected physical storage geometry".into()));
            let mut expected_geometry = vec![(physical.capacity, declaration.capacity)];
            for (bindings, expressions) in [(&physical.shape, &source.shape), (&physical.strides, &source.strides)] {
                for (&binding, expression) in bindings.iter().zip(expressions) { expected_geometry.push((binding, evaluate(expression)?)); }
            }
            let layout = if let Some(packet) = &physical.packets {
                let concrete = plan.packets(variable).ok_or_else(|| Error::Reconstruction("missing selected packet geometry".into()))?;
                expected_geometry.push((packet.physical_width, concrete.physical_width));
                expected_geometry.extend(packet.strides.iter().copied().zip(concrete.strides.iter().copied()));
                let mut total = geometry::Storage { private_elements_per_lane: 0u64, shared_elements_per_item: 0u64,
                    private_bytes_per_lane: 0u64, shared_bytes_per_group: 0u64 };
                for (part, selected) in packet.planes.iter().zip(&concrete.planes) {
                    expected_geometry.push((part.elements_per_row, selected.elements_per_row));
                    expected_geometry.push((part.elements, selected.elements));
                    let plane = seismic_realization::dispatch::TileDeclaration { symbol: declaration.symbol.clone(),
                        dtype: part.plane.dtype(), capacity: selected.elements, placement: declaration.placement.clone() }
                        .layout(dispatch).map_err(Error::Reconstruction)?;
                    let sum = |a: u64, b: u64| a.checked_add(b).ok_or_else(|| Error::Reconstruction("selected packet geometry overflow".into()));
                    total.private_elements_per_lane = sum(total.private_elements_per_lane, plane.private_elements_per_lane)?;
                    total.shared_elements_per_item = sum(total.shared_elements_per_item, plane.shared_elements_per_item)?;
                    total.private_bytes_per_lane = sum(total.private_bytes_per_lane, plane.private_bytes_per_lane)?;
                    total.shared_bytes_per_group = sum(total.shared_bytes_per_group, plane.shared_bytes_per_group)?;
                }
                total
            } else {
                let selected = declaration.layout(dispatch).map_err(Error::Reconstruction)?;
                geometry::Storage { private_elements_per_lane: selected.private_elements_per_lane,
                    shared_elements_per_item: selected.shared_elements_per_item, private_bytes_per_lane: selected.private_bytes_per_lane,
                    shared_bytes_per_group: selected.shared_bytes_per_group }
            };
            for (value, expected) in expected_geometry {
                if u64::try_from(read(value.id())?).ok() != Some(expected) {
                    return Err(Error::Reconstruction("selected capacity or stride differs from its original storage equation".into()));
                }
            }
            for (value, expected) in [
                (
                    binding.private_elements_per_lane,
                    layout.private_elements_per_lane,
                ),
                (
                    binding.shared_elements_per_item,
                    layout.shared_elements_per_item,
                ),
                (
                    binding.private_bytes_per_lane,
                    layout.private_bytes_per_lane,
                ),
                (
                    binding.shared_bytes_per_group,
                    layout.shared_bytes_per_group,
                ),
            ] {
                if u64::try_from(read(value.id())?).ok() != Some(expected) {
                    return Err(Error::Reconstruction(
                        "placement geometry differs from its emitted declaration".into(),
                    ));
                }
            }
        }
        Ok(plan)
    }
}
