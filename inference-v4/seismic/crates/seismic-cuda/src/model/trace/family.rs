//! Guarded terminal PTX construction. Each compile-time branch forks its local
//! region and rejoins the original terminal CFG once. Dynamic visits retain the
//! same original parameter; no complete source assignment is enumerated.
use super::*;

pub(in crate::model) fn derive(
    execution: &Execution,
    hardware: &CudaHardware,
    workload: &ScalarWorkload,
    limits: DerivationLimits,
    inherited: &SymbolicMemory,
    literals: &BTreeMap<usize, usize>,
    joins: &BTreeMap<usize, usize>,
) -> Result<Trace, DerivationError> {
    let plan = execution.target_plan();
    let d = execution.dispatch();
    let state_size = plan
        .registers()
        .len()
        .checked_add(plan.parameters().len())
        .and_then(|n| n.checked_mul(hardware.warp_width as usize))
        .ok_or("CUDA trace state overflow")?;
    if state_size as u64 > limits.instructions || d.dispatched_lanes() > limits.instructions {
        return Err(DerivationError::Exhausted(DerivationLimit::Instructions(
            limits.instructions,
        )));
    }
    if plan.body().len() > limits.operations {
        return Err(DerivationError::Exhausted(DerivationLimit::Operations(
            limits.operations,
        )));
    }
    let labels = plan
        .body()
        .iter()
        .enumerate()
        .filter_map(|(i, item)| {
            if let Item::Label(l) = item {
                Some((*l, i))
            } else {
                None
            }
        })
        .collect::<BTreeMap<_, _>>();
    let mut memory = Memory::new(execution, workload, hardware, limits)?;
    for (&allocation, values) in inherited {
        memory
            .allocations
            .get_mut(&Allocation::External(allocation))
            .ok_or("CUDA retained symbolic storage is missing")?
            .symbolic
            .extend(values.clone());
    }
    let mut events = Vec::new();
    let mut guards = Vec::new();
    let launch = lifecycle(
        &mut events,
        Lifecycle::Launch,
        None,
        None,
        Vec::new(),
        limits,
    )?;
    let mut block_ends = Vec::new();
    let mut instructions = 0u64;
    let warps = d.threads_per_group.div_ceil(u64::from(hardware.warp_width));
    for block in 0..d.groups {
        let admission = lifecycle(
            &mut events,
            Lifecycle::BlockAdmission,
            Some(block),
            None,
            vec![launch],
            limits,
        )?;
        let mut warp_ends = Vec::new();
        for local_warp in 0..warps {
            let warp = block
                .checked_mul(warps)
                .and_then(|n| n.checked_add(local_warp))
                .ok_or("warp index overflow")?;
            let start = lifecycle(
                &mut events,
                Lifecycle::WarpStart,
                Some(block),
                Some(warp),
                vec![admission],
                limits,
            )?;
            let first = local_warp * u64::from(hardware.warp_width);
            let last = (first + u64::from(hardware.warp_width)).min(d.threads_per_group);
            let mut lanes = Vec::new();
            for thread in first..last {
                let linear = block * d.threads_per_group + thread;
                let mut parameters = Vec::new();
                for parameter in plan.parameters() {
                    let value = match parameter.role {
                        ParameterRole::Buffers => {
                            Some(Value::Pointer(Allocation::BufferTable, 0u64.into()))
                        }
                        ParameterRole::Scalars => {
                            Some(Value::Pointer(Allocation::Scalars, 0u64.into()))
                        }
                        ParameterRole::Scratch => {
                            Some(Value::Pointer(Allocation::Scratch, 0u64.into()))
                        }
                        ParameterRole::Statuses => {
                            Some(Value::Pointer(Allocation::Statuses, 0u64.into()))
                        }
                        ParameterRole::CallArgument(_) | ParameterRole::CallResult(_) => None,
                    };
                    parameters.push(value.map(|value| Cell {
                        value,
                        writer: Vec::new(),
                    }));
                }
                lanes.push(Lane {
                    linear,
                    thread: thread as u32,
                    pc: Some(next(plan, 0)?),
                    registers: vec![None; plan.registers().len()],
                    parameters,
                    control: vec![start],
                    status: None,
                });
            }
            guards.resize(events.len(), Guard::new());
            let flow = Construction { plan, labels: &labels, literals, joins,
                events: &mut events, guards: &mut guards, instructions: &mut instructions,
                limits, block, warp, first, width: d.threads_per_group, canonical:false,
            }.run(Flow { lanes, memory, previous: vec![start], warp_events: Vec::new() }, None, &Guard::new())?;
            let Flow { lanes, memory: completed, warp_events, .. } = flow;
            memory = completed;
            for lane in &lanes {
                if lane.linear < d.participating_lanes() && lane.status != Some(0) {
                    return Err("CUDA trace returns without successful invocation status".into());
                }
                if lane.linear >= d.participating_lanes() && lane.status.is_some() {
                    return Err("padded CUDA lane wrote invocation status".into());
                }
            }
            warp_ends.push(lifecycle(
                &mut events,
                Lifecycle::WarpCompletion,
                Some(block),
                Some(warp),
                warp_events,
                limits,
            )?);
        }
        block_ends.push(lifecycle(
            &mut events,
            Lifecycle::BlockCompletion,
            Some(block),
            None,
            warp_ends,
            limits,
        )?);
    }
    if block_ends.is_empty() {
        block_ends.push(launch);
    }
    lifecycle(
        &mut events,
        Lifecycle::Completion,
        None,
        None,
        block_ends,
        limits,
    )?;
    let mut external_values = BTreeMap::new();
    let mut external_symbolic = BTreeMap::new();
    for (allocation, storage) in memory.allocations {
        if let Allocation::External(id) = allocation {
            external_values.insert(id, storage.known);
            external_symbolic.insert(id, storage.symbolic);
        }
    }
    Ok(Trace {
        guards: { guards.resize(events.len(), Guard::new()); guards },
        events,
        instructions,
        external_values,
        external_symbolic,
    })
}

/// One potential warp cohort, independently of its containing block. The PTX
/// dispatch arithmetic remains charged; only its proven linear-index result is
/// interpreted in logical lane coordinates. Native emission retains that exact
/// arithmetic and obtains the selected block coordinates at dispatch time.
pub(in crate::model) fn cohort(
    template:&ptx::family::TargetFamily,first:u64,count:u32,
    hardware:&CudaHardware,workload:&ScalarWorkload,limits:DerivationLimits,inherited:&SymbolicMemory,
)->Result<Trace,DerivationError> {
    let program=template.scalar().program();let plan=template.plan();
    if count==0 || count>hardware.warp_width {return Err("invalid CUDA terminal cohort width".into());}
    let participants=program.work_items.checked_mul(u64::from(program.participation.lanes())).ok_or("CUDA work-item lane count overflow")?;
    let size=usize::try_from(participants).map_err(|_|"CUDA participant storage overflow")?;
    let storage=crate::execution::InvocationStorage {
        buffer_table_bytes:program.buffers.len().checked_mul(8).ok_or("CUDA buffer table overflow")?,
        scalar_bytes:program.scalars.len().checked_mul(8).ok_or("CUDA scalar table overflow")?,
        scratch_bytes:program.scratch_bytes.checked_mul(size).ok_or("CUDA invocation scratch overflow")?,
        status_bytes:size.checked_mul(4).ok_or("CUDA invocation status overflow")?,
    };
    let mut memory=Memory::from_program(program,&storage,workload,hardware,limits)?;
    for (&allocation,values) in inherited {
        memory.allocations.get_mut(&Allocation::External(allocation)).ok_or("CUDA retained publication storage is missing")?.symbolic.extend(values.clone());
    }
    let labels=plan.body().iter().enumerate().filter_map(|(position,item)|match item {Item::Label(label)=>Some((*label,position)),_=>None}).collect();
    let mut events=Vec::new();
    let start=lifecycle(&mut events,Lifecycle::WarpStart,None,Some(first),Vec::new(),limits)?;
    let mut lanes=Vec::new();
    for thread in 0..count {
        let parameters=plan.parameters().iter().map(|parameter| {
            let allocation=match parameter.role {ParameterRole::Buffers=>Some(Allocation::BufferTable),ParameterRole::Scalars=>Some(Allocation::Scalars),ParameterRole::Scratch=>Some(Allocation::Scratch),ParameterRole::Statuses=>Some(Allocation::Statuses),_=>None};
            allocation.map(|allocation|Cell {value:Value::Pointer(allocation,0u64.into()),writer:Vec::new()})
        }).collect();
        lanes.push(Lane {linear:first.checked_add(u64::from(thread)).ok_or("CUDA cohort coordinate overflow")?,thread,pc:Some(next(plan,0)?),registers:vec![None;plan.registers().len()],parameters,control:vec![start],status:None});
    }
    let mut guards=vec![Guard::new()];let mut instructions=0;
    let flow=Construction {plan,labels:&labels,literals:template.literals(),joins:template.joins(),events:&mut events,guards:&mut guards,instructions:&mut instructions,limits,block:0,warp:first,first:0,width:u64::from(count),canonical:true}
        .run(Flow {lanes,memory,previous:vec![start],warp_events:Vec::new()},None,&Guard::new())?;
    for lane in &flow.lanes {
        if lane.linear<participants && lane.status!=Some(0) {return Err("CUDA cohort returns without successful invocation status".into());}
        if lane.linear>=participants && lane.status.is_some() {return Err("CUDA padded cohort lane wrote invocation status".into());}
    }
    lifecycle(&mut events,Lifecycle::WarpCompletion,None,Some(first),flow.warp_events,limits)?;guards.resize(events.len(),Guard::new());
    let mut external_values=BTreeMap::new();let mut external_symbolic=BTreeMap::new();
    for (allocation,storage) in flow.memory.allocations {
        if let Allocation::External(id)=allocation {external_values.insert(id,storage.known);external_symbolic.insert(id,storage.symbolic);}
    }
    Ok(Trace {events,guards,instructions,external_values,external_symbolic})
}

#[derive(Clone)]
struct Flow { lanes: Vec<Lane>, memory: Memory, previous: Vec<usize>, warp_events: Vec<usize> }
struct Construction<'a> {
    plan: &'a ptx::TargetPlan,
    labels: &'a BTreeMap<ptx::Label, usize>,
    literals: &'a BTreeMap<usize, usize>,
    joins: &'a BTreeMap<usize, usize>,
    events: &'a mut Vec<Event>,
    guards: &'a mut Vec<Guard>,
    instructions: &'a mut u64,
    limits: DerivationLimits,
    block: u64, warp: u64, first: u64, width: u64,
    canonical: bool,
}
impl Construction<'_> {
    fn run(&mut self, flow: Flow, stop: Option<usize>, guard: &Guard) -> Result<Flow, DerivationError> {
        let Flow { mut lanes, mut memory, mut previous, mut warp_events } = flow;
        let (plan, limits, block, warp, first) = (self.plan, self.limits, self.block, self.warp, self.first);
            while let Some(pc) = lanes.iter().filter_map(|l| l.pc).filter(|&pc| Some(pc) != stop).min() {
                resolve(&mut lanes, &mut memory, guard);
                let Item::Instruction(instruction) = &plan.body()[pc] else {
                    unreachable!()
                };
                if let Some(predicate) = instruction.predicate {
                    let selected = lanes.iter().filter(|lane| lane.pc == Some(pc)).find_map(|lane| {
                        match &lane.registers[predicate.register.0].as_ref()?.value {
                            Value::Choice { parameter, .. } => Some(*parameter), _ => None,
                        }
                    });
                    if let Some(parameter) = selected {
                        let join = self.joins.get(&pc).copied().ok_or_else(|| DerivationError::Unsupported("CUDA compile-time predicate has no retained terminal join".into()))?;
                        let join = next(plan, join)?;
                        if lanes.iter().any(|lane| lane.pc.is_some_and(|at| at != pc && Some(at) != stop)) {
                            return Err(DerivationError::Unsupported("CUDA family predicate is reached by divergent lane control".into()));
                        }
                        let input = Flow { lanes, memory, previous, warp_events };
                        let mut yes_guard = guard.clone(); yes_guard.insert(parameter, true);
                        let mut no_guard = guard.clone(); no_guard.insert(parameter, false);
                        let yes = self.run(input.clone(), Some(join), &yes_guard)?;
                        let no = self.run(input, Some(join), &no_guard)?;
                        let joined = merge(parameter, yes, no)?;
                        lanes = joined.lanes; memory = joined.memory;
                        previous = joined.previous; warp_events = joined.warp_events;
                        continue;
                    }
                }
                let cohort = lanes
                    .iter()
                    .enumerate()
                    .filter_map(|(i, l)| (l.pc == Some(pc)).then_some(i))
                    .collect::<Vec<_>>();
                *self.instructions = (*self.instructions)
                    .checked_add(cohort.len() as u64)
                    .ok_or("CUDA instruction count overflow")?;
                if *self.instructions > limits.instructions {
                    return Err(DerivationError::Exhausted(DerivationLimit::Instructions(
                        limits.instructions,
                    )));
                }
                if self.events.len() >= limits.operations {
                    return Err(DerivationError::Exhausted(DerivationLimit::Operations(
                        limits.operations,
                    )));
                }
                let event_id = self.events.len();
                let mut event = Event {
                    requirement: Requirement::Instruction(instruction.operation.primitive()),
                    block: Some(block),
                    warp: Some(warp),
                    position: Some(pc),
                    origin: Some(instruction.origin),
                    issued_lanes: Vec::new(),
                    active_lanes: Vec::new(),
                    accesses: Vec::new(),
                    predecessors: Vec::new(),
                    start_predecessors: previous.clone(),
                };
                let mut active = Vec::new();
                for &i in &cohort {
                    let lane = &lanes[i];
                    event.issued_lanes.push(lane.thread - first as u32);
                    event.predecessors.extend(lane.control.iter().copied());
                    let enabled = if let Some(p) = instruction.predicate {
                        let cell = lane.registers[p.register.0]
                            .as_ref()
                            .ok_or("undefined PTX predicate")?;
                        event.predecessors.extend(cell.writer.iter().copied());
                        cell.value.predicate()? != p.inverted
                    } else {
                        true
                    };
                    if enabled {
                        active.push(i);
                        event.active_lanes.push(lane.thread - first as u32);
                    }
                }
                // A call's body is a separate, explicitly complete opaque-body event.
                // Its memory/private/control internals are conditions of its mapping,
                // not omitted or charged as one primitive by the trace.
                let body_id = matches!(instruction.operation, Operation::Call { .. })
                    .then_some(event_id + 1)
                    .filter(|_| !active.is_empty());
                let exchange = if let Operation::Shuffle {
                    mode,
                    source,
                    lane: source_lane,
                    ..
                } = instruction.operation
                {
                    if active.len() != 32 || lanes.len() != 32 {
                        return Err(
                            "CUDA collective reaches an incomplete or divergent participant group"
                                .into(),
                        );
                    }
                    let mut values = Vec::with_capacity(32);
                    for &i in &active {
                        let source_lane =
                            operand(source_lane, &lanes[i], block, self.width)?.bits()?;
                        if source_lane >= 32 {
                            return Err("shuffle source lane exceeds its participant group".into());
                        }
                        let from = match mode {
                            ptx::ShuffleMode::Butterfly => i ^ (source_lane as usize),
                            ptx::ShuffleMode::Index => source_lane as usize,
                        };
                        let cell = lanes[from].registers[source.0]
                            .as_ref()
                            .ok_or("shuffle sources an undefined register")?;
                        event.predecessors.extend(cell.writer.iter().copied());
                        values.push(cell.value.clone());
                    }
                    Some(values)
                } else {
                    None
                };
                for &i in &active {
                    let lane = &mut lanes[i];
                    let effects = instruction.effects();
                    for r in effects
                        .register_reads
                        .iter()
                        .chain(&effects.register_writes)
                    {
                        if let Some(cell) = &lane.registers[r.0] {
                            event.predecessors.extend(cell.writer.iter().copied());
                        } else if effects.register_reads.contains(r) {
                            return Err("PTX reads an undefined register".into());
                        }
                    }
                    for p in effects
                        .parameter_reads
                        .iter()
                        .chain(&effects.parameter_writes)
                    {
                        if let Some(cell) = &lane.parameters[p.0] {
                            event.predecessors.extend(cell.writer.iter().copied());
                        } else if effects.parameter_reads.contains(p) {
                            return Err("PTX reads an undefined call parameter".into());
                        }
                    }
                    if let (Some(values), Operation::Shuffle { destination, .. }) =
                        (&exchange, &instruction.operation)
                    {
                        lane.registers[destination.0] = Some(Cell {
                            value: values[i].clone(),
                            writer: vec![event_id],
                        });
                        lane.pc = Some(next(plan, pc + 1)?);
                        continue;
                    }
                    execute(
                        instruction,
                        lane,
                        block,
                        self.width,
                        plan,
                        self.labels,
                        &mut memory,
                        &mut event,
                        event_id,
                        body_id,
                        guard,
                    )?;
                    if self.canonical && instruction.origin==ptx::Origin::Dispatch {
                        for register in instruction.effects().register_writes {
                            if plan.registers()[register.0].name==ptx::RegisterName::Abi(ptx::AbiRegister::LinearIndex) {
                                lane.registers[register.0]=Some(Cell {value:Value::Bits(lane.linear),writer:vec![event_id]});
                            }
                        }
                    }
                }
                if let Some(&parameter) = self.literals.get(&pc) {
                    let Operation::Unary { operation: Unary::Move, destination, .. } = instruction.operation else { return Err("CUDA retained literal changed form".into()); };
                    for &i in &active {
                        let value = Value::select(parameter, Value::Bits(1), Value::Bits(0)).resolve(guard);
                        lanes[i].registers[destination.0] = Some(Cell { value, writer: vec![event_id] });
                    }
                }
                for &i in &cohort {
                    if !active.contains(&i) {
                        lanes[i].pc = Some(next(plan, pc + 1)?);
                    }
                }
                event.predecessors.sort_unstable();
                event.predecessors.dedup();
                for access in &event.accesses {
                    memory.accesses.push((access.clone(), event_id, guard.clone()));
                }
                self.events.push(event);
                self.guards.push(guard.clone());
                warp_events.push(event_id);
                previous = vec![event_id];
                if let Some(body_id) = body_id {
                    if self.events.len() >= limits.operations {
                        return Err(DerivationError::Exhausted(DerivationLimit::Operations(
                            limits.operations,
                        )));
                    }
                    let Operation::Call { function, .. } = instruction.operation else {
                        unreachable!()
                    };
                    let call = &self.events[event_id];
                    let body = Event {
                        requirement: Requirement::WholeBody(ptx::Helper::for_function(function)),
                        block: Some(block),
                        warp: Some(warp),
                        position: Some(pc),
                        origin: Some(instruction.origin),
                        issued_lanes: call.active_lanes.clone(),
                        active_lanes: call.active_lanes.clone(),
                        accesses: Vec::new(),
                        predecessors: vec![event_id],
                        start_predecessors: vec![event_id],
                    };
                    self.events.push(body);
                    self.guards.push(guard.clone());
                    warp_events.push(body_id);
                    previous = vec![body_id];
                }
            }

        Ok(Flow { lanes, memory, previous, warp_events })
    }
}
fn resolve(lanes: &mut [Lane], memory: &mut Memory, guard: &Guard) {
    for lane in lanes {
        for cell in lane.registers.iter_mut().chain(&mut lane.parameters).flatten() {
            cell.value = cell.value.resolve(guard);
        }
    }
    for storage in memory.allocations.values_mut() {
        for value in storage.pointers.values_mut().chain(storage.symbolic.values_mut()) {
            *value = value.resolve(guard);
        }
    }
}
fn union<T: Ord>(left: impl IntoIterator<Item=T>, right: impl IntoIterator<Item=T>) -> Vec<T> {
    left.into_iter().chain(right).collect::<BTreeSet<_>>().into_iter().collect()
}
fn cells(parameter: usize, left: Option<Cell>, right: Option<Cell>) -> Option<Cell> {
    if left.is_none() && right.is_none() { return None; }
    let yes = left.as_ref().map_or(Value::Unknown, |cell|cell.value.clone());
    let no = right.as_ref().map_or(Value::Unknown, |cell|cell.value.clone());
    Some(Cell { value: Value::select(parameter, yes, no), writer: union(left.into_iter().flat_map(|cell|cell.writer), right.into_iter().flat_map(|cell|cell.writer)) })
}
fn merge(parameter: usize, mut yes: Flow, no: Flow) -> Result<Flow, DerivationError> {
    if yes.lanes.len() != no.lanes.len() { return Err("CUDA family changed its invocation cohort".into()); }
    for (a,b) in yes.lanes.iter_mut().zip(no.lanes) {
        if a.linear != b.linear || a.thread != b.thread || a.pc != b.pc || a.status != b.status { return Err(DerivationError::Unsupported("CUDA family alternatives do not rejoin the same invocation control".into())); }
        for (left,right) in a.registers.iter_mut().zip(b.registers) { *left = cells(parameter, left.take(), right); }
        for (left,right) in a.parameters.iter_mut().zip(b.parameters) { *left = cells(parameter, left.take(), right); }
        a.control = union(std::mem::take(&mut a.control), b.control);
    }
    if yes.memory.allocations.keys().ne(no.memory.allocations.keys()) { return Err("CUDA family alternatives changed allocation identities".into()); }
    let mut symbolic = BTreeMap::new();
    for (allocation, a) in &yes.memory.allocations {
        let b = &no.memory.allocations[allocation];
        if a.bytes != b.bytes || a.alignment != b.alignment { return Err("CUDA family changed invocation storage shape".into()); }
        let mut keys = union(a.symbolic.keys().cloned(),b.symbolic.keys().cloned());
        keys.extend(union(a.known.keys().copied(), b.known.keys().copied()).into_iter().map(|offset|(offset.into(),1)));
        keys.extend(union(a.pointers.keys().copied(), b.pointers.keys().copied()).into_iter().map(|offset|(offset.into(),8)));
        let mut values = BTreeMap::new();
        for (offset,bytes) in keys {
            let access = Access { lane:0, allocation:allocation.clone(), offset:offset.clone(), bytes, alignment:a.alignment, write:false };
            values.insert((offset,bytes),Value::select(parameter,yes.memory.load(&access),no.memory.load(&access)));
        }
        symbolic.insert(allocation.clone(),values);
    }
    for (allocation, a) in &mut yes.memory.allocations {
        let b = &no.memory.allocations[allocation];
        a.known.retain(|key,value|b.known.get(key)==Some(value));
        a.pointers.retain(|key,value|b.pointers.get(key)==Some(value));
        a.symbolic = symbolic.remove(allocation).ok_or("CUDA family lost allocation facts")?;
    }
    for access in no.memory.accesses { if !yes.memory.accesses.contains(&access) { yes.memory.accesses.push(access); } }
    yes.previous = union(yes.previous,no.previous);
    yes.warp_events = union(yes.warp_events,no.warp_events);
    Ok(yes)
}
