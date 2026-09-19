//! Conditional trace construction for one retained scalar CFG. A local choice
//! forks only its defining region and rejoins once at the recorded SSA boundary.
//! Continuations, static instruction ranks and resources are shared. This is
//! model construction; it contains no optimizer or complete-assignment traversal.
use super::*;

pub type Guard = BTreeMap<usize, bool>;
pub struct DerivedFamily {
    pub execution: DerivedModel,
    pub guards: Vec<Guard>,
    pub obligations: Vec<(Guard,String)>,
}
#[derive(Clone)]
struct Flow {
    block: Block,
    values: HashMap<Value, Binding>,
    memory: Memory,
    gate: Vec<usize>,
    completed: Vec<usize>,
}
struct Construction<'a> {
    scalar: Derivation<'a>,
    parameters: BTreeMap<Inst, usize>,
    joins: BTreeMap<Inst, Block>,
    guards: Vec<Guard>,
    obligations: Vec<(Guard,String)>,
    occurrences: HashMap<(u64,Inst),u64>,
    blocks: HashMap<(u64,Block),u64>,
}

pub fn derive(
    program: &ScalarProgram, hardware: &ScalarHardware, workload: &ScalarWorkload,
    limits: DerivationLimits, parameters: impl IntoIterator<Item=(Inst,usize)>,
    joins: impl IntoIterator<Item=(Inst,Block)>,
) -> Result<DerivedFamily, DerivationError> {
    let requirements = requirements(program)?;
    hardware.validate()?;
    let memory = Memory::new(program,workload)?;
    let graph = Graph::scalar(program);
    let static_orders = seismic_realization::scheduling::Space::new(program)?.constraints().into_iter()
        .map(|block| crate::schedule::static_order::Constraint { block:block.block, instructions:block.instructions, predecessors:block.predecessors, visits:Vec::new() }).collect();
    let mut construction = Construction {
        scalar: Derivation { program, graph:&graph, hardware, limits, memory:memory.clone(), output:DerivedModel {
            model:Model { relationship:crate::authority::ModelRelationship::hypothetical_execution(),
                identity:format!("scalar-family/{}/{}",hardware.identity,workload.identity),timebase:hardware.timebase.clone(),resources:hardware.resources.clone(),
                operations:Vec::new(),lifetimes:Vec::new(),static_orders,unmapped:Vec::new() },
            order:seismic_realization::scheduling::Order::current(program),origins:Vec::new(),accesses:Vec::new(),instructions:0,requirements,
        } }, parameters:parameters.into_iter().collect(), joins:joins.into_iter().collect(), guards:Vec::new(), obligations:Vec::new(),occurrences:HashMap::new(),blocks:HashMap::new(),
    };
    if program.dispatch == Dispatch::Sequential && program.work_items != 1 { return Err("sequential scalar family has an invalid invocation count".into()); }
    let entry = program.function.layout.entry_block().ok_or("empty scalar family")?;
    for invocation in 0..program.work_items {
        let parameters = program.function.dfg.block_params(entry);
        let expected=if program.dispatch==Dispatch::ParallelRoot {4} else {3};
        if parameters.len()!=expected || parameters.iter().any(|value|program.function.dfg.value_type(*value)!=types::I64) {
            return Err("scalar family does not match its declared invocation ABI".into());
        }
        let mut values = HashMap::new();
        let mut memory = memory.clone();
        memory.capacities.insert(AllocationIdentity::Scratch(invocation),memory.scratch_bytes);
        for (&value, allocation) in parameters.iter().take(3).zip([AllocationIdentity::BufferTable,AllocationIdentity::Scalars,AllocationIdentity::Scratch(invocation)]) {
            values.insert(value,Binding {datum:Datum::Pointer {allocation,offset:0u64.into()},ready:Vec::new()});
        }
        if parameters.len()==4 { values.insert(parameters[3],Binding {datum:Datum::Bits(invocation),ready:Vec::new()}); }
        let flow=Flow {block:entry,values,memory,gate:Vec::new(),completed:Vec::new()};
        construction.run(flow,None,&Guard::new(),invocation)?;
    }
    construction.scalar.output.model.unmapped.clear();
    Ok(DerivedFamily {execution:construction.scalar.output,guards:construction.guards,obligations:construction.obligations})
}
impl Construction<'_> {
    fn instruction(&mut self, primitive:&Primitive, dependencies:&[usize], origin:Origin, guard:&Guard, parameter:Option<usize>) -> Result<Instance,DerivationError> {
        let Some(parameter)=parameter else {
            if self.scalar.hardware.timing(primitive).is_none() {
                let reason=format!("resource timing unavailable for actual primitive {primitive:?}");
                if !self.obligations.iter().any(|(active,missing)|active==guard && missing==&reason) {self.obligations.push((guard.clone(),reason));}
            }
            let begin=self.scalar.output.model.operations.len();
            let instance=self.scalar.instantiate(primitive,dependencies,origin)?;
            self.record(begin,guard);
            return Ok(instance);
        };
        if let Some(&selected)=guard.get(&parameter) {
            let mut primitive=primitive.clone();
            primitive.modifier=Modifier::Integer(i64::from(selected));
            return self.instruction(&primitive,dependencies,origin,guard,None);
        }
        // Both immediate values use their own hardware service facts. They
        // share one static issue position and a zero-duration readiness join.
        let mut roots=Vec::new();
        let mut completions=Vec::new();
        for selected in [false,true] {
            let mut branch=guard.clone();branch.insert(parameter,selected);
            let mut primitive=primitive.clone();primitive.modifier=Modifier::Integer(i64::from(selected));
            let instance=self.instruction(&primitive,dependencies,origin.clone(),&branch,None)?;
            roots.extend(instance.roots);completions.push(instance.completion);
        }
        let begin=self.scalar.output.model.operations.len();
        let completion=self.scalar.append(Operation {name:format!("parameter.{parameter}.ready"),predecessors:completions,start_predecessors:Vec::new(),latency:0,reservations:Vec::new()},origin)?;
        self.record(begin,guard);
        Ok(Instance {completion,roots})
    }
    fn record(&mut self, begin:usize, guard:&Guard) {
        self.guards.extend((begin..self.scalar.output.model.operations.len()).map(|_|guard.clone()));
    }
    fn run(&mut self, mut flow:Flow, stop:Option<Block>, guard:&Guard, invocation:u64) -> Result<Flow,DerivationError> {
        loop {
            if Some(flow.block)==stop {return Ok(flow);}
            let block=flow.block;
            let basic=self.scalar.graph.blocks.iter().find(|b|b.id==block).ok_or("missing scalar family block")?.clone();
            let occurrence=self.blocks.entry((invocation,block)).or_default();
            let mut visit=crate::schedule::static_order::Visit {invocation,occurrence:*occurrence,roots:Vec::new()};
            *occurrence+=1;
            let mut edge=None;
            let mut fork=None;
            for index in basic.instructions {
                if self.scalar.output.instructions >= self.scalar.limits.instructions {return Err(DerivationError::Exhausted(DerivationLimit::Instructions(self.scalar.limits.instructions)));}
                self.scalar.output.instructions+=1;
                let instruction=self.scalar.graph.instructions[index].clone();
                let occurrence=self.occurrences.entry((invocation,instruction.id)).or_default();
                let current=*occurrence; *occurrence+=1;
                let count=if instruction.opcode==Opcode::Jump {0} else if instruction.opcode==Opcode::Brif {1} else {instruction.inputs.len()};
                let inputs=instruction.inputs.iter().take(count).map(|input| {
                    let id=self.scalar.program.function.dfg.resolve_aliases(input.value);
                    let mut value=flow.values.get(&id).cloned().ok_or_else(||format!("undefined family SSA operand {id}"))?;
                    value.datum=value.datum.resolve(guard); Ok(value)
                }).collect::<Result<Vec<_>,String>>()?;
                let mut dependencies=flow.gate.iter().copied().collect::<BTreeSet<_>>();
                dependencies.extend(inputs.iter().flat_map(|v|v.ready.iter().copied()));
                if instruction.opcode==Opcode::Return {dependencies.extend(flow.completed.iter().copied());}
                let (datum,instance)=self.execute(&instruction,&inputs,&flow.values,&mut flow.memory,&dependencies.into_iter().collect::<Vec<_>>(),guard,invocation,current)?;
                let completion=instance.completion;
                visit.roots.push(instance.roots);
                flow.completed.push(completion);
                if instruction.outputs.len()>1 {return Err("family primitive has unsupported result arity".into());}
                for &(value,_) in &instruction.outputs {flow.values.insert(value,Binding {datum:datum.clone(),ready:vec![completion]});}
                match instruction.encoding {
                    Data::Jump {destination,..}=>{edge=Some((destination,instruction.id,current,completion));break;},
                    Data::Brif {blocks,..}=>{
                        if let Some(value)=inputs[0].datum.bits() {edge=Some((blocks[usize::from(value==0)],instruction.id,current,completion));}
                        else if inputs[0].datum.parameter().is_some() {
                            let join=*self.joins.get(&instruction.id).ok_or("family branch has no retained reconvergence boundary")?;
                            fork=Some((inputs[0].datum.clone(),blocks,join,instruction.id,current,completion));
                        } else {return Err(DerivationError::Unsupported("scalar family control is not uniform over the declared runtime domain".into()));}
                        break;
                    },
                    Data::MultiAry {opcode:Opcode::Return,..}=>{
                        if inputs.len()!=1 || inputs[0].datum.bits()!=Some(0) {return Err("scalar family reaches an unsuccessful return".into());}
                        self.visit(block,visit); return Ok(flow);
                    },
                    _=>{},
                }
            }
            self.visit(block,visit);
            if let Some((condition,blocks,join,terminator,occurrence,completion))=fork {
                flow=self.branch(flow,&condition,blocks,join,terminator,occurrence,completion,guard,invocation)?;
            } else {
                let (destination,terminator,occurrence,completion)=edge.ok_or("scalar family block lacks a terminator")?;
                flow=self.transfer(flow,destination,terminator,occurrence,completion,guard,invocation)?;
            }
        }
    }
    fn execute(&mut self,instruction:&Instruction,inputs:&[Binding],values:&HashMap<Value,Binding>,memory:&mut Memory,dependencies:&[usize],guard:&Guard,invocation:u64,occurrence:u64)->Result<(Datum,Instance),DerivationError> {
        if inputs.iter().any(|input|!input.datum.resolve(guard).initialized()) {
            return Err("active family instruction consumes an uninitialized value".into());
        }
        let origin=Origin::Instruction {invocation,instruction:instruction.id,occurrence};
        let mut dependencies=dependencies.iter().copied().collect::<BTreeSet<_>>();
        let address=instruction.memory.as_ref().map(|access|values.get(&self.scalar.program.function.dfg.resolve_aliases(access.address)).map(|binding|binding.datum.resolve(guard)).ok_or("undefined family address")).transpose()?;
        if let Some(parameter)=address.as_ref().and_then(Datum::parameter) {
            let mut yes_guard=guard.clone();yes_guard.insert(parameter,true);
            let mut no_guard=guard.clone();no_guard.insert(parameter,false);
            let mut yes_memory=memory.clone();let mut no_memory=memory.clone();
            let dependencies=dependencies.into_iter().collect::<Vec<_>>();
            let (yes,a)=self.execute(instruction,inputs,values,&mut yes_memory,&dependencies,&yes_guard,invocation,occurrence)?;
            let (no,b)=self.execute(instruction,inputs,values,&mut no_memory,&dependencies,&no_guard,invocation,occurrence)?;
            *memory=merge_memory(parameter,yes_memory,no_memory)?;
            let begin=self.scalar.output.model.operations.len();
            let completion=self.scalar.append(Operation {name:format!("memory.choice.{parameter}.ready"),predecessors:vec![a.completion,b.completion],start_predecessors:Vec::new(),latency:0,reservations:Vec::new()},origin)?;
            self.record(begin,guard);
            return Ok((Datum::select(parameter,yes,no),Instance {completion,roots:a.roots.into_iter().chain(b.roots).collect()}));
        }
        let location=if let Some(access)=&instruction.memory {
            let (allocation,offset)=memory.address(address.as_ref().ok_or("memory instruction has no address")?,access.offset,access.bytes)?;
            for before in &self.scalar.output.accesses {
                if before.allocation!=allocation || !(before.write||access.write) || incompatible(&self.guards[before.completion],guard) {continue;}
                match offset.disjoint(u64::from(access.bytes),&before.offset,u64::from(before.bytes)) {
                    Some(true)=>{},
                    Some(false) if before.invocation==invocation=>{dependencies.insert(before.completion);},
                    Some(false)=>return Err("scalar family has conflicting parallel invocation effects".into()),
                    None=>return Err(DerivationError::Unsupported("scalar family memory dependence is not uniform over its workload domain".into())),
                }
            }
            Some((allocation,offset,access.bytes,access.write))
        } else {None};
        let requirement=primitive(instruction)?;
        let parameter=self.parameters.get(&instruction.id).copied();
        let instance=self.instruction(&requirement,&dependencies.into_iter().collect::<Vec<_>>(),origin,guard,parameter)?;
        let datum=if let Some(parameter)=parameter {Datum::Parameter(parameter).resolve(guard)}
            else if let Some((allocation,offset,width,write))=location {
                let value=if write {memory.store(&allocation,&offset,width,&inputs[0].datum.resolve(guard))?;Datum::Unknown}
                    else {memory.read(&allocation,&offset,width)?.resolve(guard)};
                if !value.initialized() {return Err("active family operation reads uninitialized private storage".into());}
                self.scalar.output.accesses.push(Access {invocation,instruction:instruction.id,occurrence,allocation,offset,bytes:width,write,completion:instance.completion});
                value
            } else {
                let inputs=inputs.iter().map(|binding|binding.datum.resolve(guard)).collect::<Vec<_>>();
                evaluate(instruction,&inputs.iter().collect::<Vec<_>>())?
            };
        Ok((datum,instance))
    }
    fn branch(&mut self,flow:Flow,condition:&Datum,blocks:[BlockCall;2],join:Block,terminator:Inst,occurrence:u64,completion:usize,guard:&Guard,invocation:u64)->Result<Flow,DerivationError> {
        let condition=condition.resolve(guard);
        if let Some(value)=condition.bits() {
            let flow=self.transfer(flow,blocks[usize::from(value==0)],terminator,occurrence,completion,guard,invocation)?;
            return self.run(flow,Some(join),guard,invocation);
        }
        let parameter=condition.parameter().ok_or_else(||DerivationError::Unsupported("scalar family control is not uniform over the declared runtime domain".into()))?;
        let mut yes_guard=guard.clone();yes_guard.insert(parameter,true);
        let mut no_guard=guard.clone();no_guard.insert(parameter,false);
        let yes=self.branch(flow.clone(),&condition,blocks,join,terminator,occurrence,completion,&yes_guard,invocation)?;
        let no=self.branch(flow,&condition,blocks,join,terminator,occurrence,completion,&no_guard,invocation)?;
        merge(parameter,yes,no)
    }
    fn visit(&mut self,block:Block,visit:crate::schedule::static_order::Visit) {
        self.scalar.output.model.static_orders.iter_mut().find(|order|order.block==block).expect("every static block has a rank domain").visits.push(visit);
    }
    fn transfer(&mut self,mut flow:Flow,edge:BlockCall,terminator:Inst,occurrence:u64,completion:usize,guard:&Guard,invocation:u64)->Result<Flow,DerivationError> {
        let begin=self.scalar.output.model.operations.len();
        let (block,assignments,gate)=self.scalar.transfer(invocation,terminator,occurrence,completion,edge,&flow.values)?;
        self.record(begin,guard);
        flow.block=block;flow.values.extend(assignments);flow.gate=vec![gate];flow.completed.push(gate);Ok(flow)
    }
}
fn incompatible(a:&Guard,b:&Guard)->bool {a.iter().any(|(id,value)|b.get(id).is_some_and(|other|other!=value))}
fn union<T:Ord>(a:impl IntoIterator<Item=T>,b:impl IntoIterator<Item=T>)->Vec<T> {a.into_iter().chain(b).collect::<BTreeSet<_>>().into_iter().collect()}
fn merge(parameter:usize,mut yes:Flow,no:Flow)->Result<Flow,DerivationError> {
    if yes.block!=no.block {return Err("retained alternatives do not reach their common continuation".into());}
    for key in union(yes.values.keys().copied(),no.values.keys().copied()) {
        let a=yes.values.remove(&key);let b=no.values.get(&key);
        let value=Binding {datum:Datum::select(parameter,a.as_ref().map_or(Datum::Uninitialized,|v|v.datum.clone()),b.map_or(Datum::Uninitialized,|v|v.datum.clone())),
            ready:union(a.into_iter().flat_map(|v|v.ready),b.into_iter().flat_map(|v|v.ready.iter().copied()))};
        yes.values.insert(key,value);
    }
    yes.memory=merge_memory(parameter,yes.memory,no.memory)?;
    yes.gate=union(yes.gate,no.gate);
    yes.completed=union(yes.completed,no.completed);
    Ok(yes)
}

fn merge_memory(parameter:usize,mut yes:Memory,no:Memory)->Result<Memory,DerivationError> {
    if yes.capacities!=no.capacities || yes.pointers!=no.pointers {return Err("source alternatives changed the invocation ABI".into());}
    let mut keys=union(yes.symbolic.keys().cloned(),no.symbolic.keys().cloned());
    keys.extend(union(yes.bytes.keys().cloned(),no.bytes.keys().cloned()).into_iter().map(|(object,offset)|(object,offset.into(),1)));
    let mut facts=BTreeMap::new();
    for (object,offset,width) in keys {
        let a=yes.read(&object,&offset,width)?;
        let b=no.read(&object,&offset,width)?;
        facts.insert((object,offset,width),Datum::select(parameter,a,b));
    }
    yes.symbolic=facts;
    yes.bytes.retain(|key,value|no.bytes.get(key)==Some(value));
    Ok(yes)
}
