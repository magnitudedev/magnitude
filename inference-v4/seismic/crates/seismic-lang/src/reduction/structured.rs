//! Coupled reduction is a source merge over tile-valued state. Its expansion uses
//! ordinary typed operations; accounting and emission see that same expansion.
pub mod primitive;
pub mod participants;
mod retention;
use crate::{
    ast::AssignOp,
    ir::{Builtin, Expr, ExprKind, Index, Stmt, StmtKind, Var, VarKind},
    span::Span,
    sym::{Atom, Sym},
    types::{DType, Shaped, Ty},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Tree {
    Ordered,
    Pairwise,
    /// Every ordered binary merge tree, selected by cuts of contiguous leaves.
    Explicit,
    /// The seed is the left root child; the right child is the contiguous
    /// pairwise tree of inputs. A compact cover of an admitted explicit tree.
    SeedThenPairwise,
}

/// Origin of a retained callback. A product callback consists entirely of its
/// resolved ordinary implementation; it has no invented source helper or
/// primitive algebraic contract.
#[derive(Clone, Debug, PartialEq)]
pub enum Callback {
    Source(Expr),
    Product,
}
impl Callback {
    pub fn source(&self) -> Option<&Expr> {
        match self { Self::Source(e) => Some(e), Self::Product => None }
    }
    pub fn source_mut(&mut self) -> Option<&mut Expr> {
        match self { Self::Source(e) => Some(e), Self::Product => None }
    }
}

/// Lifetime of acquired input data relative to fold preparation windows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PreparationScope {
    Window,
    Segment,
}

/// How an input snapshot is prepared within one selected fold segment.
/// The preparation produces ordinary typed statements during fold expansion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputPreparation {
    Direct,
    Packets { width: u32, decoder: crate::repr::PacketDecoder, coefficients: PreparationScope, words: PreparationScope },
    EncodedSnapshot,
    DecodedSnapshot { scope: PreparationScope },
}

/// A projection of the checked source operation, not a separate computation.
#[derive(Clone, Debug, PartialEq)]
pub struct Reduction {
    pub inputs: Vec<Expr>,
    pub preparation: Vec<InputPreparation>,
    pub unroll: i64,
    /// Logical preparation window; independent of the numerical fold segment.
    pub preparation_window: Option<i64>,
    pub state: Vec<Expr>,
    pub axis: usize,
    pub merge: Callback,
    pub ordered: bool,
    pub span: Span,
    pub implementation: Option<Merge>,
    pub tree: Option<Tree>,
    pub branches: Vec<Branch>,
    pub step: Option<Step>,
    pub segment: Option<i64>,
}

/// Storage of a step result relative to its private left state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StepState { Separate, Retained }

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StepOperand { Private, View }

/// Source-owned accumulation: `step(state, input, output)` and the initial
/// state of each partial segment. Regrouping is permitted only by ordered=false.
#[derive(Clone, Debug, PartialEq)]
pub struct Step {
    pub state: StepState,
    pub operands: Vec<StepOperand>,
    pub identity: Vec<Expr>,
    pub call: Callback,
    pub implementation: Option<Merge>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Branch { pub start: i64, pub cut: i64, pub end: i64 }

/// The selected fold's ordinary operations separated at their existing
/// independent segment/completion boundary. Backends refine ownership and
/// publications here; serial realization consumes these exact same operations.
#[derive(Clone, Debug)]
pub struct Segments {
    pub setup: Vec<Stmt>,
    pub index: Expr,
    pub count: i64,
    pub body: Vec<Stmt>,
    pub partial: Vec<Expr>,
    pub leaves: Vec<Expr>,
    pub publish: Vec<Stmt>,
    pub merge: Reduction,
}
enum Fold { Serial(Vec<Stmt>), Segments(Segments) }

/// Inlined implementation of the source merge. Parameter identities belong to
/// the enclosing function's ordinary variable table, just like a visible helper.
#[derive(Clone, Debug, PartialEq)]
pub struct Merge {
    pub left: Vec<Expr>,
    pub right: Vec<Expr>,
    pub output: Vec<Expr>,
    pub body: Vec<Stmt>,
}
impl Reduction {
    pub fn operands(&self) -> impl Iterator<Item = &Expr> { self.inputs.iter().chain(&self.state).chain(self.step.iter().flat_map(|s|s.identity.iter())) }
    pub fn operands_mut(&mut self) -> impl Iterator<Item = &mut Expr> { self.inputs.iter_mut().chain(&mut self.state).chain(self.step.iter_mut().flat_map(|s|s.identity.iter_mut())) }
    pub fn implementations(&self) -> impl Iterator<Item=&Merge> { self.implementation.iter().chain(self.step.iter().flat_map(|s|s.implementation.iter())) }
    pub fn implementations_mut(&mut self) -> impl Iterator<Item=&mut Merge> { self.implementation.iter_mut().chain(self.step.iter_mut().flat_map(|s|s.implementation.iter_mut())) }
    pub fn bodies(&self) -> impl Iterator<Item=&[Stmt]> { self.implementations().map(|m|m.body.as_slice()) }
    pub fn body(&self) -> &[Stmt] { self.implementation.as_ref().map_or(&[], |m|m.body.as_slice()) }
    pub fn state_variables(&self) -> impl Iterator<Item = usize> + '_ {
        self.state.iter().filter_map(|e| match e.kind {ExprKind::Var(v)=>Some(v), _=>None})
    }
    pub fn from_expr(expr: &Expr) -> Option<Self> {
        let ExprKind::Builtin { name: Builtin::Reduce, args } = &expr.kind else { return None };
        if args.len()!=5 && args.len()!=7 {return None;}
        let [fields, axis, merge, state, permission] = &args[..5] else { return None };
        let (ExprKind::Tuple(inputs), ExprKind::Tuple(state), ExprKind::Call { .. }, ExprKind::Bool(ordered)) =
            (&fields.kind, &state.kind, &merge.kind, &permission.kind) else { return None };
        Some(Self {
            inputs: inputs.clone(), preparation: vec![InputPreparation::Direct; inputs.len()], unroll: 1, preparation_window: None, state: state.clone(),
            axis: axis.sym.as_ref()?.as_constant()?.try_into().ok()?,
            merge: Callback::Source(merge.clone()), ordered: *ordered, span: expr.span,
            implementation: None, tree: None, branches:Vec::new(), segment:None,
            step: if args.len()==7 {let ExprKind::Tuple(identity)=&args[6].kind else {return None;}; Some(Step{state:StepState::Separate,operands:Vec::new(),identity:identity.clone(),call:Callback::Source(args[5].clone()),implementation:None})} else {None},
        })
    }
    pub fn extent(&self) -> &Sym { &self.inputs[0].ty.shaped().unwrap().shape[self.axis] }
    pub fn primitive(&self)->Option<crate::reduction::Contract> {self.merge.source().and_then(primitive::contract)}
    pub fn merge_name(&self) -> &str {
        match self.merge.source().map(|e| &e.kind) {Some(ExprKind::Call{callee,..})=>callee,Some(ExprKind::Builtin{name:Builtin::Reduce,..})=>"reduce",None=>"product",_=>unreachable!()}
    }
    pub fn trees(&self) -> Vec<Tree> {
        let mut trees = vec![Tree::Ordered];
        if !self.ordered && self.extent().as_constant().is_some_and(|n| n > 0 && n < i64::MAX) {
            trees.push(Tree::Pairwise);
            trees.push(Tree::Explicit);
            trees.push(Tree::SeedThenPairwise);
        }
        trees
    }
    pub fn select_tree(&mut self, tree:Tree, select:&mut dyn FnMut(&crate::lowered_ir::Decision)->Result<crate::lowered_ir::Alternative,String>) -> Result<(),String> {
        use crate::lowered_ir::{Alternative,Alternatives,Decision,DecisionKind};
        if !self.trees().contains(&tree) {return Err("reduction tree is outside its source permission".into());}
        self.branches.clear();
        self.segment=None;
        self.unroll=1;
        self.preparation_window=None;
        self.preparation.fill(InputPreparation::Direct);
        if tree!=Tree::Ordered && self.step.is_some() {
            let extent=self.extent().as_constant().ok_or("segmented fold needs a finite extent")?;
            let domain=Decision{kind:DecisionKind::ReductionSegments{extent},alternatives:Alternatives::reduction_segments(extent)?};
            let Alternative::ReductionSegment(segment)=select(&domain)? else {return Err("invalid fold segment selection".into())};
            if !(1..=extent).contains(&segment) {return Err("fold segment outside input extent".into());}
            self.segment=Some(segment);
        }
        if tree==Tree::Explicit {
            let count=self.extent().as_constant().ok_or("explicit reduction tree needs finite specialized leaves")?;
            let count=self.segment.map_or(count,|s|count/s+i64::from(count%s!=0));
            let leaves=count.checked_add(1).ok_or("reduction leaf count overflow")?;
            let mut pending=vec![(0,leaves)];
            while let Some((start,end))=pending.pop() {
                if end-start==1 {continue;}
                let domain=Decision{kind:DecisionKind::ReductionBranch{start,end,fields:self.state.iter().map(|e|e.ty.clone()).collect()},alternatives:Alternatives::reduction_cuts(start+1,end-1)?};
                let Alternative::ReductionCut(cut)=select(&domain)? else {return Err("invalid reduction cut selection".into())};
                if !(start+1..end).contains(&cut) {return Err("reduction cut does not partition its leaves".into());}
                self.branches.push(Branch{start,cut,end});
                pending.push((cut,end));pending.push((start,cut));
            }
        }
        self.tree=Some(tree);
        Ok(())
    }
    /// Fix source input values before splitting this operation into pieces.
    /// The ordinary value assignments preserve evaluation order and establish
    /// snapshot identity that every derived piece can safely share.
    pub(crate) fn capture_inputs(&mut self, vars: &mut Vec<Var>) -> Vec<Stmt> {
        let mut builder = Builder { vars, span: self.span };
        let mut body = Vec::new();
        self.inputs = self.snapshot_inputs(&mut builder, &mut body);
        body
    }
    fn snapshot_inputs(&self, builder: &mut Builder<'_>, body: &mut Vec<Stmt>) -> Vec<Expr> {
        self.inputs.iter().map(|input| {
            // Named local tiles have value semantics. Reading a binding that
            // this reduction does not replace preserves the source snapshot.
            if matches!(input.kind, ExprKind::Var(id) if !self.state_variables().any(|v| v == id)) {
                input.clone()
            } else {
                builder.snapshot(input, body)
            }
        }).collect()
    }
    /// `ordered=false` authorizes regrouping, not permutation or replacing the
    /// merge's arithmetic. No generic associativity theorem is claimed here.
    /// The initial state is one leaf, included exactly once even if non-neutral.
    pub fn expand(&self, tree: Tree, vars: &mut Vec<Var>) -> Result<Vec<Stmt>, String> {
        if !self.trees().contains(&tree) { return Err("reduction tree is not permitted by this source operation".into()); }
        if self.step.is_some() { return self.expand_fold(tree,vars); }
        let mut builder = Builder { vars, span: self.span };
        let mut body = Vec::new();
        // Snapshot inputs before modifying any state. Merge arguments themselves
        // are independent snapshots, including for a destructive source helper.
        let inputs = self.snapshot_inputs(&mut builder, &mut body);
        let (left, right, output) = if let Some(merge) = &self.implementation {
            for parameter in merge.left.iter().chain(&merge.right).chain(&merge.output) {
                body.push(builder.allocate(parameter));
            }
            (merge.left.clone(), merge.right.clone(), merge.output.clone())
        } else {
            (self.state.iter().map(|s| builder.alloc(&s.ty, &mut body)).collect(),
             self.state.iter().map(|s| builder.alloc(&s.ty, &mut body)).collect(),
             self.state.iter().map(|s| builder.alloc(&s.ty, &mut body)).collect())
        };
        match tree {
            Tree::Ordered => {
                let index = builder.index();
                let mut step = Vec::new();
                for ((l,r),(state,input)) in left.iter().zip(&right).zip(self.state.iter().zip(&inputs)) {
                    step.push(builder.copy(l, state));
                    step.push(builder.copy(r, &slice(input, self.axis, &index, self.span)));
                }
                step.extend(self.call(&left, &right, &output));
                for (state, result) in self.state.iter().zip(&output) { step.push(builder.copy(state, result)); }
                body.push(builder.range(&index, self.extent().clone(), step));
            }
            Tree::Pairwise | Tree::SeedThenPairwise => {
                let root_seed = tree == Tree::SeedThenPairwise;
                let seed_leaves = i64::from(!root_seed);
                let mut count = self.extent().as_constant().ok_or("pairwise reduction requires a specialized extent")?
                    .checked_add(seed_leaves).ok_or("reduction leaf count overflow")?;
                let mut level = Vec::new();
                for (state,input) in self.state.iter().zip(&inputs) {
                    let mut shape = state.ty.shaped().unwrap().clone();
                    shape.shape.insert(0, Sym::constant(count));
                    let buffer = builder.alloc(&Ty::Tile(shape), &mut body);
                    if !root_seed {
                        body.push(builder.copy(&slice(&buffer, 0, &integer(0, self.span), self.span), state));
                    }
                    let index = builder.index();
                    let target = slice(&buffer, 0, &symbol(index.sym.as_ref().unwrap().add(&Sym::constant(seed_leaves)), self.span), self.span);
                    let copy = builder.copy(&target, &slice(input, self.axis, &index, self.span));
                    body.push(builder.range(&index, self.extent().clone(), vec![copy]));
                    level.push(buffer);
                }
                while count > 1 {
                    let next_count = count / 2 + count % 2;
                    let mut next = Vec::new();
                    for state in &self.state {
                        let mut shape = state.ty.shaped().unwrap().clone();
                        shape.shape.insert(0, Sym::constant(next_count));
                        next.push(builder.alloc(&Ty::Tile(shape), &mut body));
                    }
                    let index = builder.index();
                    let even = symbol(index.sym.as_ref().unwrap().mul(&Sym::constant(2)), self.span);
                    let odd = symbol(even.sym.as_ref().unwrap().add(&Sym::constant(1)), self.span);
                    let mut step = Vec::new();
                    for ((l,r),buffer) in left.iter().zip(&right).zip(&level) {
                        step.push(builder.copy(l, &slice(buffer, 0, &even, self.span)));
                        step.push(builder.copy(r, &slice(buffer, 0, &odd, self.span)));
                    }
                    step.extend(self.call(&left, &right, &output));
                    for (buffer,result) in next.iter().zip(&output) {
                        step.push(builder.copy(&slice(buffer, 0, &index, self.span), result));
                    }
                    body.push(builder.range(&index, Sym::constant(count / 2), step));
                    if count % 2 == 1 {
                        for (dst,src) in next.iter().zip(&level) {
                            body.push(builder.copy(&slice(dst,0,&integer(next_count-1,self.span),self.span),
                                &slice(src,0,&integer(count-1,self.span),self.span)));
                        }
                    }
                    level = next;
                    count = next_count;
                }
                if root_seed {
                    for ((l, r), (state, buffer)) in left.iter().zip(&right).zip(self.state.iter().zip(&level)) {
                        body.push(builder.copy(l, state));
                        body.push(builder.copy(r, &slice(buffer, 0, &integer(0, self.span), self.span)));
                    }
                    body.extend(self.call(&left, &right, &output));
                    for (state, result) in self.state.iter().zip(&output) { body.push(builder.copy(state, result)); }
                } else {
                    for (state, buffer) in self.state.iter().zip(&level) {
                        body.push(builder.copy(state, &slice(buffer, 0, &integer(0, self.span), self.span)));
                    }
                }
            }
            Tree::Explicit => {
                let leaves=self.extent().as_constant().and_then(|n|n.checked_add(1)).ok_or("explicit reduction tree needs finite leaves")?;
                let branches:std::collections::BTreeMap<_,_>=self.branches.iter().map(|b|((b.start,b.end),b.cut)).collect();
                if branches.len()!=self.branches.len() {return Err("duplicate reduction tree region".into());}
                let mut pending=vec![(0,leaves,false)];
                let mut values=std::collections::BTreeMap::<(i64,i64),Vec<Expr>>::new();
                let mut visited=0;
                while let Some((start,end,ready))=pending.pop() {
                    if end-start==1 {
                        let state:Vec<_>=self.state.iter().map(|s|builder.alloc(&s.ty,&mut body)).collect();
                        for (i,value) in state.iter().enumerate() {
                            let source=if start==0 {self.state[i].clone()} else {slice(&inputs[i],self.axis,&integer(start-1,self.span),self.span)};
                            body.push(builder.copy(value,&source));
                        }
                        values.insert((start,end),state);
                        continue;
                    }
                    let cut=*branches.get(&(start,end)).ok_or("reduction tree leaves are not fully covered")?;
                    if !(start+1..end).contains(&cut) {return Err("invalid reduction tree partition".into());}
                    if !ready {
                        visited+=1;
                        pending.push((start,end,true));pending.push((cut,end,false));pending.push((start,cut,false));
                        continue;
                    }
                    let a=values.remove(&(start,cut)).ok_or("missing left merge operand")?;
                    let b=values.remove(&(cut,end)).ok_or("missing right merge operand")?;
                    for ((l,r),(a,b)) in left.iter().zip(&right).zip(a.iter().zip(&b)) {
                        body.push(builder.copy(l,a));body.push(builder.copy(r,b));
                    }
                    body.extend(self.call(&left,&right,&output));
                    let state:Vec<_>=self.state.iter().map(|s|builder.alloc(&s.ty,&mut body)).collect();
                    for (dst,src) in state.iter().zip(&output) {body.push(builder.copy(dst,src));}
                    values.insert((start,end),state);
                }
                if visited!=branches.len() {return Err("reduction tree contains unreachable regions".into());}
                let state=values.remove(&(0,leaves)).ok_or("missing completed reduction")?;
                for (dst,src) in self.state.iter().zip(&state) {body.push(builder.copy(dst,src));}
            }
        }
        Ok(body)
    }
    pub fn segments(&self,vars:&mut Vec<Var>)->Result<Segments,String> {
        if self.step.is_none() {return Err("segmentation requires a source step and identity".into());}
        let tree=self.tree.ok_or("fold tree choice remains unresolved")?;
        if !self.trees().contains(&tree) {return Err("fold tree is outside source permissions".into());}
        match self.fold(tree,vars)? {Fold::Segments(parts)=>Ok(parts),Fold::Serial(_)=>Err("selected fold is ordered and unsplit".into())}
    }
    fn expand_fold(&self,tree:Tree,vars:&mut Vec<Var>)->Result<Vec<Stmt>,String> {
        match self.fold(tree,vars)? {
            Fold::Serial(body)=>Ok(body),
            Fold::Segments(parts)=> {
                let mut body=parts.setup;
                let mut segment=parts.body;segment.extend(parts.publish);
                let b=Builder{vars,span:self.span};
                body.push(b.range(&parts.index,Sym::constant(parts.count),segment));
                body.extend(parts.merge.expand(tree,b.vars)?);
                Ok(body)
            }
        }
    }
    fn fold(&self, tree:Tree, vars:&mut Vec<Var>)->Result<Fold,String> {
        let step=self.step.as_ref().unwrap();
        let implementation = match (&step.implementation, step.state) {
            (Some(m), StepState::Retained) => Some(m.retain_state().ok_or("step state retention is not legal")?),
            (m, StepState::Separate) => m.clone(),
            (None, StepState::Retained) => return Err("retained state needs a resolved step".into()),
        };
        for (input, placement) in step.operands.iter().enumerate() {
            if *placement == StepOperand::View && !step.implementation.as_ref().is_some_and(|m| m.can_view_operand(input)) {
                return Err("step operand view is not legal".into());
            }
        }
        let mut b=Builder{vars,span:self.span};
        let mut body=Vec::new();
        // Identity and input values are fixed for the entire fold, even when
        // the author passes a state binding as an identity expression.
        let inputs = self.snapshot_inputs(&mut b, &mut body);
        let input_types:Vec<_>=inputs.iter().map(|e|slice(e,self.axis,&integer(0,self.span),self.span).ty).collect();
        let (left,right,output)=if let Some(m)=&implementation {
            let mut allocated = std::collections::HashSet::new();
            for p in m.left.iter().chain(&m.output).chain(m.right.iter().enumerate().filter_map(|(i, p)| {
                (step.operands.get(i) != Some(&StepOperand::View)).then_some(p)
            })) {
                let ExprKind::Var(id) = p.kind else { return Err("step parameter must be a private binding".into()); };
                if allocated.insert(id) { body.push(b.allocate(p)); }
            }
            (m.left.clone(),m.right.clone(),m.output.clone())
        } else {
            (self.state.iter().map(|s|b.alloc(&s.ty,&mut body)).collect::<Vec<_>>(),input_types.iter().map(|ty|b.alloc(ty,&mut body)).collect::<Vec<_>>(),self.state.iter().map(|s|b.alloc(&s.ty,&mut body)).collect::<Vec<_>>())
        };
        let index=b.index();
        let visit=|b:&mut Builder<'_>,state:&[Expr],prepared:&[Expr],offsets:&[Sym]| {
            let mut visit=Vec::new();
            for (dst,src) in left.iter().zip(state) {
                if !matches!((&dst.kind, &src.kind), (ExprKind::Var(a), ExprKind::Var(b)) if a == b) {
                    visit.push(b.copy(dst,src));
                }
            }
            let mut views = std::collections::HashMap::new();
            for (input, ((dst,src),offset)) in right.iter().zip(prepared).zip(offsets).enumerate() {
                let at=symbol(index.sym.as_ref().unwrap().sub(offset),self.span);
                let value = slice(src,self.axis,&at,self.span);
                if step.operands.get(input) == Some(&StepOperand::View) {
                    let ExprKind::Var(id) = dst.kind else { unreachable!() };
                    views.insert(id, value);
                } else {
                    visit.push(b.copy(dst,&value));
                }
            }
            if let Some(m)=&implementation {
                let mut body = m.body.clone();
                crate::composition::substitute_values(&mut body, &views);
                visit.extend(body);
            } else {
                let mut call=step.call.source().expect("product step must have an implementation").clone();
                let ExprKind::Call{args,..}=&mut call.kind else {unreachable!()};
                *args=left.iter().chain(&right).chain(&output).cloned().collect();
                visit.push(stmt(StmtKind::Expr(call),self.span));
            }
            for (dst,src) in state.iter().zip(&output) {
                if !matches!((&dst.kind, &src.kind), (ExprKind::Var(a), ExprKind::Var(b)) if a == b) {
                    visit.push(b.copy(dst,src));
                }
            }
            visit
        };
        if tree==Tree::Ordered {
            let visit=visit(&mut b,&self.state,&inputs,&vec![Sym::constant(0); inputs.len()]);
            body.push(b.range(&index,self.extent().clone(),visit));
            return Ok(Fold::Serial(body));
        }
        let extent=self.extent().as_constant().ok_or("segmented fold needs finite extent")?;
        let segment=self.segment.ok_or("fold segment choice remains unresolved")?;
        if !(1..=extent).contains(&segment) {return Err("invalid fold segment capacity".into());}
        let groups=extent/segment+i64::from(extent%segment!=0);
        let identity:Vec<_>=step.identity.iter().map(|e|{let copy=b.alloc(&e.ty,&mut body);body.push(b.copy(&copy,e));copy}).collect();
        // The step's private left parameters retain the segment accumulator.
        // They are not visible to the source or aliased by its inputs. Keeping
        // output distinct still preserves arbitrary cross-element reads/writes
        // in the callback; only the redundant copy into left disappears.
        let partial = left.clone();
        let leaves:Vec<_>=self.state.iter().map(|s| {
            let mut shape=s.ty.shaped().unwrap().clone();shape.shape.insert(0,Sym::constant(groups));b.alloc(&Ty::Tile(shape),&mut body)
        }).collect();
        let group=b.index();
        let start=group.sym.as_ref().unwrap().mul(&Sym::constant(segment));
        let mut group_body=Vec::new();
        for (dst,src) in partial.iter().zip(&identity) {group_body.push(b.copy(dst,src));}
        // Full segment is bounded by a guard on the ordinary index operation.
        // The same guard is visible to emission, accounting, and dependencies.
        let mut prepared = inputs.clone();
        let mut offsets = vec![Sym::constant(0); inputs.len()];
        for (at, preparation) in self.preparation.iter().enumerate() {
            if *preparation == InputPreparation::EncodedSnapshot {
                if extent % segment != 0 { return Err("segment snapshots require complete selected segments".into()); }
                let source = &inputs[at];
                let mut shape = source.ty.shaped().ok_or("segment snapshot input must be shaped")?.clone();
                shape.shape[self.axis] = Sym::constant(segment);
                let ty = Ty::Tile(shape);
                let mut indices = (0..source.ty.shaped().unwrap().shape.len())
                    .map(|_| Index::Slice { start: None, end: None }).collect::<Vec<_>>();
                indices[self.axis] = Index::Slice {
                    start: Some(symbol(start.clone(), self.span)),
                    end: Some(symbol(start.add(&Sym::constant(segment)), self.span)),
                };
                let view = Expr { kind: ExprKind::Index { base: Box::new(source.clone()), indices }, ty: ty.clone(), sym: None, span: self.span };
                let cache = b.local(ty.clone());
                let ExprKind::Var(id) = cache.kind else { unreachable!() };
                b.vars[id].name = format!("segment_snapshot_{id}");
                group_body.push(stmt(StmtKind::Assign {
                    target: cache.clone(), op: AssignOp::Assign,
                    value: Expr { kind: ExprKind::Load { view: Box::new(view), mode: crate::ir::LoadMode::Materialize }, ty, sym: None, span: self.span },
                }, self.span));
                prepared[at] = cache;
                offsets[at] = start.clone();
            }
        }
        for (at, preparation) in self.preparation.iter().enumerate() {
            if *preparation == (InputPreparation::DecodedSnapshot { scope: PreparationScope::Segment }) {
                prepared[at] = b.snapshot_window(&inputs[at], self.axis, start.clone(), segment, extent, &mut group_body)?;
                offsets[at] = start.clone();
            }
        }
        let window = self.preparation_window.unwrap_or(segment);
        if !(1..=segment).contains(&window) || self.unroll < 1 || self.unroll > window {
            return Err("fold preparation or traversal exceeds its selected window".into());
        }
        let mut words = Vec::with_capacity(inputs.len());
        for (source, preparation) in inputs.iter().zip(&self.preparation) {
            if matches!(preparation, InputPreparation::Packets { words: PreparationScope::Segment, .. }) {
                let (cache, producer) = crate::composition::prepare_packet_words(source, start.clone(), segment, b.vars)?;
                group_body.extend(producer);
                words.push(Some(cache));
            } else { words.push(None); }
        }
        let mut coefficients = Vec::with_capacity(inputs.len());
        for (source, preparation) in inputs.iter().zip(&self.preparation) {
            if matches!(preparation, InputPreparation::Packets { coefficients: PreparationScope::Segment, .. }) {
                let (cache, producer) = crate::composition::prepare_packet_coefficients(
                    source, start.clone(), segment, b.vars,
                )?;
                group_body.extend(producer);
                coefficients.push(Some(cache));
            } else {
                coefficients.push(None);
            }
        }
        let prepare_window = |b: &mut Builder<'_>,
                              start: Sym,
                              length: i64|
         -> Result<Vec<Stmt>, String> {
            let mut body = Vec::new();
            let mut prepared = prepared.clone();
            let mut offsets = offsets.clone();
            for (at, preparation) in self.preparation.iter().enumerate() {
                if *preparation == (InputPreparation::DecodedSnapshot { scope: PreparationScope::Window }) {
                    prepared[at] = b.snapshot_window(&inputs[at], self.axis, start.clone(), length, extent, &mut body)?;
                    offsets[at] = start.clone();
                }
                if let InputPreparation::Packets { width, decoder, .. } = preparation {
                    let source = &inputs[at];
                    let shape = source
                        .ty
                        .shaped()
                        .ok_or("packet fold input must be shaped")?;
                    let crate::types::Elem::Repr(name) = &shape.elem else {
                        return Err("packet fold input must remain encoded".into());
                    };
                    let repr = crate::repr::lookup(name).ok_or("unknown fold representation")?;
                    if shape.packed_axis != Some(self.axis)
                        || (length % i64::from(repr.group) != 0 && i64::from(repr.group) % length != 0)
                        || extent % segment != 0
                    {
                        return Err("packet fold preparation requires complete aligned windows".into());
                    }
                    let (cache, producer) = crate::composition::decode_packet_segment(
                        source,
                        start.clone(),
                        length,
                        *width,
                        *decoder,
                        coefficients[at].as_ref(),
                        words[at].as_ref(),
                        b.vars,
                    )?;
                    body.extend(producer);
                    prepared[at] = cache;
                    offsets[at] = start.clone();
                }
            }
            let visit = visit(b, &partial, &prepared, &offsets);
            let end = start.add(&Sym::constant(length));
            let cond = Expr {
                kind: ExprKind::Binary {
                    op: crate::ast::BinaryOp::Lt,
                    lhs: Box::new(index.clone()),
                    rhs: Box::new(integer(extent, self.span)),
                },
                ty: Ty::Scalar(DType::Bool),
                sym: None,
                span: self.span,
            };
            let ExprKind::Var(var) = index.kind else {
                unreachable!()
            };
            if self.unroll == 1 {
                body.push(stmt(
                    StmtKind::Range {
                        var,
                        lo: start,
                        hi: end,
                        body: vec![stmt(
                            StmtKind::If {
                                cond,
                                then: visit,
                                els: vec![],
                            },
                            self.span,
                        )],
                    },
                    self.span,
                ));
            } else {
                let chunk = b.index();
                let chunks = length / self.unroll + i64::from(length % self.unroll != 0);
                let base = if chunks == 1 {
                    start.clone()
                } else {
                    start.add(&chunk.sym.as_ref().unwrap().scale(self.unroll))
                };
                let VarKind::Index(atom) = &b.vars[var].kind else {
                    unreachable!()
                };
                let mut expanded = Vec::new();
                for offset in 0..self.unroll {
                    let at = symbol(base.add(&Sym::constant(offset)), self.span);
                    let mut body = vec![stmt(
                        StmtKind::If {
                            cond: cond.clone(),
                            then: visit.clone(),
                            els: vec![],
                        },
                        self.span,
                    )];
                    crate::widen::replace_index(&mut body, var, atom, &at);
                    if length % self.unroll != 0 {
                        let cond = Expr {
                            kind: ExprKind::Binary {
                                op: crate::ast::BinaryOp::Lt,
                                lhs: Box::new(at),
                                rhs: Box::new(symbol(end.clone(), self.span)),
                            },
                            ty: Ty::Scalar(DType::Bool),
                            sym: None,
                            span: self.span,
                        };
                        body = vec![stmt(
                            StmtKind::If {
                                cond,
                                then: body,
                                els: vec![],
                            },
                            self.span,
                        )];
                    }
                    expanded.extend(body);
                }
                if chunks == 1 {
                    body.extend(expanded);
                } else {
                    body.push(b.range(&chunk, Sym::constant(chunks), expanded));
                }
            }
            Ok(body)
        };
        let windows = segment / window;
        if windows == 1 {
            group_body.extend(prepare_window(&mut b, start.clone(), window)?);
        } else {
            let at = b.index();
            let offset = start.add(&at.sym.as_ref().unwrap().scale(window));
            let body = prepare_window(&mut b, offset, window)?;
            group_body.push(b.range(&at, Sym::constant(windows), body));
        }
        let tail = segment % window;
        if tail > 0 {
            group_body.extend(prepare_window(
                &mut b,
                start.add(&Sym::constant(segment - tail)),
                tail,
            )?);
        }
        let publish=leaves.iter().zip(&partial).map(|(dst,src)|b.copy(&slice(dst,0,&group,self.span),src)).collect();
        let mut merge=self.clone();merge.inputs=leaves.clone();merge.preparation=vec![InputPreparation::Direct; leaves.len()];merge.axis=0;merge.step=None;merge.segment=None;merge.preparation_window=None;
        Ok(Fold::Segments(Segments{setup:body,index:group,count:groups,body:group_body,partial,leaves,publish,merge}))
    }
    fn call(&self, left: &[Expr], right: &[Expr], output: &[Expr]) -> Vec<Stmt> {
        if let Some(merge) = &self.implementation { return merge.body.clone(); }
        let mut call = self.merge.source().expect("product merge must have an implementation").clone();
        let ExprKind::Call { args, .. } = &mut call.kind else { unreachable!() };
        *args = left.iter().chain(right).chain(output).cloned().collect();
        vec![stmt(StmtKind::Expr(call), self.span)]
    }
}

struct Builder<'a> { vars: &'a mut Vec<Var>, span: Span }
impl Builder<'_> {
    fn local(&mut self, ty: Ty) -> Expr {
        let id = self.vars.len();
        self.vars.push(Var { name: format!("reduction_{id}"), ty: ty.clone(), span: self.span, kind: VarKind::Local });
        Expr { kind: ExprKind::Var(id), ty, sym: None, span: self.span }
    }
    fn alloc(&mut self, ty: &Ty, body: &mut Vec<Stmt>) -> Expr {
        let target = self.local(ty.clone());
        body.push(self.allocate(&target));
        target
    }
    fn snapshot(&mut self, source: &Expr, body: &mut Vec<Stmt>) -> Expr {
        // A tile value binding evaluates the source view before allocating its
        // snapshot. In particular, dynamic slice extents inherit their capacity
        // and current length from that view instead of an unbound shape atom.
        let target = self.local(source.ty.clone());
        body.push(stmt(StmtKind::Assign {
            target: target.clone(), op: AssignOp::Assign, value: source.clone(),
        }, self.span));
        target
    }
    fn allocate(&self, target: &Expr) -> Stmt {
        let Ty::Tile(Shaped { shape, elem, .. }) = &target.ty else { unreachable!() };
        let value = Expr { kind: ExprKind::TileAlloc { shape: shape.clone(), dtype: elem.clone() }, ty: target.ty.clone(), sym: None, span: self.span };
        stmt(StmtKind::Assign { target: target.clone(), op: AssignOp::Assign, value }, self.span)
    }
    fn index(&mut self) -> Expr {
        let mut index = self.local(Ty::Scalar(DType::I32));
        let ExprKind::Var(id) = index.kind else { unreachable!() };
        let atom = Atom::Param(format!("$reduction_{id}"));
        self.vars[id].kind = VarKind::Index(atom.clone());
        index.sym = Some(Sym::atom(atom));
        index
    }
    fn range(&self, index: &Expr, hi: Sym, body: Vec<Stmt>) -> Stmt {
        let ExprKind::Var(var) = index.kind else { unreachable!() };
        stmt(StmtKind::Range { var, lo: Sym::constant(0), hi, body }, self.span)
    }
    /// A bounded, decoded snapshot. Tail elements are never read from the
    /// source and are not consumed by the fold. The source's captured shape and
    /// representation remain the authority for each live element read.
    fn snapshot_window(&mut self, source: &Expr, axis: usize, start: Sym, length: i64, extent: i64, body: &mut Vec<Stmt>) -> Result<Expr, String> {
        let mut shape = source.ty.shaped().ok_or("fold snapshot input must be shaped")?.clone();
        let dtype = shape.elem.read_dtype().ok_or("fold snapshot input must have a decoded scalar type")?;
        shape.shape[axis] = Sym::constant(length);
        shape.elem = crate::types::Elem::Dtype(dtype);
        shape.packed_axis = None;
        let cache = self.alloc(&Ty::Tile(shape.clone()), body);
        let ExprKind::Var(id) = cache.kind else { unreachable!() };
        self.vars[id].name = format!("fold_snapshot_{id}");
        let indices: Vec<_> = shape.shape.iter().map(|_| self.index()).collect();
        let at = symbol(start.add(indices[axis].sym.as_ref().unwrap()), self.span);
        let mut source_indices = indices.iter().cloned().map(Index::Point).collect::<Vec<_>>();
        source_indices[axis] = Index::Point(at.clone());
        let value = Expr { kind: ExprKind::Index { base: Box::new(source.clone()), indices: source_indices }, ty: Ty::Scalar(dtype), sym: None, span: self.span };
        let target = Expr { kind: ExprKind::Index { base: Box::new(cache.clone()), indices: indices.iter().cloned().map(Index::Point).collect() }, ty: Ty::Scalar(dtype), sym: None, span: self.span };
        let copy = stmt(StmtKind::Assign { target, op: AssignOp::Assign, value }, self.span);
        let condition = Expr { kind: ExprKind::Binary { op: crate::ast::BinaryOp::Lt, lhs: Box::new(at), rhs: Box::new(integer(extent, self.span)) }, ty: Ty::Scalar(DType::Bool), sym: None, span: self.span };
        body.push(stmt(StmtKind::Owned {
            vars: indices.iter().map(|index| { let ExprKind::Var(id) = index.kind else { unreachable!() }; id }).collect(),
            tile: cache.clone(),
            body: vec![stmt(StmtKind::If { cond: condition, then: vec![copy], els: vec![] }, self.span)],
        }, self.span));
        Ok(cache)
    }
    fn copy(&mut self, target: &Expr, value: &Expr) -> Stmt {
        let shape = target.ty.shaped().unwrap();
        let indices: Vec<_> = shape.shape.iter().map(|_| self.index()).collect();
        let point = |e: &Expr, target: bool| {
            // Flatten tile slices so scalar writes still target the owning local.
            let (base, mut ix) = match &e.kind {
                ExprKind::Index { base, indices } if target => (base.clone(), indices.clone()),
                _ => (Box::new(e.clone()), vec![Index::Slice { start: None, end: None }; shape.shape.len()]),
            };
            let mut point = indices.iter();
            for i in &mut ix { if matches!(i, Index::Slice { .. }) { *i = Index::Point(point.next().unwrap().clone()); } }
            Expr { kind: ExprKind::Index { base, indices: ix }, ty: Ty::Scalar(shape.elem.read_dtype().unwrap_or(DType::F32)), sym: None, span: self.span }
        };
        let assignment = stmt(StmtKind::Assign { target: point(target, true), op: AssignOp::Assign, value: point(value, false) }, self.span);
        stmt(StmtKind::Owned { vars: indices.iter().map(|i| { let ExprKind::Var(id)=i.kind else {unreachable!()}; id }).collect(), tile: target.clone(), body: vec![assignment] }, self.span)
    }
}
fn stmt(kind: StmtKind, span: Span) -> Stmt { Stmt { id: None, kind, span } }
pub(crate) fn integer(value: i64, span: Span) -> Expr { Expr { kind: ExprKind::Int(value), ty: Ty::Scalar(DType::I32), sym: Some(Sym::constant(value)), span } }
fn symbol(value: Sym, span: Span) -> Expr {
    // ShapeParam's spelling is diagnostic; its symbolic expression is authoritative.
    Expr { kind: ExprKind::ShapeParam(value.to_string()), ty: Ty::Scalar(DType::I32), sym: Some(value), span }
}
pub(crate) fn slice(tile: &Expr, axis: usize, index: &Expr, span: Span) -> Expr {
    let mut shaped = tile.ty.shaped().unwrap().clone();
    let rank = shaped.shape.len();
    shaped.shape.remove(axis);
    if let crate::types::Elem::Repr(_) = shaped.elem {
        let packed=shaped.packed_axis.unwrap_or(rank-1);
        if packed==axis {
            shaped.elem=crate::types::Elem::Dtype(shaped.elem.read_dtype().unwrap());
            shaped.packed_axis=None;
        } else {shaped.packed_axis=Some(packed-usize::from(packed>axis));}
    }
    let indices = (0..rank).map(|i| if i == axis { Index::Point(index.clone()) } else { Index::Slice { start: None, end: None } }).collect();
    Expr { kind: ExprKind::Index { base: Box::new(tile.clone()), indices }, ty: Ty::Tile(shaped), sym: None, span }
}

/// Final realization of retained reduction choices. The selected operation and
/// merge body determine every temporary, copy and arithmetic operation exposed to
/// backend storage planning, emission and accounting.
pub fn materialize(function: &crate::lowered_ir::LoweredIr) -> Result<crate::lowered_ir::LoweredIr, String> {
    fn block(body: &mut Vec<Stmt>, vars: &mut Vec<Var>) -> Result<(), String> {
        let mut out = Vec::new();
        for mut statement in std::mem::take(body) {
            if let StmtKind::Reduction(reduction) = &statement.kind {
                let tree = reduction.tree.ok_or("reduction execution choice remains unresolved")?;
                let mut expanded = reduction.expand(tree, vars)?;
                block(&mut expanded, vars)?;
                out.extend(expanded);
                continue;
            }
            match &mut statement.kind {
                StmtKind::Parallel {body,..} | StmtKind::Owned {body,..} |
                StmtKind::LoadLoop {body,..} | StmtKind::Range {body,..} | StmtKind::Lanes {body,..} => block(body, vars)?,
                StmtKind::If {then,els,..} => {block(then, vars)?;block(els, vars)?;}
                _ => {}
            }
            out.push(statement);
        }
        *body = out;
        Ok(())
    }
    let mut function = function.clone();
    block(&mut function.body, &mut function.vars)?;
    Ok(function)
}
