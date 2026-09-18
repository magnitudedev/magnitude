//! Coupled reduction is a source merge over tile-valued state. Its expansion uses
//! ordinary typed operations; accounting and emission see that same expansion.
pub mod primitive;
pub mod participants;
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
}

/// A projection of the checked source operation, not a separate computation.
#[derive(Clone, Debug, PartialEq)]
pub struct Reduction {
    pub inputs: Vec<Expr>,
    pub state: Vec<Expr>,
    pub axis: usize,
    pub merge: Expr,
    pub ordered: bool,
    pub span: Span,
    pub implementation: Option<Merge>,
    pub tree: Option<Tree>,
    pub branches: Vec<Branch>,
    pub step: Option<Step>,
    pub segment: Option<i64>,
}

/// Source-owned accumulation: `step(state, input, output)` and the initial
/// state of each partial segment. Regrouping is permitted only by ordered=false.
#[derive(Clone, Debug, PartialEq)]
pub struct Step {
    pub identity: Vec<Expr>,
    pub call: Expr,
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
            inputs: inputs.clone(), state: state.clone(),
            axis: axis.sym.as_ref()?.as_constant()?.try_into().ok()?,
            merge: merge.clone(), ordered: *ordered, span: expr.span,
            implementation: None, tree: None, branches:Vec::new(), segment:None,
            step: if args.len()==7 {let ExprKind::Tuple(identity)=&args[6].kind else {return None;}; Some(Step{identity:identity.clone(),call:args[5].clone(),implementation:None})} else {None},
        })
    }
    pub fn extent(&self) -> &Sym { &self.inputs[0].ty.shaped().unwrap().shape[self.axis] }
    pub fn primitive(&self)->Option<crate::reduction::Contract> {primitive::contract(&self.merge)}
    pub fn merge_name(&self) -> &str {
        match &self.merge.kind {ExprKind::Call{callee,..}=>callee,ExprKind::Builtin{name:Builtin::Reduce,..}=>"reduce",_=>unreachable!()}
    }
    pub fn trees(&self) -> Vec<Tree> {
        let mut trees = vec![Tree::Ordered];
        if !self.ordered && self.extent().as_constant().is_some_and(|n| n > 0 && n < i64::MAX) {
            trees.push(Tree::Pairwise);
            trees.push(Tree::Explicit);
        }
        trees
    }
    pub fn select_tree(&mut self, tree:Tree, select:&mut dyn FnMut(&crate::lowered_ir::Decision)->Result<crate::lowered_ir::Alternative,String>) -> Result<(),String> {
        use crate::lowered_ir::{Alternative,Alternatives,Decision,DecisionKind};
        if !self.trees().contains(&tree) {return Err("reduction tree is outside its source permission".into());}
        self.branches.clear();
        self.segment=None;
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
            Tree::Pairwise => {
                let mut count = self.extent().as_constant().ok_or("pairwise reduction requires a specialized extent")?
                    .checked_add(1).ok_or("reduction leaf count overflow")?;
                let mut level = Vec::new();
                for (state,input) in self.state.iter().zip(&inputs) {
                    let mut shape = state.ty.shaped().unwrap().clone();
                    shape.shape.insert(0, Sym::constant(count));
                    let buffer = builder.alloc(&Ty::Tile(shape), &mut body);
                    body.push(builder.copy(&slice(&buffer, 0, &integer(0, self.span), self.span), state));
                    let index = builder.index();
                    let target = slice(&buffer, 0, &symbol(index.sym.as_ref().unwrap().add(&Sym::constant(1)), self.span), self.span);
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
                for (state,buffer) in self.state.iter().zip(&level) {
                    body.push(builder.copy(state, &slice(buffer,0,&integer(0,self.span),self.span)));
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
        let mut b=Builder{vars,span:self.span};
        let mut body=Vec::new();
        // Identity and input values are fixed for the entire fold, even when
        // the author passes a state binding as an identity expression.
        let inputs = self.snapshot_inputs(&mut b, &mut body);
        let input_types:Vec<_>=inputs.iter().map(|e|slice(e,self.axis,&integer(0,self.span),self.span).ty).collect();
        let (left,right,output)=if let Some(m)=&step.implementation {
            for p in m.left.iter().chain(&m.right).chain(&m.output) {body.push(b.allocate(p));}
            (m.left.clone(),m.right.clone(),m.output.clone())
        } else {
            (self.state.iter().map(|s|b.alloc(&s.ty,&mut body)).collect::<Vec<_>>(),input_types.iter().map(|ty|b.alloc(ty,&mut body)).collect::<Vec<_>>(),self.state.iter().map(|s|b.alloc(&s.ty,&mut body)).collect::<Vec<_>>())
        };
        let index=b.index();
        let visit=|b:&mut Builder<'_>,state:&[Expr]| {
            let mut visit=Vec::new();
            for (dst,src) in left.iter().zip(state) {visit.push(b.copy(dst,src));}
            for (dst,src) in right.iter().zip(&inputs) {visit.push(b.copy(dst,&slice(src,self.axis,&index,self.span)));}
            if let Some(m)=&step.implementation {visit.extend(m.body.clone());} else {
                let mut call=step.call.clone();
                let ExprKind::Call{args,..}=&mut call.kind else {unreachable!()};
                *args=left.iter().chain(&right).chain(&output).cloned().collect();
                visit.push(stmt(StmtKind::Expr(call),self.span));
            }
            for (dst,src) in state.iter().zip(&output) {visit.push(b.copy(dst,src));}
            visit
        };
        if tree==Tree::Ordered {
            let visit=visit(&mut b,&self.state);
            body.push(b.range(&index,self.extent().clone(),visit));
            return Ok(Fold::Serial(body));
        }
        let extent=self.extent().as_constant().ok_or("segmented fold needs finite extent")?;
        let segment=self.segment.ok_or("fold segment choice remains unresolved")?;
        if !(1..=extent).contains(&segment) {return Err("invalid fold segment capacity".into());}
        let groups=extent/segment+i64::from(extent%segment!=0);
        let identity:Vec<_>=step.identity.iter().map(|e|{let copy=b.alloc(&e.ty,&mut body);body.push(b.copy(&copy,e));copy}).collect();
        let partial:Vec<_>=self.state.iter().map(|s|b.alloc(&s.ty,&mut body)).collect();
        let leaves:Vec<_>=self.state.iter().map(|s| {
            let mut shape=s.ty.shaped().unwrap().clone();shape.shape.insert(0,Sym::constant(groups));b.alloc(&Ty::Tile(shape),&mut body)
        }).collect();
        let group=b.index();
        let start=group.sym.as_ref().unwrap().mul(&Sym::constant(segment));
        let mut group_body=Vec::new();
        for (dst,src) in partial.iter().zip(&identity) {group_body.push(b.copy(dst,src));}
        // Full segment is bounded by a guard on the ordinary index operation.
        // The same guard is visible to emission, accounting, and dependencies.
        let visit=visit(&mut b,&partial);
        let end=start.add(&Sym::constant(segment));
        let cond=Expr{kind:ExprKind::Binary{op:crate::ast::BinaryOp::Lt,lhs:Box::new(index.clone()),rhs:Box::new(integer(extent,self.span))},ty:Ty::Scalar(DType::Bool),sym:None,span:self.span};
        let ExprKind::Var(var)=index.kind else {unreachable!()};
        group_body.push(stmt(StmtKind::Range{var,lo:start,hi:end,body:vec![stmt(StmtKind::If{cond,then:visit,els:vec![]},self.span)]},self.span));
        let publish=leaves.iter().zip(&partial).map(|(dst,src)|b.copy(&slice(dst,0,&group,self.span),src)).collect();
        let mut merge=self.clone();merge.inputs=leaves.clone();merge.axis=0;merge.step=None;merge.segment=None;
        Ok(Fold::Segments(Segments{setup:body,index:group,count:groups,body:group_body,partial,leaves,publish,merge}))
    }
    fn call(&self, left: &[Expr], right: &[Expr], output: &[Expr]) -> Vec<Stmt> {
        if let Some(merge) = &self.implementation { return merge.body.clone(); }
        let mut call = self.merge.clone();
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
