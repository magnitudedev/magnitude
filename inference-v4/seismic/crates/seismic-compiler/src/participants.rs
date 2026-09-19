//! Admission of replicated private values and unique uniform publications for a
//! subgroup realization. This analysis uses inlined dataflow, not function names.
use seismic_lang::{exec::ir::*, exec::lowered_ir::LoweredIr, exec::types::Ty, sym::Sym, syntax::ast::AssignOp};
use std::collections::{HashMap,HashSet};

pub(crate) fn required(function:&LoweredIr)->bool {
    fn expr(e:&Expr)->bool { match &e.kind {
        ExprKind::Intrinsic{..}|ExprKind::Lanes{..}=>true,
        ExprKind::Load{view,..}|ExprKind::Transpose(view)|ExprKind::Accessor{base:view,..}|ExprKind::Unary{expr:view,..}|ExprKind::Cast{expr:view,..}=>expr(view),
        ExprKind::Index{base,indices}=>expr(base)||indices.iter().any(|i|match i{Index::Point(e)=>expr(e),Index::Slice{start,end}=>start.iter().chain(end).any(expr)}),
        ExprKind::Binary{lhs,rhs,..}=>expr(lhs)||expr(rhs),
        ExprKind::Builtin{args,..}|ExprKind::Call{args,..}|ExprKind::Tuple(args)=>args.iter().any(expr),_=>false,
    }}
    fn body(b:&[Stmt])->bool { b.iter().any(|s|match &s.kind{
        StmtKind::Lanes{..}=>true,
        StmtKind::Parallel{body:b,..}|StmtKind::Owned{body:b,..}|StmtKind::Range{body:b,..}|StmtKind::LoadLoop{body:b,..}=>body(b),
        StmtKind::If{cond,then,els}=>expr(cond)||body(then)||body(els),
        StmtKind::Assign{target,value,..}=>expr(target)||expr(value),StmtKind::Expr(e)=>expr(e),
    })}
    body(&function.body)
}

pub(crate) fn validate(function:&LoweredIr)->Result<(),String>{
    let mut a=Analysis{function,uniform:HashMap::new(),varying_symbols:HashSet::new()};
    a.body(&function.body,false)
}
struct Analysis<'a>{function:&'a LoweredIr,uniform:HashMap<VarId,bool>,varying_symbols:HashSet<String>}
impl Analysis<'_>{
    fn symbol(&self,s:&Sym)->bool { s.eval(&|name| if self.varying_symbols.contains(name){None}else{Some(1)}).is_some() }
    fn bind_index(&mut self,var:VarId,uniform:bool){
        self.uniform.insert(var,uniform);
        if let VarKind::Index(seismic_lang::sym::Atom::Param(name))=&self.function.vars[var].kind{
            if uniform {self.varying_symbols.remove(name);}else{self.varying_symbols.insert(name.clone());}
        }
    }
    fn expression(&mut self,e:&Expr,divergent:bool)->Result<bool,String>{
        Ok(match &e.kind{
            ExprKind::Var(v)=>self.uniform.get(v).copied().unwrap_or(true),
            ExprKind::ShapeParam(name)=>!self.varying_symbols.contains(name),
            ExprKind::Int(_)|ExprKind::Float(_)|ExprKind::Bool(_)|ExprKind::TileAlloc{..}=>true,
            ExprKind::Load{view,..}|ExprKind::Transpose(view)|ExprKind::Accessor{base:view,..}|ExprKind::Unary{expr:view,..}|ExprKind::Cast{expr:view,..}=>self.expression(view,divergent)?,
            ExprKind::Index{base,indices}=>{
                let mut u=self.expression(base,divergent)?;
                for index in indices{match index{Index::Point(e)=>u &=self.expression(e,divergent)?,Index::Slice{start,end}=>for e in start.iter().chain(end){u &=self.expression(e,divergent)?;}}}u
            }
            ExprKind::Binary{lhs,rhs,..}=>self.expression(lhs,divergent)? & self.expression(rhs,divergent)?,
            ExprKind::Intrinsic{op,args}=>{
                use seismic_lang::intrinsics::Operation as I;
                if *op==I::LaneIndex {return Ok(false);}
                if divergent {return Err("subgroup intrinsic under varying control requires a convergent ownership realization".into());}
                match op {
                    I::SimdSum=>{for e in args{self.expression(e,divergent)?;}true}
                    I::ShuffleIndex=>{self.expression(&args[0],divergent)?;self.expression(&args[1],divergent)?}
                    _=>return Err(format!("subgroup scalar realization does not implement {op}")),
                }
            }
            ExprKind::Builtin{name:Builtin::Store,args}=>{
                if divergent || !self.expression(&args[0],divergent)? || !self.expression(&args[1],divergent)?{
                    return Err("subgroup publication requires a uniform value and address; distributed publication is unsupported".into());
                }true
            }
            ExprKind::Builtin{name:Builtin::Atomic,..}=>return Err("subgroup atomic publication needs explicit distributed ownership".into()),
            ExprKind::Builtin{args,..}|ExprKind::Tuple(args)=>{let mut u=true;for e in args{u &=self.expression(e,divergent)?;}u},
            ExprKind::Call{..}=>return Err("subgroup admission needs visible inlined helper bodies".into()),
            ExprKind::Lanes{..}=>return Err("lane-distributed tile views need a distributed storage realization".into()),
        })
    }
    fn body(&mut self,body:&[Stmt],divergent:bool)->Result<(),String>{
        for s in body{match &s.kind{
            StmtKind::Assign{target,op,value}=>{
                let u=self.expression(value,divergent)? && !divergent;
                if matches!(target.kind,ExprKind::Var(_)){
                    let ExprKind::Var(id)=target.kind else{unreachable!()};
                    let prior=self.uniform.get(&id).copied().unwrap_or(true);
                    self.uniform.insert(id,u && (*op==AssignOp::Assign || prior));
                }else{
                    let index_uniform=self.expression(target,divergent)?;
                    let mut root=target;while let ExprKind::Index{base,..}|ExprKind::Transpose(base)|ExprKind::Accessor{base,..}=&root.kind{root=base;}
                    let ExprKind::Var(id)=root.kind else{return Err("subgroup store has no retained value owner".into());};
                    if matches!(root.ty,Ty::Tensor(_)) && (!u||!index_uniform){return Err("subgroup tensor assignment requires uniform publication".into());}
                    if !u || !index_uniform {self.uniform.insert(id,false);}
                }
            }
            StmtKind::Expr(e)=>{self.expression(e,divergent)?;}
            StmtKind::If{cond,then,els}=>{
                let nested=divergent || !self.expression(cond,divergent)?;
                let before=self.uniform.clone();self.body(then,nested)?;let yes=self.uniform.clone();self.uniform=before.clone();self.body(els,nested)?;
                for(id,u)in yes{self.uniform.entry(id).and_modify(|v|*v &=u).or_insert(u && before.get(&id).copied().unwrap_or(true));}
            }
            StmtKind::Parallel{vars,body,..}|StmtKind::Owned{vars,body,..}=>{for &var in vars{self.bind_index(var,!divergent);}self.loop_body(body,divergent)?;}
            StmtKind::Range{var,lo,hi,body}=>{let d=divergent||!self.symbol(lo)||!self.symbol(hi);self.bind_index(*var,!d);self.loop_body(body,d)?;}
            StmtKind::Lanes{var,extent,width,body}=>{
                self.bind_index(*var,false);
                let run=Sym::constant(32*width);
                let full=extent.sub(&extent.quot(&run).mul(&run)).as_constant()==Some(0);
                self.loop_body(body,divergent||!full)?;
            }
            StmtKind::LoadLoop{vars,views,body,..}=>{for(&var,view)in vars.iter().zip(views){let u=self.expression(view,divergent)?;self.uniform.insert(var,u);}self.loop_body(body,divergent)?;}
        }}Ok(())
    }
    fn loop_body(&mut self,body:&[Stmt],divergent:bool)->Result<(),String>{
        // A later iteration can observe any varying value produced by an earlier
        // iteration. The finite two-point lattice reaches a stable conservative input.
        loop{let before=self.uniform.clone();self.body(body,divergent)?;for(&id,&u)in &before{self.uniform.entry(id).and_modify(|v|*v &=u);}
            if self.uniform==before{return Ok(());}
        }
    }
}
