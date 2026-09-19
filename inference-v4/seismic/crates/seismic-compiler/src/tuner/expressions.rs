//! Exact source polynomials over the original numeric decisions. Signed source
//! values use a nonnegative coordinate plus a fixed origin, so all products and
//! divisions remain within the shared solver's size/count vocabulary.
use magnitude_solver::model::{Arithmetic, Constraint, Domain, LinearTerm, Literal, ModelBuilder, VarId};
use seismic_accounting::algebra::{Algebra, Error, Symbolic, Value};
use seismic_lang::sym::{Atom, Sym};
use std::collections::BTreeMap;

#[derive(Clone,Copy)]
struct Integer {
    coordinate: Value,
    origin: i128,
}
impl Integer {
    fn bounds(self)->(i128,i128) {
        let (lo,hi)=self.coordinate.bounds();
        (self.origin+i128::from(lo),self.origin+i128::from(hi))
    }
}

pub struct Expressions {
    parameters:BTreeMap<String,Value>,
    values:BTreeMap<Sym,Integer>,
}
impl Expressions {
    pub fn new(parameters:BTreeMap<String,Value>)->Self {Self {parameters,values:BTreeMap::new()}}

    /// Explicit applicability requirement. A negative result excludes that
    /// assignment because the source requires a size/count here. Intermediate
    /// expressions and predicates never inherit this restriction implicitly.
    pub fn nonnegative(&mut self,builder:&mut ModelBuilder,name:&str,expression:&Sym)->Result<Value,Error> {
        let integer=self.integer(builder,name,expression)?;
        self.as_nonnegative(builder,name,integer)
    }
    /// Reify both signs without clipping the original parameter domains.
    pub fn predicate(&mut self,builder:&mut ModelBuilder,name:&str,expression:&Sym)->Result<VarId,Error> {
        let value=self.integer(builder,name,expression)?;
        let (lo,hi)=value.bounds();
        if lo>=0 || hi<0 {
            return Ok(builder.variable(format!("{name}.nonnegative"),Domain::singleton(i64::from(lo>=0))));
        }
        let active=builder.local_variable(format!("{name}.nonnegative"),Domain::boolean()).map_err(invalid)?;
        builder.guarded_constraint(vec![Literal::new(active,1)],Constraint::LinearLe {
            terms:vec![LinearTerm::new(value.coordinate.id(),-1)],rhs:value.origin,
        });
        builder.guarded_constraint(vec![Literal::new(active,0)],Constraint::LinearLe {
            terms:vec![LinearTerm::new(value.coordinate.id(),1)],rhs:value.origin.checked_neg().and_then(|n|n.checked_sub(1)).ok_or_else(overflow)?,
        });
        Ok(active)
    }
    fn integer(&mut self,builder:&mut ModelBuilder,name:&str,expression:&Sym)->Result<Integer,Error> {
        if let Some(&value)=self.values.get(expression) {return Ok(value);}
        let mut terms=Vec::new();let mut constant=0i128;
        for (monomial,coefficient) in expression.monomials() {
            if monomial.is_empty() {
                constant=constant.checked_add(i128::from(coefficient)).ok_or_else(overflow)?;
                continue;
            }
            let mut product=None;
            for (atom,exponent) in monomial {
                let mut power=self.atom(builder,name,atom)?;
                let mut exponent=*exponent;
                let mut factor=None;
                while exponent!=0 {
                    if exponent&1!=0 {factor=Some(match factor {None=>power,Some(left)=>self.product(builder,name,left,power)?});}
                    exponent>>=1;
                    if exponent!=0 {power=self.product(builder,name,power,power)?;}
                }
                if let Some(factor)=factor {product=Some(match product {None=>factor,Some(left)=>self.product(builder,name,left,factor)?});}
            }
            if let Some(product)=product {
                terms.push((product,coefficient));
            } else {constant=constant.checked_add(i128::from(coefficient)).ok_or_else(overflow)?;}
        }
        let value=self.linear(builder,name,&terms,constant)?;
        self.values.insert(expression.clone(),value);
        Ok(value)
    }
    fn linear(&self,builder:&mut ModelBuilder,name:&str,terms:&[(Integer,i64)],mut constant:i128)->Result<Integer,Error> {
        let mut variables=BTreeMap::<VarId,(Value,i128)>::new();
        for &(value,coefficient) in terms {
            constant=constant.checked_add(value.origin.checked_mul(i128::from(coefficient)).ok_or_else(overflow)?).ok_or_else(overflow)?;
            let (lo,hi)=value.coordinate.bounds();
            if lo==hi {
                constant=constant.checked_add(i128::from(lo).checked_mul(i128::from(coefficient)).ok_or_else(overflow)?).ok_or_else(overflow)?;
                continue;
            }
            let entry=variables.entry(value.coordinate.id()).or_insert((value.coordinate,0));
            entry.1=entry.1.checked_add(i128::from(coefficient)).ok_or_else(overflow)?;
        }
        variables.retain(|_,(_,coefficient)|*coefficient!=0);
        if variables.is_empty() {
            let domain=Domain::singleton(0);
            let id=builder.variable(format!("{name}.integer"),domain.clone());
            return Ok(Integer {coordinate:Value::binding(id,&domain)?,origin:constant});
        }
        if variables.len()==1 {
            let &(coordinate,coefficient)=variables.values().next().unwrap();
            if coefficient==1 {
                // An existing coordinate plus an origin is already this exact
                // integer. A second coordinate and equality add no information.
                let (lo,hi)=coordinate.bounds();
                constant.checked_add(i128::from(lo)).ok_or_else(overflow)?;
                constant.checked_add(i128::from(hi)).ok_or_else(overflow)?;
                return Ok(Integer {coordinate,origin:constant});
            }
        }
        let (mut lo,mut hi)=(constant,constant);
        let mut equation=Vec::new();
        for (_, (value,coefficient)) in variables {
            if coefficient==0 {continue;}
            let coefficient=i64::try_from(coefficient).map_err(|_|overflow())?;
            let (a,b)=value.bounds();let (a,b)=if coefficient>=0 {(a,b)} else {(b,a)};
            lo=lo.checked_add(i128::from(a).checked_mul(i128::from(coefficient)).ok_or_else(overflow)?).ok_or_else(overflow)?;
            hi=hi.checked_add(i128::from(b).checked_mul(i128::from(coefficient)).ok_or_else(overflow)?).ok_or_else(overflow)?;
            equation.push(LinearTerm::new(value.id(),coefficient));
        }
        // The origin may be signed; the solver coordinate remains nonnegative.
        let width=hi.checked_sub(lo).and_then(|n|i64::try_from(n).ok()).ok_or_else(overflow)?;
        let coordinate=Symbolic::new(builder,name).variable("integer",Domain::interval(0,width).map_err(invalid)?)?;
        for sign in [1i64,-1] {
            let mut terms=vec![LinearTerm::new(coordinate.id(),sign)];
            for term in &equation {terms.push(LinearTerm::new(term.variable,term.coefficient.checked_mul(-sign).ok_or_else(overflow)?));}
            builder.constraint(Constraint::LinearLe {terms,rhs:constant.checked_sub(lo).and_then(|n|n.checked_mul(i128::from(sign))).ok_or_else(overflow)?});
        }
        Ok(Integer {coordinate,origin:lo})
    }
    fn as_nonnegative(&self,builder:&mut ModelBuilder,name:&str,value:Integer)->Result<Value,Error> {
        if value.origin==0 {return Ok(value.coordinate);}
        let (lo,hi)=value.bounds();
        if lo==hi && lo>=0 {
            let domain=Domain::singleton(i64::try_from(lo).map_err(|_|overflow())?);
            let id=builder.variable(format!("{name}.nonnegative"),domain.clone());
            return Value::binding(id,&domain);
        }
        let lo=i64::try_from(lo.max(0)).map_err(|_|overflow())?;
        let hi=i64::try_from(hi.max(0)).map_err(|_|overflow())?;
        let result=Symbolic::new(builder,name).variable("nonnegative",Domain::interval(lo,hi).map_err(invalid)?)?;
        for sign in [1i64,-1] {
            builder.constraint(Constraint::LinearLe {terms:vec![LinearTerm::new(result.id(),sign),LinearTerm::new(value.coordinate.id(),-sign)],rhs:value.origin.checked_mul(i128::from(sign)).ok_or_else(overflow)?});
        }
        Ok(result)
    }
    fn product(&self,builder:&mut ModelBuilder,name:&str,a:Integer,b:Integer)->Result<Integer,Error> {
        for (constant,value) in [(a,b),(b,a)] {
            let (lo,hi)=constant.bounds();
            if lo==hi {
                if let Ok(coefficient)=i64::try_from(lo) {
                    return self.linear(builder,name,&[(value,coefficient)],0);
                }
            }
        }
        let product=Symbolic::new(builder,name).product(a.coordinate,b.coordinate)?;
        let coefficient_a=i64::try_from(b.origin).map_err(|_|overflow())?;
        let coefficient_b=i64::try_from(a.origin).map_err(|_|overflow())?;
        self.linear(builder,name,&[
            (Integer {coordinate:product,origin:0},1),
            (Integer {coordinate:a.coordinate,origin:0},coefficient_a),
            (Integer {coordinate:b.coordinate,origin:0},coefficient_b),
        ],a.origin.checked_mul(b.origin).ok_or_else(overflow)?)
    }
    fn atom(&mut self,builder:&mut ModelBuilder,name:&str,atom:&Atom)->Result<Integer,Error> {
        match atom {
            Atom::Param(parameter)=>self.parameters.get(parameter).copied().map(|coordinate|Integer {coordinate,origin:0}).ok_or_else(||Error::Invalid(format!("unbound execution-family parameter {parameter}"))),
            Atom::Quot(numerator,denominator)|Atom::Rem(numerator,denominator)=>{
                let expression=Sym::atom(atom.clone());
                if let Some(&value)=self.values.get(&expression) {return Ok(value);}
                let numerator=self.integer(builder,name,numerator)?;
                let denominator=self.integer(builder,name,denominator)?;
                let (minimum,maximum)=denominator.bounds();
                if minimum<=0 {return Err(Error::Unsupported("symbolic Euclidean division needs a positive divisor over its complete activation domain".into()));}
                let denominator=self.as_nonnegative(builder,name,denominator)?;
                // n + shift*d is nonnegative for every original assignment.
                // Dividing it gives the same remainder and q+shift exactly.
                let low=numerator.bounds().0;
                let shift=if low<0 {low.checked_neg().and_then(|n|n.checked_add(minimum-1)).map(|n|n/minimum).ok_or_else(overflow)?} else {0};
                let shifted=if shift==0 {numerator} else {
                    self.linear(builder,name,&[(numerator,1),(Integer {coordinate:denominator,origin:0},i64::try_from(shift).map_err(|_|overflow())?)],0)?
                };
                let shifted=self.as_nonnegative(builder,name,shifted)?;
                let quotient=Symbolic::new(builder,name).variable("quotient",Domain::interval(0,i64::try_from(i128::from(shifted.bounds().1)/minimum).map_err(|_|overflow())?).map_err(invalid)?)?;
                let remainder=Symbolic::new(builder,name).variable("remainder",Domain::interval(0,i64::try_from(i128::from(shifted.bounds().1).min(maximum-1)).map_err(|_|overflow())?).map_err(invalid)?)?;
                builder.constraint(Constraint::Arithmetic(Arithmetic::DivRem {numerator:shifted.id(),denominator:denominator.id(),quotient:quotient.id(),remainder:remainder.id()}));
                let quotient=Integer {coordinate:quotient,origin:-shift};
                let remainder=Integer {coordinate:remainder,origin:0};
                if let Atom::Quot(n,d)|Atom::Rem(n,d)=atom {
                    self.values.insert(Sym::atom(Atom::Quot(n.clone(),d.clone())),quotient);
                    self.values.insert(Sym::atom(Atom::Rem(n.clone(),d.clone())),remainder);
                }
                Ok(if matches!(atom,Atom::Quot(..)) {quotient} else {remainder})
            },
        }
    }
}
fn overflow()->Error {Error::Unsupported("source-family expression exceeds shared integer range".into())}
fn invalid(error:magnitude_solver::Error)->Error {Error::Invalid(error.to_string())}
