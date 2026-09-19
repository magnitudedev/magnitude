//! Nonnegative accounting equations with concrete and symbolic interpretations.
//! Backend service definitions use the same equations before and after selection.
//! A range that cannot be represented retains an analysis gap, never a clipped domain.
use magnitude_solver::model::{Arithmetic, Constraint, Domain, LinearTerm, ModelBuilder, VarId};
pub use seismic_realization::dispatch::geometry::Algebra;

pub trait ServiceAlgebra: Algebra {
    fn sum(&mut self, a: Self::Value, b: Self::Value) -> Result<Self::Value, Self::Error>;
}

pub struct Concrete<E>(std::marker::PhantomData<E>);
impl<E> Default for Concrete<E> {
    fn default() -> Self {
        Self(std::marker::PhantomData)
    }
}
impl<E: From<String>> Algebra for Concrete<E> {
    type Value = u64;
    type Error = E;
    fn constant(&mut self, value: u64) -> Result<u64, E> {
        Ok(value)
    }
    fn product(&mut self, a: u64, b: u64) -> Result<u64, E> {
        a.checked_mul(b)
            .ok_or_else(|| E::from("accounting product overflow".into()))
    }
    fn ceil_div(&mut self, a: u64, b: u64) -> Result<u64, E> {
        if b == 0 {
            return Err(E::from("accounting divisor must be positive".into()));
        }
        Ok(a.div_ceil(b))
    }
    fn maximum(&mut self, a: u64, b: u64) -> Result<u64, E> {
        Ok(a.max(b))
    }
}
impl<E: From<String>> ServiceAlgebra for Concrete<E> {
    fn sum(&mut self, a: u64, b: u64) -> Result<u64, E> {
        a.checked_add(b)
            .ok_or_else(|| E::from("accounting sum overflow".into()))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    Invalid(String),
    /// The original family is not narrowed to fit the solver's integer width.
    Unsupported(String),
    Reconstruction(String),
}
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for Error {}

#[derive(Clone, Copy, Debug)]
pub struct Value {
    id: VarId,
    min: u64,
    max: u64,
}
impl Value {
    /// Reuse an existing family parameter without introducing a second choice.
    /// `domain` is that variable's complete original domain in the same builder;
    /// callers must not substitute a sampled or currently narrowed subdomain.
    pub fn binding(id: VarId, domain: &Domain) -> Result<Self, Error> {
        let min = domain
            .min()
            .ok_or_else(|| Error::Invalid("empty accounting binding domain".into()))?;
        let max = domain
            .max()
            .ok_or_else(|| Error::Invalid("empty accounting binding domain".into()))?;
        if min < 0 {
            return Err(Error::Invalid("negative accounting binding domain".into()));
        }
        Ok(Self {
            id,
            min: min as u64,
            max: max as u64,
        })
    }
    pub fn id(self) -> VarId {
        self.id
    }
    pub fn bounds(self) -> (u64, u64) {
        (self.min, self.max)
    }
}
pub struct Symbolic<'a> {
    builder: &'a mut ModelBuilder,
    name: &'a str,
    next: usize,
}
impl<'a> Symbolic<'a> {
    pub fn new(builder: &'a mut ModelBuilder, name: &'a str) -> Self {
        Self {
            builder,
            name,
            next: 0,
        }
    }
    pub fn sum(&mut self, a: Value, b: Value) -> Result<Value, Error> {
        let maximum = a
            .max
            .checked_add(b.max)
            .ok_or_else(|| Error::Unsupported("accounting sum range exceeds u64".into()))?;
        let result = self.derived(a.min + b.min, maximum)?;
        for sign in [1, -1] {
            self.builder.constraint(Constraint::LinearLe {
                terms: vec![
                    LinearTerm::new(result.id, sign),
                    LinearTerm::new(a.id, -sign),
                    LinearTerm::new(b.id, -sign),
                ],
                rhs: 0,
            });
        }
        Ok(result)
    }
    pub fn variable(&mut self, label: &str, domain: Domain) -> Result<Value, Error> {
        let min = domain
            .min()
            .ok_or_else(|| Error::Invalid("empty accounting domain".into()))?;
        let max = domain.max().unwrap();
        if min < 0 {
            return Err(Error::Invalid("negative accounting domain".into()));
        }
        let id = self
            .builder
            .local_variable(format!("{}.{}.{}", self.name, label, self.next), domain)
            .map_err(|e| Error::Invalid(e.to_string()))?;
        self.next += 1;
        Ok(Value {
            id,
            min: min as u64,
            max: max as u64,
        })
    }
    fn derived(&mut self, min: u64, max: u64) -> Result<Value, Error> {
        let maximum = i64::try_from(max).map_err(|_| {
            Error::Unsupported(
                "accounting range exceeds solver i64; complete family retained as unsupported"
                    .into(),
            )
        })?;
        self.variable(
            "derived",
            Domain::interval(min as i64, maximum).map_err(|e| Error::Invalid(e.to_string()))?,
        )
    }
    pub fn positive(&mut self, label: &str, domain: Domain) -> Result<Value, Error> {
        if domain.min().is_none_or(|min| min < 1) {
            return Err(Error::Invalid(format!("{label} must be positive")));
        }
        self.variable(label, domain)
    }
}
impl Algebra for Symbolic<'_> {
    type Value = Value;
    type Error = Error;
    fn constant(&mut self, value: u64) -> Result<Value, Error> {
        self.derived(value, value)
    }
    fn product(&mut self, a: Value, b: Value) -> Result<Value, Error> {
        let maximum = a
            .max
            .checked_mul(b.max)
            .ok_or_else(|| Error::Unsupported("accounting product range exceeds u64".into()))?;
        let result = self.derived(a.min * b.min, maximum)?;
        self.builder
            .constraint(Constraint::Arithmetic(Arithmetic::Product {
                left: a.id,
                right: b.id,
                product: result.id,
            }));
        Ok(result)
    }
    fn ceil_div(&mut self, a: Value, b: Value) -> Result<Value, Error> {
        if b.min == 0 {
            return Err(Error::Invalid("accounting divisor is not positive".into()));
        }
        let result = self.derived(a.min.div_ceil(b.max), a.max.div_ceil(b.min))?;
        self.builder
            .constraint(Constraint::Arithmetic(Arithmetic::CeilDiv {
                numerator: a.id,
                denominator: b.id,
                quotient: result.id,
            }));
        Ok(result)
    }
    fn maximum(&mut self, a: Value, b: Value) -> Result<Value, Error> {
        let result = self.derived(a.min.max(b.min), a.max.max(b.max))?;
        self.builder
            .constraint(Constraint::Arithmetic(Arithmetic::Maximum {
                left: a.id,
                right: b.id,
                result: result.id,
            }));
        Ok(result)
    }
}

impl ServiceAlgebra for Symbolic<'_> {
    fn sum(&mut self, a: Value, b: Value) -> Result<Value, Error> {
        Symbolic::sum(self, a, b)
    }
}

/// Resource use from one backend service definition. Values may remain symbolic.
/// Zero demand/duration is retained symbolically and consumes no capacity.
#[derive(Clone, Debug)]
pub struct ResourceUse<V> {
    pub resource: usize,
    pub offset: u64,
    pub duration: V,
    pub units: V,
}
