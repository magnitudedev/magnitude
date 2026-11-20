//! Graph dimensions (`Dim`), scalar arguments (`Argument`), and constant
//! tables (`Table`) passed to graph builders.

use std::fmt;
use std::sync::Arc;

/// A graph dimension (`u64`) passed to a graph builder.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Dim(u64);

impl From<u64> for Dim {
    #[inline]
    fn from(value: u64) -> Self {
        Self(value)
    }
}

impl From<&Dim> for Dim {
    #[inline]
    fn from(dim: &Dim) -> Self {
        *dim
    }
}

impl fmt::Debug for Dim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Dim {
    #[inline]
    pub(crate) fn recorded(&self) -> u64 {
        self.0
    }

    /// The dimension's value.
    #[inline]
    pub fn fixed(&self) -> u64 {
        self.0
    }

    /// The dimension's value.
    #[inline]
    pub fn pin(&self) -> u64 {
        self.0
    }

    /// It as a real argument ([`Argument`]).
    #[inline]
    pub fn real(&self) -> Argument<f64> {
        Argument(self.0 as f64)
    }

    /// It as an integer argument of word `T`, `None` where it exceeds `T`.
    #[inline]
    pub fn word<T: ArgumentWord>(&self) -> Option<Argument<T>> {
        (self.0 <= <T as sealed::Word>::MAX).then(|| Argument(<T as sealed::Word>::of(self.0)))
    }

    #[inline]
    pub fn checked_add<R>(&self, right: R) -> Option<Self>
    where
        Dim: From<R>,
    {
        self.0.checked_add(Dim::from(right).0).map(Self)
    }

    #[inline]
    pub fn checked_sub<R>(&self, right: R) -> Option<Self>
    where
        Dim: From<R>,
    {
        self.0.checked_sub(Dim::from(right).0).map(Self)
    }

    #[inline]
    pub fn checked_mul<R>(&self, right: R) -> Option<Self>
    where
        Dim: From<R>,
    {
        self.0.checked_mul(Dim::from(right).0).map(Self)
    }

    #[inline]
    pub fn checked_div<R>(&self, right: R) -> Option<Self>
    where
        Dim: From<R>,
    {
        self.0.checked_div(Dim::from(right).0).map(Self)
    }

    /// `⌈self / right⌉`; panics where `right` is zero.
    #[inline]
    pub fn ceil_div<R>(&self, right: R) -> Self
    where
        Dim: From<R>,
    {
        let r = Dim::from(right).0;
        assert!(r != 0, "dimension arithmetic overflows: {} ceil_div 0", self.0);
        Self(self.0.div_ceil(r))
    }

    /// `self % right`; panics where `right` is zero.
    #[inline]
    pub fn rem<R>(&self, right: R) -> Self
    where
        Dim: From<R>,
    {
        let r = Dim::from(right).0;
        assert!(r != 0, "dimension arithmetic overflows: {} % 0", self.0);
        Self(self.0 % r)
    }

    #[inline]
    pub fn min<R>(&self, right: R) -> Self
    where
        Dim: From<R>,
    {
        Self(self.0.min(Dim::from(right).0))
    }

    #[inline]
    pub fn max<R>(&self, right: R) -> Self
    where
        Dim: From<R>,
    {
        Self(self.0.max(Dim::from(right).0))
    }

    #[inline]
    pub fn less_than<R>(&self, right: R) -> bool
    where
        Dim: From<R>,
    {
        self.0 < Dim::from(right).0
    }

    #[inline]
    pub fn at_most<R>(&self, right: R) -> bool
    where
        Dim: From<R>,
    {
        self.0 <= Dim::from(right).0
    }

    #[inline]
    pub fn greater_than<R>(&self, right: R) -> bool
    where
        Dim: From<R>,
    {
        self.0 > Dim::from(right).0
    }

    #[inline]
    pub fn at_least<R>(&self, right: R) -> bool
    where
        Dim: From<R>,
    {
        self.0 >= Dim::from(right).0
    }

    #[inline]
    pub fn equals<R>(&self, right: R) -> bool
    where
        Dim: From<R>,
    {
        self.0 == Dim::from(right).0
    }

    #[inline]
    pub fn is_zero(&self) -> bool {
        self.0 == 0
    }
}

macro_rules! dim_operator {
    ($trait:ident, $method:ident, $checked:ident, $sym:literal) => {
        impl<R> std::ops::$trait<R> for Dim
        where
            Dim: From<R>,
        {
            type Output = Dim;
            #[inline]
            fn $method(self, right: R) -> Dim {
                let r = Dim::from(right).0;
                Dim(self.0.$checked(r).unwrap_or_else(|| {
                    panic!("dimension arithmetic overflows: {} {} {r}", self.0, $sym)
                }))
            }
        }
        impl<R> std::ops::$trait<R> for &Dim
        where
            Dim: From<R>,
        {
            type Output = Dim;
            #[inline]
            fn $method(self, right: R) -> Dim {
                let r = Dim::from(right).0;
                Dim(self.0.$checked(r).unwrap_or_else(|| {
                    panic!("dimension arithmetic overflows: {} {} {r}", self.0, $sym)
                }))
            }
        }
    };
}

dim_operator!(Add, add, checked_add, "+");
dim_operator!(Sub, sub, checked_sub, "-");
dim_operator!(Mul, mul, checked_mul, "*");
dim_operator!(Div, div, checked_div, "/");

/// A model's real scalar sources and constant tables.
#[derive(Clone, Debug, Default)]
pub struct Reals {
    numbers: Vec<f64>,
    tables: Vec<Arc<[u8]>>,
}

impl Reals {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn number(&mut self, value: f64) -> usize {
        self.numbers.push(value);
        self.numbers.len() - 1
    }

    pub fn table(&mut self, bytes: Arc<[u8]>) -> usize {
        self.tables.push(bytes);
        self.tables.len() - 1
    }

    pub fn argument(&self, index: usize) -> Argument<f64> {
        Argument(*self.numbers.get(index).unwrap_or_else(|| {
            panic!("real {index} of a key outside its {} reals", self.numbers.len())
        }))
    }

    pub fn table_fact(&self, index: usize) -> Table {
        let bytes = self.tables.get(index).unwrap_or_else(|| {
            panic!("table {index} of a key outside its {} tables", self.tables.len())
        });
        Table(Arc::clone(bytes))
    }
}

/// A graph constant's table bytes.
#[derive(Clone, PartialEq, Eq)]
pub struct Table(Arc<[u8]>);

impl From<Arc<[u8]>> for Table {
    fn from(bytes: Arc<[u8]>) -> Self {
        Self(bytes)
    }
}

impl fmt::Debug for Table {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Table({} bytes)", self.0.len())
    }
}

impl Table {
    pub fn bytes(&self) -> &Arc<[u8]> {
        &self.0
    }
}

/// A kernel's scalar argument as a builder passes it to the graph.
#[derive(Clone, Copy)]
pub struct Argument<T>(T);

impl<T> From<T> for Argument<T> {
    #[inline]
    fn from(value: T) -> Self {
        Self(value)
    }
}

impl<T> fmt::Debug for Argument<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Argument")
    }
}

impl<T> Argument<T> {
    #[inline]
    pub(crate) fn value(self) -> T {
        self.0
    }
}

impl Argument<f64> {
    #[inline]
    pub fn f32(self) -> Argument<f32> {
        Argument(self.0 as f32)
    }
}

macro_rules! argument_operator {
    ($trait:ident, $method:ident) => {
        impl std::ops::$trait for Argument<f64> {
            type Output = Self;
            #[inline]
            fn $method(self, right: Self) -> Self {
                Self(std::ops::$trait::$method(self.0, right.0))
            }
        }
    };
}

argument_operator!(Add, add);
argument_operator!(Sub, sub);
argument_operator!(Mul, mul);
argument_operator!(Div, div);

/// An integer word a kernel's argument takes ([`Dim::word`]).
pub trait ArgumentWord: Copy + sealed::Word {}

mod sealed {
    pub trait Word {
        const MAX: u64;
        fn of(value: u64) -> Self;
    }
}

macro_rules! argument_word {
    ($word:ty) => {
        impl sealed::Word for $word {
            const MAX: u64 = <$word>::MAX as u64;
            #[inline]
            fn of(value: u64) -> Self {
                <$word>::try_from(value).expect("a word argument is guarded within its word")
            }
        }
        impl ArgumentWord for $word {}
    };
}

argument_word!(i32);
argument_word!(u32);
argument_word!(u64);
