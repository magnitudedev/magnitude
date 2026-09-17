//! Structural evidence attached while generating the concrete SSA program.
//! These are execution domains and memory identities, not performance formulas.
use cranelift_codegen::ir::{Block, Value};
use std::{collections::HashMap, sync::Arc};

#[derive(Clone, Debug)]
pub enum Multiplicity {
    Constant(u64),
    Product(Arc<Self>, Arc<Self>),
    PlusOne(Arc<Self>),
    /// Half-open unit-stride range. Values refer to the generated SSA itself.
    Iterations {
        lower: Value,
        upper: Value,
    },
    Predicate {
        value: Value,
        expected: bool,
    },
}
impl Multiplicity {
    pub fn product(a: Arc<Self>, b: Arc<Self>) -> Arc<Self> {
        Arc::new(Self::Product(a, b))
    }
}
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum MemoryObject {
    Buffer { parameter: String, plane: String },
    BufferTable,
    ScalarArguments,
    PrivateScratch,
}
#[derive(Clone, Debug, Default)]
pub struct ExecutionEvidence {
    /// Executions per invocation, conditional on all runtime validity guards passing.
    pub blocks: HashMap<Block, Arc<Multiplicity>>,
    pub validity_guards: Vec<Value>,
    /// Root addresses, so consumers need not guess identities from instruction names.
    pub memory_roots: HashMap<Value, MemoryObject>,
}
