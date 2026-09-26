//! Model-local reusable proofs. Variable names/identities are not semantics;
//! guards, factor definitions and every scoped domain are part of each key.
//! Only completed unconditional residual proofs enter this bounded cache.
use super::engine::{Summary, Witness};
use crate::{
    conflicts::ConflictStore,
    model::{Domain, Factor, Model, VarId},
    Result,
};
use std::{
    collections::{HashMap, VecDeque},
    hash::{Hash, Hasher},
    sync::{Arc, Mutex},
};

pub(super) type SharedKnowledge = Arc<Mutex<Knowledge>>;
#[derive(PartialEq, Eq)]
struct Definition {
    factors: Vec<Factor>,
    fingerprint: u64,
}
impl Hash for Definition {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.fingerprint.hash(state);
    }
}
#[derive(Clone, PartialEq, Eq, Hash)]
struct Shape {
    definition: Arc<Definition>,
    domains: Vec<Domain>,
}
impl Shape {
    fn bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self
                .definition
                .factors
                .iter()
                .map(Factor::retained_bytes)
                .sum::<usize>()
            + self
                .domains
                .iter()
                .map(Domain::retained_bytes)
                .sum::<usize>()
    }
}
pub(super) struct Knowledge {
    model: Arc<Model>,
    pub factors: Arc<Vec<Factor>>,
    map: Vec<VarId>,
    definitions: HashMap<Vec<usize>, Arc<Definition>>,
    definition_bytes: usize,
    proofs: HashMap<Arc<Shape>, Summary>,
    fifo: VecDeque<Arc<Shape>>,
    bytes: usize,
    conflicts: ConflictStore,
}
const CACHE_BYTES: usize = 8 * 1024 * 1024;
const CACHE_ENTRIES: usize = 512;
impl Knowledge {
    pub fn shared(model: Arc<Model>) -> SharedKnowledge {
        Arc::new(Mutex::new(Self {
            factors: Arc::new(model.factors().to_vec()),
            map: vec![VarId(0); model.variables().len()],
            definitions: HashMap::new(),
            definition_bytes: 0,
            proofs: HashMap::new(),
            fifo: VecDeque::new(),
            bytes: 0,
            conflicts: ConflictStore::new(model.clone(), 128),
            model,
        }))
    }
    pub fn belongs_to(&self, model: &Arc<Model>) -> bool {
        Arc::ptr_eq(&self.model, model)
    }
    fn shape(
        &mut self,
        factors: &[Factor],
        region: &[usize],
        context: &[(VarId, Domain)],
    ) -> Shape {
        // Original factor indices are stable across repairs. Expanded fragment
        // indices are arena-local and deliberately never use this lookup.
        let original = region.iter().all(|&id| id < self.factors.len());
        let definition = original
            .then(|| self.definitions.get(region).cloned())
            .flatten();
        let definition = definition.unwrap_or_else(|| {
            for (local, (original, _)) in context.iter().enumerate() {
                self.map[original.0] = VarId(local);
            }
            let normalized: Vec<_> = region
                .iter()
                .map(|&id| factors[id].remap(&self.map))
                .collect();
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            normalized.hash(&mut hasher);
            let definition = Arc::new(Definition {
                factors: normalized,
                fingerprint: hasher.finish(),
            });
            let bytes = definition
                .factors
                .iter()
                .map(Factor::retained_bytes)
                .sum::<usize>()
                + std::mem::size_of::<Definition>()
                + std::mem::size_of_val(region);
            if original && bytes <= CACHE_BYTES / 4 {
                if self.definitions.len() >= CACHE_ENTRIES
                    || self.definition_bytes + bytes > CACHE_BYTES / 4
                {
                    self.definitions.clear();
                    self.definition_bytes = 0;
                }
                self.definition_bytes += bytes;
                self.definitions.insert(region.to_vec(), definition.clone());
            }
            definition
        });
        Shape {
            definition,
            domains: context.iter().map(|(_, domain)| domain.clone()).collect(),
        }
    }
    pub fn lookup(
        &mut self,
        factors: &[Factor],
        region: &[usize],
        context: &[(VarId, Domain)],
    ) -> Result<(Option<Summary>, bool)> {
        if !self.proofs.is_empty() {
            let key = self.shape(factors, region, context);
            if let Some(proof) = self.proofs.get(&key) {
                let mut proof = proof.clone();
                if let Some(w) = &mut proof.witness {
                    for (local, _) in &mut w.values {
                        *local = context[*local].0 .0;
                    }
                }
                return Ok((Some(proof), false));
            }
        }
        if self.conflicts.excludes(factors, region, context)? {
            return Ok((
                Some(Summary {
                    solved: true,
                    infeasible: true,
                    ..Default::default()
                }),
                true,
            ));
        }
        Ok((None, false))
    }
    pub fn record(
        &mut self,
        factors: &[Factor],
        region: &[usize],
        context: &[(VarId, Domain)],
        proof: &Summary,
    ) -> Result<(bool, bool)> {
        if !proof.solved {
            return Ok((false, false));
        }
        let conflict = if proof.infeasible {
            self.conflicts.record_infeasible(factors, region, context)?
        } else {
            false
        };
        if self.conflicts.retained_bytes() > CACHE_BYTES / 4 {
            self.conflicts.set_capacity(0);
            self.conflicts.set_capacity(128);
        }
        let key = Arc::new(self.shape(factors, region, context));
        if self.proofs.contains_key(&key) {
            return Ok((false, conflict));
        }
        let mut normalized = proof.clone();
        if let Some(w) = &mut normalized.witness {
            for (v, _) in &mut w.values {
                *v = context
                    .binary_search_by_key(&VarId(*v), |(variable, _)| *variable)
                    .expect("residual witness escaped its complete scope");
            }
        }
        let bytes = key.bytes() + witness_bytes(&normalized);
        if bytes > CACHE_BYTES {
            return Ok((false, conflict));
        }
        while self.bytes + bytes > CACHE_BYTES || self.proofs.len() >= CACHE_ENTRIES {
            let Some(old) = self.fifo.pop_front() else {
                break;
            };
            if let Some(value) = self.proofs.remove(&old) {
                self.bytes -= old.bytes() + witness_bytes(&value);
            }
        }
        self.bytes += bytes;
        self.fifo.push_back(key.clone());
        self.proofs.insert(key, normalized);
        Ok((true, conflict))
    }
    pub fn clear(&mut self) {
        self.definitions = HashMap::new();
        self.definition_bytes = 0;
        self.proofs = HashMap::new();
        self.fifo = VecDeque::new();
        self.bytes = 0;
        self.conflicts.set_capacity(0);
        self.conflicts.set_capacity(128);
    }
    fn retained_bytes(&self) -> usize {
        self.bytes
            + self.definition_bytes
            + self.definitions.capacity() * std::mem::size_of::<(Vec<usize>, Arc<Definition>)>()
            + self.conflicts.retained_bytes()
            + self.map.capacity() * std::mem::size_of::<VarId>()
            + self.proofs.capacity() * std::mem::size_of::<(Arc<Shape>, Summary)>()
            + self.fifo.capacity() * std::mem::size_of::<Arc<Shape>>()
            + self
                .factors
                .iter()
                .map(Factor::retained_bytes)
                .sum::<usize>()
    }
}
fn witness_bytes(proof: &Summary) -> usize {
    std::mem::size_of::<Summary>()
        + proof.witness.as_ref().map_or(0, |Witness { values, .. }| {
            values.capacity() * std::mem::size_of::<(usize, i64)>()
        })
}
pub(super) fn retained_bytes(shared: &SharedKnowledge) -> usize {
    shared
        .lock()
        .expect("solver knowledge lock poisoned")
        .retained_bytes()
}
