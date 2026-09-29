//! Compiled grammars by source. Template grammars repeat across the turns of
//! a conversation, and compilation is pure, so a small most-recently-used
//! cache spares recompiling them.
use crate::{compile, Compiled, GrammarError};
use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};

pub struct CompileCache {
    entries: Mutex<VecDeque<(String, Arc<Compiled>)>>,
    max_entries: usize,
    max_bytes: usize,
}

impl CompileCache {
    pub fn new(max_entries: usize, max_bytes: usize) -> Self {
        Self {
            entries: Mutex::new(VecDeque::new()),
            max_entries,
            max_bytes,
        }
    }

    pub fn compile(&self, gbnf: &str) -> Result<Arc<Compiled>, GrammarError> {
        {
            let mut entries = self.entries.lock().expect("compile cache lock");
            if let Some(index) = entries.iter().position(|(source, _)| source == gbnf) {
                let entry = entries.remove(index).unwrap();
                let compiled = entry.1.clone();
                entries.push_back(entry);
                return Ok(compiled);
            }
        }
        let compiled = Arc::new(compile(gbnf)?);
        let bytes = |entry: &(String, Arc<Compiled>)| entry.0.len() + entry.1.grammar.lark().len();
        let entry = (gbnf.to_owned(), compiled.clone());
        if self.max_entries > 0 && bytes(&entry) <= self.max_bytes {
            let mut entries = self.entries.lock().expect("compile cache lock");
            entries.push_back(entry);
            while entries.len() > self.max_entries
                || entries.iter().map(bytes).sum::<usize>() > self.max_bytes
            {
                entries.pop_front();
            }
        }
        Ok(compiled)
    }
}
