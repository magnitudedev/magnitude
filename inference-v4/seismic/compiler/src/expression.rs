//! Authority wrappers over the entry's one expression arena.
//!
//! These wrappers do not create another expression language. They state
//! which consumer is allowed to receive an existing arena node.

use seismic_lang::expr::{AnyExpr, BoolExpr, ExprArena, SymbolKind};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct PlanningExpr<T>(T);

impl PlanningExpr<BoolExpr> {
    pub(crate) fn new(arena: &ExprArena, node: BoolExpr) -> Option<Self> {
        arena
            .free_symbols(AnyExpr::Bool(node))
            .iter()
            .all(|symbol| {
                matches!(
                    arena.symbol_kind(*symbol),
                    SymbolKind::Decision(_) | SymbolKind::TargetConstant(_)
                )
            })
            .then_some(Self(node))
    }

    pub(crate) fn node(self) -> BoolExpr {
        self.0
    }
}
