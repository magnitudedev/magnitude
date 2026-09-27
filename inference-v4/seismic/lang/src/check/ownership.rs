//! Ownership of the existing checked value product. Expression projections
//! refer back to local places; only local products hold mutable move state.
use super::{ir, Checker, LocalKind, ValueClass};
use crate::intrinsics::{PrimitiveFailure, PrimitiveId};
use crate::types::ValueType;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub(crate) struct LocalPlace {
    pub local: ir::LocalId,
    pub path: Vec<usize>,
}
impl LocalPlace {
    pub fn root(local: ir::LocalId) -> Self {
        Self {
            local,
            path: Vec::new(),
        }
    }
    pub fn overlaps(&self, other: &Self) -> bool {
        self.local == other.local
            && (self.path.starts_with(&other.path) || other.path.starts_with(&self.path))
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum TensorOwnership {
    Owned { moved: bool },
    Computed,
    Borrowed { owner: LocalPlace, exclusive: bool },
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum ValueOwnership {
    Scalar,
    Tensor(TensorOwnership),
    Tuple(Vec<ValueOwnership>),
}
impl ValueOwnership {
    pub fn shape(ty: &ValueType, tensor: &impl Fn() -> TensorOwnership) -> Self {
        match ty {
            ValueType::Tensor(_) => Self::Tensor(tensor()),
            ValueType::Tuple(parts) => {
                Self::Tuple(parts.iter().map(|p| Self::shape(p, tensor)).collect())
            }
            _ => Self::Scalar,
        }
    }
    pub fn at(&self, path: &[usize]) -> &Self {
        match path.split_first() {
            None => self,
            Some((first, rest)) => match self {
                Self::Tuple(parts) => parts[*first].at(rest),
                _ => panic!("ownership projection is not a tuple"),
            },
        }
    }
    pub fn at_mut(&mut self, path: &[usize]) -> &mut Self {
        match path.split_first() {
            None => self,
            Some((first, rest)) => match self {
                Self::Tuple(parts) => parts[*first].at_mut(rest),
                _ => panic!("ownership projection is not a tuple"),
            },
        }
    }
    pub fn visit(
        &self,
        path: &mut Vec<usize>,
        action: &mut impl FnMut(&[usize], &TensorOwnership),
    ) {
        match self {
            Self::Scalar => {}
            Self::Tensor(t) => action(path, t),
            Self::Tuple(parts) => {
                for (i, p) in parts.iter().enumerate() {
                    path.push(i);
                    p.visit(path, action);
                    path.pop();
                }
            }
        }
    }
    pub fn has_borrowed(&self) -> bool {
        let mut found = false;
        self.visit(&mut Vec::new(), &mut |_, t| {
            found |= matches!(t, TensorOwnership::Borrowed { .. })
        });
        found
    }
    pub fn has_moved(&self) -> bool {
        let mut found = false;
        self.visit(&mut Vec::new(), &mut |_, t| {
            found |= matches!(t, TensorOwnership::Owned { moved: true })
        });
        found
    }
}
impl Checker<'_> {
    pub fn value_place(&self, value: &ir::Expr) -> Option<LocalPlace> {
        match &value.kind {
            ir::ExprKind::Local(local) => Some(LocalPlace::root(*local)),
            ir::ExprKind::Primitive {
                id: PrimitiveId::TupleGet(index),
                operands,
                ..
            } => {
                let mut place = self.value_place(&operands[0])?;
                place.path.push(*index as usize);
                Some(place)
            }
            _ => None,
        }
    }
    pub fn ownership(&self, value: &ir::Expr) -> ValueOwnership {
        if let Some(place) = self.value_place(value) {
            return self.locals[place.local.index()]
                .ownership
                .at(&place.path)
                .clone();
        }
        match &value.kind {
            ir::ExprKind::Primitive {
                id: PrimitiveId::TuplePack,
                operands,
                ..
            } => ValueOwnership::Tuple(operands.iter().map(|v| self.ownership(v)).collect()),
            ir::ExprKind::Primitive {
                id: PrimitiveId::TupleGet(index),
                operands,
                ..
            } => self.ownership(&operands[0]).at(&[*index as usize]).clone(),
            ir::ExprKind::Primitive {
                id: PrimitiveId::SliceView { .. } | PrimitiveId::Transpose | PrimitiveId::Reshape,
                operands,
                ..
            } => match self.borrow_owner(&operands[0]) {
                Some(owner) => ValueOwnership::Tensor(TensorOwnership::Borrowed {
                    owner,
                    exclusive: false,
                }),
                None => ValueOwnership::shape(&value.ty, &|| TensorOwnership::Computed),
            },
            ir::ExprKind::PlaneView { base, .. } => match self.borrow_owner(base) {
                Some(owner) => ValueOwnership::Tensor(TensorOwnership::Borrowed {
                    owner,
                    exclusive: false,
                }),
                None => ValueOwnership::shape(&value.ty, &|| TensorOwnership::Computed),
            },
            ir::ExprKind::Call { .. }
            | ir::ExprKind::Primitive {
                id:
                    PrimitiveId::TensorAlloc
                    | PrimitiveId::Fill(_)
                    | PrimitiveId::Copy
                    | PrimitiveId::RepresentationConvert(_),
                ..
            } => ValueOwnership::shape(&value.ty, &|| TensorOwnership::Owned { moved: false }),
            _ => ValueOwnership::shape(&value.ty, &|| TensorOwnership::Computed),
        }
    }
    pub fn borrow_owner(&self, value: &ir::Expr) -> Option<LocalPlace> {
        match self.ownership(value) {
            ValueOwnership::Tensor(TensorOwnership::Borrowed { owner, .. }) => Some(owner),
            ValueOwnership::Tensor(TensorOwnership::Owned { .. }) => self.value_place(value),
            _ => None,
        }
    }
    pub fn ownership_class(&self, value: &ir::Expr) -> ValueClass {
        match self.ownership(value) {
            ValueOwnership::Tensor(TensorOwnership::Owned { .. }) => ValueClass::Owned,
            ValueOwnership::Tensor(TensorOwnership::Borrowed { .. }) => ValueClass::Borrowed,
            ValueOwnership::Tensor(TensorOwnership::Computed) | ValueOwnership::Tuple(_) => {
                ValueClass::Computed
            }
            ValueOwnership::Scalar => ValueClass::Scalar,
        }
    }
    pub fn is_borrowed_local(&self, local: ir::LocalId) -> bool {
        self.locals[local.index()].ownership.has_borrowed()
    }
    pub fn local_storage_root(&self, local: ir::LocalId) -> ir::LocalId {
        match &self.locals[local.index()].ownership {
            ValueOwnership::Tensor(TensorOwnership::Borrowed { owner, .. }) => owner.local,
            _ => local,
        }
    }
    pub fn moved_snapshot(&self) -> BTreeSet<LocalPlace> {
        let mut out = BTreeSet::new();
        for (i, local) in self.locals.iter().enumerate() {
            local.ownership.visit(&mut Vec::new(), &mut |path, t| {
                if matches!(t, TensorOwnership::Owned { moved: true }) {
                    out.insert(LocalPlace {
                        local: ir::LocalId::new(i as u32),
                        path: path.to_vec(),
                    });
                }
            });
        }
        out
    }
    pub fn restore_moves(&mut self, moves: &BTreeSet<LocalPlace>) {
        fn restore(
            product: &mut ValueOwnership,
            place: &mut LocalPlace,
            moves: &BTreeSet<LocalPlace>,
        ) {
            match product {
                ValueOwnership::Tensor(TensorOwnership::Owned { moved }) => {
                    *moved = moves.contains(place)
                }
                ValueOwnership::Tuple(parts) => {
                    for (i, p) in parts.iter_mut().enumerate() {
                        place.path.push(i);
                        restore(p, place, moves);
                        place.path.pop();
                    }
                }
                _ => {}
            }
        }
        for (i, local) in self.locals.iter_mut().enumerate() {
            restore(
                &mut local.ownership,
                &mut LocalPlace::root(ir::LocalId::new(i as u32)),
                moves,
            );
        }
    }
    pub fn consume_place(&mut self, place: &LocalPlace) {
        fn consume(value: &mut ValueOwnership) {
            match value {
                ValueOwnership::Tensor(TensorOwnership::Owned { moved }) => *moved = true,
                ValueOwnership::Tuple(parts) => parts.iter_mut().for_each(consume),
                _ => {}
            }
        }
        consume(
            self.locals[place.local.index()]
                .ownership
                .at_mut(&place.path),
        );
    }
    pub fn live_borrows(&self) -> Vec<(LocalPlace, LocalPlace, bool)> {
        let mut out = Vec::new();
        for local in self.scopes.iter().flat_map(|s| s.values()) {
            self.locals[local.index()]
                .ownership
                .visit(&mut Vec::new(), &mut |path, t| {
                    if let TensorOwnership::Borrowed { owner, exclusive } = t {
                        if owner.local != *local {
                            out.push((
                                LocalPlace {
                                    local: *local,
                                    path: path.to_vec(),
                                },
                                owner.clone(),
                                *exclusive,
                            ));
                        }
                    }
                });
        }
        out
    }
    pub fn exclusive_borrow_blocks(&self, local: ir::LocalId) -> bool {
        let borrows = self.live_borrows();
        let mut blocked = false;
        self.locals[local.index()]
            .ownership
            .visit(&mut Vec::new(), &mut |path, tensor| {
                if !matches!(tensor, TensorOwnership::Borrowed { .. }) {
                    let place = LocalPlace {
                        local,
                        path: path.to_vec(),
                    };
                    blocked |= borrows
                        .iter()
                        .any(|(_, owner, exclusive)| *exclusive && owner.overlaps(&place));
                }
            });
        blocked
    }
    pub fn default_ownership(
        &self,
        id: ir::LocalId,
        ty: &ValueType,
        kind: LocalKind,
    ) -> ValueOwnership {
        fn parameter(
            ty: &ValueType,
            access: &ir::Ownership,
            place: &mut LocalPlace,
        ) -> ValueOwnership {
            match (ty, access) {
                (ValueType::Tuple(types), ir::Ownership::Tuple(accesses)) => ValueOwnership::Tuple(
                    types
                        .iter()
                        .zip(accesses)
                        .enumerate()
                        .map(|(i, (ty, access))| {
                            place.path.push(i);
                            let value = parameter(ty, access, place);
                            place.path.pop();
                            value
                        })
                        .collect(),
                ),
                (ValueType::Tensor(_), ir::Ownership::Owned) => {
                    ValueOwnership::Tensor(TensorOwnership::Owned { moved: false })
                }
                (ValueType::Tensor(_), ir::Ownership::Shared | ir::Ownership::Exclusive) => {
                    ValueOwnership::Tensor(TensorOwnership::Borrowed {
                        owner: place.clone(),
                        exclusive: *access == ir::Ownership::Exclusive,
                    })
                }
                (ValueType::Tensor(_), _) => panic!("tensor parameter lost ownership"),
                _ => ValueOwnership::Scalar,
            }
        }
        if let LocalKind::Param(i) = kind {
            parameter(ty, &self.sig.params[i].ownership, &mut LocalPlace::root(id))
        } else {
            ValueOwnership::shape(ty, &|| {
                if matches!(kind, LocalKind::State) {
                    TensorOwnership::Owned { moved: false }
                } else {
                    TensorOwnership::Computed
                }
            })
        }
    }
}

impl Checker<'_> {
    /// Resolve an owned consumption from the actual value product. A direct
    /// view may transfer an available owned backing; an outstanding lexical
    /// borrow prevents that transfer, including a borrowed view local itself.
    pub fn owned_consumption(
        &self,
        value: &ir::Expr,
    ) -> Result<(ValueOwnership, BTreeSet<LocalPlace>), String> {
        let mut product = self.ownership(value);
        let source = self.value_place(value);
        let mut moves = BTreeSet::new();
        fn consume(
            checker: &Checker<'_>,
            product: &mut ValueOwnership,
            source: Option<&LocalPlace>,
            path: &mut Vec<usize>,
            moves: &mut BTreeSet<LocalPlace>,
        ) -> Result<(), String> {
            match product {
                ValueOwnership::Tuple(parts) => {
                    for (i, part) in parts.iter_mut().enumerate() {
                        path.push(i);
                        consume(checker, part, source, path, moves)?;
                        path.pop();
                    }
                }
                ValueOwnership::Tensor(TensorOwnership::Owned { moved }) => {
                    if *moved {
                        return Err("use of moved owned tensor".into());
                    }
                    if let Some(source) = source {
                        let mut source = source.clone();
                        source.path.extend(path.iter());
                        if checker
                            .live_borrows()
                            .iter()
                            .any(|(_, owner, _)| owner.overlaps(&source))
                        {
                            return Err(
                                "cannot move owned tensor while a tensor borrow is live".into()
                            );
                        }
                        moves.insert(source);
                    }
                }
                ValueOwnership::Tensor(TensorOwnership::Borrowed { owner, .. }) => {
                    let available = matches!(
                        checker.locals[owner.local.index()]
                            .ownership
                            .at(&owner.path),
                        ValueOwnership::Tensor(TensorOwnership::Owned { moved: false })
                    );
                    let borrowed = checker
                        .live_borrows()
                        .iter()
                        .any(|(_, root, _)| root.overlaps(owner));
                    if !available || borrowed {
                        return Err("borrowed tensor leaf cannot cross an owned boundary".into());
                    }
                    moves.insert(owner.clone());
                    *product = ValueOwnership::Tensor(TensorOwnership::Owned { moved: false });
                }
                _ => {}
            }
            Ok(())
        }
        // Tuple construction preserves each operand's own source place.
        if let ir::ExprKind::Primitive {
            id: PrimitiveId::TuplePack,
            operands,
            ..
        } = &value.kind
        {
            let mut parts = Vec::new();
            for operand in operands {
                let (part, taken) = self.owned_consumption(operand)?;
                parts.push(part);
                for place in taken {
                    if !moves.insert(place) {
                        return Err("owned tensor leaf is consumed more than once".into());
                    }
                }
            }
            return Ok((ValueOwnership::Tuple(parts), moves));
        }
        consume(
            self,
            &mut product,
            source.as_ref(),
            &mut Vec::new(),
            &mut moves,
        )?;
        Ok((product, moves))
    }
}

pub(crate) fn project(value: &ir::Expr, index: usize) -> ir::Expr {
    let ValueType::Tuple(parts) = &value.ty else {
        panic!("checked product projection is not a tuple")
    };
    if let ir::ExprKind::Primitive {
        id: PrimitiveId::TuplePack,
        operands,
        ..
    } = &value.kind
    {
        return operands[index].clone();
    }
    ir::Expr::new(
        ir::ExprKind::Primitive {
            id: PrimitiveId::TupleGet(index as u32),
            operands: vec![value.clone()],
            // A tuple projection has no scalar recipe, so no failure output.
            failure: PrimitiveFailure::ProvedAbsent,
        },
        parts.as_slice()[index].clone(),
        None,
        value.span,
    )
}
pub(crate) fn argument_leaves(
    access: &ir::Ownership,
    value: &ir::Expr,
    output: &mut Vec<(ir::Ownership, ir::Expr)>,
) {
    if let ir::Ownership::Tuple(parts) = access {
        for (index, part) in parts.iter().enumerate() {
            argument_leaves(part, &project(value, index), output);
        }
    } else {
        output.push((access.clone(), value.clone()));
    }
}

impl Checker<'_> {
    pub fn writable_place(&self, place: &LocalPlace) -> bool {
        fn writable(checker: &Checker<'_>, value: &ValueOwnership, place: &LocalPlace) -> bool {
            match value {
                ValueOwnership::Tensor(TensorOwnership::Borrowed { owner, exclusive }) => {
                    if owner == place {
                        *exclusive
                    } else {
                        checker.writable_place(owner)
                    }
                }
                ValueOwnership::Tuple(parts) => parts.iter().enumerate().any(|(i, part)| {
                    let mut child = place.clone();
                    child.path.push(i);
                    writable(checker, part, &child)
                }),
                _ => matches!(checker.kinds[place.local.index()], LocalKind::State),
            }
        }
        writable(
            self,
            self.locals[place.local.index()].ownership.at(&place.path),
            place,
        )
    }
    pub fn assignment_ownership(
        &mut self,
        destination: &ir::Place,
        value: &ir::Expr,
    ) -> Result<(), String> {
        fn collect(
            checker: &Checker<'_>,
            destination: &ir::Place,
            value: &ir::Expr,
            updates: &mut Vec<(LocalPlace, ValueOwnership)>,
            moves: &mut BTreeSet<LocalPlace>,
        ) -> Result<(), String> {
            match destination {
                ir::Place::Tuple(parts) => {
                    for (i, part) in parts.iter().enumerate() {
                        collect(checker, part, &project(value, i), updates, moves)?;
                    }
                }
                ir::Place::Local(place) => {
                    let (mut ownership, taken) = checker.owned_consumption(value)?;
                    if matches!(ownership, ValueOwnership::Tensor(TensorOwnership::Computed)) {
                        ownership = ValueOwnership::Tensor(TensorOwnership::Owned { moved: false });
                    }
                    for source in taken {
                        if !moves.insert(source) {
                            return Err("owned tensor leaf is assigned more than once".into());
                        }
                    }
                    updates.push((place.clone(), ownership));
                }
                ir::Place::Element { .. } => {}
            }
            Ok(())
        }
        let mut updates = Vec::new();
        let mut moves = BTreeSet::new();
        collect(self, destination, value, &mut updates, &mut moves)?;
        for source in moves {
            self.consume_place(&source);
        }
        for (place, value) in updates {
            *self.locals[place.local.index()]
                .ownership
                .at_mut(&place.path) = value;
        }
        Ok(())
    }
    pub fn binding_moves(&self, value: &ir::Expr) -> Result<BTreeSet<LocalPlace>, String> {
        let mut result = BTreeSet::new();
        fn collect(
            checker: &Checker<'_>,
            value: &ir::Expr,
            result: &mut BTreeSet<LocalPlace>,
        ) -> Result<(), String> {
            if let Some(source) = checker.value_place(value) {
                let mut duplicate = false;
                let mut borrowed = false;
                checker
                    .ownership(value)
                    .visit(&mut Vec::new(), &mut |path, t| {
                        if matches!(t, TensorOwnership::Owned { moved: false }) {
                            let mut place = source.clone();
                            place.path.extend(path);
                            borrowed |= checker
                                .live_borrows()
                                .iter()
                                .any(|(_, owner, _)| owner.overlaps(&place));
                            duplicate |= !result.insert(place);
                        }
                    });
                if borrowed {
                    return Err("cannot move owned tensor while a tensor borrow is live".into());
                }
                if duplicate {
                    return Err("owned tensor leaf is moved more than once".into());
                }
            } else if let ir::ExprKind::Primitive {
                id: PrimitiveId::TuplePack,
                operands,
                ..
            } = &value.kind
            {
                for operand in operands {
                    collect(checker, operand, result)?;
                }
            }
            Ok(())
        }
        collect(self, value, &mut result)?;
        Ok(result)
    }
}
