//! The owner context of one checked-bundle encode or decode.
//!
//! Every semantic handle is (owner, ordinal), and owners are process-local.
//! Encoding writes ordinals only: a module-owned or program-owned handle
//! writes its ordinal, and an arena- or schema-owned handle writes the dense
//! slot of its owner in first-appearance order next to its ordinal. Decoding
//! rebinds every handle to owners allocated fresh for that one decode: one
//! `ModuleId`, one `ProgramId`, and one fresh owner per arena or schema slot.
//! Two decodes of one bundle therefore yield disjoint owners, exactly like
//! two checks of one source.
//!
//! Decoding also records the extent every handle reaches, per owner and
//! handle kind. [`decode`] fails unless every referenced arena is defined
//! exactly once and covers every handle into it; the caller checks the
//! module and program extents against the decoded tables.

use crate::expr::ArenaId;
use crate::ids::{ModuleId, ProgramId, SchemaId};
use std::cell::RefCell;
use std::collections::HashMap;

/// A kind of handle an expression arena owns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ArenaHandle {
    Node,
    Symbol,
    TargetConstant,
    Decision,
    LoopBinder,
    Root,
}

impl ArenaHandle {
    const COUNT: usize = 6;

    const fn slot(self) -> usize {
        match self {
            Self::Node => 0,
            Self::Symbol => 1,
            Self::TargetConstant => 2,
            Self::Decision => 3,
            Self::LoopBinder => 4,
            Self::Root => 5,
        }
    }
}

/// How many handles of each kind one arena defines.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ArenaExtent([u32; ArenaHandle::COUNT]);

impl ArenaExtent {
    pub(crate) fn with(mut self, kind: ArenaHandle, count: usize) -> Result<Self, WireError> {
        self.0[kind.slot()] = u32::try_from(count).map_err(|_| WireError("arena extent exceeds u32"))?;
        Ok(self)
    }

    fn covers(self, reached: Self) -> bool {
        self.0.iter().zip(reached.0).all(|(defined, reached)| reached <= *defined)
    }
}

/// A kind of handle the module's semantic program owns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ProgramHandle {
    Function,
    Family,
}

/// A handle that has no meaning in this bundle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct WireError(pub(crate) &'static str);

impl std::fmt::Display for WireError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}

struct Encoding {
    module: ModuleId,
    program: ProgramId,
    arenas: HashMap<ArenaId, u32>,
    schemas: HashMap<SchemaId, u32>,
}

struct ArenaSlot {
    owner: ArenaId,
    defined: Option<ArenaExtent>,
    reached: ArenaExtent,
}

struct Decoding {
    module: ModuleId,
    program: ProgramId,
    entries: u32,
    functions: u32,
    families: u32,
    arenas: Vec<ArenaSlot>,
    schemas: Vec<SchemaId>,
    scope: Option<Scope>,
}

/// What the handles of one scoped element (a definition or a family)
/// reached: the one arena slot every arena handle in it names, the arena it
/// defined, and the extents of its body-local and dimension ordinals. The
/// caller checks them against the decoded element.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Scope {
    pub(crate) arena: Option<u32>,
    pub(crate) defined: Option<u32>,
    pub(crate) locals: u32,
    pub(crate) dimensions: u32,
}

impl Scope {
    fn name(&mut self, slot: u32) -> Result<(), WireError> {
        match *self.arena.get_or_insert(slot) == slot {
            true => Ok(()),
            false => Err(WireError("one scoped element names two arenas")),
        }
    }
}

thread_local! {
    static ENCODING: RefCell<Option<Encoding>> = const { RefCell::new(None) };
    static DECODING: RefCell<Option<Decoding>> = const { RefCell::new(None) };
}

/// Clears one context when its encode or decode ends, including by unwinding.
struct Clear(fn());
impl Drop for Clear {
    fn drop(&mut self) {
        (self.0)()
    }
}

/// Runs `encode` with the owners of one module installed.
pub(crate) fn encode<T>(module: ModuleId, program: ProgramId, encode: impl FnOnce() -> T) -> T {
    ENCODING.with(|cell| {
        let mut context = cell.borrow_mut();
        assert!(context.is_none(), "checked-bundle encodes do not nest");
        *context = Some(Encoding {
            module,
            program,
            arenas: HashMap::new(),
            schemas: HashMap::new(),
        });
    });
    let _clear = Clear(|| {
        ENCODING.with(|cell| cell.borrow_mut().take());
    });
    encode()
}

/// The owners one decode allocated, and how far module- and program-owned
/// handles reached.
pub(crate) struct Decoded {
    pub(crate) module: ModuleId,
    pub(crate) program: ProgramId,
    pub(crate) entries: u32,
    pub(crate) functions: u32,
    pub(crate) families: u32,
}

/// Runs `decode` with fresh owners installed, then checks that every
/// referenced arena is defined once and covers every handle into it.
pub(crate) fn decode<T>(
    decode: impl FnOnce() -> Result<T, WireError>,
) -> Result<(T, Decoded), WireError> {
    DECODING.with(|cell| {
        let mut context = cell.borrow_mut();
        assert!(context.is_none(), "checked-bundle decodes do not nest");
        *context = Some(Decoding {
            module: ModuleId::fresh(),
            program: ProgramId::fresh(),
            entries: 0,
            functions: 0,
            families: 0,
            arenas: Vec::new(),
            schemas: Vec::new(),
            scope: None,
        });
    });
    let _clear = Clear(|| {
        DECODING.with(|cell| cell.borrow_mut().take());
    });
    let value = decode()?;
    DECODING.with(|cell| {
        let context = cell.borrow();
        let context = context.as_ref().expect("decode context is installed");
        for arena in &context.arenas {
            match arena.defined {
                Some(extent) if extent.covers(arena.reached) => {}
                Some(_) => return Err(WireError("arena handle out of range")),
                None => return Err(WireError("handle of an undefined arena")),
            }
        }
        Ok((
            value,
            Decoded {
                module: context.module,
                program: context.program,
                entries: context.entries,
                functions: context.functions,
                families: context.families,
            },
        ))
    })
}

fn encoding<T>(use_context: impl FnOnce(&mut Encoding) -> Result<T, WireError>) -> Result<T, WireError> {
    ENCODING.with(|cell| match cell.borrow_mut().as_mut() {
        Some(context) => use_context(context),
        None => Err(WireError("semantic handles serialize only inside a checked-bundle encode")),
    })
}

fn decoding<T>(use_context: impl FnOnce(&mut Decoding) -> Result<T, WireError>) -> Result<T, WireError> {
    DECODING.with(|cell| match cell.borrow_mut().as_mut() {
        Some(context) => use_context(context),
        None => Err(WireError("semantic handles deserialize only inside a checked-bundle decode")),
    })
}

fn reach(extent: &mut u32, ordinal: u32) -> Result<(), WireError> {
    let end = ordinal.checked_add(1).ok_or(WireError("handle ordinal overflows"))?;
    *extent = (*extent).max(end);
    Ok(())
}

/// Encoding: `owner` must be the encoded module's.
pub(crate) fn encode_module(owner: ModuleId) -> Result<(), WireError> {
    encoding(|context| {
        (context.module == owner)
            .then_some(())
            .ok_or(WireError("checked module holds an entry handle of another module"))
    })
}

/// Decoding: the module owner of an entry handle with `ordinal`.
pub(crate) fn decode_module(ordinal: u32) -> Result<ModuleId, WireError> {
    decoding(|context| {
        reach(&mut context.entries, ordinal)?;
        Ok(context.module)
    })
}

/// Encoding: `owner` must be the encoded module's program.
pub(crate) fn encode_program(owner: ProgramId) -> Result<(), WireError> {
    encoding(|context| {
        (context.program == owner)
            .then_some(())
            .ok_or(WireError("checked module holds a program handle of another program"))
    })
}

/// Decoding: the program owner of a `kind` handle with `ordinal`.
pub(crate) fn decode_program(kind: ProgramHandle, ordinal: u32) -> Result<ProgramId, WireError> {
    decoding(|context| {
        let extent = match kind {
            ProgramHandle::Function => &mut context.functions,
            ProgramHandle::Family => &mut context.families,
        };
        reach(extent, ordinal)?;
        Ok(context.program)
    })
}

fn slot<Owner: Copy + Eq + std::hash::Hash>(slots: &mut HashMap<Owner, u32>, owner: Owner) -> Result<u32, WireError> {
    let next = u32::try_from(slots.len()).map_err(|_| WireError("owner slots exceed u32"))?;
    Ok(*slots.entry(owner).or_insert(next))
}

/// Encoding: the slot of an arena owner.
pub(crate) fn encode_arena(owner: ArenaId) -> Result<u32, WireError> {
    encoding(|context| slot(&mut context.arenas, owner))
}

/// Decoding: the owner of arena slot `slot`, recording that a `kind` handle
/// with `ordinal` reaches into it. Slots are dense in first-appearance order.
pub(crate) fn decode_arena(slot: u32, kind: ArenaHandle, ordinal: u32) -> Result<ArenaId, WireError> {
    decoding(|context| {
        if let Some(scope) = context.scope.as_mut() {
            scope.name(slot)?;
        }
        let arena = arena_slot(context, slot)?;
        reach(&mut arena.reached.0[kind.slot()], ordinal)?;
        Ok(arena.owner)
    })
}

/// Decoding: defines arena slot `slot` with `extent`. Each slot is defined
/// by exactly one arena.
pub(crate) fn define_arena(slot: u32, extent: ArenaExtent) -> Result<ArenaId, WireError> {
    decoding(|context| {
        if let Some(scope) = context.scope.as_mut() {
            scope.name(slot)?;
            if scope.defined.replace(slot).is_some() {
                return Err(WireError("one scoped element defines two arenas"));
            }
        }
        let arena = arena_slot(context, slot)?;
        if arena.defined.replace(extent).is_some() {
            return Err(WireError("two arenas share one identity"));
        }
        Ok(arena.owner)
    })
}

fn arena_slot(context: &mut Decoding, slot: u32) -> Result<&mut ArenaSlot, WireError> {
    let slot = usize::try_from(slot).map_err(|_| WireError("arena slot out of range"))?;
    if slot == context.arenas.len() {
        context.arenas.push(ArenaSlot {
            owner: ArenaId::fresh(),
            defined: None,
            reached: ArenaExtent::default(),
        });
    }
    context
        .arenas
        .get_mut(slot)
        .ok_or(WireError("arena slot out of first-appearance order"))
}

/// Encoding: the slot of a call-schema owner.
pub(crate) fn encode_schema(owner: SchemaId) -> Result<u32, WireError> {
    encoding(|context| slot(&mut context.schemas, owner))
}

/// Decoding: the owner of schema slot `slot`.
pub(crate) fn decode_schema(slot: u32) -> Result<SchemaId, WireError> {
    decoding(|context| {
        let slot = usize::try_from(slot).map_err(|_| WireError("schema slot out of range"))?;
        if slot == context.schemas.len() {
            context.schemas.push(SchemaId::fresh());
        }
        context
            .schemas
            .get(slot)
            .copied()
            .ok_or(WireError("schema slot out of first-appearance order"))
    })
}

fn scope(context: &mut Decoding) -> Result<&mut Scope, WireError> {
    context
        .scope
        .as_mut()
        .ok_or(WireError("scoped ordinal outside a definition or family"))
}

/// Decoding: a body-local `LocalId` with `index`.
pub(crate) fn decode_local(index: u32) -> Result<(), WireError> {
    decoding(|context| reach(&mut scope(context)?.locals, index))
}

/// Deserializes an ordinal into the dimensions of the enclosing scoped
/// element.
pub(crate) fn deserialize_dimension<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<u32, D::Error> {
    use serde::de::Error as _;
    let ordinal = <u32 as serde::Deserialize>::deserialize(deserializer)?;
    decoding(|context| reach(&mut scope(context)?.dimensions, ordinal)).map_err(D::Error::custom)?;
    Ok(ordinal)
}

/// Deserializes a sequence whose elements each decode in their own [`Scope`].
pub(crate) fn deserialize_scoped<'de, D, T>(deserializer: D) -> Result<Vec<(T, Scope)>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: serde::Deserialize<'de>,
{
    struct Elements<T>(std::marker::PhantomData<T>);
    impl<'de, T: serde::Deserialize<'de>> serde::de::Visitor<'de> for Elements<T> {
        type Value = Vec<(T, Scope)>;
        fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("a sequence of scoped elements")
        }
        fn visit_seq<A: serde::de::SeqAccess<'de>>(self, mut sequence: A) -> Result<Self::Value, A::Error> {
            use serde::de::Error as _;
            let set = |scope: Option<Scope>| {
                decoding(|context| {
                    Ok(std::mem::replace(&mut context.scope, scope))
                })
            };
            let mut elements = Vec::new();
            loop {
                if set(Some(Scope::default())).map_err(A::Error::custom)?.is_some() {
                    return Err(A::Error::custom("scoped elements do not nest"));
                }
                let element = sequence.next_element::<T>();
                let scope = set(None)
                    .map_err(A::Error::custom)?
                    .expect("the element's scope is installed");
                match element? {
                    Some(element) => elements.push((element, scope)),
                    None => return Ok(elements),
                }
            }
        }
    }
    deserializer.deserialize_seq(Elements(std::marker::PhantomData))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{ExprArena, IntExpr};

    fn encoded(handle: IntExpr, arena: &ExprArena) -> Vec<u8> {
        encode(ModuleId::fresh(), ProgramId::fresh(), || postcard::to_stdvec(&(handle, arena)))
            .expect("the pair encodes")
    }

    fn decoded(bytes: &[u8]) -> Result<(IntExpr, ExprArena), WireError> {
        decode(|| {
            postcard::from_bytes::<(IntExpr, ExprArena)>(bytes)
                .map_err(|_| WireError("does not decode"))
        })
        .map(|(pair, _)| pair)
    }

    #[test]
    fn handles_rebind_to_fresh_owners() {
        let mut arena = ExprArena::new();
        let handle = arena.int(7);
        let bytes = encoded(handle, &arena);
        let (first, first_arena) = decoded(&bytes).unwrap();
        let (second, _) = decoded(&bytes).unwrap();
        assert_ne!(first, handle);
        assert_ne!(first, second);
        assert!(matches!(first_arena.view(first.into()), crate::expr::NodeView::IntConst(7)));
        assert_eq!(encoded(first, &first_arena), bytes);
    }

    #[test]
    fn a_handle_outside_its_arena_is_rejected() {
        // A handle to node 9 of a ten-node arena, next to a one-node arena.
        let mut large = ExprArena::new();
        let handle = (0..10).map(|value| large.int(value)).last().unwrap();
        let mut small = ExprArena::new();
        small.int(1);
        let mut bytes = encoded(handle, &small);
        // `[slot 0, index 9][small's own slot 1, …]`: renaming the small
        // arena to slot 0 makes the handle name node 9 of a one-node arena.
        assert_eq!(&bytes[..3], &[0, 9, 1]);
        bytes[2] = 0;
        assert_eq!(decoded(&bytes).err(), Some(WireError("arena handle out of range")));
        // Unrenamed, the handle names an arena the bytes never define.
        let bytes = encoded(handle, &small);
        assert_eq!(decoded(&bytes).err(), Some(WireError("handle of an undefined arena")));
    }

    #[test]
    fn handles_outside_a_bundle_encode_or_decode_fail() {
        let mut arena = ExprArena::new();
        let handle = arena.int(1);
        assert!(postcard::to_stdvec(&handle).is_err());
        assert!(postcard::from_bytes::<IntExpr>(&[0, 0]).is_err());
    }
}
