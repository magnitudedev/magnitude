//! The forms a conversion recipe ([`PacketRepackRecipe`]) executes in, and
//! the values they derive from ([`PacketValues`]): each expression that only
//! places source bits is a field, routed like a bit route, and every other
//! one is flattened into a postfix program ([`RepackOp`]), the programs of
//! one shape shared.
//!
//! The CPU executes the values as compiled by `seismic_native_cpu`'s repack
//! programs, which move a whole execution of packets in vector steps. A GPU
//! executes the recipe's word form ([`WordTable`]): every word of one source
//! packet's record as runs of source bits and bits of computed values, which
//! the native ABI evaluates into each dialect's code (no kernel reads the
//! table).

use crate::external_tables::RepackTable;
use crate::registry::{
    ConversionLayout, PackedPacketLayout, PacketRepackRecipe, PlaneRepackRecipe, RepackExpr,
    RepresentationConversion, RowPlaneContent, RowPlaneRole,
};
use crate::types::DType;
use std::sync::OnceLock;

/// One postfix operation over 32-bit words (integers, or `f32` bits).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RepackOp {
    /// Push a field of `width` bits: the value's slot in the field column
    /// `prefix` slot bytes per value past its group's first.
    Field {
        prefix: u32,
        width: u32,
    },
    /// Shift left; only a shift of a computed value (a shifted field is a
    /// field).
    ShiftLeft {
        bits: u32,
    },
    /// Pop two, push their OR.
    Or,
    /// Add the value's offset `offset` (the offsets of a group's values
    /// are operands, like their fields), two's complement.
    Offset {
        offset: u32,
    },
    /// Widen the binary16 in the low half exactly.
    F16ToF32,
    I32ToF32,
    /// Pop two, push their `f32` product.
    MultiplyF32,
    /// Multiply by `2^exponent`.
    ScaleF32 {
        exponent: i32,
    },
    Lookup {
        table: RepackTable,
    },
    /// Pop the sign (0 or 1), negate the value below it where it is 1
    /// ([`negate_if`]).
    NegateIf,
}

/// `value` negated where `sign` (0 or 1) is 1, as every executor computes
/// [`RepackOp::NegateIf`]: `(value ^ -sign) + sign`, without a branch.
#[inline]
pub fn negate_if(value: u32, sign: u32) -> u32 {
    (value ^ sign.wrapping_neg()).wrapping_add(sign)
}

/// A program shape: `ops[ops..ops + length]`, reading fields of `fields`
/// slot bytes and `offsets` offsets per value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RepackShape {
    pub ops: u32,
    pub length: u32,
    pub fields: u32,
    pub offsets: u32,
}

/// The bytes of the slot of a value or field of `bits` bits.
pub fn slot(bits: u32) -> u32 {
    bits.div_ceil(8).next_power_of_two()
}

/// The deepest postfix stack a program may need.
pub const STACK: usize = 8;

/// A value of a recipe plane: its bits routed from the source, or computed
/// (`computed[index]`).
enum Value {
    Routed(Vec<(u32, u32)>),
    Computed(usize),
}

/// A computed value of one packet: its shape, the placed bits of its
/// fields, its offsets, its bits and its stack depth.
pub struct Computed {
    pub shape: usize,
    pub fields: Vec<Vec<(u32, u32)>>,
    pub offsets: Vec<i32>,
    pub width: u32,
    /// The deepest stack its evaluation reaches.
    pub depth: usize,
}

/// The type of a postfix value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Type {
    Integer,
    F32,
}

/// The bits an expression places, as (value bit, source bit), when it only
/// places source bits: a source field, shifted, or an OR of such whose bits
/// are disjoint; `None` for any other expression.
fn placed_bits(expression: &RepackExpr) -> Option<Vec<(u32, u32)>> {
    match expression {
        RepackExpr::SourceBits { bit, width } => Some(
            (0..u32::from(*width))
                .map(|offset| (offset, bit + offset))
                .collect(),
        ),
        RepackExpr::ShiftLeft { value, bits } => {
            let placed = placed_bits(value)?
                .into_iter()
                .map(|(at, source)| (at + u32::from(*bits), source))
                .collect::<Vec<_>>();
            placed.iter().all(|(at, _)| *at < 32).then_some(placed)
        }
        RepackExpr::BitOr(left, right) => {
            let mut placed = placed_bits(left)?;
            let right = placed_bits(right)?;
            if right
                .iter()
                .any(|(at, _)| placed.iter().any(|(other, _)| other == at))
            {
                return None;
            }
            placed.extend(right);
            Some(placed)
        }
        _ => None,
    }
}

/// The width of the field of `placed` bits: past its highest.
fn field_width(placed: &[(u32, u32)]) -> u32 {
    placed
        .iter()
        .map(|(at, _)| at + 1)
        .max()
        .expect("a field places bits")
}

/// One flattened expression: its ops, its fields' placed bits, the slot
/// bytes of its fields (each `Field` op's prefix is the total before it),
/// and its offsets.
#[derive(Default)]
struct Flat {
    ops: Vec<RepackOp>,
    fields: Vec<Vec<(u32, u32)>>,
    field_bytes: u32,
    offsets: Vec<i32>,
    depth: usize,
}

impl Flat {
    /// Appends `expression`'s postfix ops at stack height `height`, the
    /// interpreter's typing (`oracle::eval_repack`): bit operations on words,
    /// `F32` arithmetic on `F32` operands.
    fn push(&mut self, expression: &RepackExpr, height: usize) -> Type {
        self.depth = self.depth.max(height + 1);
        let integer = |ty: Type| {
            assert_eq!(
                ty,
                Type::Integer,
                "a recipe applies an integer operation to an f32"
            );
        };
        let float = |ty: Type| {
            assert_eq!(
                ty,
                Type::F32,
                "a recipe applies an f32 operation to an integer"
            );
        };
        if let Some(placed) = placed_bits(expression) {
            let width = field_width(&placed);
            self.ops.push(RepackOp::Field {
                prefix: self.field_bytes,
                width,
            });
            self.field_bytes += slot(width);
            self.fields.push(placed);
            return Type::Integer;
        }
        match expression {
            RepackExpr::SourceBits { .. } => unreachable!("a source field is a field"),
            RepackExpr::ShiftLeft { value, bits } => {
                integer(self.push(value, height));
                self.ops.push(RepackOp::ShiftLeft {
                    bits: u32::from(*bits),
                });
                Type::Integer
            }
            RepackExpr::BitOr(left, right) => {
                integer(self.push(left, height));
                integer(self.push(right, height + 1));
                self.ops.push(RepackOp::Or);
                Type::Integer
            }
            RepackExpr::OffsetI32 { value, offset } => {
                integer(self.push(value, height));
                self.ops.push(RepackOp::Offset {
                    offset: self.offsets.len() as u32,
                });
                self.offsets.push(*offset);
                Type::Integer
            }
            RepackExpr::F16ToF32(value) => {
                integer(self.push(value, height));
                self.ops.push(RepackOp::F16ToF32);
                Type::F32
            }
            RepackExpr::I32ToF32(value) => {
                integer(self.push(value, height));
                self.ops.push(RepackOp::I32ToF32);
                Type::F32
            }
            RepackExpr::MultiplyF32(left, right) => {
                float(self.push(left, height));
                float(self.push(right, height + 1));
                self.ops.push(RepackOp::MultiplyF32);
                Type::F32
            }
            RepackExpr::ScaleF32 { value, exponent } => {
                float(self.push(value, height));
                self.ops.push(RepackOp::ScaleF32 {
                    exponent: i32::from(*exponent),
                });
                Type::F32
            }
            RepackExpr::Lookup { index, table } => {
                integer(self.push(index, height));
                self.ops.push(RepackOp::Lookup { table: *table });
                Type::Integer
            }
            RepackExpr::NegateIf { value, sign } => {
                integer(self.push(value, height));
                integer(self.push(sign, height + 1));
                self.ops.push(RepackOp::NegateIf);
                Type::Integer
            }
        }
    }
}

/// Every value of one packet of a recipe: the computed values' shapes
/// and ops, the computed values, and the planes.
pub struct PacketValues<'a> {
    shapes: Vec<RepackShape>,
    ops: Vec<RepackOp>,
    computed: Vec<Computed>,
    planes: Vec<PlaneValues<'a>>,
}

/// One plane of a packet's values: the recipe's bit routes (its source bit
/// per plane bit), or its values of `width` bits each.
enum PlaneValues<'a> {
    Routes(&'a [u32]),
    Values { width: u32, values: Vec<Value> },
}

/// Where a packet-form bit comes from: a source bit, or bit `bit` of
/// computed value `value`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PacketBit {
    Source(u32),
    Value { value: usize, bit: u32 },
}

impl<'a> PacketValues<'a> {
    /// The values of `recipe`, into the packet form `packet`.
    pub fn of(recipe: &'a PacketRepackRecipe, packet: &PackedPacketLayout) -> Self {
        let mut shapes: Vec<RepackShape> = Vec::new();
        let mut ops = Vec::new();
        let mut computed: Vec<Computed> = Vec::new();
        let mut plane_values = Vec::new();
        for (plane, recipe) in packet.planes.iter().zip(&recipe.planes) {
            let (expressions, width, ty) = match recipe {
                PlaneRepackRecipe::BitRoutes(routes) => {
                    plane_values.push(PlaneValues::Routes(routes));
                    continue;
                }
                PlaneRepackRecipe::DenseValues(expressions) => {
                    // The interpreter casts each value to the storage dtype;
                    // the identity cast is the one recipes take.
                    let ty = match plane.storage_dtype {
                        DType::F32 => Type::F32,
                        DType::I32 | DType::U32 => Type::Integer,
                        dtype => panic!("no executable dense repack into {dtype:?} storage"),
                    };
                    (expressions, 32, ty)
                }
                PlaneRepackRecipe::PackedEntries(expressions) => {
                    (expressions, plane.entry_bits, Type::Integer)
                }
            };
            let mut values = Vec::new();
            for expression in expressions {
                if let Some(placed) = placed_bits(expression).filter(|_| ty == Type::Integer) {
                    values.push(Value::Routed(placed));
                    continue;
                }
                values.push(Value::Computed(computed.len()));
                let mut flat = Flat::default();
                assert_eq!(
                    flat.push(expression, 0),
                    ty,
                    "a recipe value's type differs from its plane's"
                );
                assert!(
                    flat.depth <= STACK,
                    "a recipe value exceeds the executor's stack"
                );
                let shape = shapes
                    .iter()
                    .position(|shape| {
                        ops[shape.ops as usize..][..shape.length as usize] == flat.ops[..]
                    })
                    .unwrap_or_else(|| {
                        shapes.push(RepackShape {
                            ops: ops.len() as u32,
                            length: flat.ops.len() as u32,
                            fields: flat.field_bytes,
                            offsets: flat.offsets.len() as u32,
                        });
                        ops.extend_from_slice(&flat.ops);
                        shapes.len() - 1
                    });
                computed.push(Computed {
                    shape,
                    fields: flat.fields,
                    offsets: flat.offsets,
                    width,
                    depth: flat.depth,
                });
            }
            plane_values.push(PlaneValues::Values { width, values });
        }
        Self {
            shapes,
            ops,
            computed,
            planes: plane_values,
        }
    }

    /// The computed values' shapes, each a run of [`ops`](Self::ops).
    pub fn shapes(&self) -> &[RepackShape] {
        &self.shapes
    }

    pub fn ops(&self) -> &[RepackOp] {
        &self.ops
    }

    pub fn computed(&self) -> &[Computed] {
        &self.computed
    }

    /// Where bit `bit` of plane `plane` of the packet form comes from;
    /// `None` for a bit a routed value does not place (zero).
    pub fn bit(&self, plane: usize, bit: u32) -> Option<PacketBit> {
        match &self.planes[plane] {
            PlaneValues::Routes(routes) => Some(PacketBit::Source(routes[bit as usize])),
            PlaneValues::Values { width, values } => match &values[(bit / width) as usize] {
                Value::Routed(placed) => placed
                    .iter()
                    .find(|(at, _)| *at == bit % width)
                    .map(|(_, source)| PacketBit::Source(*source)),
                Value::Computed(value) => Some(PacketBit::Value {
                    value: *value,
                    bit: bit % width,
                }),
            },
        }
    }
}

/// Declares [`WordOp`] from one list: its variants in code order.
macro_rules! word_ops {
    ($($op:ident,)*) => {
        /// The op of a [`WordTable`] op word, by code.
        #[repr(u32)]
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub enum WordOp {
            $($op,)*
        }

        impl WordOp {
            /// Every op, in code order.
            pub const ALL: [Self; [$(Self::$op),*].len()] = [$(Self::$op),*];
        }
    };
}

word_ops! {
    Field,
    ShiftLeft,
    Or,
    Offset,
    F16ToF32,
    I32ToF32,
    MultiplyF32,
    ScaleF32,
    Lookup,
    NegateIf,
}

impl WordOp {
    /// The op's code: bits 0..4 of its op words.
    pub const fn code(self) -> u32 {
        self as u32
    }
}

impl RepackOp {
    fn word_op(self) -> WordOp {
        match self {
            Self::Field { .. } => WordOp::Field,
            Self::ShiftLeft { .. } => WordOp::ShiftLeft,
            Self::Or => WordOp::Or,
            Self::Offset { .. } => WordOp::Offset,
            Self::F16ToF32 => WordOp::F16ToF32,
            Self::I32ToF32 => WordOp::I32ToF32,
            Self::MultiplyF32 => WordOp::MultiplyF32,
            Self::ScaleF32 { .. } => WordOp::ScaleF32,
            Self::Lookup { .. } => WordOp::Lookup,
            Self::NegateIf => WordOp::NegateIf,
        }
    }
}

/// A piece of a [`WordTable`]: `length` (1..=32) bits placed at bit `shift`
/// of a word.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Piece {
    pub bits: PieceBits,
    pub shift: u32,
    pub length: u32,
}

/// The bits a [`Piece`] places.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PieceBits {
    /// The run of source bits from this one.
    Source(u32),
    /// Bits from `bit` of the value whose record is at table word `record`.
    Value { record: u32, bit: u32 },
}

/// A piece of a [`WordTable`] that places a run of source bits: `length`
/// (1..=32) bits from source bit `bit`, at bit `shift` of a word. A field's
/// pieces, and every run a group visits ([`Group::runs`]), are such runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SourceRun {
    pub bit: u32,
    pub shift: u32,
    pub length: u32,
}

impl SourceRun {
    /// The run's word: its first bit (bits 0..16), shift (16..21) and
    /// length less one (21..26); bit 31 clear ([`Piece`]'s value flag).
    fn encode(self) -> u32 {
        assert!(self.bit < 1 << 16, "a word table's piece lies past 16 bits");
        self.bit | self.shift << 16 | (self.length - 1) << 21
    }

    fn decode(word: u32) -> Self {
        Self {
            bit: word & 0xffff,
            shift: (word >> 16) & 31,
            length: ((word >> 21) & 31) + 1,
        }
    }
}

impl Piece {
    /// A run of source bits' word ([`SourceRun`]); a value's is its record
    /// in place of the first bit, its bit at bits 26..31, and bit 31 set.
    fn encode(self) -> u32 {
        match self.bits {
            PieceBits::Source(bit) => self.run(bit).encode(),
            PieceBits::Value { record, bit } => self.run(record).encode() | bit << 26 | 1 << 31,
        }
    }

    fn decode(word: u32) -> Self {
        let SourceRun {
            bit: first,
            shift,
            length,
        } = SourceRun::decode(word);
        Self {
            bits: if word >> 31 == 0 {
                PieceBits::Source(first)
            } else {
                PieceBits::Value {
                    record: first,
                    bit: (word >> 26) & 31,
                }
            },
            shift,
            length,
        }
    }

    /// The run of source bits from `bit` this piece's shift and length
    /// place.
    fn run(self, bit: u32) -> SourceRun {
        SourceRun {
            bit,
            shift: self.shift,
            length: self.length,
        }
    }
}

/// Consecutive pieces of a [`WordTable`] value's field: runs of source bits.
#[derive(Clone, Copy, Debug)]
pub struct Pieces<'a>(&'a [u32]);

impl<'a> IntoIterator for Pieces<'a> {
    type Item = SourceRun;
    type IntoIter = std::iter::Map<std::slice::Iter<'a, u32>, fn(&u32) -> SourceRun>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter().map(|word| SourceRun::decode(*word))
    }
}

/// A run of pieces as a value's operand or a record word's span: the index
/// of the first, with their count `<< 16`.
fn encode_run(first: u32, count: u32) -> u32 {
    first | count << 16
}

fn decode_run(table: &[u32], word: u32) -> &[u32] {
    &table[(word & 0xffff) as usize..][..(word >> 16) as usize]
}

/// A lookup table of a [`WordTable`]: entries of `1 << shift` bytes, signed
/// and little-endian, from table word `first`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WordLookup {
    pub first: u32,
    pub shift: u32,
}

/// One op of a value of a [`WordTable`], with its immediate or operand.
#[derive(Clone, Copy, Debug)]
pub enum WordStep<'a> {
    /// Push the OR of a field's pieces, each a run of source bits.
    Field(Pieces<'a>),
    ShiftLeft(u32),
    Or,
    /// Add the value's offset, two's complement.
    Offset(u32),
    F16ToF32,
    I32ToF32,
    MultiplyF32,
    /// Multiply by `2^exponent`.
    ScaleF32(i32),
    Lookup(WordLookup),
    NegateIf,
}

/// The op word of `op`, whose lookup's table lies at `lookup`.
fn encode_op(op: RepackOp, lookup: impl Fn(RepackTable) -> WordLookup) -> u32 {
    let immediate = match op {
        RepackOp::ShiftLeft { bits } => bits,
        RepackOp::ScaleF32 { exponent } => exponent as u32 & 0x0fff_ffff,
        RepackOp::Lookup { table } => {
            let WordLookup { first, shift } = lookup(table);
            first | shift << 20
        }
        RepackOp::Field { .. }
        | RepackOp::Or
        | RepackOp::Offset { .. }
        | RepackOp::F16ToF32
        | RepackOp::I32ToF32
        | RepackOp::MultiplyF32
        | RepackOp::NegateIf => 0,
    };
    op.word_op().code() | immediate << 4
}

/// The record of a value: the index of its first op, with their count
/// `<< 16`; the index of its first operand.
fn encode_record(ops: u32, count: u32, operands: u32) -> [u32; 2] {
    [encode_run(ops, count), operands]
}

/// The record at table word `record`: its ops (a run), and the index of
/// its first operand.
fn decode_record(table: &[u32], record: u32) -> (u32, usize) {
    let record = record as usize;
    (table[record], table[record + 1] as usize)
}

/// The steps of the value whose record is at `record` in `table`.
fn steps(table: &[u32], record: u32) -> WordSteps<'_> {
    let (ops, operand) = decode_record(table, record);
    WordSteps {
        table,
        ops: decode_run(table, ops).iter(),
        operand,
    }
}

/// The steps of one value of a [`WordTable`], in order.
#[derive(Clone, Debug)]
pub struct WordSteps<'a> {
    table: &'a [u32],
    ops: std::slice::Iter<'a, u32>,
    /// The index of the next operand.
    operand: usize,
}

impl<'a> Iterator for WordSteps<'a> {
    type Item = WordStep<'a>;

    fn next(&mut self) -> Option<WordStep<'a>> {
        let op = *self.ops.next()?;
        let immediate = op >> 4;
        let mut operand = || {
            self.operand += 1;
            self.table[self.operand - 1]
        };
        Some(match WordOp::ALL[(op & 15) as usize] {
            WordOp::Field => WordStep::Field(Pieces(decode_run(self.table, operand()))),
            WordOp::ShiftLeft => WordStep::ShiftLeft(immediate),
            WordOp::Or => WordStep::Or,
            WordOp::Offset => WordStep::Offset(operand()),
            WordOp::F16ToF32 => WordStep::F16ToF32,
            WordOp::I32ToF32 => WordStep::I32ToF32,
            WordOp::MultiplyF32 => WordStep::MultiplyF32,
            WordOp::ScaleF32 => WordStep::ScaleF32(op as i32 >> 4),
            WordOp::Lookup => WordStep::Lookup(WordLookup {
                first: immediate & 0xfffff,
                shift: immediate >> 20,
            }),
            WordOp::NegateIf => WordStep::NegateIf,
        })
    }
}

/// A group as [`WordTable::compile`] gathers it: its pieces' structure
/// and movement over units, its representative's run, and each member's
/// parameters.
struct GroupDraft {
    key: (Vec<u32>, Vec<i64>),
    run: (PacketBit, u32, u32),
    members: Vec<Vec<i64>>,
}

/// The largest power of two, to 16, dividing `bytes` (16 for 0): the
/// alignment a byte offset or length keeps, to the widest unit a kernel
/// moves. The repack kernels receive it for the record's bytes they store
/// as ABI macros (the native ABI's record alignments).
pub fn alignment(bytes: u32) -> u32 {
    1 << bytes.trailing_zeros().min(4)
}

/// A group's word ([`Group`]): its representative's word (its index in the
/// block, bits 0..8) and its members (bits 8..24).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GroupWord {
    word: u32,
    count: u32,
}

impl GroupWord {
    fn encode(self) -> u32 {
        assert!(self.word < 1 << 8, "a block's words fit 8 bits");
        assert!(self.count < 1 << 16, "a group's members fit 16 bits");
        self.word | self.count << 8
    }

    fn decode(word: u32) -> Self {
        Self {
            word: word & 0xff,
            count: word >> 8,
        }
    }
}

/// One group of a block ([`WordTable::blocks`]): its `count` pieces of one
/// structure, which differ only by displacement. Member `m` is the
/// representative `piece` (member 0) in block word `word`, with each of
/// its parameters moved by the member's displacement of it
/// ([`displacement`](Self::displacement)): 0 its word's index in the
/// block, 1 its shift, then the first bit of each run it visits (its own,
/// or its value's fields') and each of its value's offsets, in op order.
/// Members are in order of their words. Every member's run spans as many
/// bytes as the representative's.
///
/// Metal's and CUDA's code gathers members placed alike into nested loops
/// whose parameters are affine in their counters, which their compilers
/// unroll (the runtime's `conversion::Code`): their front ends then compile
/// one member's code, not each's. Vulkan's generator emits every member:
/// NVIDIA's driver leaves `[[unroll]]` loops rolled (measured: q4_k rows16
/// 43 → 130 µs).
#[derive(Clone, Copy, Debug)]
pub struct Group<'a> {
    pub piece: Piece,
    pub word: u32,
    pub count: u32,
    /// The parameters whose displacements vary (bit `p` for parameter
    /// `p`), each member's listed from table word `list`, a row of one word
    /// per varying parameter in order.
    pub varying: u32,
    pub list: u32,
    table: &'a [u32],
}

impl Group<'_> {
    /// Member `member`'s displacement of parameter `parameter`.
    pub fn displacement(self, member: u32, parameter: u32) -> i32 {
        if self.varying >> parameter & 1 == 0 {
            return 0;
        }
        let row = self.varying.count_ones() * member;
        let column = (self.varying & ((1 << parameter) - 1)).count_ones();
        self.table[(self.list + row + column) as usize] as i32
    }

    /// The runs of source bits the representative visits, in order: its
    /// own, or its value's fields' in op order; each with its parameter.
    pub fn runs(self) -> Vec<(SourceRun, u32)> {
        match self.piece.bits {
            PieceBits::Source(bit) => vec![(self.piece.run(bit), 2)],
            PieceBits::Value { record, .. } => {
                let mut parameter = 2;
                let mut runs = Vec::new();
                for step in steps(self.table, record) {
                    match step {
                        WordStep::Field(pieces) => {
                            for piece in pieces {
                                runs.push((piece, parameter));
                                parameter += 1;
                            }
                        }
                        WordStep::Offset(_) => parameter += 1,
                        _ => {}
                    }
                }
                runs
            }
        }
    }
}

/// The groups of the block whose header is table word `header` ([`WordTable`]).
fn groups(table: &[u32], header: usize) -> impl Iterator<Item = Group<'_>> {
    let run = table[header];
    table[(run & 0xffff) as usize..][..4 * (run >> 16) as usize]
        .chunks_exact(4)
        .map(move |group| {
            let word = GroupWord::decode(group[1]);
            Group {
                piece: Piece::decode(group[0]),
                word: word.word,
                count: word.count,
                varying: group[3],
                list: group[2],
                table,
            }
        })
}

/// A slot map's entry for one run (`WordSpan`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlotEntry {
    pub slot: u32,
    /// The slot moves its runs by whole bytes in every unit of the class.
    pub whole: bool,
    /// It moves them nowhere.
    pub still: bool,
}

impl SlotEntry {
    fn encode(self) -> u32 {
        self.slot | u32::from(self.whole) << 31 | u32::from(self.still) << 30
    }

    fn decode(word: u32) -> Self {
        Self {
            slot: word & 0x3fff_ffff,
            whole: word >> 31 != 0,
            still: (word >> 30) & 1 != 0,
        }
    }
}

/// A class of a [`WordSpan`]: its representative unit and slot map.
#[derive(Clone, Copy, Debug)]
pub struct WordClass<'a> {
    pub representative: u32,
    map: &'a [u32],
}

impl WordClass<'_> {
    /// The slot of the `run`-th run the representative's words visit.
    pub fn entry(self, run: usize) -> SlotEntry {
        SlotEntry::decode(self.map[run])
    }
}

/// A unit of a [`WordSpan`]: its class and its displacements.
#[derive(Clone, Copy, Debug)]
pub struct WordUnit<'a> {
    pub class: u32,
    displacements: &'a [u32],
}

impl WordUnit<'_> {
    /// The unit's displacements of slot `slot`: in whole bytes, and in bits
    /// within a byte.
    pub fn displacement(self, slot: u32) -> (i32, i32) {
        let at = 2 * slot as usize;
        (
            self.displacements[at] as i32,
            self.displacements[at + 1] as i32,
        )
    }
}

/// The word form of a conversion: the form a GPU executes, a thread at a
/// time computing any word of the record of one source packet. It is one
/// table of words, which the native ABI evaluates, once per conversion,
/// into each dialect's code.
///
/// The record is what the destination places per source packet: the packet
/// form, for the `packet` layout; for a row layout, the storage group's
/// share of each row plane (`PackedRowLayout::group_bit`), in
/// `RowPlaneRole::ALL` order, each from a word ([`roles`](Self::roles)).
///
/// The table (every index into it absolute):
/// - From word 0, each block's header ([`blocks`](Self::blocks)): a unit's,
///   one word (unit `u`'s at word `u`), its groups (the index of the first,
///   with their count `<< 16`); then each fixed block's, two words, its
///   groups and its record words (the first, with their count `<< 16`). A
///   block is the record words a kernel computes together: a unit of the
///   span (block `u` unit `u`, its words segment by segment), or a range of
///   the other words within one role's share (a fixed block). A block's
///   words are the OR of its groups' members.
/// - A group is four words ([`Group`]): its representative piece; its word
///   in the block (bits 0..8) and members (8..24); the index of its
///   displacement list; and the parameters
///   whose displacements vary (bit p for parameter p: 0 the word, 1 the
///   shift, 2 + i the representative's i-th run or offset in op order). A
///   list is a row per member, one two's complement word per varying
///   parameter in order. A group's pieces have one structure and move
///   alike over the units of a structure.
/// - A piece places its `length` bits at bit `shift` of a word: bits
///   16..21 are `shift` and 21..26 `length - 1`. Bit 31 clear: bits 0..16
///   are the first source bit of a run of source bits. Set: bits 0..16 are
///   a computed value's record and 26..31 the value's first bit.
/// - A value's record is two words: the index of its first op, with their
///   count `<< 16`; the index of its first operand.
/// - An op is its code ([`WordOp`]) in bits 0..4 and an immediate in
///   4..32: `ShiftLeft`'s bits, `ScaleF32`'s exponent (two's complement),
///   `Lookup`'s table (the index of its first word, with log2 of its entry
///   bytes `<< 20`; entries are signed, little-endian). `Field` and
///   `Offset` take the value's next operand: the pieces of a field
///   (first, with their count `<< 16`; each a run of source bits), and the
///   offset.
///
/// A value is its ops' postfix over a stack of words. [`Group`], [`Piece`],
/// [`SourceRun`], [`WordStep`], [`WordClass`] and [`WordUnit`] read the
/// format. Routes' and fields' bits are merged into runs, so a word costs,
/// per run, the source bytes it spans and a shift and mask; pieces that
/// differ only by displacement share one representative, so a block's code
/// is its groups', not its pieces'.
///
/// It differs from the CPU's program (`seismic_native_cpu`), which moves
/// the bytes of a whole execution of packets in vector steps and evaluates
/// many values at once: a GPU thread computes the few words its layout
/// writer stores, so it reads a packet by word, not an execution by block.
/// Both derive from the recipe's values ([`PacketValues`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WordTable {
    table: Vec<u32>,
    words: u32,
    blocks: u32,
    roles: Vec<(RowPlaneRole, u32, u32)>,
    span: WordSpan,
    /// The lookup tables the values read, as laid out, in table order.
    lookups: Vec<(WordLookup, RepackTable)>,
}

/// The record words of the code planes (a row layout's code roles, or the
/// packet form's plane 0) in units of 32 columns: unit `u` is, segment by
/// segment, words `first + u * unit_words ..` of each code plane (32
/// columns of `unit_words` bits a column), the words a layout writer places
/// together, a block of the table (block `u`). Units whose blocks have one
/// structure, the same groups but for where their representatives' runs
/// start, reading values in the same pattern (a group reads the same value
/// as an earlier group of its unit exactly where it does in the other
/// units), are one class: a kernel compiles a class's representative unit
/// once, evaluating each value once, and the code planes' words of a unit
/// read the same values through the same displacements.
///
/// A unit's runs are visited in order: its groups' representatives', a
/// value's fields' pieces step by step (again at every group reading the
/// value); a member visits its representative's runs. A unit reads each run
/// its class's representative reads, moved by a displacement in whole bytes
/// and one in bits within a byte (so a moved run is the representative's
/// bytes from a moved base, shifted by a moved amount), and moves the
/// run's members alike: the runs whose displacements agree in every unit of
/// the class share a slot, so a unit holds the displacements of each slot.
///
/// In the table: `classes` is the index of two words per class, its
/// representative unit and the index of its slot map, the slot of each run
/// in visiting order ([`SlotEntry`]: bit 31 set where the slot moves its
/// runs by whole bytes in every unit of the class, bit 30 where it moves
/// them nowhere); a class's units move each run's members within the bytes
/// they span; `directory` the index of two words per unit, its class and
/// the index of its displacements, two per slot, bytes then bits (two's
/// complement).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WordSpan {
    pub segments: Vec<WordSegment>,
    pub units: u32,
    classes: u32,
    pub class_count: u32,
    directory: u32,
}

/// A code plane's words in a [`WordSpan`]'s units.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WordSegment {
    pub role: RowPlaneRole,
    pub first: u32,
    pub unit_words: u32,
}

impl WordTable {
    /// The word form of `conversion`, a registered conversion's: built once,
    /// when first asked for.
    pub fn of(conversion: &RepresentationConversion) -> &'static Self {
        static TABLES: OnceLock<Vec<OnceLock<WordTable>>> = OnceLock::new();
        TABLES.get_or_init(|| {
            crate::registry::representation_conversions()
                .iter()
                .map(|_| OnceLock::new())
                .collect()
        })[conversion.id.index()]
        .get_or_init(|| Self::build(conversion))
    }

    fn build(conversion: &RepresentationConversion) -> Self {
        let recipe = &conversion.recipe;
        match conversion.destination_layout {
            ConversionLayout::Packet(packet) => {
                let codes = &packet.planes[0];
                assert!(
                    codes.offset.is_multiple_of(4),
                    "a packet form's codes start at a word"
                );
                // Codes of whole bits a column are placed in units; others
                // (grid entries with a part below a byte) are words of the
                // packet like the rest.
                let segments = if (codes.bytes_per_group * 8).is_multiple_of(packet.group) {
                    vec![WordSegment {
                        role: RowPlaneRole::Codes,
                        first: codes.offset / 4,
                        unit_words: codes.bytes_per_group * 8 / packet.group,
                    }]
                } else {
                    Vec::new()
                };
                Self::compile(
                    recipe,
                    packet,
                    packet.packet_size.div_ceil(4),
                    |plane, bit| Some(packet.planes[plane].offset * 8 + bit),
                    Vec::new(),
                    segments,
                )
            }
            ConversionLayout::Rows(rows) => {
                // Each role's share, and the code roles' words, a unit's
                // `bits` a column.
                let (mut roles, mut segments) = (Vec::new(), Vec::new());
                let mut first = vec![0; rows.planes.len()];
                let mut words = 0;
                for role in RowPlaneRole::ALL {
                    if let Some(index) = rows.planes.iter().position(|plane| plane.role() == role) {
                        let bytes = rows.planes[index].bytes_per_group;
                        roles.push((role, words, bytes));
                        // Column codes are placed in units; a grid
                        // entry's high part is placed as a group's bytes.
                        if !matches!(
                            rows.planes[index].content,
                            RowPlaneContent::Groups { .. } | RowPlaneContent::GridHigh { .. }
                        ) {
                            segments.push(WordSegment {
                                role,
                                first: words,
                                unit_words: rows.column_bits(index),
                            });
                        }
                        first[index] = words;
                        words += bytes.div_ceil(4);
                    }
                }
                Self::compile(
                    recipe,
                    &rows.packet,
                    words,
                    |plane, bit| {
                        rows.occupies(plane, bit).then(|| {
                            let (index, share) = rows.group_bit(plane, bit);
                            first[index] * 32 + share
                        })
                    },
                    roles,
                    segments,
                )
            }
        }
    }

    /// The word form of `recipe` (into the packet form `packet`) over a
    /// record of `words` words: bit `bit` of plane `plane` of the packet
    /// form goes to record bit `destination(plane, bit)`, or nowhere for
    /// `None`; every other record bit is zero. Its code planes' words are
    /// `segments` ([`WordSpan`]).
    fn compile(
        recipe: &PacketRepackRecipe,
        packet: &PackedPacketLayout,
        words: u32,
        destination: impl Fn(usize, u32) -> Option<u32>,
        roles: Vec<(RowPlaneRole, u32, u32)>,
        segments: Vec<WordSegment>,
    ) -> Self {
        let values = PacketValues::of(recipe, packet);
        let mut inputs: Vec<Option<PacketBit>> = vec![None; words as usize * 32];
        let mut claimed = vec![false; inputs.len()];
        for (plane, info) in packet.planes.iter().enumerate() {
            for bit in 0..info.bytes_per_group * 8 {
                let Some(at) = destination(plane, bit) else {
                    continue;
                };
                let at = at as usize;
                assert!(!claimed[at], "two packet bits map to record bit {at}");
                claimed[at] = true;
                inputs[at] = values.bit(plane, bit);
            }
        }
        // Each word's runs (first input, shift, length).
        let next = |input: PacketBit, length: u32| match input {
            PacketBit::Source(bit) => PacketBit::Source(bit + length),
            PacketBit::Value { value, bit } => PacketBit::Value {
                value,
                bit: bit + length,
            },
        };
        let word_runs = inputs
            .chunks_exact(32)
            .map(|word| {
                let mut runs: Vec<(PacketBit, u32, u32)> = Vec::new();
                for (at, input) in word.iter().enumerate() {
                    let Some(input) = *input else { continue };
                    match runs.last_mut() {
                        Some((first, shift, length))
                            if *shift + *length == at as u32 && next(*first, *length) == input =>
                        {
                            *length += 1
                        }
                        _ => runs.push((input, at as u32, 1)),
                    }
                }
                runs
            })
            .collect::<Vec<_>>();
        // A field's runs, each a piece (source bit, shift, length).
        let field_runs = |placed: &[(u32, u32)]| {
            let mut placed = placed.to_vec();
            placed.sort_unstable();
            let mut runs: Vec<(u32, u32, u32)> = Vec::new();
            for (at, source) in placed {
                match runs.last_mut() {
                    Some((first, shift, length))
                        if *shift + *length == at && *first + *length == source =>
                    {
                        *length += 1
                    }
                    _ => runs.push((source, at, 1)),
                }
            }
            runs
        };
        let shape_ops = |shape: usize| {
            let shape = values.shapes[shape];
            &values.ops[shape.ops as usize..][..shape.length as usize]
        };
        // A piece's structure, and its parameters ([`Group`]): its word's
        // index in its block and its shift, then its run's first bit, or
        // per operand of its value in op order each field's runs' first
        // bits and each offset; with each run's length.
        let describe = |index: u32, (input, shift, length): (PacketBit, u32, u32)| match input {
            PacketBit::Source(bit) => (
                vec![0, length, (bit % 8 + length).div_ceil(8)],
                vec![i64::from(index), i64::from(shift), i64::from(bit)],
            ),
            PacketBit::Value { value, bit } => {
                let computed = &values.computed[value];
                let mut key = vec![1, bit, length, computed.shape as u32];
                let mut parameters = vec![i64::from(index), i64::from(shift)];
                let mut fields = computed.fields.iter();
                for op in shape_ops(computed.shape) {
                    match op {
                        RepackOp::Field { .. } => {
                            let runs = field_runs(fields.next().expect("a field per field op"));
                            key.push(runs.len() as u32);
                            for (source, at, run) in runs {
                                key.extend([at, run, (source % 8 + run).div_ceil(8)]);
                                parameters.push(i64::from(source));
                            }
                        }
                        RepackOp::Offset { offset } => {
                            parameters.push(i64::from(computed.offsets[*offset as usize]))
                        }
                        _ => {}
                    }
                }
                (key, parameters)
            }
        };
        // The blocks ([`WordTable::blocks`]): each unit of the span, its
        // words segment by segment, then the other words in ranges, each
        // within one role's share.
        assert!(
            packet.group.is_multiple_of(32),
            "a code plane's group is whole units of 32 columns"
        );
        let units = packet.group / 32;
        let mut blocks: Vec<Vec<u32>> = (0..units)
            .map(|unit| {
                segments
                    .iter()
                    .flat_map(|segment| {
                        (0..segment.unit_words)
                            .map(move |word| segment.first + unit * segment.unit_words + word)
                    })
                    .collect()
            })
            .collect();
        let spanned = blocks.concat();
        for word in (0..words).filter(|word| !spanned.contains(word)) {
            let fixed = blocks.len() > units as usize;
            let continues = fixed
                && blocks
                    .last()
                    .is_some_and(|block| block[block.len() - 1] + 1 == word)
                && !roles.iter().any(|(_, first, _)| *first == word);
            if continues {
                blocks.last_mut().unwrap().push(word);
            } else {
                blocks.push(vec![word]);
            }
        }
        // Each block's pieces, in order: its words', each word's in order;
        // each its run, structure and parameters.
        let block_pieces = blocks
            .iter()
            .map(|block| {
                block
                    .iter()
                    .enumerate()
                    .flat_map(|(index, word)| {
                        word_runs[*word as usize].iter().map(move |run| {
                            let (key, parameters) = describe(index as u32, *run);
                            (*run, key, parameters)
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        // Each piece's movement over the units of its unit's structure (its
        // pieces' structures in order): its parameters in each such unit,
        // less its own; empty for a fixed block's. Pieces of a group move
        // alike, so units of one structure differ only by where their
        // groups' representatives lie.
        let movement = |block: usize, piece: usize| -> Vec<i64> {
            if block >= units as usize {
                return Vec::new();
            }
            let structure = |block: usize| {
                block_pieces[block]
                    .iter()
                    .map(|(_, key, _)| key)
                    .collect::<Vec<_>>()
            };
            let own = &block_pieces[block][piece].2;
            (0..units as usize)
                .filter(|other| structure(*other) == structure(block))
                .flat_map(|other| {
                    block_pieces[other][piece]
                        .2
                        .iter()
                        .zip(own)
                        .map(|(moved, at)| moved - at)
                        .collect::<Vec<_>>()
                })
                .collect()
        };
        // Each block's groups: its pieces of one structure that move alike,
        // in order of the first, each with its representative's run and
        // its members' displacements of the parameters that vary
        // ([`Group`]).
        let block_groups = block_pieces
            .iter()
            .enumerate()
            .map(|(block, pieces)| {
                let mut groups: Vec<GroupDraft> = Vec::new();
                for (piece, (run, key, parameters)) in pieces.iter().enumerate() {
                    let key = (key.clone(), movement(block, piece));
                    match groups.iter_mut().find(|group| group.key == key) {
                        Some(group) => group.members.push(parameters.clone()),
                        None => groups.push(GroupDraft {
                            key,
                            run: *run,
                            members: vec![parameters.clone()],
                        }),
                    }
                }
                groups
                    .into_iter()
                    .map(|GroupDraft { run, members, .. }| {
                        let varying = (0..members[0].len())
                            .filter(|parameter| {
                                members
                                    .iter()
                                    .any(|member| member[*parameter] != members[0][*parameter])
                            })
                            .collect::<Vec<_>>();
                        assert!(members[0].len() <= 32, "a piece's parameters fit 32 bits");
                        let list = members
                            .iter()
                            .flat_map(|member| {
                                varying.iter().map(|parameter| {
                                    i32::try_from(member[*parameter] - members[0][*parameter])
                                        .expect("a displacement within i32")
                                })
                            })
                            .collect::<Vec<_>>();
                        let varying = varying
                            .iter()
                            .fold(0u32, |mask, parameter| mask | 1 << parameter);
                        (
                            run,
                            members.len() as u32,
                            members[0][0] as u32,
                            varying,
                            list,
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        // The displacement lists the groups take, each once.
        let mut lists: Vec<Vec<i32>> = Vec::new();
        for (_, _, _, _, list) in block_groups.iter().flatten() {
            if !list.is_empty() && !lists.contains(list) {
                lists.push(list.clone());
            }
        }
        // The computed values the groups read (their representatives'), the
        // shapes they take, and the lookup tables those take, each in order
        // of first use.
        let mut used: Vec<usize> = Vec::new();
        for (run, _, _, _, _) in block_groups.iter().flatten() {
            match run.0 {
                PacketBit::Value { value, .. } if !used.contains(&value) => used.push(value),
                _ => {}
            }
        }
        let mut shapes: Vec<usize> = Vec::new();
        for value in &used {
            if !shapes.contains(&values.computed[*value].shape) {
                shapes.push(values.computed[*value].shape);
            }
        }
        let mut lookups: Vec<RepackTable> = Vec::new();
        for op in shapes.iter().flat_map(|shape| shape_ops(*shape)) {
            match op {
                RepackOp::Lookup { table } if !lookups.contains(table) => lookups.push(*table),
                _ => {}
            }
        }
        // The sections' sizes, then their contents at their places.
        let operands = used
            .iter()
            .map(|value| {
                let computed = &values.computed[*value];
                computed.fields.len() + computed.offsets.len()
            })
            .sum::<usize>() as u32;
        let field_pieces = used
            .iter()
            .flat_map(|value| &values.computed[*value].fields)
            .map(|field| field_runs(field).len())
            .sum::<usize>() as u32;
        let groups_at = units + 2 * (blocks.len() as u32 - units);
        let lists_at = groups_at + 4 * block_groups.iter().map(Vec::len).sum::<usize>() as u32;
        let records_at = lists_at + lists.iter().map(Vec::len).sum::<usize>() as u32;
        let operands_at = records_at + 2 * used.len() as u32;
        let fields_at = operands_at + operands;
        let ops_at = fields_at + field_pieces;
        let lookups_at = ops_at
            + shapes
                .iter()
                .map(|shape| shape_ops(*shape).len() as u32)
                .sum::<u32>();
        let record = |value: usize| {
            records_at + 2 * used.iter().position(|used| *used == value).unwrap() as u32
        };
        let source =
            |(bit, shift, length): (u32, u32, u32)| SourceRun { bit, shift, length }.encode();
        let mut table = Vec::new();
        let mut first = groups_at;
        for (index, (block, groups)) in blocks.iter().zip(&block_groups).enumerate() {
            table.push(encode_run(first, groups.len() as u32));
            if index >= units as usize {
                table.push(encode_run(block[0], block.len() as u32));
            }
            first += 4 * groups.len() as u32;
        }
        let mut list_at = Vec::new();
        let mut at = lists_at;
        for list in &lists {
            list_at.push(at);
            at += list.len() as u32;
        }
        for (run, count, word, varying, list) in block_groups.iter().flatten() {
            let (input, shift, length) = *run;
            table.push(match input {
                PacketBit::Source(bit) => source((bit, shift, length)),
                PacketBit::Value { value, bit } => Piece {
                    bits: PieceBits::Value {
                        record: record(value),
                        bit,
                    },
                    shift,
                    length,
                }
                .encode(),
            });
            table.extend([
                GroupWord {
                    word: *word,
                    count: *count,
                }
                .encode(),
                if list.is_empty() {
                    0
                } else {
                    list_at[lists.iter().position(|known| known == list).unwrap()]
                },
                *varying,
            ]);
        }
        table.extend(
            lists
                .iter()
                .flatten()
                .map(|displacement| *displacement as u32),
        );
        let mut ops_of = Vec::new();
        let mut at = ops_at;
        for shape in &shapes {
            ops_of.push(at);
            at += shape_ops(*shape).len() as u32;
        }
        let mut operand = operands_at;
        for value in &used {
            let computed = &values.computed[*value];
            let shape = shapes
                .iter()
                .position(|shape| *shape == computed.shape)
                .unwrap();
            table.extend(encode_record(
                ops_of[shape],
                shape_ops(computed.shape).len() as u32,
                operand,
            ));
            operand += (computed.fields.len() + computed.offsets.len()) as u32;
        }
        let mut field = fields_at;
        let mut pieces = Vec::new();
        for value in &used {
            let computed = &values.computed[*value];
            let mut fields = computed.fields.iter();
            for op in shape_ops(computed.shape) {
                match op {
                    RepackOp::Field { .. } => {
                        let runs = field_runs(fields.next().expect("a field per field op"));
                        table.push(encode_run(field, runs.len() as u32));
                        field += runs.len() as u32;
                        pieces.extend(runs.into_iter().map(source));
                    }
                    RepackOp::Offset { offset } => {
                        table.push(computed.offsets[*offset as usize] as u32)
                    }
                    _ => {}
                }
            }
        }
        table.extend(pieces);
        let mut lookup_of = Vec::new();
        let mut entries = Vec::new();
        for lookup in &lookups {
            let values = lookup.values();
            // The narrowest signed entry holding every value.
            let shift = (0..3)
                .find(|shift| {
                    let bits = 8 << shift;
                    values.iter().all(|value| {
                        (-(1i64 << (bits - 1))..1i64 << (bits - 1)).contains(&i64::from(*value))
                    })
                })
                .expect("an i32 fits four bytes");
            lookup_of.push((
                WordLookup {
                    first: lookups_at + entries.len() as u32,
                    shift,
                },
                *lookup,
            ));
            let bytes = values
                .iter()
                .flat_map(|value| value.to_le_bytes()[..1 << shift].to_vec())
                .collect::<Vec<_>>();
            entries.extend(bytes.chunks(4).map(|word| {
                let mut padded = [0; 4];
                padded[..word.len()].copy_from_slice(word);
                u32::from_le_bytes(padded)
            }));
        }
        for shape in &shapes {
            for op in shape_ops(*shape) {
                table.push(encode_op(*op, |table| {
                    lookup_of
                        .iter()
                        .find(|(_, known)| *known == table)
                        .expect("every lookup's table is laid out")
                        .0
                }));
            }
        }
        assert_eq!(
            table.len() as u32,
            lookups_at,
            "a word table's sections abut"
        );
        table.extend(entries);
        let span = Self::span_of(&mut table, segments, units);
        Self {
            table,
            words,
            blocks: blocks.len() as u32,
            roles,
            span,
            lookups: lookup_of,
        }
    }

    /// The span of the code planes' `segments` of `units` units (blocks
    /// `0..units`), appended to `table`.
    fn span_of(table: &mut Vec<u32>, segments: Vec<WordSegment>, units: u32) -> WordSpan {
        // Each unit's structure and positions; the classes in order of
        // their first unit, each with its units. A unit joins a class of its
        // structure whose representative's members it moves within the
        // bytes they span (so a unit reads a member's bytes moved by its
        // byte displacement and shifted by its bit displacement more): each
        // member's first bit within its byte, moved by the unit's bit
        // displacement of the member's run, stays within the byte. Its last
        // bit then does too: a structure fixes its groups' lists, so the
        // unit's member is the representative's moved, and the bytes every
        // member of a group spans, so the two span as many bytes from their
        // first bits' bytes.
        let mut classes: Vec<(Vec<u32>, Vec<u32>)> = Vec::new();
        let mut positions = Vec::new();
        let mut class_of = Vec::new();
        let mut spreads = Vec::new();
        for unit in 0..units {
            let (key, bits, spread) = structure(table, unit);
            let within = |representative: u32| {
                let (from, members) = (
                    &positions[representative as usize],
                    &spreads[representative as usize],
                );
                bits.iter().zip(from).zip(members).all(
                    |((to, from), members): ((&u32, &u32), &Vec<u32>)| {
                        let moved = (to % 8) as i32 - (from % 8) as i32;
                        members
                            .iter()
                            .all(|bit| (0..8).contains(&((bit % 8) as i32 + moved)))
                    },
                )
            };
            let class = classes
                .iter()
                .position(|(known, members)| *known == key && within(members[0]))
                .unwrap_or_else(|| {
                    classes.push((key, Vec::new()));
                    classes.len() - 1
                });
            classes[class].1.push(unit);
            class_of.push(class);
            positions.push(bits);
            spreads.push(spread);
        }
        // Per class, its slots: the runs whose displacements from the
        // representative's, in whole bytes and in bits within a byte, agree
        // in every unit of the class share one, in order of their first run.
        // Per unit, its displacements of each slot.
        let mut slot_maps = Vec::new();
        let mut displacements = vec![Vec::new(); units as usize];
        for (_, members) in &classes {
            let representative = &positions[members[0] as usize];
            let displacement = |run: usize| {
                members
                    .iter()
                    .flat_map(|unit| {
                        let (from, to) = (representative[run], positions[*unit as usize][run]);
                        [
                            (to / 8).wrapping_sub(from / 8),
                            (to % 8).wrapping_sub(from % 8),
                        ]
                    })
                    .collect::<Vec<_>>()
            };
            let mut slots: Vec<Vec<u32>> = Vec::new();
            let map = (0..representative.len())
                .map(|run| {
                    let moved = displacement(run);
                    let whole = moved.chunks_exact(2).all(|pair| pair[1] == 0);
                    let still = moved.iter().all(|displacement| *displacement == 0);
                    let slot = slots
                        .iter()
                        .position(|known| *known == moved)
                        .unwrap_or_else(|| {
                            slots.push(moved);
                            slots.len() - 1
                        }) as u32;
                    SlotEntry { slot, whole, still }.encode()
                })
                .collect::<Vec<_>>();
            for (member, unit) in members.iter().enumerate() {
                displacements[*unit as usize] = slots
                    .iter()
                    .flat_map(|slot| [slot[2 * member], slot[2 * member + 1]])
                    .collect();
            }
            slot_maps.push(map);
        }
        // The class list, the directory, the slot maps, the displacements.
        let classes_at = table.len() as u32;
        let directory_at = classes_at + 2 * classes.len() as u32;
        let maps_at = directory_at + 2 * units;
        let displacements_at = maps_at + slot_maps.iter().map(Vec::len).sum::<usize>() as u32;
        let mut map_at = maps_at;
        for ((_, members), map) in classes.iter().zip(&slot_maps) {
            table.extend([members[0], map_at]);
            map_at += map.len() as u32;
        }
        let mut displacement_at = displacements_at;
        for (unit, class) in class_of.iter().enumerate() {
            table.extend([*class as u32, displacement_at]);
            displacement_at += displacements[unit].len() as u32;
        }
        table.extend(slot_maps.into_iter().flatten());
        table.extend(displacements.into_iter().flatten());
        assert!(
            table.len() < 1 << 16,
            "a word table addresses its records in 16 bits"
        );
        WordSpan {
            segments,
            units,
            classes: classes_at,
            class_count: classes.len() as u32,
            directory: directory_at,
        }
    }

    /// The table, as the type's docs describe it.
    #[cfg(test)]
    fn table(&self) -> &[u32] {
        &self.table
    }

    /// Words of one source packet's record.
    pub fn words(&self) -> u32 {
        self.words
    }

    /// For a row layout, per row plane in `RowPlaneRole::ALL` order: its
    /// role, the record word its share starts at, and the share's bytes
    /// (the plane's bytes per storage group). Empty for the `packet`
    /// layout.
    pub fn roles(&self) -> &[(RowPlaneRole, u32, u32)] {
        &self.roles
    }

    /// The span of the code planes (its codes and high code roles for a
    /// row layout, plane 0 for the `packet` layout).
    pub fn span(&self) -> &WordSpan {
        &self.span
    }

    /// The blocks: the span's units (block `u` unit `u`), then the fixed
    /// blocks.
    pub fn blocks(&self) -> u32 {
        self.blocks
    }

    /// The groups of block `block`.
    pub fn groups(&self, block: u32) -> impl Iterator<Item = Group<'_>> {
        groups(&self.table, self.header(block))
    }

    /// The table word of block `block`'s header: a unit's, `u`; a fixed
    /// block's, two after the units' ([`WordTable`]).
    fn header(&self, block: u32) -> usize {
        let units = self.span.units;
        (if block < units {
            block
        } else {
            units + 2 * (block - units)
        }) as usize
    }

    /// The record words of block `block`, in order: a unit's words segment
    /// by segment, or a fixed block's range.
    pub fn block_words(&self, block: u32) -> Vec<u32> {
        if block < self.span.units {
            self.span
                .segments
                .iter()
                .flat_map(|segment| {
                    (0..segment.unit_words)
                        .map(move |word| segment.first + block * segment.unit_words + word)
                })
                .collect()
        } else {
            let run = self.table[self.header(block) + 1];
            (run & 0xffff..(run & 0xffff) + (run >> 16)).collect()
        }
    }

    /// The steps of the value whose record is at table word `record` (a
    /// [`PieceBits::Value`]'s).
    pub fn steps(&self, record: u32) -> WordSteps<'_> {
        steps(&self.table, record)
    }

    /// Each lookup table the values read, in table order: where it lies,
    /// its entries, and its words.
    pub fn lookups(&self) -> impl Iterator<Item = (WordLookup, &'static [i32], &[u32])> {
        self.lookups.iter().map(|(lookup, table)| {
            let words = (table.values().len() << lookup.shift).div_ceil(4);
            (
                *lookup,
                table.values(),
                &self.table[lookup.first as usize..][..words],
            )
        })
    }

    /// Class `class` of the span.
    pub fn class(&self, class: u32) -> WordClass<'_> {
        let at = (self.span.classes + 2 * class) as usize;
        WordClass {
            representative: self.table[at],
            map: &self.table[self.table[at + 1] as usize..],
        }
    }

    /// Unit `unit` of the span.
    pub fn unit(&self, unit: u32) -> WordUnit<'_> {
        let at = (self.span.directory + 2 * unit) as usize;
        WordUnit {
            class: self.table[at],
            displacements: &self.table[self.table[at + 1] as usize..],
        }
    }

    /// The units of class `class`, in order: each its index and itself.
    pub fn class_units(&self, class: u32) -> impl Iterator<Item = (u32, WordUnit<'_>)> {
        (0..self.span.units)
            .map(|unit| (unit, self.unit(unit)))
            .filter(move |(_, unit)| unit.class == class)
    }

    /// The alignment slot `slot` keeps over the units of class `class`: the
    /// least of its byte displacements' ([`alignment`]).
    pub fn slot_alignment(&self, class: u32, slot: u32) -> u32 {
        self.class_units(class)
            .map(|(_, unit)| alignment(unit.displacement(slot).0.unsigned_abs()))
            .min()
            .expect("a class has units")
    }
}

/// The structure of unit `unit` of `table` (block `unit`), and the first
/// source bit of each run its groups' representatives visit, in order (its
/// positions). The structure is everything else: per group its word, its
/// members and their displacements, and its representative, with the
/// bytes each run spans; a value piece by its ops and by which of the
/// unit's values it is, the ordinal of its record among the unit's
/// records in order of first use. Units of one structure differ only in
/// where their representatives' runs start (their members are displaced
/// alike), and read the same values in the same pattern, so a
/// representative's value serves every unit of its class.
fn structure(table: &[u32], unit: u32) -> (Vec<u32>, Vec<u32>, Vec<Vec<u32>>) {
    let (mut key, mut positions, mut records) = (Vec::new(), Vec::new(), Vec::new());
    // A run's placement and the bytes it spans (its structure).
    let mut run = |run: SourceRun, key: &mut Vec<u32>| {
        key.extend([
            run.shift,
            run.length,
            (run.bit % 8 + run.length).div_ceil(8),
        ]);
        positions.push(run.bit);
    };
    // Each run's members' first bits.
    let members = groups(table, unit as usize)
        .flat_map(|group| {
            group.runs().into_iter().map(move |(run, parameter)| {
                (0..group.count)
                    .map(|m| {
                        run.bit
                            .wrapping_add(group.displacement(m, parameter) as u32)
                    })
                    .collect::<Vec<_>>()
            })
        })
        .collect::<Vec<_>>();
    for group in groups(table, unit as usize) {
        // Its members, and their displacements (each list once).
        key.extend([group.word, group.count, group.list, group.varying]);
        let piece = group.piece;
        match piece.bits {
            PieceBits::Source(bit) => {
                key.push(0);
                run(piece.run(bit), &mut key);
            }
            PieceBits::Value { record, bit } => {
                // The value's ops, by where they lie, and which of the
                // unit's values it is.
                let (ops, _) = decode_record(table, record);
                let ordinal = records
                    .iter()
                    .position(|known| *known == record)
                    .unwrap_or_else(|| {
                        records.push(record);
                        records.len() - 1
                    });
                key.extend([1, bit, piece.shift, piece.length, ops, ordinal as u32]);
                for step in steps(table, record) {
                    match step {
                        WordStep::Field(pieces) => {
                            let pieces = pieces.into_iter();
                            key.push(pieces.len() as u32);
                            for piece in pieces {
                                run(piece, &mut key);
                            }
                        }
                        WordStep::Offset(offset) => key.push(offset),
                        _ => {}
                    }
                }
            }
        }
    }
    (key, positions, members)
}

#[cfg(test)]
mod tests {
    use super::{negate_if, Group, PieceBits, WordLookup, WordStep, WordTable};
    use crate::registry::{self, ConversionLayout, PlaneRepackRecipe, RepresentationConversion};

    /// The words of block `block` of the source packet `packet` by the word
    /// table `words`, executed as its docs describe (as the code the native
    /// ABI generates from it computes them), each source bit it reads the bit `at` gives for it and
    /// its run: the index of the representative's run it is a member's, in
    /// the order the span's docs visit runs.
    fn block_words_at(
        words: &WordTable,
        packet: &[u8],
        block: u32,
        at: &mut dyn FnMut(usize, u32) -> u32,
    ) -> Vec<u32> {
        let at = std::cell::RefCell::new(at);
        // The index of the group's first run in the visiting order.
        let first = std::cell::Cell::new(0);
        let mask = |length: u32| u32::MAX >> (32 - length);
        let source = |run: usize, bit: u32, length: u32| {
            let bit = (at.borrow_mut())(first.get() + run, bit);
            let bytes = (bit / 8..=(bit + length - 1) / 8)
                .map(|byte| u64::from(packet[byte as usize]) << (8 * (byte - bit / 8)))
                .fold(0, |bytes, byte| bytes | byte);
            (bytes >> (bit % 8)) as u32 & mask(length)
        };
        let table = words.table();
        let moved = |at: u32, group: Group, m: u32, parameter: u32| {
            at.wrapping_add(group.displacement(m, parameter) as u32)
        };
        // Member `m` of `group`'s value: each of its parameters moved.
        let evaluate = |group: Group, record: u32, m: u32| {
            let (mut parameter, mut runs) = (2, 0);
            let mut stack = Vec::new();
            for step in words.steps(record) {
                let float = |word: u32| f32::from_bits(word);
                let top = stack.len().wrapping_sub(1);
                match step {
                    WordStep::Field(pieces) => stack.push(
                        pieces
                            .into_iter()
                            .map(|piece| {
                                (parameter, runs) = (parameter + 1, runs + 1);
                                source(
                                    runs - 1,
                                    moved(piece.bit, group, m, parameter - 1),
                                    piece.length,
                                ) << piece.shift
                            })
                            .fold(0, |value, bits| value | bits),
                    ),
                    WordStep::Offset(offset) => {
                        parameter += 1;
                        stack[top] = stack[top].wrapping_add(moved(offset, group, m, parameter - 1))
                    }
                    WordStep::ShiftLeft(bits) => stack[top] <<= bits,
                    WordStep::Or | WordStep::MultiplyF32 | WordStep::NegateIf => {
                        let right = stack.pop().unwrap();
                        let left = stack.last_mut().unwrap();
                        *left = match step {
                            WordStep::Or => *left | right,
                            WordStep::MultiplyF32 => (float(*left) * float(right)).to_bits(),
                            _ => negate_if(*left, right),
                        };
                    }
                    WordStep::F16ToF32 => {
                        stack[top] = registry::f16_to_f32(stack[top] as u16).to_bits()
                    }
                    WordStep::I32ToF32 => stack[top] = (stack[top] as i32 as f32).to_bits(),
                    WordStep::ScaleF32(exponent) => {
                        stack[top] = (float(stack[top]) * 2f32.powi(exponent)).to_bits()
                    }
                    WordStep::Lookup(WordLookup { first, shift }) => {
                        let byte = stack[top] << shift;
                        let entry = table[(first + byte / 4) as usize] >> (byte % 4 * 8);
                        let bits = 32 - (8 << shift);
                        stack[top] = ((entry << bits) as i32 >> bits) as u32;
                    }
                }
            }
            assert_eq!(stack.len(), 1);
            stack[0]
        };
        let mut values = vec![0u32; words.block_words(block).len()];
        for group in words.groups(block) {
            let piece = group.piece;
            for m in 0..group.count {
                let bits = match piece.bits {
                    PieceBits::Source(bit) => source(0, moved(bit, group, m, 2), piece.length),
                    PieceBits::Value { record, bit } => {
                        evaluate(group, record, m) >> bit & mask(piece.length)
                    }
                };
                values[moved(group.word, group, m, 0) as usize] |=
                    bits << moved(piece.shift, group, m, 1);
            }
            first.set(first.get() + group.runs().len());
        }
        values
    }

    /// In every registered conversion's word table, the units of a class
    /// read their values in their representative's pattern: two groups of
    /// a unit read one value exactly where the representative's do, so code
    /// that evaluates each of the representative's values once (as every
    /// dialect's generated code does) evaluates each of the unit's once.
    #[test]
    fn units_of_a_class_read_values_in_one_pattern() {
        for conversion in registry::representation_conversions() {
            let words = WordTable::of(conversion);
            let span = words.span();
            // The records a unit's value groups read, in visiting order.
            let reads = |unit: u32| {
                words
                    .groups(unit)
                    .filter_map(|group| match group.piece.bits {
                        PieceBits::Value { record, .. } => Some(record),
                        PieceBits::Source(_) => None,
                    })
                    .collect::<Vec<_>>()
            };
            for unit in 0..span.units {
                let class = words.class(words.unit(unit).class);
                let (own, representative) = (reads(unit), reads(class.representative));
                assert_eq!(own.len(), representative.len());
                for (i, j) in (0..own.len()).flat_map(|i| (0..i).map(move |j| (i, j))) {
                    assert_eq!(
                        own[i] == own[j],
                        representative[i] == representative[j],
                        "{} -> {}: unit {unit} groups {j} and {i}",
                        registry::representation_info(conversion.source).name,
                        registry::representation_info(conversion.destination).name,
                    );
                }
            }
        }
    }

    /// In `words`, `conversion`'s word table, a unit moves each member of its
    /// class's runs within the bytes the member spans: the kernels read a
    /// member's bytes (as many as it spans) moved by the slot's byte
    /// displacement, from bit its bit within its byte moved by the slot's
    /// bit displacement, so every bit of the member lies within them.
    fn assert_units_move_members_within_their_bytes(
        conversion: &RepresentationConversion,
        words: &WordTable,
    ) {
        for unit in 0..words.span().units {
            let moves = words.unit(unit);
            let class = words.class(moves.class);
            let mut run = 0;
            for group in words.groups(class.representative) {
                for (piece, parameter) in group.runs() {
                    let entry = class.entry(run);
                    let bits = if entry.still {
                        0
                    } else {
                        moves.displacement(entry.slot).1
                    };
                    for m in 0..group.count {
                        let member = piece.bit as i32 + group.displacement(m, parameter);
                        let (first, length) = (member % 8 + bits, piece.length as i32);
                        let spanned = (member % 8 + length + 7) / 8 * 8;
                        assert!(
                            first >= 0 && first + length <= spanned,
                            "{} -> {}: unit {unit} run {run} member {m}",
                            registry::representation_info(conversion.source).name,
                            registry::representation_info(conversion.destination).name,
                        );
                    }
                    run += 1;
                }
            }
        }
    }

    #[test]
    fn units_move_members_within_their_bytes() {
        for conversion in registry::representation_conversions() {
            assert_units_move_members_within_their_bytes(conversion, WordTable::of(conversion));
        }
    }

    /// A unit whose members, moved as its first run moves, would leave the
    /// bytes they span lies apart from the class it would otherwise join.
    /// The q4_k codes, each code's 4 bits from bit `12 c + first`: unit 0's
    /// from `first` 0 (members at bits 0 and 4 of their bytes), unit 1's
    /// from 2 (a member at bit 4 moved by 2 would need bits 6..10, two
    /// bytes, so its members split by the bytes they span), unit 2's from 8
    /// (moved by a whole byte), the rest from 0.
    #[test]
    fn units_whose_members_leave_their_bytes_lie_apart() {
        let conversion = registry::representation_conversions()
            .iter()
            .find(|conversion| {
                registry::representation_info(conversion.source).name == "gguf_q4_k"
                    && matches!(conversion.destination_layout, ConversionLayout::Packet(_))
            })
            .expect("a q4_k packet conversion is registered");
        let packet = conversion.destination_layout.packet();
        let codes = packet
            .planes
            .iter()
            .position(|plane| plane.name == "words")
            .unwrap();
        let mut conversion = conversion.clone();
        conversion.recipe.planes[codes] = PlaneRepackRecipe::BitRoutes(
            (0..packet.planes[codes].bytes_per_group * 8)
                .map(|bit| {
                    let (column, code_bit) = (bit / 4, bit % 4);
                    let first = match column / 32 {
                        1 => 2,
                        2 => 8,
                        _ => 0,
                    };
                    12 * (column % 32) + first + code_bit
                })
                .collect(),
        );
        let words = WordTable::build(&conversion);
        let class = |unit| words.unit(unit).class;
        assert_ne!(
            class(1),
            class(0),
            "unit 1's members would leave their bytes"
        );
        assert_eq!(
            class(2),
            class(0),
            "unit 2 moves its members by a whole byte"
        );
        assert_units_move_members_within_their_bytes(&conversion, &words);
        assert_writes_the_records_of_the_packet_forms(&conversion, &words);
    }

    /// `words`, `conversion`'s word table, writes each source packet's
    /// record: its packet form, placed as the destination places it.
    fn assert_writes_the_records_of_the_packet_forms(
        conversion: &RepresentationConversion,
        words: &WordTable,
    ) {
        let layout = conversion.source_layout;
        let packets = 64;
        let mut source = registry::pseudo_random_bytes(
            conversion.id.index() as u64,
            (packets * layout.packet_size) as usize,
        );
        conversion.admit_source(&mut source);
        let forms = crate::interp::packet_forms(conversion, &source);
        let packet = conversion.destination_layout.packet();
        let place = |plane: usize, bit: u32| match conversion.destination_layout {
            ConversionLayout::Packet(packet) => Some(packet.planes[plane].offset * 8 + bit),
            ConversionLayout::Rows(rows) => rows.occupies(plane, bit).then(|| {
                let (index, share) = rows.group_bit(plane, bit);
                let role = rows.planes[index].role();
                let (_, first, _) = words
                    .roles()
                    .iter()
                    .find(|(candidate, _, _)| *candidate == role)
                    .unwrap();
                first * 32 + share
            }),
        };
        for (index, (form, packet_bytes)) in forms
            .chunks_exact(packet.packet_size as usize)
            .zip(source.chunks_exact(layout.packet_size as usize))
            .enumerate()
        {
            let mut expected = vec![0u32; words.words() as usize];
            for (plane, info) in packet.planes.iter().enumerate() {
                for bit in 0..info.bytes_per_group * 8 {
                    let at = info.offset * 8 + bit;
                    let set = (form[(at / 8) as usize] >> (at % 8)) & 1 != 0;
                    if let Some(to) = place(plane, bit).filter(|_| set) {
                        expected[(to / 32) as usize] |= 1 << (to % 32);
                    }
                }
            }
            for block in 0..words.blocks() {
                let computed = block_words_at(words, packet_bytes, block, &mut |_, bit| bit);
                for (word, computed) in words.block_words(block).into_iter().zip(computed) {
                    assert_eq!(
                        computed,
                        expected[word as usize],
                        "{} -> {}: packet {index} word {word}",
                        registry::representation_info(conversion.source).name,
                        registry::representation_info(conversion.destination).name,
                    );
                }
            }
            assert_eq!(
                (0..words.blocks())
                    .map(|block| words.block_words(block).len())
                    .sum::<usize>(),
                expected.len(),
                "the blocks cover the record",
            );
            // Each unit of a span is its class's representative unit
            // reading the unit's positions.
            let span = words.span();
            for unit in 0..span.units {
                let moves = words.unit(unit);
                let class = words.class(moves.class);
                let shared = block_words_at(
                    words,
                    packet_bytes,
                    class.representative,
                    &mut |run, bit| {
                        let (bytes, bits) = moves.displacement(class.entry(run).slot);
                        ((bit / 8) as i32 + bytes) as u32 * 8 + ((bit % 8) as i32 + bits) as u32
                    },
                );
                for (word, shared) in words.block_words(unit).into_iter().zip(shared) {
                    assert_eq!(
                        shared,
                        expected[word as usize],
                        "{} -> {}: packet {index} unit {unit} word {word}",
                        registry::representation_info(conversion.source).name,
                        registry::representation_info(conversion.destination).name,
                    );
                }
            }
        }
    }

    #[test]
    fn word_tables_write_the_records_of_the_packet_forms() {
        for conversion in registry::representation_conversions() {
            assert_writes_the_records_of_the_packet_forms(conversion, WordTable::of(conversion));
        }
    }
}
