//! The execution model a Vulkan kernel may assume, checked at formation
//! (`docs/compatibility.md`, "Execution model"): Vulkan's guarantees and one
//! more, that a selection construct (if/else, switch) entered with
//! subgroup-uniform control flow rejoins at its merge block. [`check`]
//! reports whether the entry point relies on that assumption
//! ([`Findings::relies_on_reconvergence`]), and where it
//!
//! - executes a subgroup operation inside a divergent selection or a
//!   lane-varying loop, or after a lane-dependent exit, or after a
//!   lane-varying loop unless the module is formed to declare maximal
//!   reconvergence (`MaximallyReconvergesKHR`, which rejoins every
//!   invocation after the loop: [`Findings::relies_on_maximal_reconvergence`]);
//! - executes a control barrier inside a divergent selection or a
//!   lane-varying loop, or after a lane-dependent exit, or one of subgroup
//!   execution scope after a lane-varying loop unless the module is formed
//!   to declare maximal reconvergence, which rejoins the subgroup there as
//!   for a subgroup operation;
//! - executes a workgroup control barrier under control that may differ
//!   between the subgroups of the workgroup (steered by the subgroup's id,
//!   or by a subgroup operation's result that depends on it), or after an
//!   exit taken under such control: every invocation of the workgroup must
//!   reach it;
//! - gives a subgroup operation a divergent operand that SPIR-V requires to
//!   be dynamically uniform: a broadcast's or read invocation's lane, a quad
//!   broadcast's index, a cooperative-matrix load's or store's pointer,
//!   layout and stride;
//! - gives a subgroup operation a value derived from the local or global
//!   invocation id: Vulkan does not say which invocations share a subgroup,
//!   nor in which lane order;
//! - combines a value derived from the local or global invocation id with
//!   one derived from the lane order of subgroups (below), in one value or
//!   in one store's address and value: for the same reason, the result
//!   depends on how the device maps one onto the other;
//! - gives a 32-lane hardware join ([`HARDWARE_JOIN`]) a divergent value:
//!   on 32 lanes it is its argument, which then steers the subgroup;
//! - writes a buffer the ABI declares the launch only reads;
//! - stores a buffer's address, or a value derived from one (a delta
//!   between addresses), to workgroup or device memory: an address lives in
//!   an invocation's registers and variables only, where the analysis
//!   follows it ([`Provenance`]).
//!
//! An instruction the analysis does not read is refused unanalysed. A
//! refusal names, for each way the entry point breaks the model, how many
//! instructions break it so and the first (in reverse postorder), and where
//! what breaks it there comes from: the instruction that first raised the
//! offending property of its operand (a built-in's load, a read of memory with
//! a store that may race it, a subgroup operation, a load of a variable
//! written where divergent control rejoins), the construct whose divergent
//! control rejoins at a phi, or the loop that carries it out, and for
//! control, the selection or loop and the condition that steer it, and
//! where that differs from, a few constructs back.
//!
//! A value is uniform when every invocation of the hardware subgroup holds
//! it: constants and specialization constants; loads through uniform
//! pointers of uniform and push-constant memory, the argument block and
//! device memory the launch only reads; the built-in counts and ids of the
//! grid, the workgroup and its subgroups; whole-subgroup reductions, votes
//! and ballots; a broadcast or shuffle from a uniform lane; and what is
//! computed from uniform values alone. A whole-subgroup result is one value
//! for its subgroup, but which subgroup combined it shows where its operands
//! derive from the subgroup's id, or differ between invocations otherwise
//! than by their lane in the subgroup: Vulkan does not say which
//! invocations share a subgroup. Where they differ by lane alone (a lane's
//! id, a read at an address of its lane), every full subgroup combines the
//! same values, and the result is the same in every subgroup. A load through a uniform pointer of
//! workgroup memory is uniform only where no store to it can race it: no
//! store to workgroup memory that may touch its bytes lies in the same
//! barrier interval, the points reached from one legal workgroup barrier
//! ordering workgroup memory (or from the entry) without crossing another,
//! where one reaches the other in it or a branch that may part the
//! workgroup parts the ways to them; a branch the whole workgroup takes
//! alike, passed once in the interval, that parts every way to one from
//! every way to the other separates them ([`Interleaving`]). Workgroup
//! memory is followed by byte:
//! an access whose address is a constant offset into its variable (each
//! access chain index one value for every invocation, as [`Flow::live`]
//! folds it, and its type of known size) touches those bytes alone, of its
//! variable, or of the explicitly laid out memory every `Block` view shares
//! ([`Analysis::location`]); any other access may touch any byte, as a
//! store at a variable offset races every read. Without a barrier between a
//! store and a load, invocations of a subgroup may see the store or not:
//! only lockstep would make them agree. Device memory the launch writes is
//! never read alike: a barrier orders its own workgroup's stores alone, and
//! nothing shows that no other workgroup stores the bytes read. What another
//! workgroup stores holds nothing correlated with a reader's local ids (a
//! store mixing them with lane order is refused where it is made): read
//! through a uniform pointer, what a store derives from reaches the reader
//! only from a store of its own workgroup in its barrier interval, or one
//! that may run anywhere before it, which no barrier here orders. Everything
//! else is divergent: so is a phi where invocations may arrive from
//! different predecessors, but a copy (below), and a value a lane-varying
//! loop defines that may differ between its iterations, used after it (but a
//! load of memory the launch only reads, at an address the same in every
//! iteration).
//!
//! An undefined value (`OpUndef`, a variable before it is written) adds
//! nothing to what it joins: a kernel whose result depends on one is wrong
//! whatever the invocations hold, so the check assumes it holds what lets the
//! kernel be right, as LLVM's uniformity analysis does. The analysed copy
//! holds such values on paths a correct kernel never reads them through: a
//! phi of the value a function returns inside a loop, which merging its
//! returns leaves unset on the others; an array written element by element,
//! which promotion leaves in memory.
//!
//! A phi that copies a value, which every way into it brings but for the phi
//! itself, phis that copy the value and undefined values, and which is
//! defined before it on every path to it ([`Flow::copies`]), is that value
//! (matched by id, one per value: `numbering`). It holds what the value
//! holds wherever invocations arrive from, and differs between a loop's
//! iterations only as the value does.
//! Promotion leaves such phis in loops, alone or in webs among themselves,
//! for a variable a loop only reads: an outer loop's index an inner
//! lane-varying loop reads, which that loop carries out unchanged.
//!
//! Device memory the launch only reads is a buffer the ABI reads through the
//! [`INPUT_BLOCK`] (a `&` parameter), the device's tables buffer
//! ([`TABLES_BLOCK`]), the address table of a slab or record port it writes
//! through ([`ADDRESS_TABLE_BLOCK`]), a buffer whose [`ARGUMENT_BLOCK`] word
//! the launch's module never stores through (a result or scratch an earlier
//! launch wrote, which this one only reads), a port's bytes its table
//! addresses that the module never stores through, and what addresses read
//! from `&` buffers alone point to. The module itself is the one source of
//! what its launch writes; the runtime keeps every buffer a kernel writes,
//! and every port's bytes its table addresses, apart from every other buffer
//! of the launch (`api::exclusive_pairs`, `Reaching::meets`), so no store through one word
//! reaches memory read through another. An address names every buffer whose
//! address, as the argument block gives it, it is computed from, whatever
//! the computation; a value read from memory, which holds no such address
//! (none may be stored there), is an offset where it is added to one, and
//! otherwise a base: one read from a `&` buffer points into memory the
//! launch only reads by declaration, and one read from an address table into
//! its port's bytes; one read from any other memory, whether the launch
//! writes it or only reads it, may point anywhere. A condition (a boolean)
//! holds no address: one on an address carries the condition's divergence,
//! not the address.
//!
//! A value derives from the local invocation id when it is computed from
//! one, chosen by control that depends on one (a phi, of a value or of a
//! variable, where a selection or loop steered by one rejoins, a value of a
//! loop left by one, a store to shared memory under one), or read from
//! memory such a value is stored to, unless every
//! invocation of the subgroup reads one value there (through a uniform
//! pointer no store of its workgroup races): one value holds no lane order. A value derives
//! from the lane order of subgroups, in the same ways, from the subgroup's
//! id, an invocation's lane or lane masks, the result of a subgroup
//! operation that differs by lane (a scan, a clustered reduction, a shuffle
//! from a divergent lane, a relative shuffle, a quad operation, an election,
//! a cooperative-matrix fragment), and a value one for the whole subgroup (a
//! reduction, vote, ballot or broadcast) that shows which subgroup combined
//! it (above).
//!
//! A workgroup control barrier at a legal point (under control, and after
//! exits, the same for every invocation of the workgroup) is executed by
//! every invocation of the workgroup as one dynamic instance, which every hardware compiler enforces
//! wave by wave: it ends the state after a lane-varying loop for what
//! follows it. A subgroup barrier does not: Vulkan promises no reconvergence
//! after such a loop, and a subgroup barrier may compile to nothing; it is
//! legal only where control is already uniform, or where the module declares
//! maximal reconvergence, which rejoins the subgroup after the loop. Neither
//! ends the state after a lane-dependent exit, where a barrier is itself
//! refused.
//!
//! A selection is divergent when its condition is. A loop is lane-varying
//! when it is left, or continued, under a divergent condition: its exits,
//! its back edge, a `break`, `continue` or return. One left or continued
//! under a condition that differs only between subgroups is not, but what
//! it carries out differs between them as that condition does. A
//! lane-dependent exit is a return, `break` or `continue` taken under a
//! divergent condition: after it, until the merge of the construct it leaves
//! to, not every invocation runs.
//!
//! The analysis runs over SSA. It reads a copy of the module whose returns
//! are merged, whose functions but the summarised ones (below) are inlined,
//! whose locals are promoted and whose values are numbered
//! (`formation::analysed`): an inlined function's return becomes a break
//! from its body, which rejoins at the call's return as a selection does.
//!
//! Summarised functions. A pure function ([`summarised`]: what it computes
//! is a function of its arguments and the memory it reads; it writes no
//! memory but its own variables, and executes no subgroup operation or
//! barrier) stays a call, and the check analyses its body as an entry point
//! whose parameters hold what the call's arguments hold, once per such state
//! and context ([`Summaries::summary`]). A load in the body is judged as the
//! inlined body's would be, by its memory's kind and its pointer's state,
//! which the arguments give; workgroup memory and device memory the launch
//! writes, by what a load at the call may see of the caller's stores
//! ([`Context`]), which is what each load of the body sees, as the body
//! stores to neither and holds no barrier. The call then holds what the body returns;
//! what the body breaks is refused at the call, as often as there; the call
//! ends after a lane-varying loop where the body's return does, and is
//! where invocations rejoin where they rejoin in the body before its
//! return. The control the call runs under is not the body's: what it
//! decides there (a barrier's legality, what a store to shared memory or a
//! subgroup operation runs under, the state after a lane-dependent exit)
//! concerns instructions a pure body does not hold, and its exits leave
//! only constructs of its own. This is the inlined check's analysis of the same instructions,
//! but for what an analysis of each function apart cannot see, which it
//! takes the conservative way. It refuses more than an exact
//! interprocedural check would, never less, exactly where the list below
//! says. These over-approximations cost time, never a refusal: a module the
//! check refuses with summaries is checked again with every call inlined,
//! and that verdict stands (`formation::execution_model`, the one entry
//! formation checks through). Summaries are a fast path to acceptance,
//! never an authority for refusal. The over-approximations:
//!
//! - a phi of the body merges values that are one value only through the
//!   call: two arguments that are one value (`f(x, x)`), or a value the
//!   caller computed before the call. It is a merge, divergent under
//!   divergent control, where the inlined body's phi would copy that value;
//!   so two calls of one function with the same arguments are two values,
//!   not one (the contract's value rules, which numbering serves, read a
//!   copy where they are one: `numbering`);
//! - a constant reaches the body other than as an argument, as a member of
//!   a struct argument: its bits are not carried into the body, so every
//!   branch the body takes on it may go either way, and the call's result
//!   is any value where the check folds constants ([`Flow::live`]). A
//!   constant argument is carried: formation specialises a function per
//!   tuple of constant arguments its calls pass (`formation::specialised`),
//!   each copy the body with those constants in place of its parameters and
//!   the branches they decide pruned;
//! - the call is under divergent control and the body holds a selection or
//!   a loop: invocations are taken to rejoin after the call
//!   ([`Summaries::steers`]), which can only add to what the module relies
//!   on, never refuse;
//! - the call is in a loop and the body reads workgroup memory or device
//!   memory the launch writes, as a loop's load of it differs between its
//!   iterations: the call's result is taken to differ so too, whether or
//!   not it derives from what is read ([`Summary::varies`]), so it is
//!   divergent after a lane-varying loop that carries it out. Of the same
//!   arguments, a body that reads no such memory computes the same.
//!
//! A refusal at a call of a summarised function names the call and the
//! body's instruction; one that inlining the function would not give is one
//! of these, which the inlined check overrules. An edge no invocation of the module
//! can take is dropped first: a branch whose condition is computed from
//! constants, through values, structs and variables, takes one way
//! ([`Flow::live`]), as does one whose condition a constant decides alone
//! (false of `&&`, true of `||`, zero of a product or of `&`). A
//! specialization constant is no constant there: pipeline creation sets it
//! after the check, so either way may be taken. A struct's members are
//! followed apart, in values and in the variables left in memory, an
//! array's elements' members too (each joined across the elements), and a
//! variable holds at each point what the stores that reach that point put in
//! it: memory is followed as an SSA of the variables something reads
//! ([`Node`]), a node per store and per meeting of control, which each read
//! sees as the node reaching it. A variable is the invocation's own, so a
//! store to it keeps what is stored, under whatever control: where
//! invocations that took different paths meet, its phi joins what they
//! depend on, as a value's phi does, and what a lane-varying loop stores is
//! divergent where the loop is left.

use crate::seal::{self, Instruction, MalformedModule, Violation};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::rc::Rc;

pub(crate) const OP_UNDEF: u32 = 1;
pub(crate) const OP_NAME: u32 = 5;
const OP_LINE: u32 = 8;
pub(crate) const OP_EXT_INST: u32 = 12;
const OP_ENTRY_POINT: u32 = 15;
const OP_TYPE_BOOL: u32 = 20;
const OP_TYPE_INT: u32 = 21;
const OP_TYPE_VECTOR: u32 = 23;
const OP_TYPE_ARRAY: u32 = 28;
const OP_TYPE_RUNTIME_ARRAY: u32 = 29;
const OP_TYPE_STRUCT: u32 = 30;
const OP_TYPE_POINTER: u32 = 32;
const OP_CONSTANT_TRUE: u32 = 41;
const OP_CONSTANT_FALSE: u32 = 42;
const OP_CONSTANT: u32 = 43;
const OP_CONSTANT_COMPOSITE: u32 = 44;
const OP_CONSTANT_NULL: u32 = 46;
const OP_CONSTANT_COMPOSITE_REPLICATE: u32 = 4461;
pub(crate) const OP_FUNCTION: u32 = 54;
pub(crate) const OP_FUNCTION_PARAMETER: u32 = 55;
pub(crate) const OP_FUNCTION_END: u32 = 56;
pub(crate) const OP_FUNCTION_CALL: u32 = 57;
const OP_VARIABLE: u32 = 59;
pub(crate) const OP_LOAD: u32 = 61;
const OP_STORE: u32 = 62;
const OP_COPY_MEMORY: u32 = 63;
pub(crate) const OP_DECORATE: u32 = 71;
const OP_COMPOSITE_CONSTRUCT: u32 = 80;
const OP_COMPOSITE_EXTRACT: u32 = 81;
const OP_COMPOSITE_INSERT: u32 = 82;
pub(crate) const OP_COPY_OBJECT: u32 = 83;
const OP_CONVERT_U_TO_PTR: u32 = 120;
const OP_I_ADD: u32 = 128;
const OP_I_SUB: u32 = 130;
const OP_SELECT: u32 = 169;
const OP_CONTROL_BARRIER: u32 = 224;
const OP_MEMORY_BARRIER: u32 = 225;
const OP_ATOMIC_LOAD: u32 = 227;
const OP_ATOMIC_STORE: u32 = 228;
pub(crate) const OP_PHI: u32 = 245;
const OP_LOOP_MERGE: u32 = 246;
const OP_SELECTION_MERGE: u32 = 247;
const OP_LABEL: u32 = 248;
const OP_BRANCH: u32 = 249;
const OP_BRANCH_CONDITIONAL: u32 = 250;
const OP_SWITCH: u32 = 251;
const OP_KILL: u32 = 252;
const OP_RETURN: u32 = 253;
const OP_RETURN_VALUE: u32 = 254;
const OP_UNREACHABLE: u32 = 255;
const OP_NO_LINE: u32 = 317;
const OP_ATOMIC_FLAG_CLEAR: u32 = 319;
const OP_TERMINATE_INVOCATION: u32 = 4416;
const OP_COOPERATIVE_MATRIX_LOAD: u32 = 4457;
const OP_COOPERATIVE_MATRIX_STORE: u32 = 4458;

/// The terminators that leave the entry point.
const EXITS: [u32; 4] = [OP_KILL, OP_RETURN, OP_RETURN_VALUE, OP_TERMINATE_INVOCATION];

/// Every block terminator.
const TERMINATORS: [u32; 8] = [
    OP_BRANCH,
    OP_BRANCH_CONDITIONAL,
    OP_SWITCH,
    OP_KILL,
    OP_RETURN,
    OP_RETURN_VALUE,
    OP_UNREACHABLE,
    OP_TERMINATE_INVOCATION,
];

/// `OpAccessChain` and `OpInBoundsAccessChain`: a base, then indices.
const ACCESS_CHAINS: [u32; 2] = [65, 66];
/// `OpPtrAccessChain` and `OpInBoundsPtrAccessChain`: a base, an element,
/// then indices.
const ELEMENT_CHAINS: [u32; 2] = [67, 70];

/// Module-scope instructions with a result type: `OpUndef`, the constants
/// and specialization constants (`OpConstantCompositeReplicateEXT` and
/// `OpSpecConstantCompositeReplicateEXT` among them), `OpVariable`.
fn typed_declaration(opcode: u32) -> bool {
    matches!(opcode, OP_UNDEF | 41..=46 | 48..=52 | 4461 | 4462 | OP_VARIABLE)
}

/// Atomic operations with a result: `OpAtomicLoad`, `OpAtomicExchange`
/// through `OpAtomicXor`, `OpAtomicFlagTestAndSet`, `OpAtomicFMinEXT`,
/// `OpAtomicFMaxEXT`, `OpAtomicFAddEXT`. Each invocation reads its own; the
/// pointer is their first operand.
fn atomic(opcode: u32) -> bool {
    matches!(opcode, OP_ATOMIC_LOAD | 229..=242 | 318 | 5614 | 5615 | 6035)
}

/// The pointer an instruction reads memory through.
fn read_pointer(words: &[u32]) -> Option<u32> {
    match words[0] & 0xffff {
        OP_LOAD | OP_COOPERATIVE_MATRIX_LOAD => Some(words[3]),
        OP_COPY_MEMORY => Some(words[2]),
        opcode if atomic(opcode) => Some(words[3]),
        _ => None,
    }
}

/// The pointer an instruction writes memory through.
fn written_pointer(words: &[u32]) -> Option<u32> {
    match words[0] & 0xffff {
        OP_STORE | OP_COPY_MEMORY | OP_ATOMIC_STORE | OP_ATOMIC_FLAG_CLEAR | OP_COOPERATIVE_MATRIX_STORE => {
            Some(words[1])
        }
        // A variable's initializer.
        OP_VARIABLE if words.len() > 4 => Some(words[2]),
        OP_ATOMIC_LOAD => None,
        opcode if atomic(opcode) => Some(words[3]),
        _ => None,
    }
}

const DECORATION_BUILT_IN: u32 = 11;
const DECORATION_BUFFER_BLOCK: u32 = 3;
const DECORATION_BLOCK: u32 = 2;
const DECORATION_ARRAY_STRIDE: u32 = 6;
const DECORATION_OFFSET: u32 = 35;
const OP_TYPE_FLOAT: u32 = 22;
const OP_MEMBER_DECORATE: u32 = 72;

/// What a built-in holds, invocation by invocation.
fn built_in(built_in: Option<u32>) -> Taint {
    match built_in {
        // `NumWorkgroups`, `WorkgroupSize`, `WorkgroupId`, `SubgroupSize`,
        // `NumSubgroups`: every invocation of a subgroup holds them alike.
        Some(24 | 25 | 26 | 36 | 38) => Taint::UNIFORM,
        // `SubgroupId`: alike in a subgroup, but which local ids it holds
        // the device decides.
        Some(40) => Taint::SUBGROUP,
        // `LocalInvocationId`, `GlobalInvocationId`, `LocalInvocationIndex`:
        // an invocation's place in the workgroup and the grid, which does
        // not say its subgroup or lane.
        Some(27..=29) => Taint::LOCAL,
        // `SubgroupLocalInvocationId` and the `Subgroup*Mask` lane masks.
        Some(41 | 4416..=4420) => Taint::LANE,
        _ => Taint::DIVERGENT,
    }
}

/// Built-ins of an invocation's subgroup and its place in it:
/// `SubgroupSize`, `NumSubgroups`, `SubgroupId`,
/// `SubgroupLocalInvocationId`, and the `Subgroup*Mask` lane masks.
const SUBGROUP_BUILT_INS: [u32; 9] = [36, 38, 40, 41, 4416, 4417, 4418, 4419, 4420];

pub(crate) const STORAGE_UNIFORM_CONSTANT: u32 = 0;
pub(crate) const STORAGE_INPUT: u32 = 1;
pub(crate) const STORAGE_UNIFORM: u32 = 2;
const STORAGE_WORKGROUP: u32 = 4;
const STORAGE_PRIVATE: u32 = 6;
const STORAGE_FUNCTION: u32 = 7;
pub(crate) const STORAGE_PUSH_CONSTANT: u32 = 9;

/// Scopes that hold every invocation of a workgroup: `CrossDevice`,
/// `Device`, `Workgroup`, `QueueFamily`.
fn holds_workgroup(scope: Option<u32>) -> bool {
    matches!(scope, Some(0 | 1 | 2 | 5))
}

/// Memory shared between invocations: workgroup memory, and device memory
/// the launch writes.
const WORKGROUP: usize = 0;
const WRITTEN: usize = 1;

/// `semantics` order workgroup memory: `WorkgroupMemory` with
/// `AcquireRelease`, `SequentiallyConsistent`, or both `Acquire` and
/// `Release`.
fn orders_workgroup_memory(semantics: u32) -> bool {
    let ordering = semantics & 0x18 != 0 || semantics & 0x6 == 0x6;
    ordering && semantics & 0x100 != 0
}

/// The names of the block types the ABI reads a launch's argument block
/// through, which nothing in the launch writes: the addresses of the buffers
/// it may write, and the addresses of the buffers it only reads by
/// declaration (`&`) with its argument words. The addresses of the address
/// tables of the slab and record ports it may write through are read
/// through [`ADDRESS_TABLE_BLOCK`].
pub const ARGUMENT_BLOCK: &str = "seismic_argument_block_t";
pub const INPUT_BLOCK: &str = "seismic_input_block_t";
/// A buffer whose address is read through this block is an address table
/// (a slab port's slabs, a record port's records), which the launch only
/// reads: each address it holds points into bytes only that port's binding
/// reaches (the runtime's `SlabReach` and `Records`, held apart from every
/// other binding of the launch by its one binding check). That holds for
/// backed slots, under two preconditions the runtime and kernels keep: a
/// free slot names a zero slab its store shares (a pool's absent granule, a
/// copy's first slab), which a kernel reads only by a masked load whose value
/// it discards and never stores through; and the host repoints no slot that
/// work in flight reads as backed (`pool.rs`). A slab or record port bound to
/// no elements binds a plain buffer instead, whose words nothing reads.
pub const ADDRESS_TABLE_BLOCK: &str = "seismic_address_table_block_t";
/// The name of the block type the device's tables buffer is read through
/// (`seismic/lang/build.rs`, at the push constant `seismic_tables`), which
/// no launch writes.
pub const TABLES_BLOCK: &str = "seismic_tables_t";

/// The prefix of the names of the prelude's 32-lane hardware joins
/// (`seismic_hardware_any`, `seismic_steer`'s vote, and `_first`): functions returning
/// their argument, which formation checks uniform and then removes from the
/// module it ships ([`hardware_joins`]). On 64 lanes they are subgroup
/// operations, checked as such.
pub const HARDWARE_JOIN: &str = "seismic_hardware_";

/// A 32-lane hardware join of a module: a function named with
/// [`HARDWARE_JOIN`] that returns its one parameter.
pub struct HardwareJoin {
    /// The word its `OpFunction` starts at.
    pub at: usize,
    /// The ids it defines: the function, its parameter and its block.
    pub ids: [u32; 3],
}

/// The 32-lane hardware joins of `module`, and a violation for each
/// function named as one whose body is anything but the return of its one
/// parameter: not what the prelude defines, but a kernel's own function
/// under the prefix, which is not a join (and is inlined as any other).
pub fn hardware_joins(module: &[u32]) -> Result<(Vec<HardwareJoin>, Vec<Violation>), MalformedModule> {
    let instructions = seal::instructions(module)?;
    let words_of = |instruction: &Instruction| &module[instruction.start..instruction.start + instruction.words];
    let named = instructions
        .iter()
        .filter(|instruction| instruction.opcode == OP_NAME)
        .map(words_of)
        .filter_map(|words| {
            let name = seal::literal_string(&words[2..]);
            name.starts_with(HARDWARE_JOIN).then_some((words[1], name))
        })
        .collect::<HashMap<_, _>>();
    let mut joins = Vec::new();
    let mut violations = Vec::new();
    for (index, instruction) in instructions.iter().enumerate() {
        let words = words_of(instruction);
        if instruction.opcode != OP_FUNCTION || !named.contains_key(&words[2]) {
            continue;
        }
        let body = instructions[index..].iter().take(5).map(words_of).collect::<Vec<_>>();
        let identity = matches!(
            body.as_slice(),
            [function, parameter, label, returned, end]
                if parameter[0] & 0xffff == OP_FUNCTION_PARAMETER
                    && parameter[1] == function[1]
                    && label[0] & 0xffff == OP_LABEL
                    && returned[0] & 0xffff == OP_RETURN_VALUE
                    && returned[1] == parameter[2]
                    && end[0] & 0xffff == OP_FUNCTION_END
        );
        if !identity {
            let name = named[&words[2]].split('(').next().expect("a split yields its first part");
            violations.push(Violation(format!(
                "`{name}`, named as a 32-lane hardware join ({HARDWARE_JOIN}…), which returns its one argument, but whose body is otherwise"
            )));
            continue;
        }
        joins.push(HardwareJoin { at: instruction.start, ids: [words[2], body[1][2], body[2][1]] });
    }
    Ok((joins, violations))
}

/// `DontInline`, of a function's control.
pub(crate) const DONT_INLINE: u32 = 0x2;

/// The functions of `module` formation keeps as calls in the analysed copy
/// and the check summarises ([`check`]), each with, by parameter, whether it
/// is a pointer, and if so whether the function writes through it: formation
/// passes the value its variable holds in, and returns the value it ends
/// with beside its result where written (`formation::by_value`). Such a
/// function is pure: what it computes, and leaves in the variables its
/// pointer arguments point into, is a function of its arguments and of the
/// memory it reads, which it writes none of. It is neither the entry point
/// nor a hardware join; it holds only instructions the analysis reads, and
/// no subgroup operation, barrier, atomic, cooperative-matrix load or
/// store, or kill; it reads and writes memory through its own function
/// variables and its pointer parameters, each of which every call points
/// into a variable of the caller's own, into which no other pointer argument
/// of the call points (so copying the variable's value in and out is what
/// the call does to it, GLSL's own `in`, `out` and `inout`); it reads, and
/// never writes, any other memory only through module-scope variables but
/// private ones and addresses it converts to pointers, but an argument
/// block or the tables at an address a parameter gives (which the analysis
/// judges by where the address was read: `Program::argument_block`); it
/// uses no pointer but as an access chain's base, a load's and a call's
/// argument; and it calls only functions that are pure, or local, which
/// formation inlines into it: pure but for a pointer parameter some call
/// does not pass so, which the function passes it only from memory of its
/// own.
///
/// The rule reads the module as formation gives it: the cleaned module,
/// whose arguments are still pointers, and the analysed copy, where a pure
/// function's are values. A module holding an instruction of a shape the
/// analysis does not read has no summarised function: every call is
/// inlined, and the check refuses that instruction where it runs.
pub(crate) fn summarised(module: &[u32]) -> Result<HashMap<u32, Vec<Option<bool>>>, MalformedModule> {
    let instructions = seal::instructions(module)?;
    let words_of = |instruction: &Instruction| &module[instruction.start..instruction.start + instruction.words];
    let entry = entry_point(module)?;
    // Pointer type → storage class, and pointee; id → type.
    let mut pointers = HashMap::new();
    let mut pointees = HashMap::new();
    let mut types = HashMap::new();
    let mut joins = HashSet::new();
    // Module-scope variable → storage class.
    let mut globals = HashMap::new();
    struct Function<'w> {
        id: u32,
        /// Each parameter and whether it is a pointer to function storage.
        parameters: Vec<(u32, bool)>,
        /// Any other pointer parameter: the function is not pure.
        foreign: bool,
        body: Vec<&'w [u32]>,
    }
    let mut functions: Vec<Function> = Vec::new();
    let mut names = HashMap::new();
    for instruction in &instructions {
        let words = words_of(instruction);
        match instruction.opcode {
            OP_NAME => {
                names.insert(words[1], seal::literal_string(&words[2..]));
            }
            OP_TYPE_POINTER => {
                pointers.insert(words[1], words[2]);
                pointees.insert(words[1], words[3]);
            }
            OP_FUNCTION => {
                if names.get(&words[2]).is_some_and(|name: &String| name.starts_with(HARDWARE_JOIN)) {
                    joins.insert(words[2]);
                }
                functions.push(Function { id: words[2], parameters: Vec::new(), foreign: false, body: Vec::new() });
            }
            OP_FUNCTION_END => {}
            OP_FUNCTION_PARAMETER => {
                types.insert(words[2], words[1]);
                let function = functions.last_mut().expect("a parameter follows its function");
                let storage = pointers.get(&words[1]).copied();
                function.foreign |= storage.is_some_and(|storage| storage != STORAGE_FUNCTION);
                function.parameters.push((words[2], storage == Some(STORAGE_FUNCTION)));
            }
            OP_LABEL | OP_LINE | OP_NO_LINE if !functions.is_empty() => {}
            opcode if !functions.is_empty() => {
                let Some(shape) = shape(opcode) else { return Ok(HashMap::new()) };
                if shape.result {
                    types.insert(words[2], words[1]);
                }
                functions.last_mut().expect("a body follows its function").body.push(words);
            }
            opcode if typed_declaration(opcode) => {
                types.insert(words[2], words[1]);
                if opcode == OP_VARIABLE {
                    globals.insert(words[2], words[3]);
                }
            }
            _ => {}
        }
    }
    let pointer = |id: u32| types.get(&id).is_some_and(|ty| pointers.contains_key(ty));
    // The block types the ABI reads the launch's arguments and the tables
    // through (`Program::argument_block`).
    let argument_blocks = names
        .iter()
        .filter(|(_, name)| [ARGUMENT_BLOCK, INPUT_BLOCK, ADDRESS_TABLE_BLOCK, TABLES_BLOCK].contains(&name.as_str()))
        .map(|(id, _)| *id)
        .collect::<HashSet<_>>();
    // Every access chain and copy, by result → its base; every function
    // variable; every pointer converted from an address.
    let mut chains = HashMap::new();
    let mut variables = HashSet::new();
    let mut converted = HashSet::new();
    for words in functions.iter().flat_map(|function| &function.body) {
        let opcode = words[0] & 0xffff;
        if opcode == OP_VARIABLE && words[3] == STORAGE_FUNCTION {
            variables.insert(words[2]);
        }
        if opcode == OP_CONVERT_U_TO_PTR {
            converted.insert(words[2]);
        }
        if ACCESS_CHAINS.contains(&opcode) || ELEMENT_CHAINS.contains(&opcode) || opcode == OP_COPY_OBJECT {
            chains.insert(words[2], words[3]);
        }
    }
    let root = |mut id: u32| {
        while let Some(base) = chains.get(&id) {
            id = *base;
        }
        id
    };
    // `id` points into memory a load of which the analysis judges by its
    // kind and the pointer's state alone, in a body as where inlined: a
    // module-scope variable but a private one (the invocation's own, which
    // its caller may have written), or an address converted to a pointer.
    let judged = |id: u32| {
        let root = root(id);
        globals.get(&root).is_some_and(|storage| *storage != STORAGE_PRIVATE) || converted.contains(&root)
    };
    // By function and parameter: every call passes a pointer into a
    // variable of the caller's own, and none into one another pointer
    // argument of the call points into, so the call may copy the variable's
    // value in and, where the function writes it, out (GLSL's own semantics
    // of `in`, `out` and `inout`), and no other memory is written.
    let mut copies = HashMap::<(u32, usize), bool>::new();
    for words in functions.iter().flat_map(|function| &function.body) {
        if words[0] & 0xffff != OP_FUNCTION_CALL {
            continue;
        }
        let roots = words[4..].iter().map(|argument| root(*argument)).collect::<Vec<_>>();
        for (index, argument) in words[4..].iter().enumerate() {
            let copied = variables.contains(&roots[index])
                && roots.iter().enumerate().all(|(other, root)| other == index || *root != roots[index]);
            let entry = copies.entry((words[3], index)).or_insert(true);
            *entry &= copied || !pointer(*argument);
        }
    }
    let by_value = |function: &Function, index: usize| {
        function.parameters[index].1 && copies.get(&(function.id, index)).copied().unwrap_or(false)
    };
    // Local functions, to a fixed point: each falls once one it calls does.
    let index_of = functions.iter().enumerate().map(|(index, function)| (function.id, index)).collect::<HashMap<_, _>>();
    let mut local = functions
        .iter()
        .map(|function| function.id != entry && !joins.contains(&function.id) && !function.foreign)
        .collect::<Vec<_>>();
    loop {
        let mut changed = false;
        for (index, function) in functions.iter().enumerate() {
            if !local[index] {
                continue;
            }
            let parameters = function.parameters.iter().map(|(id, _)| *id).collect::<Vec<_>>();
            let own = function
                .body
                .iter()
                .filter(|words| words[0] & 0xffff == OP_VARIABLE)
                .map(|words| words[2])
                .collect::<HashSet<_>>();
            // `id`'s root is a variable or a pointer parameter of the
            // function's own.
            let owned = |id: u32| {
                let root = root(id);
                parameters.contains(&root) || own.contains(&root)
            };
            let mut holds = true;
            for words in &function.body {
                let opcode = words[0] & 0xffff;
                let shape = shape(opcode).expect("a body holds instructions of a known shape");
                if subgroup_operation(opcode).is_some()
                    || atomic(opcode)
                    || matches!(
                        opcode,
                        OP_CONTROL_BARRIER | OP_MEMORY_BARRIER | OP_ATOMIC_STORE | OP_ATOMIC_FLAG_CLEAR | OP_KILL
                            | OP_TERMINATE_INVOCATION
                    )
                {
                    holds = false;
                    break;
                }
                if opcode == OP_FUNCTION_CALL
                    && !index_of.get(&words[3]).is_some_and(|callee| local[*callee] && !joins.contains(&words[3]))
                {
                    holds = false;
                    break;
                }
                // A pointer to an argument block or the tables, converted from
                // an address the caller gives: inlined, the analysis may see it
                // read from the push constants (`Program::at_push_constant`).
                if opcode == OP_CONVERT_U_TO_PTR
                    && pointees.get(&words[1]).is_some_and(|pointee| argument_blocks.contains(pointee))
                    && parameters.contains(&root(words[3]))
                {
                    holds = false;
                    break;
                }
                for at in shape.positions(words) {
                    let id = words[at];
                    if !pointer(id) {
                        continue;
                    }
                    let derived = ACCESS_CHAINS.contains(&opcode) || ELEMENT_CHAINS.contains(&opcode);
                    let through = match opcode {
                        OP_LOAD | OP_COOPERATIVE_MATRIX_LOAD => at == 3,
                        OP_STORE => at == 1,
                        OP_COPY_MEMORY => at <= 2,
                        OP_FUNCTION_CALL => true,
                        _ if derived => at == 3,
                        OP_COPY_OBJECT => true,
                        _ => false,
                    };
                    // Read, or derived to be read, never written or passed on.
                    let read = match opcode {
                        OP_LOAD => at == 3,
                        _ if derived => at == 3,
                        OP_COPY_OBJECT => true,
                        _ => false,
                    };
                    if !(through && owned(id)) && !(read && judged(id)) {
                        holds = false;
                    }
                }
            }
            if !holds {
                local[index] = false;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    Ok(functions
        .iter()
        .enumerate()
        .filter(|(index, function)| {
            local[*index]
                && (0..function.parameters.len())
                    .all(|parameter| !function.parameters[parameter].1 || by_value(function, parameter))
        })
        .map(|(_, function)| {
            // A pointer parameter is written where the function stores
            // through it or passes it on.
            let written = function
                .body
                .iter()
                .flat_map(|words| match words[0] & 0xffff {
                    OP_FUNCTION_CALL => words[4..].to_vec(),
                    _ => written_pointer(words).into_iter().collect(),
                })
                .map(root)
                .collect::<HashSet<_>>();
            let parameters =
                function.parameters.iter().map(|(id, pointer)| pointer.then(|| written.contains(id))).collect();
            (function.id, parameters)
        })
        .collect())
}

/// The functions of `programs` (by function) that read workgroup or device
/// memory, themselves or through a function they call: two calls of one
/// with the same arguments may read different values.
pub(crate) fn shared_readers(programs: &HashMap<u32, Program<'_>>) -> HashSet<u32> {
    let reads = reads_classes(programs);
    reads.into_iter().filter(|&(_, (wg, dev))| wg || dev).map(|(f, _)| f).collect()
}

/// By function: whether it (or any callee it reaches) reads workgroup memory
/// and/or device memory.
fn reads_classes(programs: &HashMap<u32, Program<'_>>) -> HashMap<u32, (bool, bool)> {
    let mut reads = programs
        .iter()
        .map(|(&function, program)| {
            let (mut wg, mut dev) = (false, false);
            for words in program.blocks.iter().flat_map(|block| &block.instructions) {
                if let Some(pointer) = read_pointer(words) {
                    match program.place(pointer) {
                        Ok(Place::Workgroup) => wg = true,
                        Ok(Place::Device) => dev = true,
                        _ => {}
                    }
                }
            }
            (function, (wg, dev))
        })
        .collect::<HashMap<_, _>>();
    loop {
        let mut changed = false;
        for (&function, program) in programs {
            let (mut wg, mut dev) = reads[&function];
            for callee in &program.called {
                if let Some(&(cwg, cdev)) = reads.get(callee) {
                    wg |= cwg;
                    dev |= cdev;
                }
            }
            let entry = reads.get_mut(&function).expect("function in reads");
            if *entry != (wg, dev) {
                *entry = (wg, dev);
                changed = true;
            }
        }
        if !changed {
            return reads;
        }
    }
}

const GROUP_OPERATION_REDUCE: u32 = 0;

/// Which operands of an instruction are ids, past its result type and
/// result.
#[derive(Clone, Copy)]
enum Ids {
    All,
    /// The first operands; literals and enumerants follow. An optional
    /// operand is present only where those before it are, so the first
    /// operands past the required ones are the optional ids.
    Leading(usize),
    /// All but one literal: a subgroup operation's group operation, an
    /// extended instruction's number, a variable's storage class.
    Except(usize),
    /// A phi's incoming values, each paired with its parent block.
    Phi,
}

#[derive(Clone, Copy)]
pub(crate) struct Shape {
    /// A result type and a result lead the operands.
    pub(crate) result: bool,
    ids: Ids,
}

impl Shape {
    /// Where the id operands of `words`, an instruction of this shape,
    /// stand. Every operand that may name a value a function computes is
    /// among them: an id an optional operand past them names (a memory
    /// operand's scope) is a constant, as SPIR-V requires of a scope in a
    /// shader; a phi's parents and a switch's targets are blocks.
    pub(crate) fn positions(&self, words: &[u32]) -> impl Iterator<Item = usize> {
        let first = if self.result { 3 } else { 1 };
        let ids = self.ids;
        (first..words.len()).filter(move |&at| match ids {
            Ids::All => true,
            Ids::Leading(count) => at < first + count,
            Ids::Except(literal) => at != first + literal,
            Ids::Phi => (at - first).is_multiple_of(2),
        })
    }

    /// The id operands of `words`, an instruction of this shape.
    fn ids(&self, words: &[u32]) -> Vec<u32> {
        self.positions(words).map(|at| words[at]).collect()
    }
}

const VALUE: Shape = Shape { result: true, ids: Ids::All };
const GROUP: Shape = Shape { result: true, ids: Ids::Except(1) };

/// What a subgroup operation's result is, invocation by invocation.
#[derive(Clone, Copy)]
enum Yields {
    /// One value for the subgroup: a vote, a ballot, the first invocation's
    /// value.
    Uniform,
    /// The value of the invocation the id at this word names: uniform
    /// exactly when that id is (a broadcast, a shuffle, a read invocation).
    Lane(usize),
    /// Each invocation's own: an election, an inverse ballot, a relative
    /// shuffle, a quad operation, a rotation, a partition, a
    /// cooperative-matrix fragment.
    PerLane,
    /// Uniform where its operands are: a ballot's bits read.
    Operands,
    /// Uniform as a whole-subgroup reduction (group operation `Reduce`, no
    /// cluster size); a scan or a clustered or partitioned result is each
    /// invocation's own.
    Reduction,
    /// A ballot's bits counted: uniform where its operands are as a
    /// reduction, each invocation's own as a scan.
    BitCount,
}

struct SubgroupOperation {
    opcode: u32,
    name: &'static str,
    shape: Shape,
    /// None for an operation without a result.
    yields: Option<Yields>,
    /// The words SPIR-V requires to be dynamically uniform, and what each
    /// is.
    uniform: &'static [(usize, &'static str)],
}

const fn operation(opcode: u32, name: &'static str, shape: Shape, yields: Yields) -> SubgroupOperation {
    SubgroupOperation { opcode, name, shape, yields: Some(yields), uniform: &[] }
}

impl SubgroupOperation {
    const fn requiring(self, uniform: &'static [(usize, &'static str)]) -> Self {
        SubgroupOperation { uniform, ..self }
    }
}

/// The operations over the invocations of a subgroup (control barriers are
/// checked apart).
static SUBGROUP_OPERATIONS: [SubgroupOperation; 47] = [
    operation(333, "OpGroupNonUniformElect", VALUE, Yields::PerLane),
    operation(334, "OpGroupNonUniformAll", VALUE, Yields::Uniform),
    operation(335, "OpGroupNonUniformAny", VALUE, Yields::Uniform),
    operation(336, "OpGroupNonUniformAllEqual", VALUE, Yields::Uniform),
    // Scope, value, id.
    operation(337, "OpGroupNonUniformBroadcast", VALUE, Yields::Lane(5)).requiring(&[(5, "id")]),
    operation(338, "OpGroupNonUniformBroadcastFirst", VALUE, Yields::Uniform),
    operation(339, "OpGroupNonUniformBallot", VALUE, Yields::Uniform),
    operation(340, "OpGroupNonUniformInverseBallot", VALUE, Yields::PerLane),
    operation(341, "OpGroupNonUniformBallotBitExtract", VALUE, Yields::Operands),
    operation(342, "OpGroupNonUniformBallotBitCount", GROUP, Yields::BitCount),
    operation(343, "OpGroupNonUniformBallotFindLSB", VALUE, Yields::Operands),
    operation(344, "OpGroupNonUniformBallotFindMSB", VALUE, Yields::Operands),
    // Scope, value, id.
    operation(345, "OpGroupNonUniformShuffle", VALUE, Yields::Lane(5)),
    operation(346, "OpGroupNonUniformShuffleXor", VALUE, Yields::PerLane),
    operation(347, "OpGroupNonUniformShuffleUp", VALUE, Yields::PerLane),
    operation(348, "OpGroupNonUniformShuffleDown", VALUE, Yields::PerLane),
    operation(349, "OpGroupNonUniformIAdd", GROUP, Yields::Reduction),
    operation(350, "OpGroupNonUniformFAdd", GROUP, Yields::Reduction),
    operation(351, "OpGroupNonUniformIMul", GROUP, Yields::Reduction),
    operation(352, "OpGroupNonUniformFMul", GROUP, Yields::Reduction),
    operation(353, "OpGroupNonUniformSMin", GROUP, Yields::Reduction),
    operation(354, "OpGroupNonUniformUMin", GROUP, Yields::Reduction),
    operation(355, "OpGroupNonUniformFMin", GROUP, Yields::Reduction),
    operation(356, "OpGroupNonUniformSMax", GROUP, Yields::Reduction),
    operation(357, "OpGroupNonUniformUMax", GROUP, Yields::Reduction),
    operation(358, "OpGroupNonUniformFMax", GROUP, Yields::Reduction),
    operation(359, "OpGroupNonUniformBitwiseAnd", GROUP, Yields::Reduction),
    operation(360, "OpGroupNonUniformBitwiseOr", GROUP, Yields::Reduction),
    operation(361, "OpGroupNonUniformBitwiseXor", GROUP, Yields::Reduction),
    operation(362, "OpGroupNonUniformLogicalAnd", GROUP, Yields::Reduction),
    operation(363, "OpGroupNonUniformLogicalOr", GROUP, Yields::Reduction),
    operation(364, "OpGroupNonUniformLogicalXor", GROUP, Yields::Reduction),
    // Scope, value, index.
    operation(365, "OpGroupNonUniformQuadBroadcast", VALUE, Yields::PerLane).requiring(&[(5, "index")]),
    operation(366, "OpGroupNonUniformQuadSwap", VALUE, Yields::PerLane),
    operation(4421, "OpSubgroupBallotKHR", VALUE, Yields::Uniform),
    operation(4422, "OpSubgroupFirstInvocationKHR", VALUE, Yields::Uniform),
    operation(4428, "OpSubgroupAllKHR", VALUE, Yields::Uniform),
    operation(4429, "OpSubgroupAnyKHR", VALUE, Yields::Uniform),
    operation(4430, "OpSubgroupAllEqualKHR", VALUE, Yields::Uniform),
    operation(4431, "OpGroupNonUniformRotateKHR", VALUE, Yields::PerLane),
    // Value, id.
    operation(4432, "OpSubgroupReadInvocationKHR", VALUE, Yields::Lane(4)).requiring(&[(4, "id")]),
    // Pointer, layout, then an optional stride, then an optional memory
    // operand mask (a literal) and its operands: a mask stands only past a
    // stride, so the ids read are the pointer, the layout and the stride
    // where there is one, and a stride absent is a requirement unchecked
    // (`words.get`).
    operation(
        OP_COOPERATIVE_MATRIX_LOAD,
        "OpCooperativeMatrixLoadKHR",
        Shape { result: true, ids: Ids::Leading(3) },
        Yields::PerLane,
    )
    .requiring(&[(3, "pointer"), (4, "layout"), (5, "stride")]),
    // Pointer, object, layout, then optional stride and memory operands, as
    // the load's.
    SubgroupOperation {
        opcode: OP_COOPERATIVE_MATRIX_STORE,
        name: "OpCooperativeMatrixStoreKHR",
        shape: Shape { result: false, ids: Ids::Leading(4) },
        yields: None,
        uniform: &[(1, "pointer"), (3, "layout"), (4, "stride")],
    },
    // A, B, C, then the matrix operands.
    operation(4459, "OpCooperativeMatrixMulAddKHR", Shape { result: true, ids: Ids::Leading(3) }, Yields::PerLane),
    operation(5110, "OpGroupNonUniformQuadAllKHR", VALUE, Yields::PerLane),
    operation(5111, "OpGroupNonUniformQuadAnyKHR", VALUE, Yields::PerLane),
    operation(5296, "OpGroupNonUniformPartitionNV", VALUE, Yields::PerLane),
];

fn subgroup_operation(opcode: u32) -> Option<&'static SubgroupOperation> {
    SUBGROUP_OPERATIONS.iter().find(|operation| operation.opcode == opcode)
}

/// The name a violation gives an opcode: the SPIR-V name of a subgroup
/// operation and of the instructions a refusal most often stands at or
/// traces to, the number of any other.
fn opcode_name(opcode: u32) -> String {
    if let Some(operation) = subgroup_operation(opcode) {
        return operation.name.to_owned();
    }
    let name = match opcode {
        OP_UNDEF => "OpUndef",
        OP_EXT_INST => "OpExtInst",
        OP_FUNCTION_CALL => "OpFunctionCall",
        OP_VARIABLE => "OpVariable",
        OP_LOAD => "OpLoad",
        OP_STORE => "OpStore",
        OP_COPY_MEMORY => "OpCopyMemory",
        65 => "OpAccessChain",
        66 => "OpInBoundsAccessChain",
        67 => "OpPtrAccessChain",
        OP_COMPOSITE_CONSTRUCT => "OpCompositeConstruct",
        OP_COMPOSITE_EXTRACT => "OpCompositeExtract",
        OP_COMPOSITE_INSERT => "OpCompositeInsert",
        OP_COPY_OBJECT => "OpCopyObject",
        113 => "OpUConvert",
        114 => "OpSConvert",
        OP_CONVERT_U_TO_PTR => "OpConvertUToPtr",
        124 => "OpBitcast",
        OP_I_ADD => "OpIAdd",
        OP_I_SUB => "OpISub",
        132 => "OpIMul",
        134 => "OpUDiv",
        137 => "OpUMod",
        OP_SELECT => "OpSelect",
        170 => "OpIEqual",
        171 => "OpINotEqual",
        172 => "OpUGreaterThan",
        174 => "OpUGreaterThanEqual",
        176 => "OpULessThan",
        178 => "OpULessThanEqual",
        194 => "OpShiftRightLogical",
        196 => "OpShiftLeftLogical",
        197 => "OpBitwiseOr",
        199 => "OpBitwiseAnd",
        OP_CONTROL_BARRIER => "OpControlBarrier",
        OP_ATOMIC_LOAD => "OpAtomicLoad",
        OP_ATOMIC_STORE => "OpAtomicStore",
        OP_PHI => "OpPhi",
        _ => return format!("opcode {opcode}"),
    };
    name.to_owned()
}

/// The shape of a function-body instruction, from the SPIR-V grammar
/// (`spirv.core.grammar.json` of the linked SPIRV-Tools): the instructions a
/// compute kernel's body is made of. Any other is not read, never guessed.
pub(crate) fn shape(opcode: u32) -> Option<Shape> {
    if let Some(operation) = subgroup_operation(opcode) {
        return Some(operation.shape);
    }
    let value = |ids| Some(Shape { result: true, ids });
    let effect = |ids| Some(Shape { result: false, ids });
    match opcode {
        // `OpUndef`; the access chains; dynamic vector access, construction,
        // copy and transposition; conversions; arithmetic; relational and
        // logical operations; bit operations; atomics with a result;
        // `OpCopyLogical` and pointer comparison; cooperative-matrix length
        // and replicated construction.
        OP_UNDEF | 65..=67 | 70 | 77 | 78 | 80 | 83 | 84 | 109..=122 | 124 | 126..=152 | 154..=191 | 194..=205
        | 400..=403 | 4460 | 4463 => value(Ids::All),
        opcode if atomic(opcode) => value(Ids::All),
        // `OpLoad`, `OpArrayLength`, `OpCompositeExtract`,
        // `OpGenericCastToPtrExplicit`.
        OP_LOAD | 68 | 81 | 123 => value(Ids::Leading(1)),
        // `OpVectorShuffle`, `OpCompositeInsert`, `OpSDot`, `OpUDot`,
        // `OpSUDot`.
        79 | 82 | 4450..=4452 => value(Ids::Leading(2)),
        // `OpSDotAccSat`, `OpUDotAccSat`, `OpSUDotAccSat`.
        4453..=4455 => value(Ids::Leading(3)),
        OP_EXT_INST => value(Ids::Except(1)),
        // The function, then the arguments: only a call of a hardware join
        // or a summarised function is left in the analysed copy
        // (`Summaries::of`).
        OP_FUNCTION_CALL | OP_VARIABLE => value(Ids::Except(0)),
        OP_PHI => value(Ids::Phi),
        // `OpControlBarrier`, `OpMemoryBarrier`, `OpAtomicStore`,
        // `OpAtomicFlagClear`, `OpBranch`, `OpKill`, `OpReturn`,
        // `OpReturnValue`, `OpUnreachable`, `OpTerminateInvocation`.
        OP_CONTROL_BARRIER | OP_MEMORY_BARRIER | OP_ATOMIC_STORE | OP_ATOMIC_FLAG_CLEAR | OP_BRANCH
        | OP_KILL..=OP_UNREACHABLE | OP_TERMINATE_INVOCATION => effect(Ids::All),
        OP_SELECTION_MERGE => effect(Ids::Leading(1)),
        // The selector and default of `OpSwitch`, whose targets are read
        // as the block's successors.
        OP_STORE | OP_COPY_MEMORY | OP_LOOP_MERGE | OP_SWITCH => effect(Ids::Leading(2)),
        OP_BRANCH_CONDITIONAL => effect(Ids::Leading(3)),
        _ => None,
    }
}

/// Whether `module` reads its subgroups: executes a subgroup operation, or
/// reads a subgroup built-in. Only such a module sees how invocations are
/// grouped into subgroups, so only it needs its subgroups full.
pub fn reads_subgroups(module: &[u32]) -> Result<bool, MalformedModule> {
    Ok(seal::instructions(module)?.iter().any(|instruction| {
        let words = &module[instruction.start..instruction.start + instruction.words];
        subgroup_operation(instruction.opcode).is_some()
            || (instruction.opcode == OP_DECORATE
                && words[2] == DECORATION_BUILT_IN
                && SUBGROUP_BUILT_INS.contains(&words[3]))
    }))
}

/// What [`check`] finds of a module's entry point.
#[derive(Debug, PartialEq, Eq)]
pub struct Findings {
    /// Where it breaks the execution model.
    pub violations: Vec<Violation>,
    /// It relies on the execution model's assumption: it executes a
    /// subgroup operation or a subgroup barrier where invocations a
    /// divergent selection, or an exit taken under divergent control,
    /// parted have rejoined (at or after the merge they meet at, before a
    /// legal workgroup barrier). Only such a module needs
    /// `SubgroupUniformControlFlowKHR` declared.
    pub relies_on_reconvergence: bool,
    /// It relies on maximal reconvergence: it executes a subgroup operation
    /// or a subgroup barrier after a lane-varying loop (outside divergent
    /// control, before a legal workgroup barrier), which only a module declaring
    /// `MaximallyReconvergesKHR` may, as there every invocation rejoins once
    /// the loop is left. Only a module checked as one that declares it
    /// ([`check`]) holds such an operation; the seal declares it exactly
    /// where this holds.
    pub relies_on_maximal_reconvergence: bool,
}

/// Checks the entry point of `module` against the execution model.
/// `module` is the analysis copy formation makes (`formation::analysed`),
/// with one entry point: a call left in it of any function but a hardware
/// join or a summarised function ([`summarised`]) is malformed. An instruction the
/// analysis does not read is refused, unanalysed. `maximal_reconvergence`:
/// the module is formed to declare `MaximallyReconvergesKHR` where it relies
/// on it, so a subgroup operation after a lane-varying loop is accepted
/// (nothing else changes: one inside such a loop, and any barrier, is
/// checked as without it).
pub fn check(module: &[u32], maximal_reconvergence: bool) -> Result<Findings, MalformedModule> {
    let program = Program::parse(module)?;
    if !program.unread.is_empty() {
        let violations = program
            .unread
            .iter()
            .map(|opcode| {
                Violation(format!(
                    "opcode {opcode}, which the execution-model check does not read, in `{}`",
                    program.entry
                ))
            })
            .collect();
        return Ok(Findings { violations, relies_on_reconvergence: false, relies_on_maximal_reconvergence: false });
    }
    let summaries = Summaries::of(module, &program, maximal_reconvergence)?;
    let prepared = Prepared::of(&program, true)?;
    let analysis = Analysis::established(&program, &prepared, &summaries, &[], Context::default())?;
    let (found, relies_on_maximal_reconvergence) = analysis.found()?;
    let violations = found
        .0
        .into_iter()
        .map(|(kind, (count, first))| Violation(format!("{count} × {kind} in `{}`, first {first}", program.entry)))
        .collect();
    Ok(Findings {
        violations,
        relies_on_reconvergence: analysis.rejoined()?.0,
        relies_on_maximal_reconvergence,
    })
}

/// What a call of a summarised function gives its caller, for the state of
/// its arguments ([`Summaries::summary`]): its body analysed as an entry
/// point whose parameters hold that state.
struct Summary {
    /// What the result holds where the body returns it.
    result: Held,
    /// A lane-varying loop of the body, with no legal workgroup barrier
    /// after it, comes before its return: the call ends after a
    /// lane-varying loop.
    loops: bool,
    /// Invocations that divergent control parted in the body rejoin before
    /// its return.
    rejoins: bool,
    /// The body reads workgroup memory, or device memory the launch writes
    /// (itself or through its calls): what it returns may differ between
    /// two calls with the same arguments, as between the iterations of a
    /// loop around the call.
    varies: bool,
    /// Where the body breaks the execution model, by kind.
    found: Found,
}

/// What the body of a summarised function sees of its caller's memory at
/// the call: the device memory the launch writes, and, by shared memory
/// (workgroup, written device), what a load of it at the call may see of
/// the caller's stores ([`Seen`], a race the caller's: [`Racer::Caller`]).
/// The body stores to no such memory and holds no barrier, so each of its
/// loads sees what a load at the call would: the inlined body's would see
/// the same.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
struct Context {
    written: Reach,
    seen: [Seen; 2],
}

/// The summarised functions ([`summarised`]) an entry point calls, and
/// what each call of each gives, for the state of its arguments and its
/// [`Context`]: each function analysed once per such state.
struct Summaries<'a> {
    /// By function: the function, read as an entry point is.
    programs: HashMap<u32, Program<'a>>,
    /// By function: precomputed flow, copies, constructs, and workgroup
    /// locations, computed once per summarised function.
    prepared: HashMap<u32, Prepared>,
    /// By function: whether it (or its callees) reads workgroup and/or device
    /// memory (`reads_classes`); a call sees only the classes it reads.
    reads: HashMap<u32, (bool, bool)>,
    /// By function: it holds a selection or a loop.
    steers: HashMap<u32, bool>,
    /// The module declares maximal reconvergence ([`check`]).
    maximal_reconvergence: bool,
    /// The summaries made so far, by function, arguments' state and context.
    #[allow(clippy::type_complexity)]
    made: std::cell::RefCell<HashMap<(u32, Vec<Held>, Context), std::rc::Rc<Summary>>>,
}

impl<'a> Summaries<'a> {
    /// The summarised functions `program`, the entry point of `module`,
    /// calls, through each other too. A call of any other function but a
    /// hardware join is malformed: formation inlines it.
    fn of(module: &'a [u32], program: &Program<'a>, maximal_reconvergence: bool) -> Result<Self, MalformedModule> {
        let pure = summarised(module)?;
        let mut programs = HashMap::new();
        let mut waiting = program.called.iter().copied().collect::<Vec<_>>();
        while let Some(function) = waiting.pop() {
            if programs.contains_key(&function) {
                continue;
            }
            if !pure.contains_key(&function) {
                return Err(MalformedModule(format!(
                    "%{function} is called once calls are inlined, but is not a function the check summarises"
                )));
            }
            let callee = Program::parse_at(module, function)?;
            if !callee.unread.is_empty() || callee.parameters.iter().any(|parameter| callee.pointer(*parameter).is_ok()) {
                return Err(MalformedModule(format!("%{function}, which the check summarises, is not pure")));
            }
            waiting.extend(callee.called.iter().copied());
            programs.insert(function, callee);
        }
        let holds = |function: &u32, opcodes: &[u32]| {
            programs[function].blocks.iter().flat_map(|block| &block.instructions).any(|words| opcodes.contains(&(words[0] & 0xffff)))
        };
        let steers =
            programs.keys().map(|function| (*function, holds(function, &[OP_SELECTION_MERGE, OP_LOOP_MERGE, OP_FUNCTION_CALL]))).collect();
        let reads = reads_classes(&programs);
        let prepared = programs
            .iter()
            .map(|(&function, program)| Ok((function, Prepared::of(program, false)?)))
            .collect::<Result<HashMap<_, _>, MalformedModule>>()?;
        Ok(Summaries { programs, prepared, reads, steers, maximal_reconvergence, made: Default::default() })
    }

    /// What a call of `function` with arguments of `arguments`, in
    /// `context`, gives.
    fn summary(&self, function: u32, arguments: Vec<Held>, context: Context) -> Result<std::rc::Rc<Summary>, MalformedModule> {
        let (wg, dev) = self.reads.get(&function).copied().unwrap_or((false, false));
        let masked = Context {
            written: if dev { context.written } else { Reach::NONE },
            seen: [
                if wg { context.seen[WORKGROUP] } else { Seen::default() },
                if dev { context.seen[WRITTEN] } else { Seen::default() },
            ],
        };
        let key = (function, arguments, masked);
        if let Some(made) = self.made.borrow().get(&key) {
            return Ok(made.clone());
        }
        let program = &self.programs[&function];
        if key.1.len() != program.parameters.len() {
            return Err(MalformedModule(format!("a call of %{function} passes {} arguments", key.1.len())));
        }
        let analysis = Analysis::established(program, &self.prepared[&function], self, &key.1, key.2)?;
        let (found, _) = analysis.found()?;
        let exits =
            (0..program.blocks.len()).filter(|block| program.blocks[*block].exits()).collect::<Vec<_>>();
        let after = analysis.after_loops()?;
        let mut loops = false;
        for &block in &exits {
            loops |= analysis.after_each(block, after[block])?.last().copied().flatten().is_some();
        }
        let result = exits
            .iter()
            .map(|&block| {
                let terminator = program.blocks[block].terminator();
                if terminator[0] & 0xffff == OP_RETURN_VALUE {
                    analysis.value(terminator[1], block)
                } else {
                    Held::UNIFORM
                }
            })
            .fold(Held::UNIFORM, |mut result, held| {
                result.join(&held);
                result
            });
        // As a loop's load of such memory differs between its iterations
        // ([`Analysis::variants`]).
        let mut varies = false;
        for &block in &analysis.flow.dominance.order {
            for (position, words) in program.blocks[block].instructions.iter().enumerate() {
                varies |= match read_pointer(words) {
                    Some(pointer) => match program.place(pointer)? {
                        Place::Workgroup => true,
                        Place::Device => !analysis.input(pointer)?,
                        Place::Variable(_) | Place::ReadOnly | Place::BuiltIn(_) => false,
                    },
                    None => analysis.summary(words, (block, position))?.is_some_and(|summary| summary.varies),
                };
            }
        }
        let summary = std::rc::Rc::new(Summary { result, loops, rejoins: analysis.rejoined()?.1, varies, found });
        self.made.borrow_mut().insert(key, summary.clone());
        Ok(summary)
    }
}

/// A lattice value the analysis raises to a fixed point: it only grows, by `join`.
trait Join: Copy + PartialEq {
    fn join(self, other: Self) -> Self;
}

impl Join for bool {
    fn join(self, other: bool) -> bool {
        self || other
    }
}

/// How many constructs a refusal follows back from what it refuses, each
/// steered by a condition that differs because control parted before it
/// ([`Origin::Rejoined`]).
const ORIGIN_DEPTH: usize = 4;

/// An instruction of the entry point: its block, and its index there.
type At = (usize, usize);

/// A store that may race a load: one of the function analysed, or, in a
/// summarised function's body, one of its caller that may race the call
/// ([`Context`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Racer {
    Store(At),
    Caller,
}

/// A store that may race a load, once one is found: the first found stays,
/// so a load names one.
impl Join for Option<Racer> {
    fn join(self, other: Option<Racer>) -> Option<Racer> {
        self.or(other)
    }
}

/// What a load of shared memory (workgroup, written device) at an
/// instruction may see of its workgroup's stores to it ([`Analysis::seen`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
struct Seen {
    /// A store that may race it: one in its barrier interval, which may run
    /// beside it; in written device memory, which no barrier here orders,
    /// also one that may run before it.
    raced: Option<Racer>,
    /// What the stores it may see keep ([`ValueState::kept`]): those in its
    /// barrier interval and those that may run before it. A store that can
    /// run neither (on a path no invocation of the workgroup takes to it)
    /// keeps nothing it reads.
    kept: ValueState,
}

impl Join for Seen {
    fn join(self, other: Seen) -> Seen {
        Seen { raced: self.raced.join(other.raced), kept: self.kept.join(other.kept) }
    }
}

/// What places workgroup memory's bytes ([`Analysis::location`]): the
/// bytes of scalar and vector types, and the explicit layout's strides and
/// offsets, which a type without them takes from its members' bytes.
#[derive(Default)]
struct Layout {
    /// Scalar and vector types → their bytes.
    bytes: HashMap<u32, u64>,
    /// Array types → their `ArrayStride`.
    strides: HashMap<u32, u64>,
    /// Struct types and members → the member's `Offset`.
    offsets: HashMap<(u32, u32), u64>,
    /// Struct types decorated `Block`: a workgroup variable of one is a view
    /// of the workgroup's explicitly laid out memory, which every such view
    /// shares from its first byte (`WorkgroupMemoryExplicitLayoutKHR`).
    blocks: HashSet<u32>,
}

/// Workgroup memory an access may land in, as far as its address tells:
/// the views of the explicitly laid out memory, all one memory, or one
/// variable of its own.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Region {
    Views,
    Variable(u32),
}

/// The bytes of workgroup memory an access lands on, `start..end` of a
/// region, where its address is a constant offset into it; none where it is
/// not, which may be anywhere.
type Location = Option<(Region, u64, u64)>;

/// Two accesses of workgroup memory may touch one byte.
fn overlap(first: Location, second: Location) -> bool {
    match (first, second) {
        (Some((region, start, end)), Some((other, other_start, other_end))) => {
            region == other && start < other_end && other_start < end
        }
        _ => true,
    }
}

/// Stores of shared memory, each joined where it lands ([`Location`]), so a
/// load sees those that may touch its own bytes alone.
#[derive(Clone, Debug, Default, PartialEq)]
struct Stores(BTreeMap<Location, Seen>);

impl Stores {
    /// Add `seen`, of stores landing at `location`.
    fn add(&mut self, location: Location, seen: Seen) {
        let at = self.0.entry(location).or_default();
        *at = at.join(seen);
    }

    /// Add every store of `other`; whether anything rose.
    fn raise(&mut self, other: &Stores) -> bool {
        let before = self.clone();
        for (location, seen) in &other.0 {
            self.add(*location, *seen);
        }
        *self != before
    }

    /// What a load at `location` sees of them.
    fn seen(&self, location: Location) -> Seen {
        self.0
            .iter()
            .filter(|(stored, _)| overlap(**stored, location))
            .fold(Seen::default(), |seen, (_, more)| seen.join(*more))
    }
}

/// How much a value may differ between the invocations of a subgroup, and
/// which of the two unrelated orders of a workgroup's invocations it derives
/// from: their local ids, or their lanes in subgroups (which subgroup, and
/// which lane in it). A set of the bits below: every
/// value is made by joining them, so none holds a bit without the others of
/// its kind (a local value is divergent, a lane's id varies).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
struct Taint(u8);

/// The bits of a taint a refusal traces to where each was first raised
/// ([`Analysis::origins`]), by index; [`Taint::SCATTERED`], which none
/// traces, follows them.
#[derive(Clone, Copy)]
enum Traced {
    /// May differ between the invocations of a subgroup.
    Varies,
    /// Derived from the local or global invocation id.
    Local,
    /// Derived from which subgroup an invocation is in.
    Subgroup,
    /// Derived from an invocation's lane in its subgroup.
    Lane,
}

impl Traced {
    const ALL: [Traced; 4] = [Traced::Varies, Traced::Local, Traced::Subgroup, Traced::Lane];

    /// Its bit of a taint.
    const fn bit(self) -> u8 {
        1 << self as u8
    }
}

impl Taint {
    const UNIFORM: Taint = Taint(0);
    /// May differ between the invocations of a subgroup.
    const VARIES: Taint = Taint(Traced::Varies.bit());
    /// May differ between the invocations of a subgroup other than as a
    /// function of their lanes in it, which a full subgroup holds alike
    /// whichever invocations the device groups in it.
    const SCATTERED: Taint = Taint(1 << Traced::ALL.len());
    /// May differ between the invocations of a subgroup in any way.
    const DIVERGENT: Taint = Taint(Taint::VARIES.0 | Taint::SCATTERED.0);
    /// Divergent, and derived from the local or global invocation id.
    const LOCAL: Taint = Taint(Taint::DIVERGENT.0 | Traced::Local.bit());
    /// Derived from which subgroup an invocation is in: alike in a subgroup,
    /// but which invocations share it the device decides.
    const SUBGROUP: Taint = Taint(Traced::Subgroup.bit());
    /// Derived from an invocation's lane in its subgroup, and varying by it.
    const LANE: Taint = Taint(Taint::VARIES.0 | Traced::Lane.bit());

    fn varies(self) -> bool {
        self.holds(Taint::VARIES)
    }

    /// Holds every bit `other` does.
    fn holds(self, other: Taint) -> bool {
        self.0 & other.0 == other.0
    }

    /// Derives from the lane order of subgroups: which subgroup an
    /// invocation is in, or its lane there.
    fn lane_ordered(self) -> bool {
        self.0 & (Traced::Subgroup.bit() | Traced::Lane.bit()) != 0
    }

    /// Derives from both the local id and the lane order.
    fn mixes(self) -> bool {
        self.holds(Taint::LOCAL) && self.lane_ordered()
    }

    /// What one value for a whole subgroup combined from values of this
    /// taint (a reduction, vote, ballot, broadcast) holds: which subgroup
    /// combined it shows where they derive from the subgroup, or differ
    /// otherwise than by lane; where they differ by lane alone, every full
    /// subgroup combines the same values.
    fn gathered(self) -> Taint {
        if self.0 & (Taint::SUBGROUP.0 | Taint::SCATTERED.0) != 0 {
            Taint::SUBGROUP
        } else {
            Taint::UNIFORM
        }
    }
}

impl Join for Taint {
    fn join(self, other: Taint) -> Taint {
        Taint(self.0 | other.0)
    }
}

/// Memory of the launch: buffers, by how their addresses are read.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
struct Reach {
    /// Buffers the launch only reads by declaration: read through the
    /// [`INPUT_BLOCK`].
    input: bool,
    /// The [`ARGUMENT_BLOCK`] words, by index, of the buffers it may be.
    words: u128,
    /// The [`ADDRESS_TABLE_BLOCK`] words, by index, of the address tables it
    /// may be, which the launch only reads.
    address_tables: u128,
    /// The address tables, by word, into whose ports' bytes it may point:
    /// what an address read from one of them addresses.
    regions: u128,
    /// Any memory.
    anywhere: bool,
}

impl Reach {
    const NONE: Reach = Reach { input: false, words: 0, address_tables: 0, regions: 0, anywhere: false };
    const INPUT: Reach = Reach { input: true, ..Reach::NONE };
    const ANYWHERE: Reach = Reach { anywhere: true, ..Reach::NONE };

    fn join(self, other: Reach) -> Reach {
        Reach {
            input: self.input || other.input,
            words: self.words | other.words,
            address_tables: self.address_tables | other.address_tables,
            regions: self.regions | other.regions,
            anywhere: self.anywhere || other.anywhere,
        }
    }

    fn none(self) -> bool {
        self == Reach::NONE
    }

    /// This memory is none that `written` names. A buffer read through the
    /// [`INPUT_BLOCK`], and an address table, is none the launch writes,
    /// wherever it writes: a write through either is refused, and the
    /// runtime keeps every buffer a kernel writes, and every port's bytes its
    /// table addresses, apart from every other.
    fn apart(self, written: Reach) -> bool {
        let writable = self.words | self.regions;
        !self.anywhere
            && self.words & written.words == 0
            && self.regions & written.regions == 0
            && (!written.anywhere || writable == 0)
    }

    /// What memory of this reach holds addresses into: a `&` buffer's,
    /// memory the launch only reads by declaration (a `&` slab view's or
    /// record port's bytes, which its table reaches); an address table's, its
    /// port's bytes. What any other memory holds may point anywhere, whether
    /// the launch writes it or only reads it (a buffer whose argument word
    /// nothing stores through, a port's bytes).
    fn holding(self) -> Reach {
        if self.anywhere || self.words != 0 || self.regions != 0 {
            Reach::ANYWHERE
        } else {
            Reach { input: self.input, regions: self.address_tables, ..Reach::NONE }
        }
    }
}

/// Where the addresses a value is derived from point.
///
/// A buffer's address is read from the argument block into an invocation's
/// registers: a value computed from it, by any operation, names its buffer
/// (`names`), so a delta between two addresses names both, and so does an
/// address rebuilt from one. Such a value is never stored to workgroup or
/// device memory (refused, [`Analysis::violations`]), so what memory holds
/// is no address the launch was given: a value read from it (`loaded`) is
/// an offset where it is added to a named address ([`Provenance::sum`]):
/// every kernel addresses a buffer from its own address (`SEISMIC_PTR`, an
/// `Out`) at offsets within it. Otherwise it is a base that may point into
/// what the memory read may hold addresses into ([`Reach::holding`]): what
/// memory a buffer the launch only reads by declaration holds points into
/// such memory (a `&` slab view's or record port's table); what an address
/// table of a port the launch writes through holds, into that port's bytes
/// (the runtime holds either apart as the view's own bytes,
/// `api::exclusive_pairs`); an address read from any other memory may
/// point anywhere. A function or private variable is the invocation's own,
/// as its registers are: what it holds keeps its names. A condition holds
/// none ([`ValueState::condition`]).
///
/// The default is what nothing is known of yet, which the analysis raises
/// from; a known value derived from no address and read from no memory
/// (constants, ids, the launch's words) is [`Provenance::PLAIN`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
struct Provenance {
    /// Something is known of it.
    known: bool,
    /// The buffers whose addresses, as the argument block gives them, it is
    /// computed from; `anywhere` for a word the analysis cannot name.
    names: Reach,
    /// The memory values read from memory that it is computed from may
    /// address.
    loaded: Reach,
}

impl Provenance {
    /// Derived from no address, and from no value read from memory.
    const PLAIN: Provenance = Provenance { known: true, names: Reach::NONE, loaded: Reach::NONE };

    /// A value read from memory that may hold addresses into `reach`.
    const fn loaded(reach: Reach) -> Provenance {
        Provenance { known: true, names: Reach::NONE, loaded: reach }
    }

    /// A buffer's address, as the argument block gives it.
    const fn named(reach: Reach) -> Provenance {
        Provenance { known: true, names: reach, loaded: Reach::NONE }
    }

    fn join(self, other: Provenance) -> Provenance {
        Provenance {
            known: self.known || other.known,
            names: self.names.join(other.names),
            loaded: self.loaded.join(other.loaded),
        }
    }

    /// It is computed from a buffer's address.
    fn is_named(self) -> bool {
        !self.names.none()
    }

    /// The sum or difference of `self` and `other` (`OpIAdd`, `OpISub`):
    /// both operands' names, and what either has read from memory but where
    /// the other is named, which makes it an offset. One of which nothing
    /// is known yet may still be named, so the sum is not known before both
    /// are. Which operand is named can change as the analysis raises them,
    /// but a value keeps every answer it was given (`Held::raise`), so a sum
    /// names at least the memory its final operands give it.
    fn sum(self, other: Provenance) -> Provenance {
        let offset = |of: Provenance, to: Provenance| if to.is_named() { Reach::NONE } else { of.loaded };
        Provenance {
            known: self.known && other.known,
            names: self.names.join(other.names),
            loaded: if self.known && other.known { offset(self, other).join(offset(other, self)) } else { Reach::NONE },
        }
    }

    /// A pointer of this provenance indexed by `index`, an offset into what
    /// it points to: its names, but not what it has read from memory.
    fn indexed(self, index: Provenance) -> Provenance {
        Provenance { names: self.names.join(index.names), ..self }
    }

    /// What memory shared between invocations keeps of a value of this
    /// provenance, read back: a value read from memory, which may hold
    /// addresses into what it named or read.
    fn read(self) -> Provenance {
        Provenance { known: self.known, names: Reach::NONE, loaded: self.loaded.join(self.names) }
    }

    /// The memory a pointer of this provenance addresses: one derived from
    /// no address (a descriptor's, or one made of constants and ids) may be
    /// any; one of which nothing is known yet, none yet.
    fn attributed(self) -> Reach {
        let reach = self.names.join(self.loaded);
        if self.known && reach.none() {
            Reach::ANYWHERE
        } else {
            reach
        }
    }
}

/// What is known of a value.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
struct ValueState {
    taint: Taint,
    provenance: Provenance,
}

impl ValueState {
    /// A value of `taint`, of which nothing else is known yet.
    const fn of(taint: Taint) -> ValueState {
        ValueState { taint, provenance: Provenance { known: false, names: Reach::NONE, loaded: Reach::NONE } }
    }

    /// A value of `taint` derived from no address and no memory.
    const fn plain(taint: Taint) -> ValueState {
        ValueState { taint, provenance: Provenance::PLAIN }
    }

    /// What memory shared between invocations keeps of a value placed in
    /// it ([`Analysis::placing`]) through an address of taint `address`:
    /// whether it derives from the local id and from the lane order beyond
    /// what the address does, and where it points. Its divergence is the
    /// read's. A byte holds its one writer's value, and which invocation
    /// that is the address decides: a taint bit the address holds too leaves the
    /// value a function of the byte, whichever invocation wrote it (a store
    /// staging `x[t]` at `y[t]`). A value placed by one order and deriving
    /// from the other is refused where it is stored ([`MIXING`]).
    fn kept(self, address: Taint) -> ValueState {
        let taint = [Taint::LOCAL, Taint::SUBGROUP, Taint::LANE]
            .into_iter()
            .filter(|bit| self.taint.holds(*bit) && !address.holds(*bit))
            .fold(Taint::UNIFORM, Taint::join);
        ValueState { taint, provenance: self.provenance.read() }
    }

    /// What a condition (a boolean) of this state holds: its taint, but no
    /// address, nor anything read from memory that one could address. One
    /// bit chooses; it points nowhere.
    fn condition(self) -> ValueState {
        ValueState { taint: self.taint, provenance: Provenance { known: self.provenance.known, ..Provenance::default() } }
    }
}

impl Join for ValueState {
    fn join(self, other: ValueState) -> ValueState {
        ValueState { taint: self.taint.join(other.taint), provenance: self.provenance.join(other.provenance) }
    }
}

/// What is known of the bits of a value, to find the edges no invocation of
/// the module can take ([`Flow::live`]): nothing yet (not yet reached),
/// one value (zero-extended from its type's width), or any. An undefined
/// value is any: each invocation may see different bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Constant {
    #[default]
    Unknown,
    Value(u64),
    Varying,
}

impl Join for Constant {
    fn join(self, other: Constant) -> Constant {
        match (self, other) {
            (Constant::Unknown, known) | (known, Constant::Unknown) => known,
            (Constant::Value(mine), Constant::Value(theirs)) if mine == theirs => Constant::Value(mine),
            _ => Constant::Varying,
        }
    }
}

/// What a value, or what a variable holds, is made of: a struct's members
/// apart, anything else whole. An array, vector or matrix is its elements'
/// summary, what any of them holds ([`Program::members`]), so a struct
/// inside one keeps its members apart.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Shaped<T> {
    Whole(T),
    Members(Vec<Shaped<T>>),
}

/// The taint and provenance of a value ([`ValueState`]), by member.
type Held = Shaped<ValueState>;
/// The bits of a value ([`Constant`]), by member.
type Known = Shaped<Constant>;

impl<T: Join + Default> Shaped<T> {
    /// Every member joined.
    fn leaf(&self) -> T {
        match self {
            Shaped::Whole(leaf) => *leaf,
            Shaped::Members(members) => members.iter().map(Shaped::leaf).fold(T::default(), T::join),
        }
    }

    fn join(&mut self, other: &Self) {
        match (&mut *self, other) {
            (Shaped::Whole(mine), Shaped::Whole(theirs)) => *mine = mine.join(*theirs),
            (Shaped::Members(mine), Shaped::Members(theirs)) => {
                for (mine, theirs) in mine.iter_mut().zip(theirs) {
                    mine.join(theirs);
                }
            }
            (Shaped::Members(mine), Shaped::Whole(_)) => {
                for mine in mine {
                    mine.join(other);
                }
            }
            (Shaped::Whole(mine), Shaped::Members(theirs)) => {
                let mine = Shaped::Whole(*mine);
                *self = Shaped::Members(
                    theirs
                        .iter()
                        .map(|theirs| {
                            let mut member = theirs.clone();
                            member.join(&mine);
                            member
                        })
                        .collect(),
                );
            }
        }
    }

    /// Raise to `value`; whether anything rose.
    fn raise(&mut self, value: &Self) -> bool {
        match (&mut *self, value) {
            (Shaped::Whole(mine), Shaped::Whole(theirs)) => {
                let joined = mine.join(*theirs);
                let rose = joined != *mine;
                *mine = joined;
                rose
            }
            (Shaped::Members(mine), Shaped::Members(theirs)) => {
                let mut rose = false;
                for (mine, theirs) in mine.iter_mut().zip(theirs) {
                    rose |= mine.raise(theirs);
                }
                rose
            }
            (Shaped::Members(mine), Shaped::Whole(_)) => {
                let mut rose = false;
                for mine in mine {
                    rose |= mine.raise(value);
                }
                rose
            }
            (Shaped::Whole(_), Shaped::Members(_)) => {
                self.join(value);
                true
            }
        }
    }

    /// The member at `path`; past a whole value, that value.
    fn get(&self, path: impl IntoIterator<Item = usize>) -> Self {
        let mut held = self;
        for index in path {
            match held {
                Shaped::Members(members) => held = &members[index],
                Shaped::Whole(_) => break,
            }
        }
        held.clone()
    }

    /// Write `value` to the member at `path` (each step a member and its
    /// struct's member count): replacing it where `whole`, joined into it,
    /// member by member, where the write reaches only some of what it
    /// summarizes (an element of an array, a component of a vector).
    fn write(&mut self, path: &[(usize, usize)], value: Self, whole: bool) {
        match path.split_first() {
            None if whole => *self = value,
            None => self.join(&value),
            Some(((member, count), rest)) => {
                if let Shaped::Whole(leaf) = *self {
                    *self = Shaped::Members(vec![Shaped::Whole(leaf); *count]);
                }
                let Shaped::Members(members) = self else { unreachable!("a whole value was just split") };
                members[*member].write(rest, value, whole);
            }
        }
    }
}

impl Held {
    /// Nothing yet: what the analysis raises from, and what an undefined
    /// value (`OpUndef`, a variable before it is written) holds, which adds
    /// nothing to what it joins.
    const UNIFORM: Held = Shaped::Whole(ValueState::of(Taint::UNIFORM));
    /// A uniform value derived from no address.
    const CONSTANT: Held = Shaped::Whole(ValueState::plain(Taint::UNIFORM));

    fn state(&self) -> ValueState {
        self.leaf()
    }

    fn taint(&self) -> Taint {
        self.state().taint
    }

    fn joined(mut self, taint: Taint) -> Held {
        if taint != Taint::UNIFORM {
            self.join(&Shaped::Whole(ValueState::of(taint)));
        }
        self
    }
}

impl Known {
    const UNKNOWN: Known = Shaped::Whole(Constant::Unknown);
    const VARYING: Known = Shaped::Whole(Constant::Varying);

    /// What a value held whole is, as one constant; one whose members are
    /// held apart is any value as one.
    fn value(&self) -> Constant {
        match self {
            Shaped::Whole(constant) => *constant,
            Shaped::Members(_) => Constant::Varying,
        }
    }
}

#[derive(Clone, Copy)]
enum Merge {
    Selection { merge: usize },
    Loop { merge: usize, continue_target: usize },
}

pub(crate) struct Block<'a> {
    /// Its label's id.
    label: u32,
    /// The instructions after the label, the terminator last.
    pub(crate) instructions: Vec<&'a [u32]>,
    /// By instruction: a control barrier whose scopes hold the workgroup
    /// and whose semantics, with those of the memory barriers before it in
    /// the block, order workgroup memory.
    fences: Vec<bool>,
    merge: Option<Merge>,
    /// Distinct, ascending.
    successors: Vec<usize>,
    /// A switch's cases: each literal, zero-extended, and its target.
    cases: Vec<(u64, usize)>,
}

impl Block<'_> {
    fn terminator(&self) -> &[u32] {
        self.instructions.last().copied().expect("a parsed block ends in its terminator")
    }

    /// The block leaves the entry point.
    fn exits(&self) -> bool {
        EXITS.contains(&(self.terminator()[0] & 0xffff))
    }
}

/// A pointer derived from another.
struct Chain<'a> {
    base: u32,
    /// Indices into what the base points to.
    indices: &'a [u32],
    /// The pointer steps over elements of the base first
    /// (`OpPtrAccessChain`), so it selects no member of a variable whole.
    element: bool,
}

/// What a pointer into a variable selects of it.
pub(crate) struct Selection {
    root: u32,
    /// The struct members selected, each with its struct's member count:
    /// an index into an array steps to its elements' summary, past which the
    /// path goes on into the element's members.
    path: Vec<(usize, usize)>,
    /// What is selected is all of what it is held as: no index selects one
    /// of several elements of an array, nor a component of a vector or
    /// matrix.
    whole: bool,
}

/// What a pointer addresses. A read of [`Place::ReadOnly`] or
/// [`Place::BuiltIn`] memory reads what nothing in the launch writes: the
/// one test of that, for the analysis and the value numbering
/// (`numbering`) alike.
pub(crate) enum Place {
    /// A function or private variable, each invocation's own.
    Variable(Selection),
    /// Workgroup memory, which the invocations of a workgroup share.
    Workgroup,
    /// Storage-buffer and physical-storage-buffer memory, and any other
    /// memory invocations may share.
    Device,
    /// Uniform (but a `BufferBlock`'s, a storage buffer), uniform-constant
    /// and push-constant memory.
    ReadOnly,
    /// An input variable: the built-in it holds.
    BuiltIn(Option<u32>),
}

/// A block type the ABI reads the launch's arguments, or the device's
/// tables, through.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ArgumentBlock {
    /// [`ARGUMENT_BLOCK`]: the addresses of the buffers the launch may write.
    Argument,
    /// [`INPUT_BLOCK`]: the addresses of the buffers the launch only reads by
    /// declaration, and past the addresses, the launch's words.
    Input,
    /// [`ADDRESS_TABLE_BLOCK`]: the addresses of the address tables of the
    /// slab and record ports the launch may write through.
    AddressTable,
    /// [`TABLES_BLOCK`]: the device's tables, which hold no addresses.
    Tables,
}

/// The entry point of a module and the declarations it reads.
pub(crate) struct Program<'a> {
    /// The function's name, which violations name as their holder.
    entry: String,
    /// The entry point's function.
    pub(crate) function: u32,
    /// The entry point's blocks, its first block first.
    pub(crate) blocks: Vec<Block<'a>>,
    labels: HashMap<u32, usize>,
    /// The block and instruction of each result of the entry point.
    pub(crate) defined: HashMap<u32, (usize, &'a [u32])>,
    types: HashMap<u32, u32>,
    /// Pointer type → storage class and pointee type.
    pointers: HashMap<u32, (u32, u32)>,
    /// Struct type → member types.
    structs: HashMap<u32, Vec<u32>>,
    /// Array type → element type, and whether it holds one element.
    arrays: HashMap<u32, (u32, bool)>,
    /// Integer scalar and vector types → width.
    ints: HashMap<u32, u32>,
    /// Boolean scalar and vector types.
    booleans: HashSet<u32>,
    /// Constants (not specialization constants, which pipeline creation
    /// sets after the check) → their bits.
    known: HashMap<u32, Known>,
    /// `OpName`s but empty ones, which name nothing (glslang gives an
    /// anonymous block's instance one).
    names: HashMap<u32, String>,
    /// Variable → the built-in it holds.
    built_ins: HashMap<u32, u32>,
    /// Struct types decorated `BufferBlock`: uniform memory of one is a
    /// storage buffer.
    buffer_blocks: HashSet<u32>,
    /// Where workgroup memory's bytes lie ([`Program::location`]).
    layout: Layout,
    /// Module-scope variables and the entry point's.
    variables: HashSet<u32>,
    /// Module-scope variables with an initializer, a constant.
    initialized: HashSet<u32>,
    chains: HashMap<u32, Chain<'a>>,
    /// The entry point's phis → their incoming values, each with its parent
    /// block.
    phi_incoming: HashMap<u32, Vec<(u32, usize)>>,
    /// Opcodes in the entry point the analysis does not read.
    pub(crate) unread: BTreeSet<u32>,
    /// The [`ARGUMENT_BLOCK`], [`INPUT_BLOCK`], [`ADDRESS_TABLE_BLOCK`] and
    /// [`TABLES_BLOCK`] types → which each is.
    argument_blocks: HashMap<u32, ArgumentBlock>,
    /// The hardware joins the entry point calls → their names.
    joins: HashMap<u32, String>,
    /// Module-scope `OpUndef`s.
    undefined: Vec<u32>,
    /// The function's parameters, in order: none for an entry point.
    parameters: Vec<u32>,
    /// The functions it calls but hardware joins: summarised functions
    /// ([`summarised`]).
    pub(crate) called: BTreeSet<u32>,
    bound: usize,
}

/// The function of `module`'s one entry point.
fn entry_point(module: &[u32]) -> Result<u32, MalformedModule> {
    let entries = seal::instructions(module)?
        .iter()
        .filter(|instruction| instruction.opcode == OP_ENTRY_POINT)
        .map(|instruction| module[instruction.start + 2])
        .collect::<Vec<_>>();
    match entries[..] {
        [entry] => Ok(entry),
        [] => Err(MalformedModule("no entry point".into())),
        _ => Err(MalformedModule("more than one entry point".into())),
    }
}

impl<'a> Program<'a> {
    /// The entry point of `module`, its one entry point.
    pub(crate) fn parse(module: &'a [u32]) -> Result<Self, MalformedModule> {
        Self::parse_at(module, entry_point(module)?)
    }

    /// The function `entry` of `module`, read as the analysis reads an entry
    /// point: a summarised function ([`summarised`]) is analysed so, its
    /// parameters values the call gives it.
    pub(crate) fn parse_at(module: &'a [u32], entry: u32) -> Result<Self, MalformedModule> {
        let instructions = seal::instructions(module)?;
        let words_of = |instruction: &Instruction| -> &'a [u32] {
            &module[instruction.start..instruction.start + instruction.words]
        };
        let mut program = Program {
            parameters: Vec::new(),
            called: BTreeSet::new(),
            entry: String::new(),
            function: entry,
            blocks: Vec::new(),
            labels: HashMap::new(),
            defined: HashMap::new(),
            types: HashMap::new(),
            pointers: HashMap::new(),
            structs: HashMap::new(),
            arrays: HashMap::new(),
            ints: HashMap::new(),
            booleans: HashSet::new(),
            known: HashMap::new(),
            names: HashMap::new(),
            built_ins: HashMap::new(),
            buffer_blocks: HashSet::new(),
            layout: Layout::default(),
            variables: HashSet::new(),
            initialized: HashSet::new(),
            chains: HashMap::new(),
            phi_incoming: HashMap::new(),
            unread: BTreeSet::new(),
            argument_blocks: HashMap::new(),
            joins: HashMap::new(),
            undefined: Vec::new(),
            bound: module[3] as usize,
        };
        // Each block's label and instructions.
        let mut blocks: Vec<(u32, Vec<&'a [u32]>)> = Vec::new();
        let mut in_entry = false;
        // A block whose terminator is still to come.
        let mut open = false;
        for instruction in &instructions {
            let words = words_of(instruction);
            let opcode = instruction.opcode;
            if !in_entry {
                match opcode {
                    OP_NAME => {
                        let name = seal::literal_string(&words[2..]);
                        if !name.is_empty() {
                            program.names.insert(words[1], name);
                        }
                    }
                    OP_TYPE_BOOL => {
                        program.booleans.insert(words[1]);
                    }
                    OP_TYPE_INT => {
                        program.ints.insert(words[1], words[2]);
                        program.layout.bytes.insert(words[1], u64::from(words[2] / 8));
                    }
                    OP_TYPE_FLOAT => {
                        program.layout.bytes.insert(words[1], u64::from(words[2] / 8));
                    }
                    // A vector is held as its components' summary, of their
                    // type.
                    OP_TYPE_VECTOR => {
                        if program.booleans.contains(&words[2]) {
                            program.booleans.insert(words[1]);
                        }
                        if let Some(&int) = program.ints.get(&words[2]) {
                            program.ints.insert(words[1], int);
                        }
                        if let Some(&bytes) = program.layout.bytes.get(&words[2]) {
                            program.layout.bytes.insert(words[1], bytes * u64::from(words[3]));
                        }
                    }
                    OP_DECORATE if words[2] == DECORATION_BLOCK => {
                        program.layout.blocks.insert(words[1]);
                    }
                    OP_DECORATE if words[2] == DECORATION_ARRAY_STRIDE => {
                        program.layout.strides.insert(words[1], u64::from(words[3]));
                    }
                    OP_MEMBER_DECORATE if words[3] == DECORATION_OFFSET => {
                        program.layout.offsets.insert((words[1], words[2]), u64::from(words[4]));
                    }
                    OP_TYPE_ARRAY => {
                        program.arrays.insert(words[1], (words[2], program.constant(words[3]) == Some(1)));
                    }
                    OP_TYPE_RUNTIME_ARRAY => {
                        program.arrays.insert(words[1], (words[2], false));
                    }
                    OP_TYPE_STRUCT => {
                        program.structs.insert(words[1], words[2..].to_vec());
                    }
                    OP_TYPE_POINTER => {
                        program.pointers.insert(words[1], (words[2], words[3]));
                    }
                    OP_DECORATE if words[2] == DECORATION_BUILT_IN => {
                        program.built_ins.insert(words[1], words[3]);
                    }
                    OP_DECORATE if words[2] == DECORATION_BUFFER_BLOCK => {
                        program.buffer_blocks.insert(words[1]);
                    }
                    OP_FUNCTION => in_entry = words[2] == entry,
                    opcode if typed_declaration(opcode) => {
                        program.types.insert(words[2], words[1]);
                        if opcode == OP_VARIABLE {
                            program.variables.insert(words[2]);
                            if words.len() > 4 {
                                program.initialized.insert(words[2]);
                            }
                        }
                        if opcode == OP_UNDEF {
                            program.undefined.push(words[2]);
                        }
                        if let Some(known) = program.declared(opcode, words) {
                            program.known.insert(words[2], known);
                        }
                    }
                    _ => {}
                }
                continue;
            }
            match opcode {
                // Debug line information, which may stand anywhere.
                OP_LINE | OP_NO_LINE => {}
                OP_FUNCTION_END => {
                    if open {
                        return Err(MalformedModule("the entry point's last block has no terminator".into()));
                    }
                    in_entry = false;
                }
                OP_LABEL => {
                    if open {
                        return Err(MalformedModule(format!("the block before %{} has no terminator", words[1])));
                    }
                    blocks.push((words[1], Vec::new()));
                    open = true;
                }
                // The function's parameters precede its first block.
                OP_FUNCTION_PARAMETER if blocks.is_empty() => {
                    program.types.insert(words[2], words[1]);
                    program.parameters.push(words[2]);
                }
                _ if blocks.is_empty() => {}
                _ if !open => {
                    return Err(MalformedModule(format!("opcode {opcode} follows a block's terminator")));
                }
                _ => {
                    let block = blocks.len() - 1;
                    match shape(opcode) {
                        Some(shape) if shape.result => {
                            program.types.insert(words[2], words[1]);
                            program.defined.insert(words[2], (block, words));
                        }
                        Some(_) => {}
                        None => {
                            program.unread.insert(opcode);
                        }
                    }
                    if opcode == OP_VARIABLE {
                        program.variables.insert(words[2]);
                    }
                    // Formation inlines every call but a hardware join's and a
                    // summarised function's ([`summarised`]), which `check`
                    // holds it to.
                    if opcode == OP_FUNCTION_CALL {
                        match program.names.get(&words[3]).filter(|name| name.starts_with(HARDWARE_JOIN)) {
                            Some(name) => {
                                program.joins.insert(words[3], name.clone());
                            }
                            None => {
                                program.called.insert(words[3]);
                            }
                        }
                    }
                    if ACCESS_CHAINS.contains(&opcode) || ELEMENT_CHAINS.contains(&opcode) || opcode == OP_COPY_OBJECT
                    {
                        let element = ELEMENT_CHAINS.contains(&opcode);
                        let indices = if opcode == OP_COPY_OBJECT { &[][..] } else { &words[4 + element as usize..] };
                        program.chains.insert(words[2], Chain { base: words[3], indices, element });
                    }
                    blocks[block].1.push(words);
                    open = !TERMINATORS.contains(&opcode);
                }
            }
        }
        if blocks.is_empty() {
            return Err(MalformedModule("the entry point has no body".into()));
        }
        program.entry = program.names.get(&entry).cloned().unwrap_or_else(|| format!("%{entry}"));
        for (id, name) in &program.names {
            let block = match name.as_str() {
                ARGUMENT_BLOCK => ArgumentBlock::Argument,
                INPUT_BLOCK => ArgumentBlock::Input,
                ADDRESS_TABLE_BLOCK => ArgumentBlock::AddressTable,
                TABLES_BLOCK => ArgumentBlock::Tables,
                _ => continue,
            };
            if program.structs.contains_key(id) {
                program.argument_blocks.insert(*id, block);
            }
        }
        program.labels = blocks.iter().enumerate().map(|(index, (label, _))| (*label, index)).collect();
        let label = |id: u32| {
            program
                .labels
                .get(&id)
                .copied()
                .ok_or_else(|| MalformedModule(format!("%{id} is not a block of the entry point")))
        };
        let mut parsed = Vec::with_capacity(blocks.len());
        let mut phi_incoming = HashMap::new();
        for (block_label, instructions) in blocks {
            let mut merge = None;
            // The semantics of the memory barriers since the last control
            // barrier.
            let mut pending = 0;
            let mut fences = Vec::with_capacity(instructions.len());
            for words in &instructions {
                let constant = |id: u32| program.constant(id);
                fences.push(match words[0] & 0xffff {
                    OP_SELECTION_MERGE => {
                        merge = Some(Merge::Selection { merge: label(words[1])? });
                        false
                    }
                    OP_LOOP_MERGE => {
                        merge = Some(Merge::Loop {
                            merge: label(words[1])?,
                            continue_target: label(words[2])?,
                        });
                        false
                    }
                    OP_PHI => {
                        let incoming =
                            words[3..].chunks_exact(2).map(|pair| label(pair[1]).map(|parent| (pair[0], parent)));
                        phi_incoming.insert(words[2], incoming.collect::<Result<Vec<_>, _>>()?);
                        false
                    }
                    OP_MEMORY_BARRIER => {
                        if holds_workgroup(constant(words[1])) {
                            pending |= constant(words[2]).unwrap_or(0);
                        }
                        false
                    }
                    OP_CONTROL_BARRIER => {
                        let semantics = std::mem::take(&mut pending) | constant(words[3]).unwrap_or(0);
                        program.rejoins(words)
                            && holds_workgroup(constant(words[2]))
                            && orders_workgroup_memory(semantics)
                    }
                    _ => false,
                });
            }
            let terminator = *instructions.last().expect("a closed block holds its terminator");
            let mut cases = Vec::new();
            let targets = match terminator[0] & 0xffff {
                OP_BRANCH => vec![terminator[1]],
                OP_BRANCH_CONDITIONAL => vec![terminator[2], terminator[3]],
                OP_SWITCH => {
                    // Each case is a literal of the selector's width, low
                    // word first, then its target.
                    let width = program
                        .types
                        .get(&terminator[1])
                        .and_then(|ty| program.ints.get(ty))
                        .copied()
                        .filter(|width| *width <= 64)
                        .ok_or_else(|| {
                            MalformedModule(format!("switch selector %{} is not an integer", terminator[1]))
                        })?;
                    let case = width.div_ceil(32) as usize + 1;
                    let literals = &terminator[3..];
                    if literals.len() % case != 0 {
                        return Err(MalformedModule("a switch's cases are not whole".into()));
                    }
                    for pair in literals.chunks_exact(case) {
                        let literal =
                            pair[..case - 1].iter().rev().fold(0u64, |value, word| value << 32 | u64::from(*word));
                        cases.push((truncated(literal, width), label(pair[case - 1])?));
                    }
                    std::iter::once(terminator[2])
                        .chain(literals.chunks_exact(case).map(|pair| pair[case - 1]))
                        .collect()
                }
                _ => Vec::new(),
            };
            let mut successors = targets.into_iter().map(label).collect::<Result<Vec<_>, _>>()?;
            successors.sort_unstable();
            successors.dedup();
            parsed.push(Block { label: block_label, instructions, fences, merge, successors, cases });
        }
        program.blocks = parsed;
        program.phi_incoming = phi_incoming;
        Ok(program)
    }

    /// `words` is a control barrier whose execution scope holds the
    /// workgroup: every invocation of the workgroup executes it as one
    /// dynamic instance, so control is uniform after it.
    fn rejoins(&self, words: &[u32]) -> bool {
        words[0] & 0xffff == OP_CONTROL_BARRIER && holds_workgroup(self.constant(words[1]))
    }

    /// The pointer type of `pointer`: its storage class and pointee.
    fn pointer(&self, pointer: u32) -> Result<(u32, u32), MalformedModule> {
        self.types
            .get(&pointer)
            .and_then(|ty| self.pointers.get(ty))
            .copied()
            .ok_or_else(|| MalformedModule(format!("%{pointer} is not a pointer of known storage")))
    }

    pub(crate) fn place(&self, pointer: u32) -> Result<Place, MalformedModule> {
        Ok(match self.pointer(pointer)?.0 {
            STORAGE_FUNCTION | STORAGE_PRIVATE => Place::Variable(self.selection(pointer)?),
            STORAGE_WORKGROUP => Place::Workgroup,
            STORAGE_UNIFORM if self.buffer_block(pointer)? => Place::Device,
            STORAGE_UNIFORM_CONSTANT | STORAGE_UNIFORM | STORAGE_PUSH_CONSTANT => Place::ReadOnly,
            STORAGE_INPUT => Place::BuiltIn(self.built_ins.get(&self.selection(pointer)?.root).copied()),
            _ => Place::Device,
        })
    }

    /// Whether `pointer` addresses a `BufferBlock`'s memory: its variable's
    /// type, or the element type of an array of them, is one.
    fn buffer_block(&self, pointer: u32) -> Result<bool, MalformedModule> {
        let pointee = self.pointer(self.root(pointer))?.1;
        let pointee = self.arrays.get(&pointee).map_or(pointee, |&(element, _)| element);
        Ok(self.buffer_blocks.contains(&pointee))
    }

    /// What a read through `pointer` gives, if it reads the argument block
    /// or the tables, which every launch reads alike. The buffers'
    /// addresses, read from the block at the address the push constants
    /// hold, name their buffers: a buffer the launch only reads by
    /// declaration through the [`INPUT_BLOCK`], else the buffer or address
    /// table of the [`ARGUMENT_BLOCK`] or [`ADDRESS_TABLE_BLOCK`] word it
    /// reads, `w[index]` at a constant index (any buffer at another). The
    /// launch's words, read through a block at an address past the buffers'
    /// (`seismic_words`, `abi::vulkan::tail`), and the tables hold no
    /// addresses.
    fn argument_block(&self, pointer: u32) -> Result<Option<Provenance>, MalformedModule> {
        let base = self.root(pointer);
        let Some(&block) = self.argument_blocks.get(&self.pointer(base)?.1) else { return Ok(None) };
        if block == ArgumentBlock::Tables || !self.at_push_constant(base)? {
            return Ok(Some(Provenance::PLAIN));
        }
        if block == ArgumentBlock::Input {
            return Ok(Some(Provenance::named(Reach::INPUT)));
        }
        let word = match self.chains.get(&pointer) {
            Some(Chain { base: block, indices: [_, index], element: false }) if *block == base => {
                self.constant(*index).filter(|word| *word < u128::BITS)
            }
            _ => None,
        };
        Ok(Some(Provenance::named(match word {
            Some(word) if block == ArgumentBlock::Argument => Reach { words: 1 << word, ..Reach::NONE },
            Some(word) => Reach { address_tables: 1 << word, ..Reach::NONE },
            None => Reach::ANYWHERE,
        })))
    }

    /// The pointer `pointer` is derived from by access chains and copies.
    fn root(&self, mut pointer: u32) -> u32 {
        while let Some(chain) = self.chains.get(&pointer) {
            pointer = chain.base;
        }
        pointer
    }

    /// `pointer` is converted from an address read from push-constant or
    /// uniform memory as it is, with no offset.
    fn at_push_constant(&self, pointer: u32) -> Result<bool, MalformedModule> {
        let definition = |id: u32, opcode: u32| {
            self.defined.get(&id).map(|(_, words)| *words).filter(|words| words[0] & 0xffff == opcode)
        };
        let Some(converted) = definition(pointer, OP_CONVERT_U_TO_PTR) else { return Ok(false) };
        let Some(load) = definition(converted[3], OP_LOAD) else { return Ok(false) };
        Ok(matches!(self.place(load[3])?, Place::ReadOnly))
    }

    /// What `pointer` selects of the variable it addresses.
    fn selection(&self, pointer: u32) -> Result<Selection, MalformedModule> {
        let Some(chain) = self.chains.get(&pointer) else {
            return if self.variables.contains(&pointer) {
                Ok(Selection { root: pointer, path: Vec::new(), whole: true })
            } else {
                Err(MalformedModule(format!("pointer %{pointer} addresses no variable")))
            };
        };
        let mut selection = self.selection(chain.base)?;
        // An element chain steps over elements of what its base points to
        // first, then into one of them.
        let (members, whole) = self.members(self.pointer(chain.base)?.1, chain.indices, false)?;
        selection.path.extend(members);
        selection.whole &= whole && !chain.element;
        Ok(selection)
    }

    /// The struct members `indices` select from a value of type `ty`, each
    /// with its struct's member count, and whether they select all of what
    /// they are held as ([`Shaped`]): an index into an array steps to its
    /// elements' summary, selecting all of it only where the array holds one
    /// element, and the path goes on into the element; one into a vector or
    /// matrix selects a component of what is held whole. An access chain's
    /// indices are constants' ids, an extraction's literals.
    fn members(
        &self,
        mut ty: u32,
        indices: &[u32],
        literal: bool,
    ) -> Result<(Vec<(usize, usize)>, bool), MalformedModule> {
        let mut path = Vec::new();
        let mut whole = true;
        for index in indices {
            if let Some(&(element, single)) = self.arrays.get(&ty) {
                whole &= single;
                ty = element;
                continue;
            }
            let Some(members) = self.structs.get(&ty) else {
                return Ok((path, false));
            };
            let member = if literal {
                *index
            } else {
                self.constant(*index)
                    .ok_or_else(|| MalformedModule(format!("struct member index %{index} is not a constant")))?
            } as usize;
            ty = *members
                .get(member)
                .ok_or_else(|| MalformedModule(format!("struct %{ty} has no member {member}")))?;
            path.push((member, members.len()));
        }
        Ok((path, whole))
    }

    /// The type of `id`.
    fn type_of(&self, id: u32) -> Result<u32, MalformedModule> {
        self.types.get(&id).copied().ok_or_else(|| MalformedModule(format!("%{id} has no type")))
    }

    /// Every block's successors.
    pub(crate) fn successors(&self) -> Vec<Vec<usize>> {
        self.blocks.iter().map(|block| block.successors.clone()).collect()
    }

    /// The incoming values of `phi`, a phi of `block`, each with its parent
    /// block, along the edges of `edges` (by block, those control may take
    /// from it).
    fn incoming<'s>(
        &'s self,
        phi: u32,
        block: usize,
        edges: &'s [Vec<usize>],
    ) -> impl Iterator<Item = (u32, usize)> + 's {
        self.phi_incoming[&phi].iter().copied().filter(move |&(_, parent)| edges[parent].contains(&block))
    }

    /// The value of `id` where it is a constant whose bits fit 32 (not a
    /// specialization constant).
    fn constant(&self, id: u32) -> Option<u32> {
        match self.known.get(&id)?.value() {
            Constant::Value(bits) => u32::try_from(bits).ok(),
            Constant::Unknown | Constant::Varying => None,
        }
    }

    /// The bits of a module-scope constant declared by `words`, an
    /// instruction of `opcode`; none for any other declaration. A
    /// specialization constant is none: pipeline creation sets it after the
    /// check, so the check takes it as any value.
    fn declared(&self, opcode: u32, words: &[u32]) -> Option<Known> {
        let ty = words[1];
        let constituent = |id: &u32| self.known.get(id).cloned().unwrap_or(Known::VARYING);
        Some(match opcode {
            OP_CONSTANT_TRUE => Shaped::Whole(Constant::Value(1)),
            OP_CONSTANT_FALSE => Shaped::Whole(Constant::Value(0)),
            // An integer's bits, low word first, of its width (a narrow
            // signed one's word is sign-extended); a float's as they are.
            OP_CONSTANT if words.len() <= 5 => {
                let bits = words[3..].iter().rev().fold(0u64, |value, word| value << 32 | u64::from(*word));
                Shaped::Whole(Constant::Value(self.ints.get(&ty).map_or(bits, |width| truncated(bits, *width))))
            }
            OP_CONSTANT_NULL => self.null(ty),
            OP_CONSTANT_COMPOSITE => self.composite(ty, words[3..].iter().map(constituent).collect()),
            OP_CONSTANT_COMPOSITE_REPLICATE => {
                self.composite(ty, vec![constituent(&words[3]); self.structs.get(&ty).map_or(1, Vec::len)])
            }
            _ => return None,
        })
    }

    /// The bits of the null value of type `ty`.
    fn null(&self, ty: u32) -> Known {
        if let Some(members) = self.structs.get(&ty) {
            Shaped::Members(members.iter().map(|member| self.null(*member)).collect())
        } else if let Some(&(element, _)) = self.arrays.get(&ty) {
            self.null(element)
        } else if self.ints.contains_key(&ty) || self.booleans.contains(&ty) {
            Shaped::Whole(Constant::Value(0))
        } else {
            Known::VARYING
        }
    }

    /// The bits of a composite of type `ty` made of `constituents`: a
    /// struct's members apart; an array's or vector's summary, what any of
    /// them holds.
    fn composite(&self, ty: u32, constituents: Vec<Known>) -> Known {
        if self.structs.contains_key(&ty) {
            return Shaped::Members(constituents);
        }
        constituents
            .into_iter()
            .reduce(|mut summary, constituent| {
                summary.join(&constituent);
                summary
            })
            .unwrap_or(Known::UNKNOWN)
    }
}

/// `value` truncated to its low `width` bits.
fn truncated(value: u64, width: u32) -> u64 {
    if width >= u64::BITS {
        value
    } else {
        value & ((1u64 << width) - 1)
    }
}

/// A structured construct: the blocks its header dominates, short of its
/// exits. The constructs holding a block are a chain, innermost first
/// (`Analysis::enclosing`): a construct holds a block exactly where it is
/// on the block's chain, which is where it holds the block's innermost
/// construct in its `nested` places.
struct Construct {
    header: usize,
    merge: usize,
    /// A loop's continue target; none for a selection.
    continue_target: Option<usize>,
    /// Its merge, and the merges and outer continue constructs of those
    /// around it.
    exits: Vec<usize>,
    /// The blocks it holds, in reverse postorder.
    members: Vec<usize>,
    parent: Option<usize>,
    /// The places, in a preorder of the tree of constructs (by `parent`),
    /// of it, first, and of the constructs inside it.
    nested: std::ops::Range<usize>,
}

impl Construct {
    fn is_loop(&self) -> bool {
        self.continue_target.is_some()
    }
}

/// A node of the memory SSA: a definition of what a function or private
/// variable something reads holds. A variable is defined at the entry, at
/// each store to it, and where control from different predecessors meets
/// (a phi, at the iterated dominance frontier of its stores); each read of
/// it sees the node reaching it along the dominator tree. A node holds one
/// [`Held`], so memory takes a node per store and meeting of a variable
/// something reads, not a state of every variable at every block.
enum Node {
    /// What the variable holds where the entry point starts: its
    /// module-scope initializer, a constant (`initialized`), or unwritten:
    /// undefined, as `OpUndef` (an initializer in the entry point is a store
    /// there).
    Entry { initialized: bool },
    /// A store: the node it overwrites, with the store's write applied
    /// (`Shaped::write`).
    Store { over: usize },
    /// Where control meets: the join of the nodes reaching it from each
    /// predecessor, with that predecessor, and of what invocations
    /// arriving from different predecessors depend on.
    Phi(Vec<(usize, usize)>),
}

impl Node {
    /// What an entry node holds, and what any other holds before anything
    /// is established of it.
    fn initial(&self) -> Held {
        match self {
            Node::Entry { initialized: true } => Held::CONSTANT,
            Node::Entry { initialized: false } | Node::Store { .. } | Node::Phi(_) => Held::UNIFORM,
        }
    }
}

/// Where a bit of a value's taint was first raised: an instruction (a
/// built-in's load, a subgroup operation), a read of shared memory and a
/// store of the workgroup that may race it, a construct whose divergent
/// control rejoins where a phi is, or a loop that carries the value out.
#[derive(Clone, Copy, Debug)]
enum Origin {
    Value(u32),
    Raced { load: u32, store: Racer },
    Rejoined(usize),
    Loop(usize),
}

/// By traced bit of a taint a refusal traces ([`Traced`]): where it was first
/// raised.
type Origins = [Option<Origin>; Traced::ALL.len()];

/// Memory as the instructions of one block see it, one after another: what
/// the nodes reaching each hold, where the block's own stores, carried
/// under the values as they are now, stand in for what is established of
/// them.
struct Walk<'m> {
    block: usize,
    /// The instruction being carried.
    at: usize,
    /// By node: what it holds.
    held: &'m [Held],
    /// The block's stores carried so far, by node: what each holds.
    stored: HashMap<usize, Held>,
}

impl Walk<'_> {
    /// What `node` holds as the instruction being carried sees it, but for
    /// the loops it is carried out of ([`Analysis::carried_out`]): a store of
    /// the block as carried; otherwise what is established of it.
    fn seen(&self, node: usize) -> &Held {
        self.stored.get(&node).unwrap_or(&self.held[node])
    }
}

/// The blocks searches of one pass have reached, each by the last search
/// that reached it: one array for every search.
struct Marks {
    seen: Vec<usize>,
    search: usize,
}

/// The point opening a barrier interval of workgroup memory: the entry
/// (`None`), or a legal barrier ordering workgroup memory, by block and
/// instruction.
type Opening = Option<(usize, usize)>;

/// For one barrier interval, the parting branches whose single outgoing edge
/// every way from the interval's opening to a point must take, sorted by
/// branch block (at most one outgoing edge per branch).
type Ways = Vec<Edge>;

/// Two points are parted where some branch in both signatures requires
/// different outgoing edges.
fn parted(first: &[Edge], second: &[Edge]) -> bool {
    let (mut i, mut j) = (0, 0);
    while i < first.len() && j < second.len() {
        match first[i].0.cmp(&second[j].0) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                if first[i].1 != second[j].1 {
                    return true;
                }
                i += 1;
                j += 1;
            }
        }
    }
    false
}

/// Which accesses of one barrier interval of workgroup memory may run
/// beside each other, under one pass's control state
/// ([`Analysis::memory`]). Invocations may be at two points of an interval
/// at once unless a branch the whole workgroup takes alike, which the
/// interval passes once at most, parts every way from the interval's
/// opening to one from every way to the other: every invocation there took
/// the same way at it, so none took the way to the other point (and neither
/// point can reach the other in the interval, since that would extend a
/// way past the branch without taking the other edge). This holds by
/// execution order: invocations that first part at a branch read its
/// condition before either way's stores run, so those stores race nothing
/// it reads.
struct Interleaving<'s> {
    /// By block: where control goes from it.
    successors: &'s [Vec<usize>],
    /// By block: the legal barriers ordering workgroup memory its
    /// instructions and its end follow ([`Analysis::fenced`]).
    fenced: &'s [(Vec<Option<usize>>, Option<usize>)],
    /// By block: the openings whose intervals reach its start.
    opened: &'s [BTreeSet<Opening>],
    /// Blocks with at least two successors whose branch is taken alike by
    /// the whole workgroup (uniform terminator and legal control), sorted.
    agreed_branches: Vec<usize>,
    /// By branch block: the blocks whose start its end reaches in its interval.
    onward: RefCell<HashMap<usize, Rc<Vec<bool>>>>,
    /// By opening: for each block, its parting-branch [`Ways`] signature.
    ways: RefCell<HashMap<Opening, Rc<Vec<Ways>>>>,
}

/// An edge of control, from a block to one of its successors.
type Edge = (usize, usize);

impl<'s> Interleaving<'s> {
    fn new(
        successors: &'s [Vec<usize>],
        fenced: &'s [(Vec<Option<usize>>, Option<usize>)],
        opened: &'s [BTreeSet<Opening>],
        agreed_branches: Vec<usize>,
    ) -> Self {
        Interleaving {
            successors,
            fenced,
            opened,
            agreed_branches,
            onward: RefCell::default(),
            ways: RefCell::default(),
        }
    }

    /// The blocks whose start is reached from `starts`' starts without
    /// crossing a legal barrier ordering workgroup memory, and without the
    /// edge `without`.
    fn walk(&self, mut starts: Vec<usize>, without: Option<Edge>) -> Vec<bool> {
        let successors = self.successors;
        let mut reached = vec![false; successors.len()];
        while let Some(block) = starts.pop() {
            if std::mem::replace(&mut reached[block], true) || self.fenced[block].1.is_some() {
                continue;
            }
            starts.extend(successors[block].iter().copied().filter(|successor| without != Some((block, *successor))));
        }
        reached
    }

    /// The blocks whose start the end of `block` reaches in its interval.
    fn onward(&self, block: usize) -> Rc<Vec<bool>> {
        if let Some(onward) = self.onward.borrow().get(&block) {
            return Rc::clone(onward);
        }
        let onward = Rc::new(if self.fenced[block].1.is_some() {
            vec![false; self.fenced.len()]
        } else {
            self.walk(self.successors[block].clone(), None)
        });
        self.onward.borrow_mut().insert(block, Rc::clone(&onward));
        onward
    }

    /// The blocks whose start `opening` reaches in its interval without `without`.
    fn opened_without(&self, opening: Opening, without: Edge) -> Vec<bool> {
        let starts = match opening {
            None => vec![0],
            Some((block, fence)) if self.fenced[block].1 == Some(fence) => self.successors[block]
                .iter()
                .copied()
                .filter(|successor| without != (block, *successor))
                .collect(),
            Some(_) => Vec::new(),
        };
        self.walk(starts, Some(without))
    }

    /// For each block, its parting-branch [`Ways`] signature in `opening`.
    fn table(&self, opening: Opening) -> Rc<Vec<Ways>> {
        if let Some(table) = self.ways.borrow().get(&opening) {
            return Rc::clone(table);
        }
        let count = self.fenced.len();
        let mut table = vec![Ways::new(); count];
        // A barrier before its block's last opens an intra-block interval
        // that reaches no block's start.
        let reaches_starts = match opening {
            None => true,
            Some((block, fence)) => self.fenced[block].1 == Some(fence),
        };
        if reaches_starts {
            for &branch in &self.agreed_branches {
                let is_opening = matches!(opening, Some((block, _)) if block == branch);
                let onward = if is_opening {
                    None
                } else {
                    if !self.opened[branch].contains(&opening) || self.fenced[branch].1.is_some() {
                        continue;
                    }
                    let onward = self.onward(branch);
                    if onward[branch] {
                        continue;
                    }
                    Some(onward)
                };
                let mut seen_succ = Vec::new();
                for &succ in &self.successors[branch] {
                    if seen_succ.contains(&succ) {
                        continue;
                    }
                    seen_succ.push(succ);
                    let without = self.opened_without(opening, (branch, succ));
                    for block in 0..count {
                        if !without[block]
                            && self.opened[block].contains(&opening)
                            && onward.as_ref().is_none_or(|onward| onward[block])
                        {
                            table[block].push((branch, succ));
                        }
                    }
                }
            }
        }
        let table = Rc::new(table);
        self.ways.borrow_mut().insert(opening, Rc::clone(&table));
        table
    }

    /// The parting-branch [`Ways`] signature of a point in `block` after
    /// `fence` within `opening`'s interval.
    fn ways(&self, opening: Opening, block: usize, fence: Option<usize>) -> Ways {
        // A point after a legal barrier in its block is reached only
        // through it: no branch of the interval parts it from another.
        if fence.is_some() || self.agreed_branches.is_empty() {
            return Ways::new();
        }
        self.table(opening)[block].clone()
    }
}

/// How control and memory flow through the entry point, where control goes
/// from each block to its successors: what is reachable, dominance, and the
/// memory SSA. [`Flow::live`] reads it on every edge to find those some
/// invocation may take; the analysis reads it on those alone.
struct Flow<'p, 'a> {
    program: &'p Program<'a>,
    /// By block: the blocks control may go to from it; none from a block
    /// control never reaches.
    successors: Vec<Vec<usize>>,
    /// Over those edges.
    dominance: Dominance,
    /// The memory SSA ([`Node`]), its entries first, one per variable.
    nodes: Vec<Node>,
    /// By node: the block it is defined at.
    defined_at: Vec<usize>,
    /// By block: the phi nodes where it starts.
    phis: Vec<Vec<usize>>,
    /// By block and instruction reading a variable something reads: the
    /// node reaching it.
    reaching: HashMap<(usize, usize), usize>,
    /// By block and instruction storing to a variable something reads: the
    /// node it defines.
    defining: HashMap<(usize, usize), usize>,
    /// The loops entered at least once ([`Entered`], [`Flow::live`]).
    entered: Vec<Entered>,
    /// By id: its bits, as [`Flow::live`] found them along the edges some
    /// invocation may take.
    known: Vec<Known>,
}

/// What [`Flow::live`] finds: by block, the edges some invocation may take
/// from it; the loops entered at least once; and by id, its bits along
/// those edges.
struct Live {
    taken: Vec<Vec<usize>>,
    entered: Vec<Entered>,
    known: Vec<Known>,
}

/// A loop whose first test holds along the edge entering it: from that
/// edge, `test`'s way to `exit`, the loop's merge, is taken only after an
/// iteration (a counted loop, `for (i = 0; i < 3; ++i)`).
#[derive(Clone, Copy, Debug)]
struct Entered {
    header: usize,
    test: usize,
    exit: usize,
}

/// The state of the entry point the analysis raises to a fixed point, where
/// control and memory flow as its [`Flow`] has them.
/// Mapping from SPIR-V SSA ids to dense slots in `Analysis::values`,
/// `Analysis::origins`, and `Prepared::defined_block`. Entry points use direct
/// indexing over `0..bound`; summarised callees index only their own
/// parameters and defined ids (tens of entries instead of the module's bound).
enum IdIndex {
    Dense(usize),
    Sparse(HashMap<u32, usize>),
}

impl IdIndex {
    fn len(&self) -> usize {
        match self {
            IdIndex::Dense(bound) => *bound,
            IdIndex::Sparse(map) => map.len(),
        }
    }

    fn get(&self, id: u32) -> Option<usize> {
        match self {
            IdIndex::Dense(_) => Some(id as usize),
            IdIndex::Sparse(map) => map.get(&id).copied(),
        }
    }
}

/// Precomputed function-invariant flow, constructs, copy phis, and workgroup
/// locations, built once per function ([`Prepared::of`]) and borrowed by every
/// [`Analysis`] of it.
struct Prepared {
    ids: IdIndex,
    successors: Vec<Vec<usize>>,
    dominance: Dominance,
    nodes: Vec<Node>,
    defined_at: Vec<usize>,
    phis: Vec<Vec<usize>>,
    reaching: HashMap<(usize, usize), usize>,
    defining: HashMap<(usize, usize), usize>,
    entered: Vec<Entered>,
    known: Vec<Known>,
    initial: Vec<Held>,
    copies: HashMap<u32, u32>,
    constructs: Vec<Construct>,
    innermost: Vec<Option<usize>>,
    enclosing_loop: Vec<Option<usize>>,
    defined_block: Vec<u32>,
    landing: Vec<Vec<Location>>,
    reading: Vec<Vec<Location>>,
    has_loops: bool,
    has_shared_stores: bool,
}

/// Borrowed view of a [`Prepared`] flow paired with its [`Program`].
struct PreparedFlow<'p, 'a> {
    program: &'p Program<'a>,
    data: &'p Prepared,
}

impl std::ops::Deref for PreparedFlow<'_, '_> {
    type Target = Prepared;
    fn deref(&self) -> &Prepared {
        self.data
    }
}

impl PreparedFlow<'_, '_> {
    fn initial(&self) -> Vec<Held> {
        self.data.initial.clone()
    }
}

/// Where `pointer`, into workgroup memory, lands ([`Location`]) using
/// `known` constant-folding state.
fn workgroup_location(program: &Program<'_>, known: &[Known], pointer: u32) -> Location {
    let mut chains = Vec::new();
    let mut root = pointer;
    while let Some(chain) = program.chains.get(&root) {
        if chain.element {
            return None;
        }
        chains.push(chain.indices);
        root = chain.base;
    }
    let mut ty = program.pointer(root).ok()?.1;
    let region = if program.layout.blocks.contains(&ty) { Region::Views } else { Region::Variable(root) };
    let mut start = 0u64;
    for index in chains.into_iter().rev().flatten() {
        let Some(Constant::Value(index)) = known.get(*index as usize).map(Known::value) else {
            return None;
        };
        if let Some(members) = program.structs.get(&ty) {
            let member = u32::try_from(index).ok()?;
            let next = *members.get(member as usize)?;
            let offset = match program.layout.offsets.get(&(ty, member)) {
                Some(offset) => *offset,
                None => members[..member as usize]
                    .iter()
                    .map(|member| program.layout.bytes.get(member).copied())
                    .sum::<Option<u64>>()?,
            };
            start = start.checked_add(offset)?;
            ty = next;
        } else if let Some(&(element, _)) = program.arrays.get(&ty) {
            let stride = program.layout.strides.get(&ty).or_else(|| program.layout.bytes.get(&element)).copied()?;
            start = start.checked_add(index.checked_mul(stride)?)?;
            ty = element;
        } else {
            return None;
        }
    }
    let bytes = *program.layout.bytes.get(&ty)?;
    let end = start.checked_add(bytes)?;
    Some((region, start, end))
}

impl Prepared {
    fn of(program: &Program<'_>, dense: bool) -> Result<Self, MalformedModule> {
        let Live { taken, entered, known } = Flow::new(program, program.successors())?.live()?;
        let mut flow = Flow::new(program, taken)?;
        flow.entered = entered;
        flow.known = known;
        let initial = flow.initial();
        let copies = flow.copies();
        let blocks = &program.blocks;
        let count = blocks.len();
        let mut constructs: Vec<Construct> = Vec::new();
        let mut innermost: Vec<Option<usize>> = vec![None; count];
        for &block in &flow.dominance.preorder {
            let mut around = match block {
                0 => None,
                _ => {
                    let dominator = flow.dominance.dominator[block];
                    innermost[dominator.expect("a reachable block has a dominator")]
                }
            };
            while let Some(open) = around.filter(|open| constructs[*open].exits.contains(&block)) {
                around = constructs[open].parent;
            }
            if let Some(merge) = blocks[block].merge {
                let (merge, continue_target) = match merge {
                    Merge::Selection { merge } => (merge, None),
                    Merge::Loop { merge, continue_target } => (merge, Some(continue_target)),
                };
                let mut exits = vec![merge];
                let mut walk = around;
                while let Some(outer) = walk {
                    let outer_c = &constructs[outer];
                    exits.push(outer_c.merge);
                    if let Some(target) = outer_c.continue_target {
                        if !flow.dominance.dominates(target, block) {
                            exits.push(target);
                        }
                    }
                    walk = outer_c.parent;
                }
                let holds_header = !exits.iter().any(|exit| flow.dominance.dominates(*exit, block));
                constructs.push(Construct {
                    header: block,
                    merge,
                    continue_target,
                    exits,
                    members: Vec::new(),
                    parent: around,
                    nested: 0..0,
                });
                if holds_header {
                    around = Some(constructs.len() - 1);
                }
            }
            innermost[block] = around;
        }
        for &block in &flow.dominance.order {
            let mut around = innermost[block];
            while let Some(construct) = around {
                constructs[construct].members.push(block);
                around = constructs[construct].parent;
            }
        }
        let mut sizes = vec![1; constructs.len()];
        for construct in (0..constructs.len()).rev() {
            if let Some(parent) = constructs[construct].parent {
                sizes[parent] += sizes[construct];
            }
        }
        let mut next = vec![0; constructs.len()];
        let mut outermost = 0;
        for (construct, size) in sizes.into_iter().enumerate() {
            let free = match constructs[construct].parent {
                Some(parent) => &mut next[parent],
                None => &mut outermost,
            };
            let start = *free;
            *free += size;
            next[construct] = start + 1;
            constructs[construct].nested = start..start + size;
        }
        let mut enclosing_loop = vec![None; count];
        for block in 0..count {
            let mut walk = innermost[block];
            while let Some(c) = walk {
                if constructs[c].is_loop() {
                    enclosing_loop[block] = Some(c);
                    break;
                }
                walk = constructs[c].parent;
            }
        }
        let ids = if dense {
            IdIndex::Dense(program.bound)
        } else {
            let mut map = HashMap::with_capacity(program.parameters.len() + program.defined.len());
            for &parameter in &program.parameters {
                let next = map.len();
                map.entry(parameter).or_insert(next);
            }
            for &id in program.defined.keys() {
                let next = map.len();
                map.entry(id).or_insert(next);
            }
            IdIndex::Sparse(map)
        };
        let mut defined_block = vec![u32::MAX; ids.len()];
        for (&id, &(block, _)) in &program.defined {
            if let Some(slot) = ids.get(id) {
                defined_block[slot] = block as u32;
            }
        }
        let mut landing = vec![Vec::new(); count];
        let mut reading = vec![Vec::new(); count];
        let mut has_shared_stores = false;
        for &block in &flow.dominance.order {
            let instructions = &blocks[block].instructions;
            landing[block] = instructions
                .iter()
                .map(|words| {
                    Ok(match written_pointer(words) {
                        Some(pointer) => match program.place(pointer)? {
                            Place::Workgroup => {
                                has_shared_stores = true;
                                workgroup_location(program, &flow.known, pointer)
                            }
                            Place::Device => {
                                has_shared_stores = true;
                                None
                            }
                            _ => None,
                        },
                        None => None,
                    })
                })
                .collect::<Result<Vec<_>, MalformedModule>>()?;
            reading[block] = instructions
                .iter()
                .map(|words| {
                    Ok(match read_pointer(words) {
                        Some(pointer) if matches!(program.place(pointer)?, Place::Workgroup) => {
                            workgroup_location(program, &flow.known, pointer)
                        }
                        _ => None,
                    })
                })
                .collect::<Result<Vec<_>, MalformedModule>>()?;
        }
        let has_loops = constructs.iter().any(Construct::is_loop);
        Ok(Prepared {
            ids,
            successors: flow.successors,
            dominance: flow.dominance,
            nodes: flow.nodes,
            defined_at: flow.defined_at,
            phis: flow.phis,
            reaching: flow.reaching,
            defining: flow.defining,
            entered: flow.entered,
            known: flow.known,
            initial,
            copies,
            constructs,
            innermost,
            enclosing_loop,
            defined_block,
            landing,
            reading,
            has_loops,
            has_shared_stores,
        })
    }
}

struct Analysis<'p, 'a> {
    flow: PreparedFlow<'p, 'a>,
    /// Ordered so that each construct follows those around it.
    constructs: &'p [Construct],
    /// By block: the innermost construct holding it.
    innermost: &'p [Option<usize>],
    /// By block: the innermost loop holding it, if any.
    enclosing_loop: &'p [Option<usize>],
    /// By id: the block defining it, or `u32::MAX` if not defined in this function.
    defined_block: &'p [u32],
    /// By block and instruction: workgroup memory location written / read (invariant across iterations).
    landing: &'p [Vec<Location>],
    reading: &'p [Vec<Location>],
    /// By loop: the values defined in it that may differ between its
    /// iterations.
    variant: Vec<HashSet<u32>>,
    /// Whether `variant` has been computed at least once.
    variants_computed: bool,
    /// By phi: the value it copies ([`Flow::copies`]).
    copies: &'p HashMap<u32, u32>,
    /// By node: what `memory` has established it holds.
    held: Vec<Held>,
    /// By id.
    values: Vec<Held>,
    /// By id: where each traced bit of its taint a refusal traces was first
    /// raised.
    origins: Vec<Origins>,
    /// By block: how its terminator sends invocations.
    terminator: Vec<Taint>,
    /// By construct: a selection's condition, or what a loop is left or
    /// continued under.
    construct: Vec<Taint>,
    /// By block: what invocations arriving from different predecessors
    /// depend on.
    join: Vec<Taint>,
    /// By block: after a lane-dependent exit, before the merge it rejoins
    /// at.
    exited: Vec<Taint>,
    /// By shared memory (workgroup, written device), block and instruction:
    /// what a load of it there may see of the workgroup's stores ([`Seen`]).
    seen: [Vec<Vec<Seen>>; 2],
    /// By block and instruction: what a store to shared memory there keeps
    /// of the values it places ([`ValueState::kept`]).
    kept: Vec<Vec<ValueState>>,
    /// The device memory the launch writes: what the pointers it stores
    /// through address.
    written: Reach,
    /// The summarised functions it may call, and what their calls give.
    summaries: &'p Summaries<'a>,
}

/// Raise `flag` to `value`; whether it rose.
fn raise<T: Join>(flag: &mut T, value: T) -> bool {
    let joined = flag.join(value);
    let rose = joined != *flag;
    *flag = joined;
    rose
}

fn raise_all<T: Join>(flags: &mut [T], values: &[T]) -> bool {
    flags.iter_mut().zip(values).fold(false, |rose, (flag, value)| raise(flag, *value) || rose)
}

/// Dominance over the blocks of a function entered at its first block: the
/// one implementation the analysis ([`Flow`]) and value numbering
/// (`numbering`) read.
pub(crate) struct Dominance {
    /// Reachable blocks in reverse postorder.
    order: Vec<usize>,
    /// By block: the reachable blocks control may come to it from.
    predecessors: Vec<Vec<usize>>,
    /// By block: its immediate dominator, the entry's itself; none for a
    /// block control never reaches.
    dominator: Vec<Option<usize>>,
    /// By block: the blocks it immediately dominates.
    children: Vec<Vec<usize>>,
    /// Reachable blocks in dominator-tree preorder.
    pub(crate) preorder: Vec<usize>,
    /// Dominator-tree preorder and postorder numbers.
    pre: Vec<usize>,
    post: Vec<usize>,
}

impl Dominance {
    /// The dominance where control goes from each block to its
    /// `successors`, of which a block control never reaches is cleared: it
    /// takes no edge.
    pub(crate) fn new(successors: &mut [Vec<usize>]) -> Dominance {
        let count = successors.len();
        let mut reachable = vec![false; count];
        let mut postorder = Vec::with_capacity(count);
        reachable[0] = true;
        let mut stack = vec![(0usize, 0usize)];
        while let Some(&(block, next)) = stack.last() {
            match successors[block].get(next) {
                Some(&successor) => {
                    stack.last_mut().expect("the stack holds the block").1 += 1;
                    if !reachable[successor] {
                        reachable[successor] = true;
                        stack.push((successor, 0));
                    }
                }
                None => {
                    postorder.push(block);
                    stack.pop();
                }
            }
        }
        // A block control never reaches takes no edge.
        for (targets, reached) in successors.iter_mut().zip(&reachable) {
            if !*reached {
                targets.clear();
            }
        }
        let order = postorder.into_iter().rev().collect::<Vec<_>>();
        let mut predecessors = vec![Vec::new(); count];
        for &block in &order {
            for &successor in &successors[block] {
                predecessors[successor].push(block);
            }
        }

        // Immediate dominators (Cooper, Harvey and Kennedy).
        let mut position = vec![usize::MAX; count];
        for (index, block) in order.iter().enumerate() {
            position[*block] = index;
        }
        let mut dominator = vec![None; count];
        dominator[0] = Some(0);
        let mut changed = true;
        while changed {
            changed = false;
            for &block in &order[1..] {
                let mut found: Option<usize> = None;
                for &predecessor in &predecessors[block] {
                    if dominator[predecessor].is_none() {
                        continue;
                    }
                    found = Some(match found {
                        None => predecessor,
                        Some(mut other) => {
                            let mut finger = predecessor;
                            while finger != other {
                                while position[finger] > position[other] {
                                    finger = dominator[finger].expect("a processed block has a dominator");
                                }
                                while position[other] > position[finger] {
                                    other = dominator[other].expect("a processed block has a dominator");
                                }
                            }
                            finger
                        }
                    });
                }
                if found != dominator[block] {
                    dominator[block] = found;
                    changed = true;
                }
            }
        }
        let mut children = vec![Vec::new(); count];
        for &block in &order[1..] {
            children[dominator[block].expect("a reachable block has a dominator")].push(block);
        }
        let (mut pre, mut post) = (vec![0; count], vec![0; count]);
        let mut preorder = vec![0usize];
        let mut clock = 1;
        let mut stack = vec![(0usize, 0usize)];
        while let Some(&(block, next)) = stack.last() {
            match children[block].get(next) {
                Some(&child) => {
                    stack.last_mut().expect("the stack holds the block").1 += 1;
                    pre[child] = clock;
                    clock += 1;
                    preorder.push(child);
                    stack.push((child, 0));
                }
                None => {
                    post[block] = clock;
                    clock += 1;
                    stack.pop();
                }
            }
        }
        Dominance { order, predecessors, dominator, children, preorder, pre, post }
    }

    /// Whether control reaches `block`.
    fn reachable(&self, block: usize) -> bool {
        self.dominator[block].is_some()
    }

    pub(crate) fn dominates(&self, dominator: usize, block: usize) -> bool {
        self.reachable(dominator)
            && self.reachable(block)
            && self.pre[dominator] <= self.pre[block]
            && self.post[block] <= self.post[dominator]
    }
}

impl<'p, 'a> Flow<'p, 'a> {
    /// The flow through `program`'s entry point where control goes from each
    /// block to its `successors`.
    fn new(program: &'p Program<'a>, mut successors: Vec<Vec<usize>>) -> Result<Self, MalformedModule> {
        let blocks = &program.blocks;
        let count = blocks.len();
        let dominance = Dominance::new(&mut successors);

        // The memory SSA ([`Node`]) of the function and private variables
        // something reads: what a variable nothing reads holds is never
        // seen, though its stores are checked as any. A pointer `place`
        // refuses is refused where `propagate` first carries its
        // instruction, as any other refusal.
        let root_of = |pointer: u32| match program.place(pointer) {
            Ok(Place::Variable(selection)) => Some(selection.root),
            _ => None,
        };
        // By block: its instructions reading or writing a variable, with
        // the variable each reads and writes.
        let mut accesses = vec![Vec::new(); count];
        let mut read = HashSet::new();
        for &block in &dominance.order {
            for (position, words) in blocks[block].instructions.iter().enumerate() {
                let reads = read_pointer(words).and_then(root_of);
                let writes = written_pointer(words).and_then(root_of);
                read.extend(reads);
                if reads.is_some() || writes.is_some() {
                    accesses[block].push((position, reads, writes));
                }
            }
        }
        let mut slotted = Vec::new();
        for variable in &program.variables {
            if matches!(program.pointer(*variable)?.0, STORAGE_FUNCTION | STORAGE_PRIVATE) && read.contains(variable) {
                slotted.push(*variable);
            }
        }
        slotted.sort_unstable();
        let slots = slotted.iter().enumerate().map(|(slot, variable)| (*variable, slot)).collect::<HashMap<_, _>>();
        let slot_for = |root: Option<u32>| root.and_then(|root| slots.get(&root).copied());
        // Each variable's entry, as its slot.
        let mut nodes = slotted
            .iter()
            .map(|variable| Node::Entry { initialized: program.initialized.contains(variable) })
            .collect::<Vec<_>>();
        // By node: its variable's slot, and the block it is defined at.
        let mut slot_of = (0..slotted.len()).collect::<Vec<_>>();
        let mut defined_at = vec![0; slotted.len()];

        // Dominance frontiers (Cooper, Harvey and Kennedy). Where the entry
        // block starts, a variable holds what it holds at the entry,
        // whatever branches there (which SPIR-V forbids): it is no
        // frontier.
        let mut frontiers = vec![Vec::new(); count];
        for &block in &dominance.order[1..] {
            if dominance.predecessors[block].len() < 2 {
                continue;
            }
            let idom = dominance.dominator[block].expect("a reachable block has a dominator");
            for &predecessor in &dominance.predecessors[block] {
                let mut runner = predecessor;
                while runner != idom {
                    if frontiers[runner].last() != Some(&block) {
                        frontiers[runner].push(block);
                    }
                    runner = dominance.dominator[runner].expect("a reachable block has a dominator");
                }
            }
        }
        // Phis at the iterated dominance frontier of each variable's
        // stores.
        let mut stored_in = vec![Vec::new(); slotted.len()];
        for (block, accesses) in accesses.iter().enumerate() {
            for &(_, _, writes) in accesses {
                if let Some(slot) = slot_for(writes) {
                    if stored_in[slot].last() != Some(&block) {
                        stored_in[slot].push(block);
                    }
                }
            }
        }
        let mut phis = vec![Vec::new(); count];
        // By block: the last slot given a phi there, and the last whose
        // search reached it.
        let (mut placed, mut searched) = (vec![usize::MAX; count], vec![usize::MAX; count]);
        for (slot, stores) in stored_in.iter().enumerate() {
            let mut work = stores.clone();
            for block in &work {
                searched[*block] = slot;
            }
            while let Some(block) = work.pop() {
                for &frontier in &frontiers[block] {
                    if placed[frontier] == slot {
                        continue;
                    }
                    placed[frontier] = slot;
                    phis[frontier].push(nodes.len());
                    nodes.push(Node::Phi(Vec::new()));
                    slot_of.push(slot);
                    defined_at.push(frontier);
                    if searched[frontier] != slot {
                        searched[frontier] = slot;
                        work.push(frontier);
                    }
                }
            }
        }
        // One walk of the dominator tree, each variable's nodes stacked
        // along it: a read records the node reaching it, a store the node it
        // overwrites, a phi the node reaching it from each predecessor.
        let mut stacks = (0..slotted.len()).map(|slot| vec![slot]).collect::<Vec<_>>();
        fn top(stacks: &[Vec<usize>], slot: usize) -> usize {
            *stacks[slot].last().expect("a variable's stack holds its entry")
        }
        let (mut reaching, mut defining) = (HashMap::new(), HashMap::new());
        // The blocks entered, each with the next child to enter and the
        // slots it stacked nodes for.
        let mut walk: Vec<(usize, usize, Vec<usize>)> = Vec::new();
        let mut entering = Some(0);
        loop {
            if let Some(block) = entering.take() {
                let mut stacked = Vec::new();
                for &phi in &phis[block] {
                    stacks[slot_of[phi]].push(phi);
                    stacked.push(slot_of[phi]);
                }
                for &(position, reads, writes) in &accesses[block] {
                    if let Some(slot) = slot_for(reads) {
                        reaching.insert((block, position), top(&stacks, slot));
                    }
                    if let Some(slot) = slot_for(writes) {
                        let over = top(&stacks, slot);
                        defining.insert((block, position), nodes.len());
                        stacks[slot].push(nodes.len());
                        nodes.push(Node::Store { over });
                        slot_of.push(slot);
                        defined_at.push(block);
                        stacked.push(slot);
                    }
                }
                for &successor in &successors[block] {
                    for &phi in &phis[successor] {
                        let node = top(&stacks, slot_of[phi]);
                        let Node::Phi(incoming) = &mut nodes[phi] else { unreachable!("a block's phis are phi nodes") };
                        incoming.push((block, node));
                    }
                }
                walk.push((block, 0, stacked));
            }
            let Some(&(block, next, _)) = walk.last() else { break };
            match dominance.children[block].get(next) {
                Some(&child) => {
                    walk.last_mut().expect("the walk holds the block").1 += 1;
                    entering = Some(child);
                }
                None => {
                    let (_, _, stacked) = walk.pop().expect("the walk holds the block");
                    for slot in stacked {
                        stacks[slot].pop();
                    }
                }
            }
        }
        Ok(Flow {
            program,
            successors,
            dominance,
            nodes,
            defined_at,
            phis,
            reaching,
            defining,
            entered: Vec::new(),
            known: Vec::new(),
        })
    }

    /// By block: the edges some invocation of the module may take from it,
    /// where control goes from each block to its successors in this flow. A
    /// branch on a constant takes one way alone: a condition computed from
    /// constants (not specialization constants, which pipeline creation sets
    /// after the check, so a device may take either way), through values, a
    /// struct's members and the variables something reads, folding
    /// comparisons, logical operations, selections and plain integer
    /// arithmetic of constants. Constants are carried forward from
    /// the entry along the edges found taken (sparse conditional constant
    /// propagation): a phi, of a value or of a variable, joins only what
    /// arrives along them. A block no taken edge reaches is unreachable. An
    /// undefined value (`OpUndef`, a variable before it is written) is any
    /// value, so a condition it reaches, along even one taken edge, takes
    /// every way.
    fn live(&self) -> Result<Live, MalformedModule> {
        let program = self.program;
        let count = program.blocks.len();
        // By id: what its bits are. A value the entry point computes is
        // raised from nothing; any other is a constant, or any value (an
        // undefined one too).
        let mut known = (0..program.bound as u32)
            .map(|id| {
                if program.defined.contains_key(&id) {
                    Known::UNKNOWN
                } else {
                    program.known.get(&id).cloned().unwrap_or(Known::VARYING)
                }
            })
            .collect::<Vec<_>>();
        // By node: what the variable's bits are. An initializer is not
        // followed, and a variable not yet written is undefined: any value.
        let mut stored = self
            .nodes
            .iter()
            .map(|node| match node {
                Node::Entry { .. } => Known::VARYING,
                Node::Store { .. } | Node::Phi(_) => Known::UNKNOWN,
            })
            .collect::<Vec<_>>();
        let mut executable = vec![false; count];
        executable[0] = true;
        let mut taken = vec![Vec::new(); count];
        loop {
            let mut changed = false;
            for &block in &self.dominance.order {
                if !executable[block] {
                    continue;
                }
                for &phi in &self.phis[block] {
                    let Node::Phi(incoming) = &self.nodes[phi] else { unreachable!("a block's phis are phi nodes") };
                    let mut joined = Known::UNKNOWN;
                    for &(from, node) in incoming {
                        if taken[from].contains(&block) {
                            joined.join(&stored[node]);
                        }
                    }
                    changed |= stored[phi].raise(&joined);
                }
                for (position, words) in program.blocks[block].instructions.iter().enumerate() {
                    let opcode = words[0] & 0xffff;
                    let shape = shape(opcode).expect("an analysed program holds only instructions of a known shape");
                    if shape.result {
                        let value = self.folded(words, block, position, &known, &stored, &taken)?;
                        changed |= known[words[2] as usize].raise(&value);
                    }
                    if let Some(&node) = self.defining.get(&(block, position)) {
                        let Node::Store { over } = self.nodes[node] else {
                            unreachable!("a store defines a store node")
                        };
                        let pointer = written_pointer(words).expect("a store node is defined by a write");
                        let Place::Variable(Selection { path, whole, .. }) = program.place(pointer)? else {
                            unreachable!("a store node writes a function or private variable")
                        };
                        let value = match opcode {
                            OP_STORE => known[words[2] as usize].clone(),
                            OP_VARIABLE => known[words[4] as usize].clone(),
                            OP_COPY_MEMORY => self.recalled(words[2], block, position, &stored)?,
                            // An atomic's or a cooperative-matrix store's.
                            _ => Known::VARYING,
                        };
                        let mut held = stored[over].clone();
                        held.write(&path, value, whole);
                        changed |= stored[node].raise(&held);
                    }
                }
                for target in self.decided(block, &known).unwrap_or_default() {
                    if !taken[block].contains(&target) {
                        taken[block].push(target);
                        executable[target] = true;
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
        // Nothing raised from nothing stays nothing where it is reached: an
        // executable block's operands are defined in blocks dominating it,
        // or are phis joining an edge taken into it.
        for &block in &self.dominance.order {
            assert!(
                !executable[block] || self.decided(block, &known).is_some(),
                "a reachable block's condition is known at the fixed point"
            );
        }
        // The loops whose first test holds along the edge entering them:
        // the header's phis at what that edge brings, the blocks on to the
        // test folded again from them (and put back after).
        let mut entered = Vec::new();
        for &header in &self.dominance.order {
            let Some(Merge::Loop { merge, .. }) = program.blocks[header].merge else { continue };
            if !executable[header] {
                continue;
            }
            let mut restore = Vec::new();
            for words in &program.blocks[header].instructions {
                if words[0] & 0xffff != OP_PHI {
                    continue;
                }
                let mut joined = Known::UNKNOWN;
                for (value, parent) in program.incoming(words[2], header, &taken) {
                    if !self.dominance.dominates(header, parent) {
                        joined.join(&known[value as usize]);
                    }
                }
                restore.push((words[2], std::mem::replace(&mut known[words[2] as usize], joined)));
            }
            let mut block = header;
            let test = loop {
                for (position, words) in program.blocks[block].instructions.iter().enumerate() {
                    let opcode = words[0] & 0xffff;
                    let shape = shape(opcode).expect("an analysed program holds only instructions of a known shape");
                    if shape.result && !(block == header && opcode == OP_PHI) {
                        let value = self.folded(words, block, position, &known, &stored, &taken)?;
                        restore.push((words[2], std::mem::replace(&mut known[words[2] as usize], value)));
                    }
                }
                let ways = self.decided(block, &known);
                if program.blocks[block].terminator()[0] & 0xffff == OP_BRANCH_CONDITIONAL {
                    break match ways.as_deref() {
                        Some(&[way]) if way != merge && taken[block].contains(&merge) => Some(block),
                        _ => None,
                    };
                }
                match ways.as_deref() {
                    // Down the dominator tree, so the walk ends.
                    Some(&[next]) if next != block && next != merge && self.dominance.dominates(block, next) => {
                        block = next;
                    }
                    _ => break None,
                }
            };
            for (id, value) in restore.into_iter().rev() {
                known[id as usize] = value;
            }
            if let Some(test) = test {
                entered.push(Entered { header, test, exit: merge });
            }
        }
        for targets in &mut taken {
            targets.sort_unstable();
        }
        Ok(Live { taken, entered, known })
    }

    /// The successors of `block` its terminator takes, where values' bits
    /// are `known`; none where its condition is not known yet.
    fn decided(&self, block: usize, known: &[Known]) -> Option<Vec<usize>> {
        let program_block = &self.program.blocks[block];
        let terminator = program_block.terminator();
        let label = |id: u32| self.program.labels[&id];
        let opcode = terminator[0] & 0xffff;
        if !matches!(opcode, OP_BRANCH_CONDITIONAL | OP_SWITCH) {
            return Some(self.successors[block].clone());
        }
        match known[terminator[1] as usize].value() {
            Constant::Unknown => None,
            Constant::Varying => Some(self.successors[block].clone()),
            Constant::Value(condition) if opcode == OP_BRANCH_CONDITIONAL => {
                Some(vec![label(terminator[if condition != 0 { 2 } else { 3 }])])
            }
            Constant::Value(selector) => Some(vec![program_block
                .cases
                .iter()
                .find(|(literal, _)| *literal == selector)
                .map_or_else(|| label(terminator[2]), |(_, target)| *target)]),
        }
    }

    /// The bits a read through `pointer`, instruction `position` of `block`,
    /// gives, where the variables' nodes hold `stored`: a variable's at the
    /// member read; any other memory's, any.
    fn recalled(
        &self,
        pointer: u32,
        block: usize,
        position: usize,
        stored: &[Known],
    ) -> Result<Known, MalformedModule> {
        Ok(match self.program.place(pointer)? {
            Place::Variable(Selection { path, .. }) => {
                let node = *self
                    .reaching
                    .get(&(block, position))
                    .expect("a read of a function or private variable has a node reaching it");
                stored[node].get(path.iter().map(|(member, _)| *member))
            }
            Place::Workgroup | Place::Device | Place::ReadOnly | Place::BuiltIn(_) => Known::VARYING,
        })
    }

    /// The bits of the result of `words`, instruction `position` of
    /// `block`, where values hold `known`, variables' nodes `stored`, and
    /// the edges `taken` are taken.
    fn folded(
        &self,
        words: &[u32],
        block: usize,
        position: usize,
        known: &[Known],
        stored: &[Known],
        taken: &[Vec<usize>],
    ) -> Result<Known, MalformedModule> {
        let program = self.program;
        let of = |id: u32| known[id as usize].clone();
        Ok(match words[0] & 0xffff {
            OP_UNDEF => Known::VARYING,
            OP_COPY_OBJECT => of(words[3]),
            OP_LOAD => self.recalled(words[3], block, position, stored)?,
            OP_PHI => {
                let mut joined = Known::UNKNOWN;
                for (value, _) in program.incoming(words[2], block, taken) {
                    joined.join(&of(value));
                }
                joined
            }
            OP_COMPOSITE_CONSTRUCT => program.composite(words[1], words[3..].iter().map(|id| of(*id)).collect()),
            OP_COMPOSITE_EXTRACT => {
                let (path, _) = program.members(program.type_of(words[3])?, &words[4..], true)?;
                of(words[3]).get(path.iter().map(|(member, _)| *member))
            }
            OP_COMPOSITE_INSERT => {
                let (path, whole) = program.members(program.type_of(words[4])?, &words[5..], true)?;
                let mut composite = of(words[4]);
                composite.write(&path, of(words[3]), whole);
                composite
            }
            OP_SELECT => match of(words[3]).value() {
                Constant::Value(condition) => of(words[if condition != 0 { 4 } else { 5 }]),
                Constant::Unknown => Known::UNKNOWN,
                Constant::Varying => {
                    let mut either = of(words[4]);
                    either.join(&of(words[5]));
                    either
                }
            },
            opcode => Shaped::Whole(self.arithmetic(opcode, words, known)),
        })
    }

    /// The bits of the result of `words`, an instruction of `opcode`, where
    /// values hold `known`: an integer or boolean operation of known
    /// operands folded, bit for bit at the operands' and the result's
    /// widths; any other result, any value.
    fn arithmetic(&self, opcode: u32, words: &[u32], known: &[Known]) -> Constant {
        let program = self.program;
        // `OpSConvert`, `OpUConvert`, `OpSNegate`, `OpLogicalNot`, `OpNot`.
        let unary = matches!(opcode, 113 | 114 | 126 | 168 | 200);
        // `OpIAdd`, `OpISub`, `OpIMul`, the logical and integer
        // comparisons, the shifts and the bitwise operations.
        let binary = matches!(opcode, 128 | 130 | 132 | 164..=167 | 170..=179 | 194..=199);
        if !unary && !binary {
            return Constant::Varying;
        }
        // The width of an integer or boolean type.
        let width = |ty: Option<&u32>| {
            let ty = ty?;
            program.ints.get(ty).copied().or_else(|| program.booleans.contains(ty).then_some(1))
        };
        let (Some(bits), Some(from)) = (width(Some(&words[1])), width(program.types.get(&words[3]))) else {
            return Constant::Varying;
        };
        let operands = &words[3..if unary { 4 } else { 5 }];
        let constants = operands.iter().map(|id| known[*id as usize].value()).collect::<Vec<_>>();
        // An operand that decides the result alone (false of `&&`, true of
        // `||`, zero of a product or of `&`, all ones of `|`) decides it
        // whatever the other holds.
        let ones = truncated(u64::MAX, bits);
        let deciding = match opcode {
            132 | 167 | 199 => Some(0),
            166 => Some(1),
            197 => Some(ones),
            _ => None,
        };
        if let Some(decides) = deciding {
            if constants.iter().any(|constant| matches!(constant, Constant::Value(value) if *value == decides)) {
                return Constant::Value(decides);
            }
        }
        let mut values = [0u64; 2];
        let mut unknown = false;
        for (value, constant) in values.iter_mut().zip(&constants) {
            match constant {
                Constant::Value(bits) => *value = *bits,
                Constant::Unknown => unknown = true,
                Constant::Varying => return Constant::Varying,
            }
        }
        if unknown {
            return Constant::Unknown;
        }
        let [a, b] = values;
        // Sign-extended from `width` bits.
        let signed = |value: u64, width: u32| {
            let shift = u64::BITS - width;
            ((value << shift) as i64) >> shift
        };
        let truth = |holds: bool| u64::from(holds);
        let folded = match opcode {
            113 => a,
            114 => signed(a, from) as u64,
            126 => a.wrapping_neg(),
            168 => truth(a == 0),
            200 => !a,
            128 => a.wrapping_add(b),
            130 => a.wrapping_sub(b),
            132 => a.wrapping_mul(b),
            164 => truth(a == b),
            165 => truth(a != b),
            166 => truth(a != 0 || b != 0),
            167 => truth(a != 0 && b != 0),
            170 => truth(a == b),
            171 => truth(a != b),
            172 => truth(a > b),
            173 => truth(signed(a, from) > signed(b, from)),
            174 => truth(a >= b),
            175 => truth(signed(a, from) >= signed(b, from)),
            176 => truth(a < b),
            177 => truth(signed(a, from) < signed(b, from)),
            178 => truth(a <= b),
            179 => truth(signed(a, from) <= signed(b, from)),
            // A shift by the width or more is undefined.
            194..=196 if b >= u64::from(bits) => return Constant::Varying,
            194 => a >> b,
            195 => (signed(a, bits) >> b) as u64,
            196 => a << b,
            197 => a | b,
            198 => a ^ b,
            199 => a & b,
            _ => unreachable!("only the operations above are folded"),
        };
        Constant::Value(truncated(folded, bits))
    }

    /// By memory node: what it holds before anything is established, but
    /// what the entries hold.
    fn initial(&self) -> Vec<Held> {
        self.nodes.iter().map(Node::initial).collect()
    }

    /// By phi: the value it copies. A phi copies a value where every way
    /// into it some invocation may take brings that value, apart from the
    /// phi itself, phis that copy the value, and undefined values, and the
    /// value is defined before the phi on every path to it: outside the
    /// entry point, or in a block strictly dominating the phi's. Promotion
    /// leaves such phis in loops, of a variable a loop only reads, alone or
    /// as webs of phis among themselves. This is what invocations hold on
    /// the ways they may take (`live`), which the module's ids
    /// (`numbering`, numbered over every edge) cannot say: a phi one of
    /// whose ways no invocation takes is the value the others bring.
    ///
    /// The copies are the most a consistent assignment allows. Each round
    /// takes every phi not yet found to merge values for a copy, raising
    /// what each brings from nothing (`Brought`) until nothing rises; a phi
    /// two different values reach is found to merge them, as is one whose
    /// value is not defined before it, and the next round takes those as
    /// values of their own. A phi's value never changes but to many, so a
    /// copy's two ways in never bring different values: a phi is found to
    /// merge only where it does.
    fn copies(&self) -> HashMap<u32, u32> {
        let program = self.program;
        let undefined = |id: u32| {
            program.undefined.contains(&id)
                || program.defined.get(&id).is_some_and(|(_, words)| words[0] & 0xffff == OP_UNDEF)
        };
        let mut phis = Vec::new();
        for &block in &self.dominance.order {
            for words in &program.blocks[block].instructions {
                if words[0] & 0xffff == OP_PHI {
                    let phi = words[2];
                    let incoming = program
                        .incoming(phi, block, &self.successors)
                        .map(|(value, _)| value)
                        .filter(|&value| value != phi && !undefined(value))
                        .collect::<Vec<_>>();
                    phis.push((phi, block, incoming));
                }
            }
        }
        if phis.is_empty() {
            return HashMap::new();
        }
        let bound = program.bound;
        let mut is_phi = vec![false; bound];
        for &(phi, _, _) in &phis {
            is_phi[phi as usize] = true;
        }
        // The phis found to merge values, each a value of its own.
        let mut merging = vec![false; bound];
        let mut brought = vec![Brought::Nothing; bound];
        let mut found = vec![false; bound];
        loop {
            for &(phi, _, _) in &phis {
                brought[phi as usize] = Brought::Nothing;
                found[phi as usize] = false;
            }
            let mut changed = true;
            while changed {
                changed = false;
                for (phi, _, incoming) in &phis {
                    let phi = *phi;
                    if merging[phi as usize] {
                        continue;
                    }
                    let mut joined = brought[phi as usize];
                    for &value in incoming {
                        let arriving = if !is_phi[value as usize] || merging[value as usize] {
                            Brought::One(value)
                        } else {
                            brought[value as usize]
                        };
                        if let (Brought::One(held), Brought::One(other)) = (joined, arriving) {
                            if held != other {
                                found[phi as usize] = true;
                            }
                        }
                        joined = joined.join(arriving);
                    }
                    if joined != brought[phi as usize] {
                        brought[phi as usize] = joined;
                        changed = true;
                    }
                }
            }
            // A phi two different values reach merges them (one that many
            // reach only through other phis may copy one of those); one
            // bringing a value not defined before it is its own.
            let mut merged = false;
            for &(phi, block, _) in &phis {
                if merging[phi as usize] {
                    continue;
                }
                let own = match brought[phi as usize] {
                    Brought::Nothing => false,
                    Brought::One(value) => program
                        .defined
                        .get(&value)
                        .is_some_and(|&(defined, _)| defined == block || !self.dominance.dominates(defined, block)),
                    Brought::Many => found[phi as usize],
                };
                if own {
                    merging[phi as usize] = true;
                    merged = true;
                }
            }
            if !merged {
                return phis
                    .iter()
                    .filter_map(|&(phi, _, _)| match brought[phi as usize] {
                        Brought::One(value) => Some((phi, value)),
                        Brought::Nothing | Brought::Many => None,
                    })
                    .collect();
            }
        }
    }
}

/// What the ways into a phi bring in a round of [`Flow::copies`], each
/// apart from the phi itself and undefined values.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Brought {
    /// Nothing yet.
    Nothing,
    /// One value, or the value one phi copies.
    One(u32),
    /// More than one.
    Many,
}

impl Brought {
    fn join(self, other: Brought) -> Brought {
        match (self, other) {
            (Brought::Nothing, brought) | (brought, Brought::Nothing) => brought,
            (Brought::One(one), Brought::One(another)) if one == another => self,
            _ => Brought::Many,
        }
    }
}

impl<'p, 'a> Analysis<'p, 'a> {
    /// The analysis of the entry point where control and memory flow as
    /// `flow` has them, from nothing known.
    /// The analysis of `program`, its parameters holding `arguments`,
    /// raised to its fixed point: divergent values make constructs
    /// divergent and memory racy, which make more values divergent; all
    /// only grow. An entry point's edges no invocation can take go first: a
    /// branch on a constant takes one way alone.
    fn established(
        program: &'p Program<'a>,
        prepared: &'p Prepared,
        summaries: &'p Summaries<'a>,
        arguments: &[Held],
        context: Context,
    ) -> Result<Self, MalformedModule> {
        let flow = PreparedFlow { program, data: prepared };
        let mut analysis = Analysis::new(flow, summaries);
        for (parameter, held) in program.parameters.iter().zip(arguments) {
            let slot = prepared.ids.get(*parameter).expect("parameter has a slot");
            analysis.values[slot] = held.clone();
        }
        // A summarised function's body stores to no shared memory: what its
        // caller's do is what it sees.
        analysis.written = context.written;
        for (class, seen) in context.seen.into_iter().enumerate() {
            for block in &mut analysis.seen[class] {
                block.fill(seen);
            }
        }
        while analysis.propagate()? | analysis.control() | analysis.memory()? {}
        Ok(analysis)
    }

    fn new(flow: PreparedFlow<'p, 'a>, summaries: &'p Summaries<'a>) -> Self {
        let program = flow.program;
        let blocks = &program.blocks;
        let count = blocks.len();
        let data = flow.data;

        // What the entry point does not compute is a constant or a variable,
        // which holds no address, or undefined.
        let mut values = match &data.ids {
            IdIndex::Dense(bound) => {
                let mut v = (0..*bound)
                    .map(|slot| if data.defined_block[slot] != u32::MAX { Held::UNIFORM } else { Held::CONSTANT })
                    .collect::<Vec<_>>();
                for undefined in &program.undefined {
                    v[*undefined as usize] = Held::UNIFORM;
                }
                v
            }
            IdIndex::Sparse(map) => vec![Held::UNIFORM; map.len()],
        };
        let _ = &mut values;

        Self {
            // Nothing is established before `memory` first runs but what the
            // entries hold.
            held: data.initial.clone(),
            copies: &data.copies,
            constructs: &data.constructs,
            innermost: &data.innermost,
            enclosing_loop: &data.enclosing_loop,
            defined_block: &data.defined_block,
            landing: &data.landing,
            reading: &data.reading,
            flow,
            variant: vec![HashSet::new(); data.constructs.len()],
            variants_computed: false,
            values,
            origins: vec![Origins::default(); data.ids.len()],
            terminator: vec![Taint::UNIFORM; count],
            construct: vec![Taint::UNIFORM; data.constructs.len()],
            join: vec![Taint::UNIFORM; count],
            exited: vec![Taint::UNIFORM; count],
            seen: [WORKGROUP, WRITTEN]
                .map(|_| blocks.iter().map(|block| vec![Seen::default(); block.instructions.len()]).collect()),
            kept: blocks.iter().map(|block| vec![ValueState::default(); block.instructions.len()]).collect(),
            written: Reach::NONE,
            summaries,
        }
    }

    /// By loop: the values defined in it that may differ between its
    /// iterations. A loop's iterations differ by its phis and what memory
    /// holds: a value is variant when computed from those, by a subgroup
    /// operation, or from variant values. Memory the launch never writes is
    /// the same in every iteration, so a load of it at an address the same
    /// in every iteration is not: what the launch writes ([`Analysis::input`])
    /// only grows as the analysis rises, and so do these sets.
    fn variants(&self) -> Result<Vec<HashSet<u32>>, MalformedModule> {
        let program = self.flow.program;
        let mut variant = vec![HashSet::new(); self.constructs.len()];
        for (index, construct) in self.constructs.iter().enumerate().filter(|(_, construct)| construct.is_loop()) {
            let set = &mut variant[index];
            for &block in &construct.members {
                for (position, words) in program.blocks[block].instructions.iter().enumerate() {
                    let opcode = words[0] & 0xffff;
                    let shape = shape(opcode).expect("an analysed program holds only instructions of a known shape");
                    if !shape.result {
                        continue;
                    }
                    let changes = match opcode {
                        // A copy differs between iterations as its value
                        // does.
                        OP_PHI => self.copies.get(&words[2]).is_none_or(|copied| set.contains(copied)),
                        OP_LOAD => match program.place(words[3])? {
                            Place::ReadOnly | Place::BuiltIn(_) => false,
                            Place::Device => !self.input(words[3])?,
                            Place::Variable(_) | Place::Workgroup => true,
                        },
                        // A summarised function's call, as its body's reads
                        // do: of the same arguments, it computes the same.
                        OP_FUNCTION_CALL => self.summary(words, (block, position))?.is_some_and(|summary| summary.varies),
                        opcode => atomic(opcode) || subgroup_operation(opcode).is_some(),
                    };
                    if changes || shape.ids(words).iter().any(|id| set.contains(id)) {
                        set.insert(words[2]);
                    }
                }
            }
        }
        Ok(variant)
    }

    /// `construct` and the constructs around it, innermost first.
    fn outward(&self, construct: Option<usize>) -> impl Iterator<Item = usize> + '_ {
        std::iter::successors(construct, move |construct| self.constructs[*construct].parent)
    }

    /// The constructs holding `block`, innermost first.
    fn enclosing(&self, block: usize) -> impl Iterator<Item = usize> + '_ {
        self.outward(self.innermost[block])
    }

    /// `construct` holds `block`: it is one of the constructs holding it,
    /// which is where it holds the innermost.
    fn contains(&self, construct: usize, block: usize) -> bool {
        let nested = &self.constructs[construct].nested;
        self.innermost[block].is_some_and(|innermost| nested.contains(&self.constructs[innermost].nested.start))
    }

    /// An edge to `target` from within `construct` leaves it other than
    /// through its merge.
    fn left_by(&self, construct: usize, target: usize) -> bool {
        !self.contains(construct, target) && target != self.constructs[construct].merge
    }

    /// The constructs whose control holds `block`'s instructions, innermost
    /// first: those holding it, but a selection it heads. A selection
    /// header's instructions run before its branch, under the control
    /// around it; a loop header's run in every iteration, under the loop's.
    fn controlling(&self, block: usize) -> impl Iterator<Item = usize> + '_ {
        self.enclosing(block).filter(move |construct| {
            let construct = &self.constructs[*construct];
            construct.is_loop() || construct.header != block
        })
    }

    /// What the control `block`'s instructions run under depends on: the
    /// constructs controlling it.
    fn control_of(&self, block: usize) -> Taint {
        self.controlling(block).map(|construct| self.construct[construct]).fold(Taint::UNIFORM, Taint::join)
    }

    /// A workgroup control barrier in `block` is at a legal point: every
    /// invocation of the workgroup reaches it, under control, and after
    /// exits, that neither differ between the invocations of a subgroup nor
    /// between its subgroups.
    fn legal(&self, block: usize) -> bool {
        self.control_of(block).join(self.exited[block]) == Taint::UNIFORM
    }

    /// `block` holds a workgroup control barrier at a legal point, after
    /// which control is uniform again.
    fn synchronizes(&self, block: usize) -> bool {
        self.legal(block)
            && self.flow.program.blocks[block].instructions.iter().any(|words| self.flow.program.rejoins(words))
    }

    /// `block` is a selection header merging at `target`.
    fn selects_into(&self, block: usize, target: usize) -> bool {
        matches!(self.flow.program.blocks[block].merge, Some(Merge::Selection { merge }) if merge == target)
    }

    /// The loops holding `defined` that `block` is outside of: those a
    /// value defined at `defined` is carried out of where `block` uses it.
    fn left_loops(&self, defined: usize, block: usize) -> impl Iterator<Item = usize> + '_ {
        self.enclosing(defined)
            .filter(move |construct| self.constructs[*construct].is_loop() && !self.contains(*construct, block))
    }

    /// The loops a value `id` defines that may differ between its
    /// iterations is carried out of where `block` uses it.
    fn carried_out_of(&self, id: u32, block: usize) -> impl Iterator<Item = usize> + '_ {
        self.flow
            .program
            .defined
            .get(&id)
            .into_iter()
            .flat_map(move |(defined, _)| self.left_loops(*defined, block))
            .filter(move |construct| self.variant[*construct].contains(&id))
    }

    /// The stored `Held` for `id`, or `CONSTANT`/`UNIFORM` for external ids
    /// outside a sparse callee's local slot table.
    fn stored_value(&self, id: u32) -> &Held {
        match self.flow.ids.get(id) {
            Some(slot) => &self.values[slot],
            None => {
                if self.flow.program.undefined.contains(&id) {
                    &Held::UNIFORM
                } else {
                    &Held::CONSTANT
                }
            }
        }
    }

    /// Temporal taint of `id` used in `block` from any lane-varying loop it
    /// is carried out of, with an O(1) fast path when `id` is defined in
    /// `block` or not carried out of any enclosing loop.
    fn temporal(&self, id: u32, block: usize) -> Taint {
        let Some(slot) = self.flow.ids.get(id) else {
            return Taint::UNIFORM;
        };
        let defined = self.defined_block[slot];
        if defined == u32::MAX || defined as usize == block {
            return Taint::UNIFORM;
        }
        let defined = defined as usize;
        let Some(first_loop) = self.enclosing_loop[defined] else {
            return Taint::UNIFORM;
        };
        if self.contains(first_loop, block) {
            return Taint::UNIFORM;
        }
        self.left_loops(defined, block)
            .filter(|construct| self.variant[*construct].contains(&id))
            .map(|construct| self.construct[construct])
            .fold(Taint::UNIFORM, Taint::join)
    }

    /// What `id`, used in `block`, holds. A value a lane-varying loop
    /// defines that may differ between its iterations is divergent after
    /// it: invocations left it at different iterations.
    fn value(&self, id: u32, block: usize) -> Held {
        let temporal = self.temporal(id, block);
        self.stored_value(id).clone().joined(temporal)
    }

    /// What a variable's memory node `node`, seen from `block`, gains from
    /// the loops it is defined in and `block` is outside of: as a value they
    /// define, it may differ between their iterations, which invocations
    /// left at different ones.
    fn carried_out(&self, node: usize, block: usize) -> Taint {
        let defined = self.flow.defined_at[node];
        if defined == block {
            return Taint::UNIFORM;
        }
        let Some(first_loop) = self.enclosing_loop[defined] else {
            return Taint::UNIFORM;
        };
        if self.contains(first_loop, block) {
            return Taint::UNIFORM;
        }
        self.left_loops(defined, block)
            .map(|construct| self.construct[construct])
            .fold(Taint::UNIFORM, Taint::join)
    }

    fn value_state(&self, id: u32, block: usize) -> ValueState {
        let mut state = self.stored_value(id).state();
        let temporal = self.temporal(id, block);
        if temporal != Taint::UNIFORM {
            state.taint = state.taint.join(temporal);
        }
        state
    }

    fn operand(&self, id: u32, block: usize) -> Taint {
        self.value_state(id, block).taint
    }

    /// What `words`' id operands, in `block`, are together.
    fn operands(&self, words: &[u32], block: usize) -> ValueState {
        let shape = shape(words[0] & 0xffff).expect("an analysed program holds only instructions of a known shape");
        shape.positions(words).map(|at| self.value_state(words[at], block)).fold(ValueState::default(), ValueState::join)
    }

    /// `pointer` addresses device memory the launch only reads: the
    /// argument block, a buffer read through the input block or through an
    /// argument word nothing in the launch writes through (`written`), or
    /// memory addressed from addresses read from such buffers alone.
    fn input(&self, pointer: u32) -> Result<bool, MalformedModule> {
        Ok(self.flow.program.argument_block(pointer)?.is_some() || self.addressed(pointer).apart(self.written))
    }

    /// The memory `pointer` addresses: wherever it is used, as where it
    /// points does not change with the iteration a loop leaves at.
    fn addressed(&self, pointer: u32) -> Reach {
        self.stored_value(pointer).state().provenance.attributed()
    }

    /// The shared memory (workgroup, written device) `words` stores to.
    fn shared_written(&self, words: &[u32]) -> Result<Option<usize>, MalformedModule> {
        let Some(pointer) = written_pointer(words) else { return Ok(None) };
        Ok(match self.flow.program.place(pointer)? {
            Place::Workgroup => Some(WORKGROUP),
            Place::Device if !self.input(pointer)? => Some(WRITTEN),
            _ => None,
        })
    }

    /// What a read through `pointer` in `block` gives, where memory is
    /// `memory`, at the instruction it carries. Shared memory a store of the
    /// workgroup may race with ([`Analysis::raced`]) gives each invocation
    /// the stored value or not.
    fn read(&self, pointer: u32, block: usize, memory: &Walk<'_>) -> Result<Held, MalformedModule> {
        // Where no store races it, every invocation reading one address
        // reads one value, which is uniform whatever stored it; reading
        // apart, each reads what its own address holds.
        let scattered = self.operand(pointer, block).varies();
        // What the values stored to shared memory keep reaches the read
        // where it reads apart or a store may race it; a race, divergence.
        let seen = |class: usize| self.seen[class][memory.block][memory.at];
        let racing = |class: usize| {
            let raced = seen(class).raced.is_some();
            let read = if raced || scattered { seen(class).kept.taint } else { Taint::UNIFORM };
            if raced {
                read.join(Taint::DIVERGENT)
            } else {
                read
            }
        };
        let held = match self.flow.program.place(pointer)? {
            // The invocation's own, as its registers are: what the store
            // reaching the read put at its member (any element's, past an
            // array index).
            Place::Variable(Selection { path, .. }) => {
                let node = *self
                    .flow
                    .reaching
                    .get(&(memory.block, memory.at))
                    .expect("a read of a function or private variable has a node reaching it");
                memory.seen(node).get(path.iter().map(|(member, _)| *member)).joined(self.carried_out(node, block))
            }
            // What memory shared between invocations keeps is a value read
            // back, whether or not anything is stored.
            Place::Workgroup => Held::Whole(ValueState {
                taint: racing(WORKGROUP),
                provenance: Provenance::PLAIN.join(seen(WORKGROUP).kept.provenance),
            }),
            Place::Device => match self.flow.program.argument_block(pointer)? {
                // The argument block holds the buffers' addresses and the
                // words; the tables, values every launch reads alike.
                Some(provenance) => Held::Whole(ValueState { taint: Taint::UNIFORM, provenance }),
                // What memory the launch only reads by declaration holds
                // addresses into such memory alone, and an address table
                // into its port's bytes; any other memory's may point
                // anywhere (`Reach::holding`).
                None if self.input(pointer)? => {
                    let addressed = self.addressed(pointer);
                    let provenance = if addressed.none() {
                        // Nothing is known yet of where the pointer points.
                        Provenance::default()
                    } else {
                        Provenance::loaded(addressed.holding())
                    };
                    Held::Whole(ValueState { taint: Taint::UNIFORM, provenance })
                }
                // Memory the launch writes is never read alike: nothing
                // shows that no other workgroup stores the bytes read, and
                // no barrier orders its stores. What another workgroup
                // stores derives from its own invocations' places, none the
                // reader holds, so it adds the race alone; the reader's own
                // workgroup's stores race as they do in workgroup memory.
                None => Held::Whole(ValueState {
                    taint: racing(WRITTEN).join(Taint::DIVERGENT),
                    provenance: Provenance::loaded(Reach::ANYWHERE),
                }),
            },
            Place::ReadOnly => Held::CONSTANT,
            Place::BuiltIn(which) => Held::Whole(ValueState::plain(built_in(which))),
        };
        Ok(held.joined(self.operand(pointer, block)))
    }

    /// What the operands of `words`, in `block`, but the pointer it writes
    /// or reads through (its first id operand) are together: an atomic's
    /// value and scope, a cooperative-matrix store's object.
    fn carried(&self, words: &[u32], block: usize) -> ValueState {
        let shape = shape(words[0] & 0xffff).expect("an analysed program holds only instructions of a known shape");
        shape.ids(words).into_iter().skip(1).map(|id| self.value_state(id, block)).fold(ValueState::default(), ValueState::join)
    }

    /// The pointer `words`, in `block`, stores through and the value it
    /// stores, before what places it ([`Analysis::placing`]) joins it, where
    /// memory is `memory`.
    fn stored(&self, words: &[u32], block: usize, memory: &Walk<'_>) -> Result<Option<(u32, Held)>, MalformedModule> {
        let Some(pointer) = written_pointer(words) else { return Ok(None) };
        let value = match words[0] & 0xffff {
            OP_STORE => self.value(words[2], block),
            OP_VARIABLE => self.value(words[4], block),
            OP_COPY_MEMORY => self.read(words[2], block, memory)?,
            // An atomic's, a flag's or a cooperative-matrix store's: what it
            // stores is made of its operands besides the pointer, and of
            // what memory held there.
            _ => {
                let content = match read_pointer(words) {
                    Some(pointer) => self.read(pointer, block, memory)?.state(),
                    None => ValueState::default(),
                };
                Held::Whole(self.carried(words, block).join(content))
            }
        };
        Ok(Some((pointer, value)))
    }

    /// What a store through `pointer`, to `place`, in `block` joins to the
    /// value it stores: where each value lands on its address; and in
    /// memory invocations share, which invocations store, which the control
    /// that chose them decides. A variable is each invocation's own: what
    /// one holds differs by the control invocations took only where those
    /// that took different paths meet, which its memory SSA's phis join
    /// ([`Analysis::memory`]), as a value's phis do.
    fn placing(&self, pointer: u32, place: &Place, block: usize) -> Taint {
        let at = self.operand(pointer, block);
        match place {
            Place::Variable(_) => at,
            Place::Workgroup | Place::Device | Place::ReadOnly | Place::BuiltIn(_) => {
                at.join(self.control_of(block)).join(self.exited[block])
            }
        }
    }

    /// Carry the instruction `memory` is at, in `block`, through `memory`,
    /// where it stores `stored` ([`Analysis::stored`]); the value it places
    /// in workgroup or device memory, if it stores there.
    fn step(
        &self,
        stored: Option<(u32, Held)>,
        block: usize,
        memory: &mut Walk<'_>,
    ) -> Result<Option<Held>, MalformedModule> {
        let Some((pointer, value)) = stored else { return Ok(None) };
        let place = self.flow.program.place(pointer)?;
        let value = value.joined(self.placing(pointer, &place, block));
        match place {
            Place::Variable(Selection { path, whole, .. }) => {
                // A variable nothing reads has no node: what it holds is
                // never seen.
                if let Some(&node) = self.flow.defining.get(&(memory.block, memory.at)) {
                    let Node::Store { over } = self.flow.nodes[node] else {
                        unreachable!("a store defines a store node")
                    };
                    let mut held = memory.seen(over).clone().joined(self.carried_out(over, block));
                    held.write(&path, value, whole);
                    memory.stored.insert(node, held);
                }
            }
            Place::ReadOnly | Place::BuiltIn(_) => {
                return Err(MalformedModule(format!("%{pointer} is written, but addresses read-only memory")));
            }
            // A write to device memory the launch only reads is refused
            // (`violations`).
            Place::Workgroup | Place::Device => return Ok(Some(value)),
        }
        Ok(None)
    }

    /// By instruction of `block`: the legal barrier ordering workgroup
    /// memory in the block it last follows, if any; and the one the block's
    /// end last follows.
    fn fenced(&self, block: usize) -> (Vec<Option<usize>>, Option<usize>) {
        let program_block = &self.flow.program.blocks[block];
        let legal = self.legal(block);
        let mut last = None;
        let mut each = Vec::with_capacity(program_block.fences.len());
        for (index, fence) in program_block.fences.iter().enumerate() {
            each.push(last);
            if legal && *fence {
                last = Some(index);
            }
        }
        (each, last)
    }

    /// What the result of `words`, in `block`, holds, where memory is
    /// `memory`.
    fn result(&self, opcode: u32, words: &[u32], block: usize, memory: &Walk<'_>) -> Result<Held, MalformedModule> {
        let held = match opcode {
            OP_UNDEF => Held::UNIFORM,
            OP_LOAD => self.read(words[3], block, memory)?,
            // An address and an offset (`Provenance::sum`).
            OP_I_ADD | OP_I_SUB => {
                let first = self.value_state(words[3], block);
                let second = self.value_state(words[4], block);
                Held::Whole(ValueState {
                    taint: first.taint.join(second.taint),
                    provenance: first.provenance.sum(second.provenance),
                })
            }
            // A pointer into what its base pointer addresses: its element and
            // indices are offsets (`Provenance::indexed`).
            opcode if ACCESS_CHAINS.contains(&opcode) || ELEMENT_CHAINS.contains(&opcode) => {
                let base = self.value_state(words[3], block);
                let mut taint = base.taint;
                let mut provenance = base.provenance;
                for &index in &words[4..] {
                    let index = self.value_state(index, block);
                    taint = taint.join(index.taint);
                    provenance = provenance.indexed(index.provenance);
                }
                Held::Whole(ValueState { taint, provenance })
            }
            // A variable's address is every invocation's own copy of it.
            OP_VARIABLE => Held::UNIFORM,
            // A summarised function's call, what its body returns of its
            // arguments; a 32-lane hardware join's, its argument.
            OP_FUNCTION_CALL => match self.summary(words, (block, memory.at))? {
                Some(summary) => summary.result.clone(),
                None => self.value(words[4], block),
            },
            // A copy holds its value, wherever invocations arrive from.
            OP_PHI if self.copies.contains_key(&words[2]) => self.value(self.copies[&words[2]], block),
            OP_PHI => {
                let mut held = Held::Whole(ValueState::of(self.join[block]));
                for (value, parent) in self.flow.program.incoming(words[2], block, &self.flow.successors) {
                    held.join(&self.value(value, parent));
                }
                held
            }
            // A struct's members apart; an array's or vector's elements'
            // summary.
            OP_COMPOSITE_CONSTRUCT if self.flow.program.structs.contains_key(&words[1]) => {
                Held::Members(words[3..].iter().map(|id| self.value(*id, block)).collect())
            }
            OP_COMPOSITE_CONSTRUCT => words[3..].iter().fold(Held::UNIFORM, |mut summary, id| {
                summary.join(&self.value(*id, block));
                summary
            }),
            OP_COMPOSITE_EXTRACT => {
                let (path, _) = self.flow.program.members(self.flow.program.type_of(words[3])?, &words[4..], true)?;
                self.value(words[3], block).get(path.iter().map(|(member, _)| *member))
            }
            OP_COMPOSITE_INSERT => {
                let (path, whole) = self.flow.program.members(self.flow.program.type_of(words[4])?, &words[5..], true)?;
                let mut held = self.value(words[4], block);
                held.write(&path, self.value(words[3], block), whole);
                held
            }
            OP_COPY_OBJECT => self.value(words[3], block),
            OP_SELECT => {
                let mut held = self.value(words[4], block);
                held.join(&self.value(words[5], block));
                held.joined(self.operand(words[3], block))
            }
            // Each invocation reads its own, of what memory holds; a
            // cooperative-matrix fragment, by its lane. It is what memory
            // held, wherever the pointer points.
            opcode if atomic(opcode) || opcode == OP_COOPERATIVE_MATRIX_LOAD => {
                let operands = self.operands(words, block);
                let content = self.read(words[3], block, memory)?.state();
                let own = if opcode == OP_COOPERATIVE_MATRIX_LOAD {
                    operands.taint.join(Taint::LANE)
                } else {
                    Taint::DIVERGENT
                };
                Held::Whole(ValueState {
                    taint: operands.taint.join(content.taint).join(own),
                    provenance: content.provenance,
                })
            }
            opcode => {
                let operands = self.operands(words, block);
                let whole = |taint: Taint| Held::Whole(ValueState { taint, provenance: operands.provenance });
                let per_lane = operands.taint.join(Taint::LANE);
                let gathered = operands.taint.gathered();
                match subgroup_operation(opcode)
                    .map(|operation| operation.yields.expect("a subgroup operation with a result says what it yields"))
                {
                    None | Some(Yields::Operands) => Held::Whole(operands),
                    Some(Yields::Uniform) => whole(gathered),
                    // From a lane that differs by lane, a value that does; from
                    // one lane, one value for the subgroup.
                    Some(Yields::Lane(id)) => {
                        whole(if self.operand(words[id], block).varies() { per_lane } else { gathered })
                    }
                    Some(Yields::PerLane) => whole(per_lane),
                    // Scope, group operation, value, then the cluster size of a
                    // clustered reduction.
                    Some(Yields::Reduction) if words[4] == GROUP_OPERATION_REDUCE && words.len() == 6 => whole(gathered),
                    Some(Yields::BitCount) if words[4] == GROUP_OPERATION_REDUCE => Held::Whole(operands),
                    Some(Yields::Reduction | Yields::BitCount) => whole(per_lane),
                }
            }
        };
        // A condition holds no address: one on an address carries its
        // divergence, but not the address.
        Ok(if self.flow.program.booleans.contains(&words[1]) { Held::Whole(held.state().condition()) } else { held })
    }

    /// What `words`, at `at`, gives where it calls a summarised function, of
    /// the arguments it passes and the memory it sees there.
    fn summary(&self, words: &[u32], (block, position): At) -> Result<Option<std::rc::Rc<Summary>>, MalformedModule> {
        if words[0] & 0xffff != OP_FUNCTION_CALL || !self.summaries.programs.contains_key(&words[3]) {
            return Ok(None);
        }
        let arguments = words[4..].iter().map(|id| self.value(*id, block)).collect();
        let seen = |class: usize| {
            let seen: Seen = self.seen[class][block][position];
            Seen { raced: seen.raced.map(|_| Racer::Caller), kept: seen.kept }
        };
        let context = Context { written: self.written, seen: [seen(WORKGROUP), seen(WRITTEN)] };
        self.summaries.summary(words[3], arguments, context).map(Some)
    }

    /// Carry what values hold through the entry point under the current
    /// control and memory state; whether anything changed.
    fn propagate(&mut self) -> Result<bool, MalformedModule> {
        let program = self.flow.program;
        let mut changed_at_all = false;
        loop {
            let mut changed = false;
            for index in 0..self.flow.dominance.order.len() {
                let block = self.flow.dominance.order[index];
                let mut memory = Walk { block, at: 0, held: &self.held, stored: HashMap::new() };
                for (position, words) in program.blocks[block].instructions.iter().enumerate() {
                    memory.at = position;
                    let opcode = words[0] & 0xffff;
                    let shape = shape(opcode).expect("an analysed program holds only instructions of a known shape");
                    if shape.result {
                        let held = self.result(opcode, words, block, &memory)?;
                        let slot = self.flow.ids.get(words[2]).expect("defined result has a slot");
                        let risen = held.taint().0 & !self.values[slot].taint().0;
                        if risen != 0 {
                            let origins = self.origins_of(risen, words, (block, position));
                            for (origin, raised) in self.origins[slot].iter_mut().zip(origins) {
                                if raised.is_some() {
                                    *origin = raised;
                                }
                            }
                        }
                        changed |= self.values[slot].raise(&held);
                    }
                    self.step(self.stored(words, block, &memory)?, block, &mut memory)?;
                }
            }
            if !changed {
                return Ok(changed_at_all);
            }
            changed_at_all = true;
        }
    }

    /// Where each bit of `risen`, first raised in the result of `words` at
    /// `at`, comes from: where it was raised in the first operand that holds
    /// it (a phi's, arriving along an edge some invocation may take; a
    /// copy's, its value), or else this instruction, which raises it: a read
    /// with the store that may race it, where one may.
    fn origins_of(&self, risen: u8, words: &[u32], at: At) -> Origins {
        let block = at.0;
        let opcode = words[0] & 0xffff;
        // Each operand, with the block it is used in: a copy's is its value.
        let operands = match opcode {
            OP_PHI => match self.copies.get(&words[2]) {
                Some(&copied) => vec![(copied, block)],
                None => self.flow.program.incoming(words[2], block, &self.flow.successors).collect(),
            },
            _ => {
                let shape = shape(opcode).expect("an analysed program holds only instructions of a known shape");
                shape.ids(words).iter().map(|id| (*id, block)).collect()
            }
        };
        // A phi where invocations parted by control that holds the bit
        // rejoin holds it by that control, whatever its operands hold: the
        // construct merging there.
        let rejoined = |traced: Traced| {
            let bit = Taint(traced.bit());
            (opcode == OP_PHI && !self.copies.contains_key(&words[2]) && self.join[block].holds(bit))
                .then(|| {
                    (0..self.constructs.len())
                        .find(|&construct| self.constructs[construct].merge == block && self.construct[construct].holds(bit))
                })
                .flatten()
                .map(Origin::Rejoined)
        };
        let mut origins = Origins::default();
        for (origin, traced) in origins.iter_mut().zip(Traced::ALL).filter(|(_, traced)| risen & traced.bit() != 0) {
            *origin = Some(
                rejoined(traced)
                    .or_else(|| operands.iter().find_map(|&(id, used)| self.origin(id, used, traced)))
                    .unwrap_or_else(|| self.raised(words, at)),
            );
        }
        origins
    }

    /// The origin of a bit the result of `words`, at `at`, raises itself: a
    /// read of shared memory with the store that may race it, where one may;
    /// else the instruction.
    fn raised(&self, words: &[u32], at: At) -> Origin {
        let store = read_pointer(words).and_then(|pointer| {
            let class = match self.flow.program.place(pointer).expect("a pointer the analysis read has a place") {
                Place::Workgroup => WORKGROUP,
                Place::Device if !self.input(pointer).expect("a pointer the analysis read has a place") => WRITTEN,
                _ => return None,
            };
            self.seen[class][at.0][at.1].raced
        });
        match store {
            Some(store) => Origin::Raced { load: words[2], store },
            None => Origin::Value(words[2]),
        }
    }

    /// Where the traced bit `traced` of `id`, used in `block`, was raised, if `id`
    /// holds it there: where its value first held it, or a loop it is
    /// carried out of.
    fn origin(&self, id: u32, block: usize, traced: Traced) -> Option<Origin> {
        let bit = Taint(traced.bit());
        if let Some(slot) = self.flow.ids.get(id) {
            if self.values[slot].taint().holds(bit) {
                return self.origins[slot][traced as usize];
            }
        }
        self.carried_out_of(id, block).find(|construct| self.construct[*construct].holds(bit)).map(Origin::Loop)
    }

    /// What a loop is left or continued under.
    fn lane_varying(&self, loop_: usize) -> Taint {
        let construct = &self.constructs[loop_];
        construct
            .members
            .iter()
            .map(|&block| {
                // The constructs inside the loop holding `block` whose
                // control is not uniform: within the subgroup, or between
                // subgroups, which the values the loop carries out keep.
                let inner = self
                    .enclosing(block)
                    .take_while(|inner| *inner != loop_)
                    .filter(|inner| self.construct[*inner] != Taint::UNIFORM)
                    .collect::<Vec<_>>();
                let program_block = &self.flow.program.blocks[block];
                if program_block.exits() {
                    return inner.iter().map(|inner| self.construct[*inner]).fold(Taint::UNIFORM, Taint::join);
                }
                self.flow.successors[block]
                    .iter()
                    .filter(|&&target| {
                        !self.contains(loop_, target)
                            || Some(target) == construct.continue_target
                            || target == construct.header
                    })
                    .map(|&target| {
                        let branch =
                            if self.selects_into(block, target) { Taint::UNIFORM } else { self.terminator[block] };
                        inner
                            .iter()
                            .filter(|inner| self.left_by(**inner, target))
                            .map(|inner| self.construct[*inner])
                            .fold(branch, Taint::join)
                    })
                    .fold(Taint::UNIFORM, Taint::join)
            })
            .fold(Taint::UNIFORM, Taint::join)
    }

    /// Mark in `into`, with `taint`, the blocks reachable from `seeds` within
    /// `within` (the whole entry point where none), a search of `marks`.
    fn reach(&self, seeds: &[usize], within: Option<usize>, into: &mut [Taint], taint: Taint, marks: &mut Marks) {
        let inside = |block: usize| {
            self.flow.dominance.reachable(block) && within.is_none_or(|within| self.contains(within, block))
        };
        marks.search += 1;
        let search = marks.search;
        let mut stack = seeds.iter().copied().filter(|block| inside(*block)).collect::<Vec<_>>();
        while let Some(block) = stack.pop() {
            if std::mem::replace(&mut marks.seen[block], search) == search {
                continue;
            }
            raise(&mut into[block], taint);
            stack.extend(
                self.flow.successors[block]
                    .iter()
                    .copied()
                    .filter(|successor| inside(*successor) && marks.seen[*successor] != search),
            );
        }
    }

    /// Establish the control state the current values give; whether any
    /// rose.
    fn control(&mut self) -> bool {
        let program = self.flow.program;
        let count = program.blocks.len();
        let mut changed = false;
        for index in 0..self.flow.dominance.order.len() {
            let block = self.flow.dominance.order[index];
            let terminator = program.blocks[block].terminator();
            let taint = if self.flow.successors[block].len() > 1
                && matches!(terminator[0] & 0xffff, OP_BRANCH_CONDITIONAL | OP_SWITCH)
            {
                self.operand(terminator[1], block)
            } else {
                Taint::UNIFORM
            };
            changed |= raise(&mut self.terminator[block], taint);
        }
        for construct in 0..self.constructs.len() {
            if !self.constructs[construct].is_loop() {
                let taint = self.terminator[self.constructs[construct].header];
                changed |= raise(&mut self.construct[construct], taint);
            }
        }
        // A loop is lane-varying through a divergent construct inside it,
        // which may be a loop: until none rises.
        loop {
            let mut rose = false;
            for construct in 0..self.constructs.len() {
                if self.constructs[construct].is_loop() {
                    let taint = self.lane_varying(construct);
                    rose |= raise(&mut self.construct[construct], taint);
                }
            }
            if !rose {
                break;
            }
            changed = true;
        }

        let mut joins = vec![Taint::UNIFORM; count];
        for (index, construct) in self.constructs.iter().enumerate() {
            raise(&mut joins[construct.merge], self.construct[index]);
        }
        for &block in &self.flow.dominance.order {
            for &target in &self.flow.successors[block] {
                let taint = self
                    .enclosing(block)
                    .filter(|construct| self.left_by(*construct, target))
                    .map(|construct| self.construct[construct])
                    .fold(self.terminator[block], Taint::join);
                raise(&mut joins[target], taint);
            }
        }
        changed |= raise_all(&mut self.join, &joins);

        // A lane-dependent exit leaves its constructs other than through
        // their merges: from each of those merges, and from the other ways
        // its own branch goes, only some invocations run, up to the merge of
        // the construct it leaves to (or to the end, for a return). So do
        // only some subgroups after an exit taken under control that differs
        // between them, which a workgroup barrier may not follow.
        let mut exited = vec![Taint::UNIFORM; count];
        let mut marks = Marks { seen: vec![0; count], search: 0 };
        for &block in &self.flow.dominance.order {
            let taint = self.terminator[block].join(self.control_of(block));
            if taint == Taint::UNIFORM {
                continue;
            }
            if program.blocks[block].exits() {
                let merges =
                    self.enclosing(block).map(|construct| self.constructs[construct].merge).collect::<Vec<_>>();
                self.reach(&merges, None, &mut exited, taint, &mut marks);
                continue;
            }
            for &target in &self.flow.successors[block] {
                let mut seeds = Vec::new();
                let mut within = None;
                for construct in self.enclosing(block) {
                    if !self.left_by(construct, target) {
                        within = Some(construct);
                        break;
                    }
                    seeds.push(self.constructs[construct].merge);
                }
                if self.terminator[block] != Taint::UNIFORM {
                    seeds.extend(self.flow.successors[block].iter().copied().filter(|other| *other != target));
                }
                // Within control at least as dependent, the region is
                // checked as such.
                let checked = within.is_some_and(|within| {
                    self.outward(Some(within)).any(|construct| self.construct[construct].holds(taint))
                });
                if !seeds.is_empty() && !checked {
                    self.reach(&seeds, within, &mut exited, taint, &mut marks);
                }
            }
        }
        changed |= raise_all(&mut self.exited, &exited);
        changed
    }

    /// Establish what memory holds through the entry point under the current
    /// values and control state; whether anything rose.
    fn memory(&mut self) -> Result<bool, MalformedModule> {
        let program = self.flow.program;
        let count = program.blocks.len();
        if !self.flow.has_shared_stores && !self.flow.has_loops && self.flow.nodes.is_empty() {
            return Ok(false);
        }
        let mut written_changed = false;
        if self.flow.has_shared_stores {
            let mut written = self.written;
            for &block in &self.flow.dominance.order {
                for words in &program.blocks[block].instructions {
                    if let Some(pointer) = written_pointer(words) {
                        if matches!(program.place(pointer)?, Place::Device) {
                            written = written.join(self.addressed(pointer));
                        }
                    }
                }
            }
            written_changed = written != self.written;
            self.written = written;
        }
        let mut rose = written_changed;
        if self.flow.has_loops && (!self.variants_computed || written_changed) {
            // Less memory the launch only reads leaves more loads that may
            // differ between a loop's iterations.
            let variant = self.variants()?;
            rose |= variant != self.variant;
            self.variant = variant;
            self.variants_computed = true;
        }

        // By block and instruction: the shared memory (workgroup, written
        // device) it stores to.
        let mut classes: Vec<Vec<Option<usize>>> = Vec::new();
        if self.flow.has_shared_stores {
            classes = vec![Vec::new(); count];
            // By block: the legal barriers ordering workgroup memory its
            // instructions and its end follow ([`Analysis::fenced`]).
            let mut fenced: Vec<(Vec<Option<usize>>, Option<usize>)> = vec![(Vec::new(), None); count];
            for &block in &self.flow.dominance.order {
                let instructions = &program.blocks[block].instructions;
                classes[block] =
                    instructions.iter().map(|words| self.shared_written(words)).collect::<Result<Vec<_>, _>>()?;
                fenced[block] = self.fenced(block);
            }

            // The barrier intervals of workgroup memory the start of each block
            // is in: those opened by the entry or a legal barrier ordering
            // workgroup memory that reach it without crossing another.
            // Each comes with the entered loop ([`Entered`]) it came into along
            // its entering edge and has not gone around since: along it, that
            // loop's first test does not leave.
            let entered = |header: usize| self.flow.entered.iter().find(|entered| entered.header == header);
            let mut arriving = vec![BTreeSet::<(Opening, Option<usize>)>::new(); count];
            arriving[0].insert((None, None));
            loop {
                let mut changed = false;
                for &block in &self.flow.dominance.order {
                    let end = match fenced[block].1 {
                        Some(fence) => BTreeSet::from([(Some((block, fence)), None)]),
                        None => arriving[block].clone(),
                    };
                    for &successor in &self.flow.successors[block] {
                        for &(opening, fresh) in &end {
                            let first = fresh.and_then(entered);
                            if first.is_some_and(|first| first.test == block && first.exit == successor) {
                                continue;
                            }
                            let fresh = if entered(successor).is_some() && !self.flow.dominance.dominates(successor, block) {
                                Some(successor)
                            } else if fresh == Some(successor) {
                                // Around the loop.
                                None
                            } else {
                                fresh
                            };
                            changed |= arriving[successor].insert((opening, fresh));
                        }
                    }
                }
                if !changed {
                    break;
                }
            }
            let opened = arriving
                .iter()
                .map(|arriving| arriving.iter().map(|(opening, _)| *opening).collect::<BTreeSet<_>>())
                .collect::<Vec<_>>();
            // The intervals a store to shared memory is in, by its class: a
            // load in one of them may run beside it ([`Interleaving`]).
            let interval = |block: usize, fence: Option<usize>| match fence {
                Some(fence) => BTreeSet::from([Some((block, fence))]),
                None => opened[block].clone(),
            };
            // A store of the workgroup at `at`, of what it keeps.
            let store = |(block, index): At| Seen { raced: Some(Racer::Store((block, index))), kept: self.kept[block][index] };
            let agreed_branches = (0..count)
                .filter(|&block| {
                    self.flow.successors[block].len() >= 2 && self.terminator[block] == Taint::UNIFORM && self.legal(block)
                })
                .collect();
            let interleaving = Interleaving::new(&self.flow.successors, &fenced, &opened, agreed_branches);
            // By class, by barrier interval, by parting-branch signature: its
            // stores, summarized by where they land.
            let mut stored = [BTreeMap::<Opening, BTreeMap<Ways, Stores>>::new(), BTreeMap::new()];
            for &block in &self.flow.dominance.order {
                for (index, (class, fence)) in classes[block].iter().zip(&fenced[block].0).enumerate() {
                    if let Some(class) = class {
                        for opening in interval(block, *fence) {
                            let ways = interleaving.ways(opening, block, *fence);
                            stored[*class]
                                .entry(opening)
                                .or_default()
                                .entry(ways)
                                .or_default()
                                .add(self.landing[block][index], store((block, index)));
                        }
                    }
                }
            }
            // What the load in `block` after `fence`, of `location`, sees of the
            // stores of `class` in one of `intervals`: those that may touch its
            // bytes and are not parted from it by an agreed branch.
            let beside = |class: usize, intervals: &BTreeSet<Opening>, block: usize, fence: Option<usize>, location: Location| {
                intervals.iter().fold(Seen::default(), |seen, opening| {
                    let Some(buckets) = stored[class].get(opening) else {
                        return seen;
                    };
                    let ways = interleaving.ways(*opening, block, fence);
                    buckets
                        .iter()
                        .filter(|(store_ways, _)| !parted(store_ways, &ways))
                        .fold(seen, |seen, (_, stores)| seen.join(stores.seen(location)))
                })
            };
            // By class, by block: the stores of the workgroup that may run
            // before its start, whatever orders them.
            let mut after = [vec![Stores::default(); count], vec![Stores::default(); count]];
            loop {
                let mut changed = false;
                for &block in &self.flow.dominance.order {
                    for (class, after) in after.iter_mut().enumerate() {
                        let mut end = after[block].clone();
                        for (index, stores) in classes[block].iter().enumerate() {
                            if *stores == Some(class) {
                                end.add(self.landing[block][index], store((block, index)));
                            }
                        }
                        for &successor in &self.flow.successors[block] {
                            changed |= after[successor].raise(&end);
                        }
                    }
                }
                if !changed {
                    break;
                }
            }
            for index in 0..self.flow.dominance.order.len() {
                let block = self.flow.dominance.order[index];
                let mut before = [after[WORKGROUP][block].clone(), after[WRITTEN][block].clone()];
                let mut seen = [Vec::new(), Vec::new()];
                for (index, (class, fence)) in classes[block].iter().zip(&fenced[block].0).enumerate() {
                    // What memory an instruction sees is read only where it
                    // reads memory, or calls a summarised function, which reads
                    // it there ([`Analysis::read`], [`Analysis::summary`]).
                    let words = &program.blocks[block].instructions[index];
                    if read_pointer(words).is_some() || words[0] & 0xffff == OP_FUNCTION_CALL {
                        let intervals = interval(block, *fence);
                        let location = self.reading[block][index];
                        // A barrier orders workgroup memory: a store before it
                        // is seen, not raced. Nothing here orders device memory.
                        let ordered = Seen { raced: None, kept: before[WORKGROUP].seen(location).kept };
                        let written = before[WRITTEN].seen(location);
                        seen[WORKGROUP].push(ordered.join(beside(WORKGROUP, &intervals, block, *fence, location)));
                        seen[WRITTEN].push(written.join(beside(WRITTEN, &intervals, block, *fence, location)));
                    } else {
                        seen[WORKGROUP].push(Seen::default());
                        seen[WRITTEN].push(Seen::default());
                    }
                    if let Some(class) = class {
                        before[*class].add(self.landing[block][index], store((block, index)));
                    }
                }
                for (class, seen) in seen.iter().enumerate() {
                    rose |= raise_all(&mut self.seen[class][block], seen);
                }
            }
        }

        if self.flow.nodes.is_empty() && !self.flow.has_shared_stores {
            return Ok(rose);
        }

        // Forward, from nothing: what each node holds.
        let mut held = self.flow.initial();
        let mut kept = self.kept.clone();
        loop {
            let mut changed = false;
            for index in 0..self.flow.dominance.order.len() {
                let block = self.flow.dominance.order[index];
                // Where invocations from different predecessors meet, what a
                // variable holds depends on the control that parted them, as
                // a value's phi does; and what a loop defines, on the
                // iteration each left it at.
                for &phi in &self.flow.phis[block] {
                    let Node::Phi(incoming) = &self.flow.nodes[phi] else {
                        unreachable!("a block's phis are phi nodes")
                    };
                    let mut joined = Held::Whole(ValueState::of(self.join[block]));
                    for &(from, node) in incoming {
                        joined.join(&held[node].clone().joined(self.carried_out(node, from)));
                    }
                    if held[phi] != joined {
                        held[phi] = joined;
                        changed = true;
                    }
                }
                let mut memory = Walk { block, at: 0, held: &held, stored: HashMap::new() };
                for (position, words) in program.blocks[block].instructions.iter().enumerate() {
                    memory.at = position;
                    let placed = self.step(self.stored(words, block, &memory)?, block, &mut memory)?;
                    if let Some(value) = placed {
                        if self.flow.has_shared_stores && classes[block][position].is_some() {
                            let address = written_pointer(words)
                                .expect("an instruction storing to shared memory stores through a pointer");
                            let kept_state = value.state().kept(self.operand(address, block));
                            kept[block][position] = kept[block][position].join(kept_state);
                        }
                    }
                }
                let Walk { stored, .. } = memory;
                for (node, stored) in stored {
                    if held[node] != stored {
                        held[node] = stored;
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
        for (known, established) in self.held.iter_mut().zip(&held) {
            rose |= known.raise(established);
        }
        if self.flow.has_shared_stores {
            for (known, established) in self.kept.iter_mut().zip(&kept) {
                rose |= raise_all(known, established);
            }
        }
        Ok(rose)
    }

    /// Where invocations that divergent control parted rejoin, and what
    /// follows there before a legal workgroup barrier makes the workgroup's
    /// control uniform again: whether the function executes a subgroup
    /// operation or a control barrier (an entry point that does relies on
    /// the execution model's assumption), and whether it returns (a call of
    /// a summarised function that does is where its caller's invocations
    /// rejoin). They rejoin at or after the merge of a divergent selection,
    /// the block an exit taken under divergent control leaves to, and a call
    /// of a summarised function after which they do, or which steers under
    /// divergent control ([`Summaries::steers`]: the edges of its body that
    /// leave its constructs are taken under that control).
    fn rejoined(&self) -> Result<(bool, bool), MalformedModule> {
        let program = self.flow.program;
        // Where to look from: a block, and its instruction to start at.
        let mut stack = self
            .constructs
            .iter()
            .enumerate()
            .filter(|(index, construct)| !construct.is_loop() && self.construct[*index].varies())
            .map(|(_, construct)| (construct.merge, 0))
            .collect::<Vec<_>>();
        for &block in &self.flow.dominance.order {
            for (position, words) in program.blocks[block].instructions.iter().enumerate() {
                if let Some(summary) = self.summary(words, (block, position))? {
                    if summary.rejoins || (self.control_of(block).varies() && self.summaries.steers[&words[3]]) {
                        stack.push((block, position + 1));
                    }
                }
            }
            if program.blocks[block].exits() || !self.terminator[block].join(self.control_of(block)).varies() {
                continue;
            }
            for &target in &self.flow.successors[block] {
                if self.enclosing(block).any(|construct| self.left_by(construct, target)) {
                    stack.push((target, 0));
                }
            }
        }
        stack.retain(|(block, _)| self.flow.dominance.reachable(*block));
        // By block: the first instruction looked from.
        let mut from = vec![usize::MAX; program.blocks.len()];
        let (mut executes, mut returns) = (false, false);
        while let Some((block, start)) = stack.pop() {
            if start >= from[block] {
                continue;
            }
            from[block] = start;
            let legal = self.legal(block);
            let instructions = &program.blocks[block].instructions;
            let rejoined = instructions[start..].iter().position(|words| legal && program.rejoins(words));
            let reached = &instructions[start..rejoined.map_or(instructions.len(), |end| start + end)];
            executes |= reached.iter().any(|words| {
                let opcode = words[0] & 0xffff;
                subgroup_operation(opcode).is_some() || opcode == OP_CONTROL_BARRIER
            });
            if rejoined.is_none() {
                returns |= program.blocks[block].exits();
                stack.extend(self.flow.successors[block].iter().map(|successor| (*successor, 0)));
            }
        }
        Ok((executes, returns))
    }

    /// Where the entry point breaks the execution model: one violation per
    /// kind, with how many instructions break it so and the first (in
    /// reverse postorder) that does, and where what breaks it there comes
    /// from; and whether it relies on maximal reconvergence
    /// ([`Findings::relies_on_maximal_reconvergence`]): a subgroup operation
    /// or barrier after a lane-varying loop, accepted where the module
    /// declares it.
    fn found(&self) -> Result<(Found, bool), MalformedModule> {
        let mut found = Found::default();
        let mut relies_on_maximal_reconvergence = false;
        let after_loops = self.after_loops()?;
        for &block in &self.flow.dominance.order {
            let inside = |looping: bool| {
                self.controlling(block).find(|construct| {
                    self.construct[*construct].varies() && self.constructs[*construct].is_loop() == looping
                })
            };
            let (place, region) = if let Some(selection) = inside(false) {
                (Some("inside a divergent selection"), Some(selection))
            } else if let Some(loop_) = inside(true) {
                (Some("inside a lane-varying loop"), Some(loop_))
            } else if self.exited[block].varies() {
                (Some("after a lane-dependent exit"), None)
            } else {
                (None, None)
            };
            let after_each = self.after_each(block, after_loops[block])?;
            // Memory as each instruction sees it, carried as `propagate`
            // carries it.
            let mut memory = Walk { block, at: 0, held: &self.held, stored: HashMap::new() };
            for (position, words) in self.flow.program.blocks[block].instructions.iter().enumerate() {
                memory.at = position;
                let after_loop = after_each[position];
                let opcode = words[0] & 0xffff;
                let shape = shape(opcode).expect("an analysed program holds only instructions of a known shape");
                let ids = shape.ids(words);
                let at = || format!("at {}", self.instruction(words));
                // A summarised function's call: its body reports what it
                // breaks there.
                let summarised = opcode == OP_FUNCTION_CALL && self.summaries.programs.contains_key(&words[3]);
                // Where the local id and the lane order come from, of `ids`.
                let orders = |ids: &[u32]| {
                    let local = ids.iter().find_map(|id| self.origin(*id, block, Traced::Local));
                    let lane = ids.iter().find_map(|id| {
                        self.origin(*id, block, Traced::Lane).or_else(|| self.origin(*id, block, Traced::Subgroup))
                    });
                    format!("{}{}{}", at(), self.traced(local, "the local id"), self.traced(lane, "the lane order"))
                };
                // Where a value first derives from both orders: none of
                // its operands does, or its stored value and address alone.
                if shape.result
                    && !summarised
                    && self.stored_value(words[2]).taint().mixes()
                    && !ids.iter().any(|id| self.operand(*id, block).mixes())
                {
                    found.report(MIXING.to_owned(), || orders(std::slice::from_ref(&words[2])));
                }
                let stored = self.stored(words, block, &memory)?;
                if let Some(&(pointer, ref value)) = stored.as_ref() {
                    let target = self.flow.program.place(pointer)?;
                    let taint = value.taint();
                    if taint.join(self.placing(pointer, &target, block)).mixes()
                        && !taint.mixes()
                        && !self.operand(pointer, block).mixes()
                    {
                        found.report(MIXING.to_owned(), || orders(ids.as_slice()));
                    }
                    let common = match target {
                        Place::Device => {
                            if self.input(pointer)? {
                                found.report("a write to memory the launch only reads".to_owned(), at);
                            }
                            true
                        }
                        Place::Workgroup => true,
                        Place::Variable(_) | Place::ReadOnly | Place::BuiltIn(_) => false,
                    };
                    // An address leaves the registers and variables, where
                    // the analysis follows it, for no memory (`Provenance`).
                    if common && value.state().provenance.is_named() {
                        found.report(
                            "a store of a buffer's address, or of a value derived from one, to workgroup or device memory"
                                .to_owned(),
                            at,
                        );
                    }
                }
                if let Some(operation) = subgroup_operation(opcode) {
                    let name = operation.name;
                    if let Some(place) = place {
                        found.report(format!("{name} {place}"), || format!("{}{}", at(), self.region(region, "in")));
                    } else if let Some(after) = after_loop {
                        if self.summaries.maximal_reconvergence {
                            relies_on_maximal_reconvergence = true;
                        } else {
                            found.report(format!("{name} after a lane-varying loop"), || {
                                format!("{}{}", at(), self.after(after))
                            });
                        }
                    }
                    for (word, what) in operation.uniform {
                        if let Some(&id) = words.get(*word).filter(|id| self.operand(**id, block).varies()) {
                            found.report(format!("{name} with a divergent {what}"), || {
                                format!("{}{}", at(), self.traced(self.origin(id, block, Traced::Varies), "it varies"))
                            });
                        }
                    }
                    let operands = operation.shape.ids(words);
                    if operands.iter().any(|id| self.operand(*id, block).holds(Taint::LOCAL)) {
                        found.report(format!("{name} on a value derived from the local or global invocation id"), || {
                            let local = operands.iter().find_map(|id| self.origin(*id, block, Traced::Local));
                            format!("{}{}", at(), self.traced(local, "the local id"))
                        });
                    }
                } else if opcode == OP_CONTROL_BARRIER {
                    if let Some(place) = place {
                        found.report(format!("OpControlBarrier {place}"), || {
                            format!("{}{}", at(), self.region(region, "in"))
                        });
                    } else if self.flow.program.rejoins(words) && !self.legal(block) {
                        // Every invocation of a subgroup reaches it, but not
                        // every subgroup of the workgroup.
                        found.report(
                            "OpControlBarrier of the workgroup under control that differs between subgroups".to_owned(),
                            || {
                                let steering = self
                                    .controlling(block)
                                    .find(|construct| self.construct[*construct] != Taint::UNIFORM);
                                format!("{}{}", at(), self.region(steering, "in"))
                            },
                        );
                    } else if self.flow.program.rejoins(words) {
                        // Every invocation of the workgroup executes it
                        // together, which ends the state after a loop
                        // (`after_each`).
                    } else if let Some(after) = after_loop {
                        // A subgroup barrier needs its subgroup together,
                        // which only maximal reconvergence rejoins after the
                        // loop, as for a subgroup operation there.
                        if self.summaries.maximal_reconvergence {
                            relies_on_maximal_reconvergence = true;
                        } else {
                            found.report("OpControlBarrier of a subgroup after a lane-varying loop".to_owned(), || {
                                format!("{}{}", at(), self.after(after))
                            });
                        }
                    }
                } else if summarised {
                    let summary =
                        self.summary(words, (block, position))?.expect("a summarised function's call has a summary");
                    let callee = &self.summaries.programs[&words[3]].entry;
                    for (kind, (count, first)) in &summary.found.0 {
                        found.report_many(kind.clone(), *count, || format!("{} of `{callee}`, there {first}", at()));
                    }
                } else if opcode == OP_FUNCTION_CALL && self.operand(words[4], block).varies() {
                    let name = &self.flow.program.joins[&words[3]];
                    let name = name.split('(').next().expect("a split yields its first part");
                    found.report(format!("`{name}` of a value that differs within the hardware subgroup"), || {
                        format!("{}{}", at(), self.traced(self.origin(words[4], block, Traced::Varies), "it differs"))
                    });
                }
                self.step(stored, block, &mut memory)?;
            }
        }
        Ok((found, relies_on_maximal_reconvergence))
    }

    /// An instruction as a refusal names it: its result and opcode, and the
    /// name of its result, or of the variable its pointer is into.
    fn instruction(&self, words: &[u32]) -> String {
        let opcode = words[0] & 0xffff;
        let shape = shape(opcode).expect("an analysed program holds only instructions of a known shape");
        let result = shape.result.then_some(words[2]);
        let pointer = read_pointer(words).or_else(|| written_pointer(words));
        let name = result
            .and_then(|id| self.flow.program.names.get(&id))
            .or_else(|| pointer.and_then(|pointer| self.flow.program.names.get(&self.flow.program.root(pointer))));
        // As SPIR-V assembly writes it: an instruction with a result by its
        // id, one without by the ids it takes.
        let text = match result {
            Some(id) => format!("%{id} = {}", opcode_name(opcode)),
            None => shape.ids(words).iter().fold(opcode_name(opcode), |text, id| format!("{text} %{id}")),
        };
        match name {
            Some(name) => format!("{text} (`{name}`)"),
            None => text,
        }
    }

    /// `id` as a refusal names it: the instruction defining it, or the id
    /// and its name.
    fn named(&self, id: u32) -> String {
        match self.flow.program.defined.get(&id) {
            Some((_, words)) => self.instruction(words),
            None => match self.flow.program.names.get(&id) {
                Some(name) => format!("%{id} (`{name}`)"),
                None => format!("%{id}"),
            },
        }
    }

    /// `, {what} from {origin}`, where there is an origin.
    fn traced(&self, origin: Option<Origin>, what: &str) -> String {
        self.traced_within(origin, what, ORIGIN_DEPTH)
    }

    /// [`Analysis::traced`], following constructs' conditions back through
    /// at most `depth` more constructs (they may form a cycle).
    fn traced_within(&self, origin: Option<Origin>, what: &str, depth: usize) -> String {
        match origin {
            Some(Origin::Rejoined(construct)) if depth > 0 => {
                format!(", {what} where control rejoins{}", self.region_within(Some(construct), "after", depth - 1))
            }
            Some(Origin::Rejoined(construct)) => format!(
                ", {what} where control rejoins after the construct at %{}",
                self.flow.program.blocks[self.constructs[construct].header].label
            ),
            Some(Origin::Value(id)) => format!(", {what} from {}", self.named(id)),
            Some(Origin::Raced { load, store: Racer::Store((block, index)) }) => format!(
                ", {what} from {}, which {} may race",
                self.named(load),
                self.instruction(&self.flow.program.blocks[block].instructions[index])
            ),
            Some(Origin::Raced { load, store: Racer::Caller }) => {
                format!(", {what} from {}, which a store of the caller may race", self.named(load))
            }
            Some(Origin::Loop(loop_)) => {
                format!(", {what} from the loop at %{}", self.flow.program.blocks[self.constructs[loop_].header].label)
            }
            None => String::new(),
        }
    }

    /// `, {preposition} the selection or loop at %{header}`, steered by its
    /// condition and where that differs from, where there is a construct.
    fn region(&self, construct: Option<usize>, preposition: &str) -> String {
        self.region_within(construct, preposition, ORIGIN_DEPTH)
    }

    /// [`Analysis::region`], its condition's origin traced within `depth`
    /// ([`Analysis::traced_within`]).
    fn region_within(&self, construct: Option<usize>, preposition: &str, depth: usize) -> String {
        let Some(construct) = construct else { return String::new() };
        let header = self.constructs[construct].header;
        let kind = if self.constructs[construct].is_loop() { "loop" } else { "selection" };
        let mut text = format!(", {preposition} the {kind} at %{}", self.flow.program.blocks[header].label);
        if let Some((block, condition)) = self.steering(construct) {
            text += &format!(" steered by {}", self.named(condition));
            let taint = self.construct[construct];
            let traced = if taint.varies() { Traced::Varies } else { Traced::Subgroup };
            text += &self.traced_within(self.origin(condition, block, traced), "which differs", depth);
        }
        text
    }

    /// The block and condition steering `construct`: a selection's header's;
    /// a loop's first branch, in reverse postorder, under a condition that
    /// is not uniform.
    fn steering(&self, construct: usize) -> Option<(usize, u32)> {
        let condition = |block: usize| {
            let terminator = self.flow.program.blocks[block].terminator();
            matches!(terminator[0] & 0xffff, OP_BRANCH_CONDITIONAL | OP_SWITCH).then_some((block, terminator[1]))
        };
        let construct = &self.constructs[construct];
        if construct.is_loop() {
            construct
                .members
                .iter()
                .copied()
                .find(|block| self.terminator[*block] != Taint::UNIFORM)
                .and_then(condition)
        } else {
            condition(construct.header)
        }
    }

    /// By block: the first lane-varying loop, or call ending after one
    /// ([`Summary::loops`]), it is entered after, with no legal workgroup
    /// control barrier since. Each search runs onward from a loop's merge,
    /// or from the end of a block holding such a call, up to such a
    /// barrier, after the earlier ones': a block an earlier one reached, it
    /// reached onward from too.
    fn after_loops(&self) -> Result<Vec<Option<After>>, MalformedModule> {
        let mut after = vec![None; self.flow.program.blocks.len()];
        let mut stack = Vec::new();
        for loop_ in 0..self.constructs.len() {
            let merge = self.constructs[loop_].merge;
            if self.constructs[loop_].is_loop()
                && self.construct[loop_].varies()
                && self.flow.dominance.reachable(merge)
                && after[merge].is_none()
            {
                after[merge] = Some(After::Loop(loop_));
                stack.push(merge);
            }
        }
        for &block in &self.flow.dominance.order {
            if after[block].is_none() && self.after_each(block, None)?.last().copied().flatten().is_some() {
                stack.push(block);
            }
        }
        while let Some(block) = stack.pop() {
            let Some(end) = self.after_each(block, after[block])?.last().copied().flatten() else { continue };
            for &successor in &self.flow.successors[block] {
                if after[successor].is_none() {
                    after[successor] = Some(end);
                    stack.push(successor);
                }
            }
        }
        Ok(after)
    }

    /// By instruction of `block`, entered after `entered`
    /// ([`Analysis::after_loops`]), then at its end: the lane-varying loop,
    /// or call ending after one, it follows with no legal workgroup control
    /// barrier since.
    fn after_each(&self, block: usize, entered: Option<After>) -> Result<Vec<Option<After>>, MalformedModule> {
        let program = self.flow.program;
        let legal = self.legal(block);
        let mut after = entered;
        let mut each = Vec::with_capacity(program.blocks[block].instructions.len() + 1);
        for (position, words) in program.blocks[block].instructions.iter().enumerate() {
            each.push(after);
            if legal && program.rejoins(words) {
                after = None;
            } else if self.summary(words, (block, position))?.is_some_and(|summary| summary.loops) {
                after = Some(After::Call(words[2]));
            }
        }
        each.push(after);
        Ok(each)
    }

    /// `, after` the loop or call `after`, as a refusal names it.
    fn after(&self, after: After) -> String {
        match after {
            After::Loop(loop_) => self.region(Some(loop_), "after"),
            After::Call(call) => format!(", after {}, whose function ends after a lane-varying loop", self.named(call)),
        }
    }
}

/// What control is after ([`Analysis::after_loops`]): a lane-varying loop,
/// or a call of a summarised function whose body ends after one (by its
/// result).
#[derive(Clone, Copy, Debug)]
enum After {
    Loop(usize),
    Call(u32),
}

const MIXING: &str = "an instruction combining the local or global invocation id with the lane order of subgroups";

/// The violations found, by kind: how many instructions break the model so,
/// and the first.
#[derive(Default)]
struct Found(BTreeMap<String, (usize, String)>);

impl Found {
    /// One more instruction breaks the model as `kind`; `first` says where,
    /// if it is the first.
    fn report(&mut self, kind: String, first: impl FnOnce() -> String) {
        self.report_many(kind, 1, first);
    }

    /// `count` more instructions break the model as `kind`; `first` says
    /// where the first does, if they are the first.
    fn report_many(&mut self, kind: String, count: usize, first: impl FnOnce() -> String) {
        match self.0.entry(kind) {
            std::collections::btree_map::Entry::Occupied(mut entry) => entry.get_mut().0 += count,
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert((count, first()));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{check, Findings, Flow, Program};

    /// Memory takes a node per store to and meeting of a variable something
    /// reads, whatever else the entry point holds: an entry point inlining
    /// `CALLS` calls, each with `LOCALS` locals stored to and one of them
    /// stored again in a branch and read after it, holds per call that
    /// local's entry, its two stores and the phi where the branch rejoins,
    /// however many locals and blocks there are (a state of every variable
    /// at every block would be (3 × `CALLS` + 2) × `CALLS` × `LOCALS`).
    #[test]
    fn memory_takes_a_node_per_definition_of_a_variable_read() {
        const CALLS: u32 = 48;
        const LOCALS: u32 = 24;
        fn op(module: &mut Vec<u32>, opcode: u32, operands: &[u32]) {
            module.push(((operands.len() as u32 + 1) << 16) | opcode);
            module.extend_from_slice(operands);
        }
        // "main", its terminating null in the word after.
        let name = u32::from_le_bytes(*b"main");
        let mut next = 1;
        let mut id = || {
            next += 1;
            next - 1
        };
        let (void, signature, int, boolean, pointer, truth, seven, main) =
            (id(), id(), id(), id(), id(), id(), id(), id());
        let locals = (0..CALLS).map(|_| (0..LOCALS).map(|_| id()).collect::<Vec<_>>()).collect::<Vec<_>>();
        // Each call's header, branch, merge and load; then the exit.
        let calls = (0..CALLS).map(|_| [id(), id(), id(), id()]).collect::<Vec<_>>();
        let (entry, exit) = (id(), id());
        let bound = id();

        let mut module = vec![0x0723_0203, 0x0001_0300, 0, bound, 0];
        // `OpCapability Shader`, `OpMemoryModel Logical GLSL450`,
        // `OpEntryPoint GLCompute`, `OpExecutionMode LocalSize 1 1 1`.
        op(&mut module, 17, &[1]);
        op(&mut module, 14, &[0, 1]);
        op(&mut module, 15, &[5, main, name, 0]);
        op(&mut module, 16, &[main, 17, 1, 1, 1]);
        op(&mut module, 5, &[main, name, 0]);
        op(&mut module, 19, &[void]);
        op(&mut module, 33, &[signature, void]);
        op(&mut module, 21, &[int, 32, 0]);
        op(&mut module, 20, &[boolean]);
        // A pointer to a function variable.
        op(&mut module, 32, &[pointer, 7, int]);
        op(&mut module, 41, &[boolean, truth]);
        op(&mut module, 43, &[int, seven, 7]);
        op(&mut module, 54, &[void, main, 0, signature]);
        op(&mut module, 248, &[entry]);
        for local in locals.iter().flatten() {
            op(&mut module, 59, &[pointer, *local, 7]);
        }
        op(&mut module, 249, &[calls[0][0]]);
        for (call, [header, branch, merge, load]) in calls.iter().enumerate() {
            op(&mut module, 248, &[*header]);
            for local in &locals[call] {
                op(&mut module, 62, &[*local, seven]);
            }
            op(&mut module, 247, &[*merge, 0]);
            op(&mut module, 250, &[truth, *branch, *merge]);
            op(&mut module, 248, &[*branch]);
            op(&mut module, 62, &[locals[call][0], seven]);
            op(&mut module, 249, &[*merge]);
            op(&mut module, 248, &[*merge]);
            op(&mut module, 61, &[int, *load, locals[call][0]]);
            op(&mut module, 249, &[calls.get(call + 1).map_or(exit, |following| following[0])]);
        }
        op(&mut module, 248, &[exit]);
        op(&mut module, 253, &[]);
        op(&mut module, 56, &[]);

        let program = Program::parse(&module).expect("module");
        // Every edge kept, as the constant `truth` would otherwise drop one.
        let flow = Flow::new(&program, program.successors()).expect("flow");
        assert_eq!(flow.nodes.len(), 4 * CALLS as usize);
        assert_eq!(
            check(&module, false),
            Ok(Findings {
                violations: Vec::new(),
                relies_on_reconvergence: false,
                relies_on_maximal_reconvergence: false
            })
        );
    }
}
