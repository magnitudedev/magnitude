# Numerical applicability

Numerical construction determines where a physical implementation satisfies the entry's precision policy. It composes the meaning of the actual operations with construction; it does not attach approval to an independently created executable.

| Input | Output |
| --- | --- |
| Reference semantics, selected physical operations, legal input/state domain and precision policy | Established whole-entry applicability regions and their derived explanation, established exclusions, or pending reasoning |

## The contract

The first applicable portable body defines reference behavior, including conversions, accumulation, association, rounding and exceptional values. Policies are:

| Policy | Requirement |
| --- | --- |
| Exact | Agreement with the reference contract, including its permitted outcomes and representation rules. |
| Bounded | Declared whole-entry floating deviation, with explicit absolute/relative/ULP and special-value rules. |
| Unconstrained | Explicit exploration without a bounded floating guarantee. Structural, discrete, effect, memory and progress requirements still apply. |

Returns and final writable state belong to the same contract. Read-only bytes remain unchanged. Integer/Boolean behavior, checks, discrete decisions and effects do not acquire floating tolerance. A local approximation that changes an index or check must satisfy the resulting whole-entry behavior, not just a local error estimate.

Scalar operations have one language-owned typed-bit recipe implementation. The reference interpreter
evaluates it and reference kernel construction instantiates it into terminal unsigned word arithmetic,
Boolean selection and payload transport. Floating arithmetic rounds once at each declared output;
narrow FMA forms the exact product-plus-addend before its single rounding. Transcendental recipes
retain their ordered primitive steps and use that same owner. Exactness is not inferred from an
empty list of numerical effects or a factory-supplied flag.

Typed payloads preserve NaN bits and signed zero during transport. Abs, Neg and identity casts act
at the original representation width; arithmetic invalid results and nonidentity NaN conversions
use the declared canonical NaN. Decimal tokens retain the lexer’s parsed F64 meaning and quantize
once in the checker to their contextual dtype. Source `exp_fast` normalizes to the same semantic
exponential as `exp`; an approximate exponential is an explicit physical choice governed by the
caller’s policy. Numeric source integers and internal natural/index values retain distinct types.

Native narrow scalar registers retain the original 16 payload bits: CPU and CUDA use the low
16 bits of a U32 carrier, and Metal uses `ushort`. Loads, selections, joins, carries, stores and
result publication move those bits directly. An explicitly selected physical numerical operation
decodes and re-encodes only at its operation boundary. F32 transport likewise retains its original
32 bits without a host F64 conversion.

Ordinary packed reads instantiate the registry decode sequence before kernel closure. A typed field
read returns either raw U32 code bits or the coefficient’s storage dtype. Code interpretation,
floating-code lookup, multiplication, FMA and final conversion then become the same scalar recipe
operations used by the interpreter. A packed vector read guards each lane before any field load,
produces positive zero for inactive lanes, and assembles the resulting payloads in lane order.
Native field extraction follows the packet’s little-endian byte layout and reads only touched bytes;
it performs no numerical decoding. Dense vector reads retain their native operation.

A scalar recipe’s partial integer result includes actual failure predicates. Source traversal
publishes the cause and enters the existing structured success continuation before binding that
result. Its total eager terminal graph cannot trap in an unused exceptional arm. Physical native
arithmetic remains a distinct realization under the same operation owner; native lowering does not
silently select it after closure.

## Construct meaning with operations

Primitive constructors derive numerical relations from the selected operation, operands, dtype, rounding and target mode. Sequence composes state transitions; calls substitute actual operands/results; branches compose guarded alternatives; loops compose recurrences; stores include destination rounding and final state. Reassociation, contraction, FTZ, approximation and representation changes are explicit choices.

The general reference recipe preserves that semantics directly through construction. It does not need a general-purpose equivalence search before ordinary preparation can succeed.

Supported alternative relations follow actual bound operands and selected child bodies. Scalar
recipe comparison uses the same terminal word/Boolean graph with actual operand substitution;
it does not infer equality from a body label, matching type, or empty numerical-effect list.
Tensor maps, stateful loops and native operations require their own complete operation relation.
Until one exists for a selected alternative, that alternative remains Pending. A missing required
source operation is a construction gap to fix.

Optional alternatives require whole-entry composition. Giving each helper the full entry tolerance is unsound. Bounds must account for accumulation, cancellation, subsequent operations, mutable state and control consequences.

Content assumptions come from the checked contract and actual control paths. A caller-supplied search range does not establish a tensor-content property. Adding an input scanner to justify an otherwise unsafe transformation is not a substitute for source-preserving construction.

## Universal applicability

Let `m` be invocation metadata, `x` all legal input contents and initial state, `p` a physical outcome, and `r` a permitted reference outcome. For an admitted metadata region:

```text
for every legal x and every permitted physical outcome p:
    there is a permitted reference outcome r
    such that the whole-entry policy relation holds between r and p
```

This is more demanding than matching one interpreter representative. Safety, effect ordering and fair progress must also hold; a relation containing only terminating outputs cannot establish termination.

The exact relation can be expressed as absence of a counterexample:

```text
Legal(x) AND Physical(x, p)
         AND NOT EXISTS r: (Reference(x, r) AND Policy(r, p))
```

Quantification over `x,p` searches all legal contents and permitted physical outcomes. Failure states are explicit outcomes. Tensor contents do not become sampled “qualification points.” Metadata guards may mention only information available through the invocation binding contract.

## Resolution strategy

The required source body executes the checked operations through the source constructor. Its complete
value products and continuation carry the result, state and failures; it does not wait for a second
whole-program outcome replay to establish its own meaning. Selecting a different body requires an
actual relation to that required computation. The supported scalar and bit-recipe comparisons can
establish some exact alternatives; unsupported tensor, effect, failure or progress relations remain
Pending. A different parallel mapping also needs an exact realization of its mathematical quantities
and participation rules before it is selectable.

The current implementation has no general nonzero whole-entry error bound and no general symbolic
equivalence engine for arbitrary tensors or controllers. A Bounded policy therefore accepts Exact
applicability but cannot use an unproved nonzero-error alternative. Unconstrained policy can accept
a proved discrete relation with unresolved floating values; it cannot bypass memory, checks,
failure or progress obligations. Analysis limitations affect optimization, not availability of the
required source construction. Numerical applicability belongs to actual construction and its
selected children, not an attachable effect list or a second approval artifact.

## Execution boundary and observations

An applicable candidate owns its accepted regions and explanations derived together from its construction. Neither a raw guard nor an independent “evidence” or certificate object can widen them. Native formation may remain private while reasoning is pending. Timing and retained execution expose only applicable, natively reconciled candidates.

Preparation returns a numerical-pending outcome when its current analysis establishes no selectable
region. It does not create a trial handle, cache a semantic rejection, or remove the structural
alternative. Analytical completion reports that pending reasoning; feedback reports pending
preparation attempts. Native artifacts may remain shared for later reasoning. Bounded policies have
no evidence mode, and validation corpus results have no accepted-scope constructor.

The interpreter and comparator are testing tools. A completed interpreter outcome owns outputs, final input backing and the allowed-outcome relation. One comparison operation consumes this complete outcome and the actual completed native observation. Read-only storage is compared exactly; unsupported comparisons remain unsupported.

Source termination is either a complete returned product or an actual source failure; both retain
observable final input backing. Invalid invocation, reference allowance exhaustion and device
failure remain separate errors. A failed source outcome preserves writes before its failing event
and has no successful return product. Diagnostic comparison checks the complete representative,
including termination and immutable bytes. A differing parallel failure prefix is Unsupported
unless the allowed-outcome relation can decide it; merely observing that both runs failed never
establishes agreement. Observation and comparison cannot grant numerical applicability.

Source failures identify the actual checked failure event in its stable body namespace, plus its
typed cause. Rebuilding an entry preserves that identity; fresh program handles, diagnostic text,
and native addresses do not define it. Distinct authored bodies require an established event
correspondence before an otherwise matching failed observation can be accepted. Source display
locations remain separate from this comparison identity.

Native source stops pass through terminal completion while ordinary resource ownership remains
live. Controlled trials copy final input state before dropping or restoring their private buffers;
public diagnostic calls retain their private inputs and gates through observation. Completed source
stops have no timing sample. Device failures and incomplete native state never become source stops.

A definite mismatch from an admitted candidate is a compiler/backend defect. It cannot be treated as routine qualification failure, quietly removed from the portfolio, or repaired by testing more inputs. A comparison success never grants applicability. See [feedback evaluation](feedback-evaluation.md) for controlled performance observations.
