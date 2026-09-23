# Seismic

Seismic turns authored computational structure into implementations selected for a particular hardware and backend. Its language describes the computation, its compiler constructs and evaluates permitted implementations, and its runtime binds and executes the selected implementations. The standard library supplies reusable computations and authored alternatives.

## The optimality equation

Fix a computation, its allowed behavior and precision, a hardware/backend deployment, and an objective `J` to minimize. For example, `J` can be invocation latency, or expected latency under an explicitly supplied workload distribution. Let `J*` be the best attainable value for that fixed problem.

Seismic's central requirement is:

```text
sufficient authored structure
  + exhaustive permitted physical and numerical variations
  + optimal native emission for each fixed physical contract
  + exact evaluation and global selection
  = the best attainable implementation

J(Form(Select*(CandidateDomain))) = J*
```

`Select*` denotes ideal global selection; `Form` realizes the selected physical and emission choices.

The premises separate these responsibilities:

| Premise | Responsibility |
| --- | --- |
| The authored structure can express an optimal implementation. | The kernel author supplies the mathematical structure and any alternative algorithms. Ordinary reasonable structural spellings must qualify; hidden recognition of a particular helper name is insufficient. |
| The candidate domain contains every potentially optimal permitted realization of that structure. | The compiler represents physical mappings, storage, communication, scheduling and numerical choices without arbitrary template or search caps. |
| Emission is optimal within a fixed physical contract. | The domain exposes permitted emission alternatives, the evaluator compares them, and the backend forms the selected requests while preserving the contract. |
| Evaluation selects the actual global best. | The evaluator compares the domain under the stated objective. |

The domain should also be **minimal**: it should not independently choose derived facts or retain redundant encodings of the same decisions. Distinct emissions of one physical contract remain alternatives. Minimality avoids unnecessary search; duplicates do not themselves invalidate the equation.

The first three premises are exact design requirements. Native compilation alone does not establish optimal emission. Practical evaluation searches physical and emission alternatives together: analytical models have prediction error, and feedback measures only part of the space under imperfect conditions. The target is near-optimal selection. Neither a finite measurement budget nor a good benchmark score establishes a general near-optimality bound.

“100%” means the attainable optimum for this computation and contract. It does not mean simultaneously saturating nominal FLOPs, bandwidth and every hardware resource. Where the best cost is an unattained infimum, the corresponding goal is approaching it; exact selection cannot manufacture an attaining implementation.

## The system

```text
authored Seismic
       |
       v
checked language + concrete entry types
       |
       v
LogicalEntry + target + precision policy
       |
       v
CandidateDomain <---------- CandidateEvaluator
       |                    requests construction,
       |                    estimates or measures,
       v                    retains and selects
physical IR -> native executable ------^
                                      |
                                      v
                               SelectionPolicy
                                      |
                                      v
                                PreparedKernel
                                      |
                          actual invocation arguments
                                      |
                                      v
                         binding -> workflow -> execution
```

A candidate fixes the physical and emission choices for one whole-entry implementation. It can contain several kernels, transfers, loops and commands. A prepared kernel can retain several candidates and select between them using invocation metadata. Preparation does not produce a separate compiled program for every future invocation.

## Rules across the system

The author owns mathematical work; the compiler owns its physical realization. Tiling a matrix multiplication or using a matching matrix primitive is physical choice. Replacing the authored algorithm with an unprovided factorization or a table of answers is not.

The entry's legal invocation domain comes from its source contract and the external target ABI. Internal temporary sizes, search budgets, sampled shapes and incomplete compiler reasoning cannot narrow that domain. A general reference construction must cover it, subject to real resource and service availability.

Guarantees belong to the main construction paths and their inputs and outputs. An operation creates its values, effects and meaning together; an executable owns its resource structure; an invocation owns its actual bindings. Attaching an “evidence” object, certificate, hash or successful test result cannot make unrelated code conform. Tests help find defects; exhaustiveness requires an argument covering the constructors and their composition.

## Reading map

| Area | Start here | Questions answered |
| --- | --- | --- |
| Language | [Language overview](language/overview.md) | What should authors express, and what freedom does that leave the compiler? |
| Compiler | [Compiler overview](compiler/overview.md) | What are the pipeline entities, their guarantees and their ownership boundaries? |
| Execution | [Execution overview](execution/overview.md) | How do prepared implementations become actual resource-owning executions? |
