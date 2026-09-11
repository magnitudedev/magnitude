# Weights

**A container is forgotten at residency. Kernels read representations; one owner
holds every resident weight; the layout decision is made once, from the stored
layout, the shape and the endpoint.**

## From container to kernel

```text
container ──► stored weight ──► decision ──► resident representation ──► views ──► commands
  format        blocks, dense,    stored ×       dense / blocked /        every         retain the
  parses and    affine planes     shape ×        planar affine             consumer      allocation
  validates                       capability                               shares one    past the owner
```

| Stage | Owns | Never knows |
|---|---|---|
| Format | Parsing, validation, content identity, what is stored for a role | Devices, layouts kernels want |
| Decision | Which resident layout a stored layout becomes here | Which model asked, which operation will read |
| Residency | Upload, repack, conversion, the one allocation, views, groups | Container identity |
| Kernel | Reading one representation | Where it came from |

## Representations

A representation is parameters, not a name. Two containers that yield the same
parameters share every kernel that reads them.

| Representation | Parameters | Read cost | Bits per value |
|---|---|---|---|
| Dense | dtype | none | the dtype |
| Encoded blocks | codec and block geometry | codec-specific decode | the declared block geometry |
| Planar affine | bits, high bits, group, coefficient width, sign, bias | word loads of one plane set; one affine correction per group | bits + coefficient bits over the group |
| Hierarchical affine | code bits, group and supergroup geometry, local coefficient width, sign, bias, packing | packed code dot; local coefficients interpreted in registers; one correction per group | the compact block geometry |

Planar affine storage is one allocation with a fixed plane order, and the offsets
of those planes have one definition shared by the kernel that writes them, the
upload that places a container's planes at the same offsets, and every kernel
that reads them. Hierarchical affine storage preserves compact local and superblock
coefficients; its packing is part of the execution representation, not the name of
the container that supplied it.

## The decision

| Stored | Becomes | Because |
|---|---|---|
| Affine planes | Planar affine, coefficients as stored | The container already is the resident layout |
| Hierarchical affine blocks | Hierarchical affine, coefficients as stored | Decode applies compact local and superblock coefficients inside the contraction; no persistent expansion is paid on every token |
| Other encoded blocks | Encoded blocks | The codec is an execution fact even when no affine hierarchy describes it |
| Dense floats | Dense FP32 | Parameters are read by every layer in FP32; the declared transform is applied here, once |

The decision reads capability, never a backend. A resident weight records its
representation, and run records report it, so a layout change is visible as such.

## Ownership

| Rule | Reason |
|---|---|
| One allocation per weight or compatible row-concatenated group | Every consumer views the same bytes; there is no second copy to keep coherent |
| A group is declared before any member is asked for alone | Grouping after individual residency would mean a copy; declared first, a member is a row-offset view |
| Commands retain the allocation | The container and the owner may retire while work that reads the weight is still submitted |
| Residency checks the artifact identity the description came from | A description and a container from different artifacts fail at binding, never as wrong numbers |
| Tokenizer metadata never reaches residency | It travels in the same files but is an input concern |

## Extension

A format brings a parser that yields stored weights and, per model, a mapping
from its names to the model's roles. Wire type numbers and recipe labels stop at
that parser. If its stored layout is one a representation already covers, no
kernel changes. A representation is new only when no parameterization of an
existing one describes its numerical meaning and physical bytes, and it brings
the kernels that read it.
