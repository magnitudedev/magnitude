# ICN model fit assessment

ICN assesses model fit through one native planning pipeline shared by downloaded inventory models
and remote previews. `POST /v1/models/preview` resolves immutable artifact metadata without
downloading full weights, inspects the complete component set, and evaluates every requested
execution profile against `GET /v1/hardware`'s normalized memory domains.

A complete result is `Fits` or `DoesNotFit`. Each result identifies the resolved profile and reports
per-domain model, context, compute, auxiliary, required, available, and margin bytes. ACN does not
estimate memory, reserve capacity, inspect the OS, or maintain architecture-specific formulas.

Runtime load reassesses the concrete requested profile before allocating resources. Preview is
therefore accurate before download while load remains the final safety boundary. Both use the same
artifact identities, native build, hardware topology, execution policy, and capacity policy in
their cache keys.
