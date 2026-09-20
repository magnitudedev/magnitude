# Azure worker preparation

These files deploy the coordinator infrastructure and prepare disposable Ubuntu 24.04 CPU
workers for the existing outward worker protocol. A successful deployment is not proof of
a completed app test. Windows, GPU driver setup, other distributions and Hermes provisioning
require separate qualification.

Use subscription `5304c4b3-d605-4193-b0cb-766c065acfa6` and resource group `magnitude-ci`
explicitly; the Azure CLI's default subscription may be different.

`network.bicep` creates a private worker subnet and explicit NAT egress. Workers have no
public IP. This network is shared infrastructure, not a per-run lease: delete its virtual
network, NAT gateway and egress public IP when retiring the deployment. NAT and its public
IP incur charges even without workers.

```sh
az deployment group create \
  --subscription 5304c4b3-d605-4193-b0cb-766c065acfa6 \
  --resource-group magnitude-ci --name lab-network \
  --template-file packages/testing-lab/infra/network.bicep \
  --parameters location=westus2
```

Create the **trusted test runtime** from an administrator-reviewed checkout, separately
from source candidates submitted by developers or CI:

```sh
bun packages/testing-lab/scripts/create-worker-runtime.ts . /tmp/lab-runtime
```

This captures dirty source through the normal snapshot validator, writes `runtime.tar.gz`
and records its source identity, SHA-256 and size in `runtime.json`. The archive contains
source and frozen dependency locks, not local dependencies, credentials or application data.
Keep the output directory outside the checkout. Upload the archive to private blob storage
and obtain a read-only HTTPS capability valid through the intended allocation window.

Prepare a private JSON configuration with this shape. Every download requires its exact
archive length and SHA-256; the renderer rejects HTTP, URL user credentials and fragments.

```json
{
  "adminUsername": "labworker",
  "architecture": "x64",
  "runtime": { "url": "https://…", "sha256": "<64 lowercase hex characters>", "bytes": 123 },
  "node": { "url": "https://…", "sha256": "<64 lowercase hex characters>", "bytes": 123 },
  "rustup": { "url": "https://…", "sha256": "<64 lowercase hex characters>", "bytes": 123 }
}
```

Node must be a matching Linux tar.xz distribution with Node 24 or newer. Rustup must be a
matching native executable from a versioned release. The setup installs the repository's
pinned Bun and Rust toolchain, frozen lab dependencies, locked Pi/OpenCode clients, build
dependencies, and an Xvfb/Openbox/D-Bus desktop session. It verifies that the outward worker
imports successfully. This is an X11 automation environment; it does not establish Wayland
or physical-display qualification.

```sh
bun packages/testing-lab/scripts/prepare-ubuntu-worker.ts \
  /private/path/initialization.json /private/path/cloud-init.yml
```

The command prints only `{ "file": "…", "sha256": "…" }`. Add that object as
`initialization` on the corresponding Azure image in the coordinator configuration.
The setup file contains private download capabilities: keep it private and out of Git.
Use the same `adminUsername` in allocation and initialization configuration. Set the
image's disk to at least 128 GiB for source builds.

The matching guest runtime uses `executable: "/opt/magnitude-lab-worker"`, `args: []`,
`root: "/home/labworker/lab-runs"`, and `disposable: true`. The launcher supplies the
display session, actual Node executable and pinned harness paths. Its outward worker
receives only the attempt-scoped credential delivered by the existing Azure bootstrap.

Cloud-init runs before credential delivery. Initialization failure fails allocation;
the scheduler owns deletion through the normal lease and reconciler. A successful VM
provisioning state is insufficient: the allocator checks the pinned initialization tag
and waits for `cloud-init status --wait` to exit successfully. Native setup diagnostics
are available in `lab-initialize` and `/var/log/cloud-init-output.log` while the VM exists.


## Coordinator deployment

Deploy `foundation.bicep` first: it creates the private registry, coordinator managed identity,
Container Apps environment, network and DNS. Deploy `database.bicep` next: it creates PostgreSQL
16 with private network access through VNet integration and network peering. The actual subscription
rejects PostgreSQL in West US 2; the database defaults to West US 3 while the coordinator and
workers stay in West US 2. Pass the database password through a private secure-parameter file,
never a shell argument or Git file. PostgreSQL backup retention is seven days.

Bundle `src/server.ts` with the repository-pinned Bun using `bun build --target=bun`. Place the
result as `coordinator.js` beside `Dockerfile` and `entrypoint.sh` in a private build context.
Build that context using `az acr build --registry magnitudelab5304 --platform linux/amd64`.
Only this generated context is uploaded; do not send the entire working directory or secrets.
Resolve the resulting image digest, then deploy `coordinator.bicep` using that immutable reference.
Supply a unique `revision` suffix whenever the image or configuration changes. Updating a
Container Apps secret alone does not restart the process that read its previous value.

Service parameters contain a TLS-verifying database URL, base64-encoded server configuration,
base64-encoded worker cloud-init, and a private operator token. Supply them through a mode-0600
parameter file. Container Apps stores these as secrets; the entrypoint creates private files and
logs in with the coordinator managed identity. Run credentials are still delivered separately
through the protected Azure worker bootstrap. The coordinator has one always-running replica,
2 vCPUs and 4 GiB memory, with database/object state outside the replica. Entra developer
credentials remain supported by the ordinary server configuration.

The managed identity can allocate resources within magnitude-ci, pull the private image and
access artifact blobs. Worker VMs do not receive that identity. Foundation resources are durable
service infrastructure and intentionally outlive individual runs. Their ongoing costs include
PostgreSQL, the coordinator replica, registry, NAT and logs; worker VMs remain per-run leases.

Artifact transfer uses the [Azure Blob REST API](https://learn.microsoft.com/en-us/rest/api/storageservices/put-blob)
over a shared HTTP client. Azure CLI acquires a renewable Entra credential; it is not invoked
for every object. Conditional writes preserve immutable content addresses, and conditional reads
plus byte limits and SHA-256 verification protect downloads. Source manifests verify owner-scoped
object metadata in batches so admission does not require a database round trip per source file.

The initial runtime download capability is time-limited. Its expiry must be tracked and its
configuration renewed before subsequent allocations; automatic capability renewal or prepared
image publication remains required before unattended long-term operation. Never present a
one-time valid URL as a permanently provisioned worker image.
