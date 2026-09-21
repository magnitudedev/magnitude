# Azure worker preparation

These files deploy the coordinator infrastructure and prepare disposable Ubuntu 24.04, Debian 13, Fedora 44 and Red Hat 10 CPU
workers for the existing outward worker protocol. A successful deployment is not proof of
a completed app test. See [the coverage ledger](../COVERAGE.md) for actual qualification.
Red Hat's Azure image leaves its home logical volume at 1 GiB; preparation verifies the expected
XFS/LVM layout and expands that volume to 64 GiB before installing dependencies.
Windows preparation installs pinned tools and runtime dependencies, creates the admitted user's
interactive desktop, removes one-shot login credentials, and verifies live readiness. It is
integrated with allocation and natively proven on a disposable Windows Server diagnostic;
Windows 10/11 still require eligible client licensing and their own application qualification.
GPU driver preparation is implemented but native GPU qualification awaits quota. CUDA source producers have a separate pinned
SDK preparation phase; its native qualification status is recorded in the coverage ledger.
Windows delivery now supports an already prepared interactive desktop user: its configured
runtime must be a native `.exe` with absolute local paths. The bootstrap creates a temporary
Interactive scheduled task, verifies the actual user and nonzero session, observes its exit,
and removes the system-owned launcher and scoped credential. It does not create a login
session or install build dependencies. The launcher has native Windows Server validation;
Windows 10/11 app coverage is not established by that diagnostic. RHEL 10
uses a private headless GNOME Wayland session because [Red Hat removed the X.Org server](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/10/html/10.0_release_notes/removed-features);
the worker waits for a logical monitor before starting. Its compositor and sockets are removed on
exit. Bounded compositor logs accompany failed execution. Electron inherits `XDG_SESSION_TYPE=wayland`;
no inference backend override is applied. See [Electron’s native Wayland behavior](https://www.electronjs.org/blog/tech-talk-wayland).

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
  "distribution": { "os": "ubuntu", "version": "24.04" },
  "adminUsername": "labworker",
  "architecture": "x64",
  "runtime": { "url": "https://…", "sha256": "<64 lowercase hex characters>", "bytes": 123 },
  "node": { "url": "https://…", "sha256": "<64 lowercase hex characters>", "bytes": 123 },
  "rustup": { "url": "https://…", "sha256": "<64 lowercase hex characters>", "bytes": 123 }
}
```

Use `distribution: { "os": "debian", "version": "13" }` for Debian or
`distribution: { "os": "fedora", "version": "44" }` for Fedora, or
`distribution: { "os": "redhat", "version": "10" }` for Red Hat (including its 10.x minor versions). The allocator rejects a
recipe/target mismatch before allocating; the guest rejects an image/recipe mismatch before
installing dependencies. Existing Ubuntu recipes must be migrated from `kind: "ubuntu"` to
`kind: "linux"` with an explicit distribution when deploying this version. Do not change a
running coordinator merely to migrate the recipe.

The official Debian image pins verified in West US 2 are
`Debian:debian-13:13-gen2:0.20260914.2601` (x64) and
`Debian:debian-13:13-arm64:0.20260914.2601` (ARM64). Both use generation 2 and have no
marketplace purchase plan. These pins establish image availability, not worker qualification.
Fedora publishes official Azure VHDs for both architectures at its [Cloud download page](https://www.fedoraproject.org/cloud/download/).
They require a verified import into a versioned Azure gallery image before allocation; the
allocator already accepts an explicit gallery version ID. No third-party marketplace image
or additional compute provider is required. Fedora's native packages include
[Xvfb with xvfb-run](https://packages.fedoraproject.org/pkgs/xorg-x11-server/xorg-x11-server-Xvfb/fedora-44.html)
and [Openbox](https://packages.fedoraproject.org/pkgs/openbox/openbox/). The pinned Hermes
release requires Python below 3.14, so Fedora uses its parallel
[Python 3.13 package](https://packages.fedoraproject.org/pkgs/python3.13/python3.13/); the OS Python is unchanged.

The Red Hat x64 image pin `RedHat:RHEL:10_2-gen2:10.2.2026080415` is available in West US 2,
uses generation 2, and has no marketplace purchase plan. A disposable native probe installed
the recipe's packages, created a GNOME Wayland logical monitor, mapped a real GTK window,
and verified display socket cleanup. A deliberate worker exit 17 preserved that exit and left
no compositor process or private socket directory. This proves the desktop mechanism, not packaged-app
or inference qualification. Red Hat's recipe omits the removed X11 screensaver library.

The verified Fedora imports are gallery `magnitude_lab`, definitions `fedora44-x64` and
`fedora44-arm64`, version `44.1.7`, in `magnitude-ci`. Both are replicated in West US 2.
These are persistent base images, not qualified application workers. Tag persistent image
resources with `lab-owner=magnitude-testing-lab-images-v1`. The marker
`magnitude-testing-lab-v1` is reserved for disposable lease resources with `lab-machine` and
`lab-lease` metadata; using it on gallery resources prevents cleanup inventory from succeeding.

Node must be a matching Linux tar.xz distribution with Node 24 or newer. Rustup must be a
matching native executable from a versioned release. The setup installs the repository's
pinned Bun and Rust toolchain, frozen lab dependencies, locked Pi/OpenCode clients, build
dependencies and a private desktop session: Xvfb/Openbox/D-Bus on Ubuntu, Debian and Fedora;
GNOME/Wayland/D-Bus on Red Hat 10. It verifies that the outward worker imports successfully.
Neither virtual desktop establishes physical-display qualification.

```sh
bun packages/testing-lab/scripts/prepare-linux-worker.ts \
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
and waits for `cloud-init status --wait` to exit successfully. On preparation failure, bounded cloud-init diagnostics are redacted and retained as run
evidence before VM deletion. Retrieve their digest through `bun lab evidence`; access is
limited to the run owner. The guest also retains `lab-initialize` status and
`/var/log/cloud-init-output.log` during its lifetime.


## Coordinator deployment

Deploy `foundation.bicep` first: it creates the private registry, coordinator managed identity,
Container Apps environment, network and DNS. Deploy `database.bicep` next: it creates PostgreSQL
16 with private network access through VNet integration and network peering. The actual subscription
rejects PostgreSQL in West US 2; the database defaults to West US 3 while the coordinator and
workers stay in West US 2. Pass the database password through a private secure-parameter file,
never a shell argument or Git file. PostgreSQL backup retention is seven days.

Bundle `src/server.ts` with the repository-pinned Bun using `bun build --target=bun`. Place the
result as `coordinator.js` beside `Dockerfile`, `entrypoint.sh`, `linux-worker.sh`,
`macos-worker.sh`, `windows-tools.ps1`, `windows-runtime.ps1`, `windows-desktop.ps1`,
`tailscale-start.sh`, `spark-docker.sh` and `spark-ssh.sh` in a private build context.
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

For trusted Spark runs, set the Spark allocator executable to `/opt/lab/spark-docker.sh`,
host to `ssh://tom@sparky`, and image to an immutable digest already available to that
Docker daemon. Supply `tailscaleAuthKey` through the secure deployment parameter file.
Use a reusable key with ephemeral devices so coordinator replacements enroll automatically
and offline replicas disappear. This is a separate coordinator credential, never a worker
environment variable. It must be rotated before its enrollment expiry; an expired key cannot
start a replacement coordinator. The existing tailnet policy must authorize that identity to
SSH to Sparky as `tom`. Enrollment does not narrow the organization's existing network policy.
Do not copy a developer's personal SSH private key into the service.

The coordinator runs Tailscale in userspace with no inbound listener, subnet route or privileged
container. Tailscale's SSH wrapper resolves MagicDNS and authenticates the remote host key;
only the Spark Docker client uses this route. Enrollment material is written to a private
temporary file, removed after enrollment and unset before the application starts. Tailscale
startup failures prevent that revision from becoming ready; other healthy revisions must not
be removed before the new revision is ready.

Build `Spark.Dockerfile` with the pinned worker archive as `runtime.tar.gz` and
`spark-prepare.sh` and `spark-update.rules` under their original names in the context. Use a native ARM64 Azure builder: the qualified
Bun version crashed under the registry builder's ARM emulation. Image construction, dependency
installation and candidate source compilation do not run on the office Spark. Its configured
runtime launches `/bin/bash /opt/lab/runtime/packages/testing-lab/infra/spark-worker.sh`, with
`/lab` as the workspace and an explicit small model. Native generation remains a separate
qualification from successfully building or pulling the image.
The container owns a private network namespace with `NET_ADMIN` for scoped network fault tests,
plus its own system D-Bus and Polkit service for the packaged updater. It has no host network,
host filesystem mounts or privileged-container mode; these services never run on Sparky's host.

The managed identity can allocate resources within magnitude-ci, pull the private image and
access artifact blobs. Worker VMs do not receive that identity. Foundation resources are durable
service infrastructure and intentionally outlive individual runs. Their ongoing costs include
PostgreSQL, the coordinator replica, registry, NAT and logs; worker VMs remain per-run leases.

Artifact transfer uses the [Azure Blob REST API](https://learn.microsoft.com/en-us/rest/api/storageservices/put-blob)
over a shared HTTP client. Azure CLI acquires a renewable Entra credential; it is not invoked
for every object. Conditional writes preserve immutable content addresses, and conditional reads
plus byte limits and SHA-256 verification protect downloads. Source manifests verify owner-scoped
object metadata in batches so admission does not require a database round trip per source file.

For unattended allocations, use an initialization recipe with `kind: "linux"`, a pinned
`setup: { file, sha256 }`, `distribution`, `adminUsername`, `architecture`, the existing `node`/`rustup`
downloads, and `runtime: { account, container, blob, sha256, bytes }`. The runtime blob name
must be `worker-runtime/<sha256>.tar.gz`. The coordinator uses its managed identity to issue
a fresh one-hour user-delegation SAS with read permission for only that blob. It validates
the returned scope and expiry before provisioning. No subscription credential enters the guest.
The VM's preparation identity pins the recipe and setup bytes independently of each capability.
The managed identity needs blob data access and permission to issue user-delegation keys at
the storage account; Storage Blob Data Contributor provides these permissions.

The setup writes its root-owned completion receipt last. Admission waits for cloud-init and
that receipt; cloud-init fatal errors and incomplete setup always fail. Completed cloud-init
with recoverable platform warnings may proceed only when the lab setup itself completed.
Detailed cloud-init status remains on the guest, and failure diagnostics are captured before
cleanup. Static cloud-init remains useful for explicitly managed preparation inputs, but any
capabilities embedded in a static file must be renewed by its owner.

For unattended update qualification, disposable Ubuntu workers authorize only their configured
worker user running the packaged `_install-application-update` command through real `pkexec`.
The initializer installs a Polkit rule matching the executable and command line; it does not
replace the updater, its signature checks, or the native package transaction. This qualifies
preauthorized installation, not interactive password-prompt handling. The rule exists only on
the disposable worker and disappears with it. The matching variables are documented by
[Polkit](https://polkit.pages.freedesktop.org/polkit/polkit.8.html).

### Windows worker preparation

Configure a Windows image with a `kind: "windows"` initialization recipe. Pin each of
`toolsSetup`, `runtimeSetup` and `desktopSetup` by file path and SHA-256; the coordinator image
contains these scripts under `/opt/lab/`. Include the decoded native download pins from
`tools/windows-downloads.json`, the exact client distribution (`windows` version `10` or `11`),
`architecture: "x64"`, the allocation's administrator username, and the same content-addressed
runtime blob descriptor used by Linux. The recipe must match the target before allocation.

Allocation installs native tools, grants a temporary blob-only runtime download capability,
prepares dependencies and a disposable interactive desktop, reboots once, and verifies the live
desktop plus removal of temporary login credentials. Managed command identities allow observation
to resume without reinstalling or rebooting a ready desktop. Only readiness is refreshed after
success. Failed stages retain execution diagnostics in the run's evidence and leave the owned
lease discoverable for cleanup. Windows initialization does not use Linux cloud-init.

Server 2025 diagnostic preparation is a separate mechanism test, not Windows 10/11 qualification.
Do not configure a client image until its licensing eligibility is established. The implementation
requires `windowsLicense` on a client image: `visual-studio-dev-test` for verified Visual Studio
and subscription eligibility, or `multitenant` for verified Windows hosting rights. Only the latter
emits Azure's `licenseType: Windows_Client`. Omitting this field blocks client allocation before
resources are created; it is not inferred from credits or image availability.

### Namespace coordinator authentication

The coordinator image includes Devbox 0.0.189, verified against the vendor's published Linux
archive digest. Supply a dedicated operator login in the secure `namespaceCredentialBase64`
deployment parameter. The entrypoint creates a private `NSC_TOKEN_FILE` and removes the encoded
credential from its environment. No Namespace credential enters a guest or a submitted source
snapshot. The login remains bound to its Namespace workspace membership and expiration; renew
it through the named coordinator keychain and deploy a new secret/revision before expiration.

The tested named keychain is `magnitude-testing-lab-coordinator`. Machine inventory and image
catalog reads succeed through its explicit credential file. The current CLI rejected opaque
revocable-token credentials for Devbox operations; those diagnostic tokens were revoked. Do not
substitute `devbox auth check` for a real API check: it reports the local user login, even when
`NSC_TOKEN_FILE` is configured for another credential.


Namespace configuration requires a `preparation` recipe alongside `executable` and the image
locks. Pin the SHA-256 of `/opt/lab/macos-worker.sh`, the trusted Azure runtime blob, and the
Node, Rustup and macOS Tirith downloads. `azureExecutable` and `subscription` identify the
coordinator's existing blob-read authority. Allocation verifies the locked guest version and
build before preparation; no test source or run credential is sent until readiness succeeds.

The script requires the locked image's `/opt/homebrew/bin/python3`, Xcode, CMake, a non-root
`runner` account with passwordless sudo, and that same user's active console desktop. It installs
under `/Users/runner/lab-runtime`; the transport runtime executable is
`/Users/runner/lab-runtime/worker` with no fixed arguments. Use a separate per-attempt transfer
root, such as `/Users/runner/lab-work`, and `artifactHost: darwin-arm64`.
The root-owned readiness receipt binds the preparation recipe. An incomplete installation fails
and the owned worker is released rather than reused. Preparation failure logs are retained in
the normal run evidence before cleanup.

## CUDA compiler preparation

CUDA source builds select the existing release CUDA 12.9 pack. Before candidate compilation,
the trusted worker installs the pinned NVIDIA redistributable components from
`tools/cuda-toolkit.json` into that build's private SDK directory. It verifies each download's
length and SHA-256, verifies NVCC identity and required headers, compiles a kernel to PTX, and
links a shared library against cuBLAS. The phase is retained with normal build evidence.
`CUDA_PATH`, `CUDACXX` and the SDK binary path reach the source build through its explicit
environment. CPU and Metal builds do not install this SDK.

This compiler installation does not install a GPU driver or require GPU quota. A10 and RTX PRO
6000 consumer drivers have separate pinned preparation in `tools/nvidia-drivers.json`. Configure
that model's OS-specific entry as `gpu` in the initialization recipe. The allocator checks hardware,
VM family and supported guest OS before provisioning; current recipes permit Ubuntu 24.04 and
Windows 11, plus Windows Server diagnostics. Driver installation and real CUDA generation remain
unverified at zero GPU quota. Other catalog OS/GPU combinations require a supported driver recipe. Linux x64 SDK compilation/linking has
native Azure CPU evidence; Windows and Linux ARM SDK execution remain unverified.
