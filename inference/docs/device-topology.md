# Device topology

**One model of the machine: where computation runs, where bytes live, and what each executor may
place where.** `seismic-topology` discovers it from standard interfaces; it measures nothing, and
every value in it is one an interface reports. How fast work runs on it is measured apart from it
(`seismic-measure`, below).

## The graph

```mermaid
flowchart LR
    subgraph GPU[Physical device: GPU]
        CU[Executor: CUDA]
        VK[Executor: Vulkan]
    end
    subgraph CPUD[Physical device: CPU]
        CP[Executor: CPU]
    end
    VR[(Store: device memory)]
    RAM[(Store: host RAM)]
    VOL[(Store: volume)]
    CU -- access, grant --> VR
    VK -- access, grant --> VR
    CU -- access: upload --> RAM
    CP -- access --> RAM
    VOL -. transfer .-> RAM
    RAM -. transfer .-> VR
```

| Node or edge | What it is |
| --- | --- |
| Physical device | One piece of hardware, identified by the identity its drivers share |
| Executor | One backend's handle on a physical device (CPU, Metal, CUDA, Vulkan); what runs kernels and what selection names |
| Store | A place bytes live, with one ledger: **host RAM** (memory the OS manages), **device memory** (memory the OS does not manage, owned by one physical device), **volume** (a filesystem) |
| Access path | An executor reading and writing a store directly, under the **grant** its driver gives this process there |
| Transfer path | Bytes moved between stores by an executor's copy mechanism: a volume into RAM, RAM into device memory, one GPU to another |
| Placement | For each storage kind an executor allocates, the ordered access paths its backend allocates from |

**Memory the OS manages is host RAM, whoever uses it.** A GPU whose memory is the host's (Apple
silicon, GB10, integrated GPUs) has access paths into host RAM under its driver's grant, so its
charges and the CPU's are one sum. Memory the OS does not manage, a discrete GPU's or one an
integrated GPU's adapter reports as dedicated, is a device memory store; such a GPU places there
first and then in host RAM. Whose memory a GPU's is, discovery decides once per physical device
from all its backends' reports (Vulkan's device type, CUDA's integrated or NUMA-node memory,
Metal's unified memory); every executor of the device places by that one decision, and reports that
disagree make its executors unavailable.

An access path also names its route: the memory its executor allocates along it. A discrete Vulkan
heap is a route of its own. An integrated Vulkan adapter's dedicated and shared grants share one
route, the same memory, while their charges and remaining budgets are partitioned in placement
order. The remaining grants sum to at most the driver's remaining budget.

Stable selectors name native executors across revisions. A GPU's physical identity is its device
UUID, which its CUDA and Vulkan views report alike and which survives a reboot; the OS adapter LUID,
new each boot under WDDM, only matches the dedicated memory the OS reports to the adapter.

**Process limits are grants too.** A cgroup, address-space or job limit bounds every path into host
RAM.

**Revisions are immutable.** A refresh that finds a different machine builds a new revision; old
identifiers become stale and are never renumbered.

The normalized graph uses stable selectors and store identities, independent of revision and
enumeration order. Hardware fingerprints exclude timestamped status.

## Reported values

| Layer | What | How | Lives |
| --- | --- | --- | --- |
| Structure | nodes, paths, identities, executor kinds, whose memory each device's is, capacities, grants, limits | reported by APIs at discovery | one topology revision |
| Status | each store's headroom, what is left of each grant | reported by APIs in a sample | one timestamped sample |

Every value is a **reported value**, carrying the API that reported it. Nothing is assumed.

**APIs report structure and limits.** Discovery reads only standard interfaces (Vulkan, CUDA, Metal,
the OS's memory, filesystem and adapter interfaces) and branches only on what they report, never on
a vendor or model. A placement is what the backend actually allocates from, so the grant reported
and the memory used agree. Where no interface reports a value, the topology takes the conservative
reading: memory it cannot place outside the OS counts as host RAM. A discrete CUDA device's peak
memory bandwidth, its memory clock times its bus width, is reported on its CUDA executor.

**The host's identity is its hardware**: the reported processor, its ISA and its memory, not
merely the OS build.

## Measuring work

Kernel timing has one owner. `seismic-measure` times work and depends on nothing else in Seismic;
the runtime gives it a formed kernel's calls over a rotation of argument sets.

- **One session.** A sample is one submission of a calibrated number of passes queued behind an
  untimed lead; tuning's comparisons and every measured point take their times this way.
- **The sustained clock.** The device is raised to its sustained clock, gauged with the caller's
  own work, before a session.
- **Statistics.** Tuning's comparisons resolve by an anytime-valid paired sign test; a point's own
  time is the median of the samples its caller asks for.

Which work is timed is its consumer's: tuning's search times its candidates through it, and the
kernel tests time their points. Measuring opens executors and runs kernels, so nothing measures
implicitly.

## API

| Call | Returns |
| --- | --- |
| `topology()` | the reported structure of the current revision |
| `refresh()` | the revision after discovering again: the same one when the machine reads the same, a new one when it does not |
| `sample(topology)` | a timestamped status sample of a revision's stores and grants |
| `host_status()` | one reading of host RAM alone, as a sample reads it |
| `volume_available(path)` | the free space of one path's volume, as a sample reads it |

The topology describes the machine. It knows nothing of models, loads or reserves.

## Ownership

| Owner | Responsibility |
| --- | --- |
| `seismic-topology` | What the machine is, end to end: the graph, discovery through the backend crates and the OS, and status samples |
| `seismic-measure` | Timing work: the session, the sustained clock, the statistics and deadlines |
| Seismic runtime | Opening executors by identity, allocation and ledgers, the measurable work (kernel calls), and tuning |
| Engine | Interpreting the topology for inference, in one place per use (below) |

`seismic-topology` and `seismic-measure` are internal to Seismic and published through the
`seismic` crate, like every Seismic crate; consumers, including anything that only needs to know the
machine, depend on `seismic`. Inside Seismic, `seismic-runtime` depends on `seismic-topology`
(which depends on the backend crates) and on `seismic-measure` (which depends on no Seismic crate).

## Consumption

The engine reads the topology in two functions, each with one owner, and the rest of the engine
goes through them rather than interpreting grants, stores or rates itself:
- **Fit** (memory policy): whether a demand fits, from the topology, a sample and the engine's
  reserves. Admission, the device heap, first-load tuning, assessment, the hardware view and
  downloads use it.
- **Cost** (estimates): a model's decode speed over the selected executor's memory bandwidth, the
  peak its API reports (CUDA, for a discrete device), else a published figure for its name, else
  its class's. Speed estimates use it.

## What it enables

- Correct capacity on every unified-memory device, from its driver's grants.
- Host RAM accounted once across every device and model in the process.
- Choosing among physical devices by fit and cost; loads spanning executors; a discrete GPU's data
  in host RAM; streaming from a volume. The graph expresses these; using them is engine work on top.
