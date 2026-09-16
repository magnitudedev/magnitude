# Installed configuration detection

Research and local experiments, 2026-09-16. This distinguishes implemented observations from candidate probes; published specifications are not measurements.

## Implemented observations

The existing ICN backend inventory supplies CPU model, inference-visible accelerator names, physical/shared memory domains, and memory capacity. It does not enumerate every installed GPU: an unavailable driver or uncompiled backend can leave an installed GPU absent. Consequently “CPU inference” means no accelerator available to this runtime, not proof that the machine has no GPU.

Physical CPU cores now travel from ICN through the generated protocol and ACN projection to the hardware card as an optional observation. The existing sysinfo 0.38.4 dependency reads `hw.physicalcpu` on macOS, `GetLogicalProcessorInformationEx` on Windows, and a bounded `/proc/cpuinfo` topology reader on Linux. The result is cached once per ICN process; a failed/zero query remains absent. The separate `available_parallelism()` value remains internal scheduling metadata and is not displayed on the hardware card, because affinity, containers, and scheduling restrictions can make it smaller than the installed thread count. Linux counts distinct (socket ID, core ID) pairs. Missing, malformed, or incomplete topology remains unknown; it never falls back to counting logical processor records. ARM Linux commonly omits these IDs, in which case catalog facts remain available. Virtualization can expose virtual topology rather than host physical cores.

On the development Mac, 1,000 direct `hw.physicalcpu` calls returned 16 cores, median 0.916 microseconds and p99 1.209 microseconds. These are local query timings, not cross-platform latency guarantees. No utilization refresh, subprocess, network request, or sampling interval is added.

Sources: [Apple sysctl](https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man3/sysctlbyname.3.html), [Windows processor topology](https://learn.microsoft.com/en-us/windows/win32/api/sysinfoapi/nf-sysinfoapi-getlogicalprocessorinformationex), [Linux CPU topology](https://www.kernel.org/doc/html/latest/admin-guide/cputopology.html).

## GPU queries investigated, not yet integrated

| Platform | Candidate observation | Constraints |
| --- | --- | --- |
| Apple Silicon | Read `gpu-core-count` from the matching IOAccelerator registry entry | Returned 40 on the development Mac; first query 73 microseconds, subsequent five 10–18 microseconds. This driver property is not a public Metal compatibility guarantee. Missing/wrong-typed properties must remain unknown; match registry ID, never array position. |
| NVIDIA | NVML `nvmlDeviceGetNumGpuCores`, memory info, PCI identity | Explicit core-count query exists. Requires driver/library availability; MIG partitions need instance-specific information, not the parent GPU's full count. Loading NVML solely for display needs separate cold-start measurements. |
| AMD Vulkan | `VkPhysicalDeviceShaderCoreProperties2AMD.activeComputeUnitCount` | Reports enabled CUs when the extension is supported. Extension presence is mandatory; do not substitute workgroup limits for compute units. |
| Generic Vulkan/Metal | Existing device identity and memory observations | No universal GPU core-count property. Core counts, CUs, shader processors, and CUDA multiprocessors are different units. |

Sources: [Fastfetch's Apple detector implementation](https://github.com/fastfetch-cli/fastfetch/blob/dev/src/detection/gpu/gpu_apple.c), [Metal device API](https://developer.apple.com/documentation/metal/mtldevice), [NVIDIA NVML device queries](https://docs.nvidia.com/deploy/nvml-api/api/group__nvmlDeviceQueries.html), [AMD Vulkan extension](https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDeviceShaderCoreProperties2AMD.html).

An eventual GPU supplement should reuse initialized device handles, run once outside rendering, associate results with physical identity, and fail independently of the initial hardware card. Do not introduce `system_profiler`, WMI/PowerShell, `nvidia-smi`, a workload, or a driver initialization on the card's critical path. A cached optional observation can resolve a published variant; a laptop family or image alone cannot.
