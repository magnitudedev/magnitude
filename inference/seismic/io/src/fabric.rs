//! Physical ontology of the machine's data-movement fabric: memory/storage
//! domains, directed links with measured performance curves, shared contention
//! buses, connected-component [`CoupledSetSpec`] partitioning, and multi-hop
//! [`RoutePlan`] construction (including conditional 1-hop UMA collapse).

use std::collections::{HashMap, HashSet};
use std::fmt;

/// Storage volume identifier (maps 1:1 to a physical block volume).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VolumeId(pub u64);

/// Host NUMA node identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NumaNodeId(pub u32);

/// Compute device identifier (GPU, accelerator, or CPU device).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DeviceId(pub u64);

/// PCIe root complex identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RootComplexId(pub u32);

/// Unified-memory SoC fabric identifier (e.g., Apple Silicon, GB10).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SocId(pub u32);

/// Ordered producer queue identifier scoped to one [`DeviceId`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ProducerId(pub u64);

/// Unique identifier of a directed physical [`LinkSpec`] within a [`Fabric`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LinkId(pub u32);

/// Unique identifier of a connected component of links sharing at least one
/// [`SharedBusId`]. Each coupled set has one atomic admission owner.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CoupledSetId(pub u32);

/// Opaque data representation identifier (quantization format / memory layout).
/// `seismic-io` never interprets quantization formats; it compares
/// [`RepresentationId`]s for exact equality (join-in-flight & UMA collapse) and
/// delegates mismatches to executor-supplied [`crate::plane::TransformTask`]s.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RepresentationId(pub u32);

/// A physical memory or storage domain where bytes reside.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum DomainId {
    Storage(VolumeId),
    HostRam(NumaNodeId),
    DeviceVram(DeviceId),
    /// Physically unified host+device memory (Apple Silicon, GB10, iGPUs).
    Unified(DeviceId),
}

/// Physical mechanism moving bytes along a directed edge `src -> dst`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum LinkKind {
    /// Direct uncached block I/O (`O_DIRECT` / `F_NOCACHE` / `FILE_FLAG_NO_BUFFERING` / `io_uring`).
    NvmeDirect,
    /// 1-hop collapse on UMA where storage DMA lands directly into GPU-visible unified pages.
    UnifiedDirect,
    /// Hardware async DMA copy engine (CUDA async memcpy, Vulkan transfer queue).
    CopyEngineH2D,
    CopyEngineD2H,
    CopyEngineD2D,
    /// GPU SMs reading pinned host RAM directly over PCIe (modeled for quoting & bus debit).
    SmZeroCopy,
    /// Host CPU memory copy (e.g. staging unpack or NUMA migration; debits `HostDdr` 2x).
    HostMemcpy,
    /// Direct storage-to-VRAM DMA (future extension, zero schema change).
    GpuDirectStorage,
    /// Peer RDMA across nodes/devices (future extension, zero schema change).
    PeerRdma,
}

/// Physical contention bottleneck shared across multiple Links and non-IO compute.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum SharedBusId {
    HostDdr(NumaNodeId),
    PcieRoot(RootComplexId),
    UnifiedFabric(SocId),
}

/// Direction of traffic placed on a [`SharedBusId`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BusDirection {
    Read,
    Write,
    FullDuplexH2D,
    FullDuplexD2H,
    Bidirectional,
}

/// Coupling between a physical Link and one [`SharedBusId`] it loads.
///
/// Example: `HostMemcpy` couples to `HostDdr` with `weight = 2.0` (read + write).
/// Example: `CopyEngineH2D` couples to `HostDdr(Read, 1.0)` + `PcieRoot(FullDuplexH2D, 1.0)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BusCoupling {
    pub bus: SharedBusId,
    pub direction: BusDirection,
    pub weight: f32,
}

/// Measured specification of a [`SharedBusId`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SharedBusSpec {
    pub id: SharedBusId,
    /// Asymptotic aggregate bandwidth of the bus (bytes/sec).
    pub peak_bps: f64,
    /// Relative measurement spread (e.g., 0.05 for 5%).
    pub uncertainty_rel: f64,
}

/// Measured performance curve of a physical Link (pushed by Discovery/Assessment).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LinkCurve {
    /// Fixed per-submission latency `λ` (seconds).
    pub latency_s: f64,
    /// Asymptotic line rate `P_peak` (bytes/sec).
    pub peak_bps: f64,
    /// Narrowest in-flight window reaching `peak_bps` within noise.
    pub knee_bytes: u64,
    /// Window holding a full `Blocking`/`Streaming` demand burst.
    pub due_depth_bytes: u64,
    /// Hardware queue depth / OS worker ceiling derived from the probed curve.
    pub max_in_flight_ops: u32,
    /// Required offset/buffer alignment (e.g., 4096 for `O_DIRECT`).
    pub alignment_bytes: u64,
    /// Measurement spread / relative uncertainty.
    pub uncertainty_rel: f64,
}

impl LinkCurve {
    /// Residual speculative service-time bound `τ_hol = max(2λ, min_quantum_s)`.
    /// Ensures an arriving `Blocking` miss is never delayed by more than one
    /// bounded speculative quantum regardless of expert size or link speed.
    pub fn tau_hol_s(&self, min_quantum_s: f64) -> f64 {
        (2.0 * self.latency_s).max(min_quantum_s)
    }
}

/// Specification of one directed physical Link in a [`FabricSpec`].
#[derive(Clone, Debug, PartialEq)]
pub struct LinkSpec {
    pub id: LinkId,
    pub kind: LinkKind,
    pub src: DomainId,
    pub dst: DomainId,
    pub curve: LinkCurve,
    pub couplings: Vec<BusCoupling>,
}

/// Immutable topology & curve snapshot pushed by Topology & Discovery.
#[derive(Clone, Debug, PartialEq)]
pub struct FabricSpec {
    pub revision: u64,
    pub buses: Vec<SharedBusSpec>,
    pub links: Vec<LinkSpec>,
}

/// One connected component of links that share at least one [`SharedBusId`],
/// governed by a single atomic admission owner.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CoupledSetSpec {
    pub id: CoupledSetId,
    pub links: Vec<LinkId>,
    pub buses: Vec<SharedBusId>,
}

/// A single stage of a multi-hop [`RoutePlan`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RouteStage {
    /// Physical transfer across `link` (`src -> dst`).
    Transfer {
        link: LinkId,
        kind: LinkKind,
        src: DomainId,
        dst: DomainId,
    },
    /// Opaque CPU transform stage (`src_repr -> dst_repr`) executing in `domain`
    /// and debiting `host_bus` at `weight = 2.0` (read + write).
    Transform {
        domain: DomainId,
        src_repr: RepresentationId,
        dst_repr: RepresentationId,
        host_bus: Option<SharedBusId>,
    },
}

/// Physical pipeline plan constructed by [`Fabric::route`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RoutePlan {
    pub stages: Vec<RouteStage>,
    /// True iff `Storage -> Unified` collapsed into a single zero-copy `UnifiedDirect` hop.
    pub uma_collapsed: bool,
}

/// Why a [`FabricSpec`] or [`Fabric::route`] call is invalid.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FabricError {
    DuplicateBus(SharedBusId),
    DuplicateLink(LinkId),
    UnknownBus { link: LinkId, bus: SharedBusId },
    InvalidBusSpec { bus: SharedBusId, reason: &'static str },
    InvalidLinkCurve { link: LinkId, reason: &'static str },
    InvalidCouplingWeight { link: LinkId, bus: SharedBusId },
    NoRoute { src: DomainId, dst: DomainId },
}

impl fmt::Display for FabricError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateBus(bus) => write!(f, "duplicate shared bus {bus:?}"),
            Self::DuplicateLink(link) => write!(f, "duplicate link {link:?}"),
            Self::UnknownBus { link, bus } => write!(f, "link {link:?} references unknown bus {bus:?}"),
            Self::InvalidBusSpec { bus, reason } => write!(f, "invalid spec for bus {bus:?}: {reason}"),
            Self::InvalidLinkCurve { link, reason } => write!(f, "invalid curve for link {link:?}: {reason}"),
            Self::InvalidCouplingWeight { link, bus } => {
                write!(f, "link {link:?} coupling to {bus:?} must have finite positive weight")
            }
            Self::NoRoute { src, dst } => write!(f, "no fabric route from {src:?} to {dst:?}"),
        }
    }
}

impl std::error::Error for FabricError {}

/// Immutable, revisioned data-movement fabric graph with pre-computed
/// [`CoupledSetSpec`] connected components.
#[derive(Clone, Debug)]
pub struct Fabric {
    revision: u64,
    buses: Vec<SharedBusSpec>,
    bus_index: HashMap<SharedBusId, usize>,
    links: Vec<LinkSpec>,
    link_index: HashMap<LinkId, usize>,
    coupled_sets: Vec<CoupledSetSpec>,
    link_to_set: HashMap<LinkId, CoupledSetId>,
    bus_to_set: HashMap<SharedBusId, CoupledSetId>,
}

impl Fabric {
    /// Validate `spec` and partition its links into connected [`CoupledSetSpec`]s
    /// by shared [`SharedBusId`]s.
    pub fn build(spec: FabricSpec) -> Result<Self, FabricError> {
        let mut bus_index = HashMap::with_capacity(spec.buses.len());
        for (idx, bus) in spec.buses.iter().enumerate() {
            if !bus.peak_bps.is_finite() || bus.peak_bps <= 0.0 {
                return Err(FabricError::InvalidBusSpec {
                    bus: bus.id,
                    reason: "peak_bps must be finite and > 0",
                });
            }
            if !bus.uncertainty_rel.is_finite() || bus.uncertainty_rel < 0.0 {
                return Err(FabricError::InvalidBusSpec {
                    bus: bus.id,
                    reason: "uncertainty_rel must be finite and >= 0",
                });
            }
            if bus_index.insert(bus.id, idx).is_some() {
                return Err(FabricError::DuplicateBus(bus.id));
            }
        }

        let mut link_index = HashMap::with_capacity(spec.links.len());
        for (idx, link) in spec.links.iter().enumerate() {
            validate_curve(link.id, &link.curve)?;
            if link_index.insert(link.id, idx).is_some() {
                return Err(FabricError::DuplicateLink(link.id));
            }
            for coupling in &link.couplings {
                if !bus_index.contains_key(&coupling.bus) {
                    return Err(FabricError::UnknownBus {
                        link: link.id,
                        bus: coupling.bus,
                    });
                }
                if !coupling.weight.is_finite() || coupling.weight <= 0.0 {
                    return Err(FabricError::InvalidCouplingWeight {
                        link: link.id,
                        bus: coupling.bus,
                    });
                }
            }
        }

        // Partition links into connected components via union-find over shared buses.
        let n = spec.links.len();
        let mut parent: Vec<usize> = (0..n).collect();
        let mut bus_first_link: HashMap<SharedBusId, usize> = HashMap::new();
        for (idx, link) in spec.links.iter().enumerate() {
            for coupling in &link.couplings {
                if let Some(&first) = bus_first_link.get(&coupling.bus) {
                    uf_union(&mut parent, first, idx);
                } else {
                    bus_first_link.insert(coupling.bus, idx);
                }
            }
        }

        // Group links by root representative in deterministic order.
        let mut root_to_set: HashMap<usize, usize> = HashMap::new();
        let mut coupled_sets: Vec<CoupledSetSpec> = Vec::new();
        let mut set_bus_seen: Vec<HashSet<SharedBusId>> = Vec::new();
        let mut link_to_set = HashMap::with_capacity(n);
        let mut bus_to_set = HashMap::with_capacity(spec.buses.len());

        for (idx, link) in spec.links.iter().enumerate() {
            let root = uf_find(&mut parent, idx);
            let set_idx = match root_to_set.get(&root) {
                Some(&existing) => existing,
                None => {
                    let id = CoupledSetId(coupled_sets.len() as u32);
                    let pos = coupled_sets.len();
                    root_to_set.insert(root, pos);
                    coupled_sets.push(CoupledSetSpec {
                        id,
                        links: Vec::new(),
                        buses: Vec::new(),
                    });
                    set_bus_seen.push(HashSet::new());
                    pos
                }
            };
            let set_id = coupled_sets[set_idx].id;
            coupled_sets[set_idx].links.push(link.id);
            link_to_set.insert(link.id, set_id);
            for coupling in &link.couplings {
                if set_bus_seen[set_idx].insert(coupling.bus) {
                    coupled_sets[set_idx].buses.push(coupling.bus);
                    bus_to_set.insert(coupling.bus, set_id);
                }
            }
        }

        // Any shared bus not yet coupled to any link gets its own bus-only set so
        // non-IO `BusReservation`s on that bus still have a single admission owner.
        for bus in &spec.buses {
            bus_to_set.entry(bus.id).or_insert_with(|| {
                let id = CoupledSetId(coupled_sets.len() as u32);
                coupled_sets.push(CoupledSetSpec {
                    id,
                    links: Vec::new(),
                    buses: vec![bus.id],
                });
                id
            });
        }

        Ok(Self {
            revision: spec.revision,
            buses: spec.buses,
            bus_index,
            links: spec.links,
            link_index,
            coupled_sets,
            link_to_set,
            bus_to_set,
        })
    }

    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn buses(&self) -> &[SharedBusSpec] {
        &self.buses
    }

    pub fn bus(&self, id: SharedBusId) -> Option<&SharedBusSpec> {
        self.bus_index.get(&id).map(|&idx| &self.buses[idx])
    }

    pub fn bus_index(&self, id: SharedBusId) -> Option<usize> {
        self.bus_index.get(&id).copied()
    }

    pub fn links(&self) -> &[LinkSpec] {
        &self.links
    }

    pub fn link(&self, id: LinkId) -> Option<&LinkSpec> {
        self.link_index.get(&id).map(|&idx| &self.links[idx])
    }

    pub fn link_index(&self, id: LinkId) -> Option<usize> {
        self.link_index.get(&id).copied()
    }

    pub fn coupled_sets(&self) -> &[CoupledSetSpec] {
        &self.coupled_sets
    }

    pub fn coupled_set_of_link(&self, link: LinkId) -> Option<CoupledSetId> {
        self.link_to_set.get(&link).copied()
    }

    pub fn coupled_set_of_bus(&self, bus: SharedBusId) -> Option<CoupledSetId> {
        self.bus_to_set.get(&bus).copied()
    }

    /// Construct a physical multi-hop [`RoutePlan`] from `src` to `dst`,
    /// converting `src_repr` to `dst_repr` when they differ.
    ///
    /// - **UMA (`Storage -> Unified`)**: collapses to 1-hop `UnifiedDirect`
    ///   **only** when `uma_direct_ok && src_repr == dst_repr`. When a repack
    ///   is needed (`src_repr != dst_repr`), inserts an explicit
    ///   [`RouteStage::Transform`] stage in the unified domain (zero PCIe hop,
    ///   zero format logic in `seismic-io`).
    /// - **Discrete GPU (`Storage -> DeviceVram`)**: routes
    ///   `NvmeDirect(Storage -> HostRam)` -> optional `Transform(HostRam)` ->
    ///   `CopyEngineH2D(HostRam -> DeviceVram)`.
    /// - **Direct 1-hop**: selects the fastest non-`SmZeroCopy` transfer link
    ///   connecting `src -> dst` (with optional `Transform` when `src_repr != dst_repr`).
    pub fn route(
        &self,
        src: DomainId,
        dst: DomainId,
        src_repr: RepresentationId,
        dst_repr: RepresentationId,
        uma_direct_ok: bool,
    ) -> Result<RoutePlan, FabricError> {
        let needs_transform = src_repr != dst_repr;

        // 1. Direct UMA Storage -> Unified route.
        if let (DomainId::Storage(_), DomainId::Unified(_)) = (src, dst) {
            if let Some(link) = self.best_link(src, dst, Some(LinkKind::UnifiedDirect)) {
                let mut stages = vec![RouteStage::Transfer {
                    link: link.id,
                    kind: link.kind,
                    src,
                    dst,
                }];
                let uma_collapsed = uma_direct_ok && !needs_transform;
                if needs_transform {
                    let host_bus = link.couplings.first().map(|c| c.bus);
                    stages.push(RouteStage::Transform {
                        domain: dst,
                        src_repr,
                        dst_repr,
                        host_bus,
                    });
                }
                return Ok(RoutePlan { stages, uma_collapsed });
            }
        }

        // 2. Direct 1-hop route (e.g. Storage -> HostRam, HostRam -> DeviceVram,
        //    DeviceVram -> HostRam, DeviceVram -> DeviceVram, GpuDirectStorage).
        if let Some(link) = self.best_transfer_link(src, dst) {
            if !needs_transform {
                return Ok(RoutePlan {
                    stages: vec![RouteStage::Transfer {
                        link: link.id,
                        kind: link.kind,
                        src,
                        dst,
                    }],
                    uma_collapsed: false,
                });
            }
            // Transform must execute in a host-accessible domain (HostRam or Unified).
            let transform_domain = match (src, dst) {
                (_, DomainId::HostRam(_)) | (_, DomainId::Unified(_)) => Some((dst, true)),
                (DomainId::HostRam(_), _) | (DomainId::Unified(_), _) => Some((src, false)),
                _ => None,
            };
            if let Some((dom, after_transfer)) = transform_domain {
                let host_bus = self.host_bus_for_domain(dom);
                let transfer = RouteStage::Transfer {
                    link: link.id,
                    kind: link.kind,
                    src,
                    dst,
                };
                let transform = RouteStage::Transform {
                    domain: dom,
                    src_repr,
                    dst_repr,
                    host_bus,
                };
                let stages = if after_transfer {
                    vec![transfer, transform]
                } else {
                    vec![transform, transfer]
                };
                return Ok(RoutePlan {
                    stages,
                    uma_collapsed: false,
                });
            }
        }

        // 3. 2-hop route via intermediate HostRam (e.g., Storage -> HostRam -> DeviceVram).
        let mut best_two_hop: Option<(f64, &LinkSpec, &LinkSpec)> = None;
        for hop0 in self.links.iter().filter(|l| l.src == src && l.kind != LinkKind::SmZeroCopy) {
            if !matches!(hop0.dst, DomainId::HostRam(_) | DomainId::Unified(_)) {
                continue;
            }
            for hop1 in self
                .links
                .iter()
                .filter(|l| l.src == hop0.dst && l.dst == dst && l.kind != LinkKind::SmZeroCopy)
            {
                let bottleneck = hop0.curve.peak_bps.min(hop1.curve.peak_bps);
                if best_two_hop.is_none_or(|(best_bps, _, _)| bottleneck > best_bps) {
                    best_two_hop = Some((bottleneck, hop0, hop1));
                }
            }
        }

        if let Some((_, hop0, hop1)) = best_two_hop {
            let mut stages = Vec::with_capacity(if needs_transform { 3 } else { 2 });
            stages.push(RouteStage::Transfer {
                link: hop0.id,
                kind: hop0.kind,
                src: hop0.src,
                dst: hop0.dst,
            });
            if needs_transform {
                stages.push(RouteStage::Transform {
                    domain: hop0.dst,
                    src_repr,
                    dst_repr,
                    host_bus: self.host_bus_for_domain(hop0.dst),
                });
            }
            stages.push(RouteStage::Transfer {
                link: hop1.id,
                kind: hop1.kind,
                src: hop1.src,
                dst: hop1.dst,
            });
            return Ok(RoutePlan {
                stages,
                uma_collapsed: false,
            });
        }

        Err(FabricError::NoRoute { src, dst })
    }

    fn best_link(&self, src: DomainId, dst: DomainId, kind: Option<LinkKind>) -> Option<&LinkSpec> {
        self.links
            .iter()
            .filter(|l| l.src == src && l.dst == dst && kind.is_none_or(|k| l.kind == k))
            .max_by(|a, b| a.curve.peak_bps.total_cmp(&b.curve.peak_bps))
    }

    fn best_transfer_link(&self, src: DomainId, dst: DomainId) -> Option<&LinkSpec> {
        self.links
            .iter()
            .filter(|l| l.src == src && l.dst == dst && l.kind != LinkKind::SmZeroCopy)
            .max_by(|a, b| a.curve.peak_bps.total_cmp(&b.curve.peak_bps))
    }

    fn host_bus_for_domain(&self, domain: DomainId) -> Option<SharedBusId> {
        match domain {
            DomainId::HostRam(node) => {
                let candidate = SharedBusId::HostDdr(node);
                self.bus_index.contains_key(&candidate).then_some(candidate)
            }
            DomainId::Unified(_) => self
                .buses
                .iter()
                .find(|b| matches!(b.id, SharedBusId::UnifiedFabric(_)))
                .map(|b| b.id),
            _ => None,
        }
    }
}

fn validate_curve(link: LinkId, curve: &LinkCurve) -> Result<(), FabricError> {
    if !curve.latency_s.is_finite() || curve.latency_s < 0.0 {
        return Err(FabricError::InvalidLinkCurve {
            link,
            reason: "latency_s must be finite and >= 0",
        });
    }
    if !curve.peak_bps.is_finite() || curve.peak_bps <= 0.0 {
        return Err(FabricError::InvalidLinkCurve {
            link,
            reason: "peak_bps must be finite and > 0",
        });
    }
    if curve.knee_bytes == 0 {
        return Err(FabricError::InvalidLinkCurve {
            link,
            reason: "knee_bytes must be > 0",
        });
    }
    if curve.due_depth_bytes == 0 {
        return Err(FabricError::InvalidLinkCurve {
            link,
            reason: "due_depth_bytes must be > 0",
        });
    }
    if curve.max_in_flight_ops == 0 {
        return Err(FabricError::InvalidLinkCurve {
            link,
            reason: "max_in_flight_ops must be > 0",
        });
    }
    if !curve.uncertainty_rel.is_finite() || curve.uncertainty_rel < 0.0 {
        return Err(FabricError::InvalidLinkCurve {
            link,
            reason: "uncertainty_rel must be finite and >= 0",
        });
    }
    Ok(())
}

fn uf_find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

fn uf_union(parent: &mut [usize], a: usize, b: usize) {
    let ra = uf_find(parent, a);
    let rb = uf_find(parent, b);
    if ra != rb {
        parent[rb] = ra;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn curve(peak_gbps: f64) -> LinkCurve {
        LinkCurve {
            latency_s: 5e-6,
            peak_bps: peak_gbps * 1e9,
            knee_bytes: 4 * 1024 * 1024,
            due_depth_bytes: 16 * 1024 * 1024,
            max_in_flight_ops: 8,
            alignment_bytes: 4096,
            uncertainty_rel: 0.05,
        }
    }

    #[test]
    fn coupled_sets_partition_by_shared_bus_connected_components() {
        let ddr0 = SharedBusId::HostDdr(NumaNodeId(0));
        let pcie0 = SharedBusId::PcieRoot(RootComplexId(0));
        let ddr1 = SharedBusId::HostDdr(NumaNodeId(1));

        let spec = FabricSpec {
            revision: 1,
            buses: vec![
                SharedBusSpec { id: ddr0, peak_bps: 60e9, uncertainty_rel: 0.03 },
                SharedBusSpec { id: pcie0, peak_bps: 32e9, uncertainty_rel: 0.03 },
                SharedBusSpec { id: ddr1, peak_bps: 60e9, uncertainty_rel: 0.03 },
            ],
            links: vec![
                // Link 0: NVMe -> HostRam(0), coupled to ddr0 + pcie0
                LinkSpec {
                    id: LinkId(0),
                    kind: LinkKind::NvmeDirect,
                    src: DomainId::Storage(VolumeId(0)),
                    dst: DomainId::HostRam(NumaNodeId(0)),
                    curve: curve(6.0),
                    couplings: vec![
                        BusCoupling { bus: ddr0, direction: BusDirection::Write, weight: 1.0 },
                        BusCoupling { bus: pcie0, direction: BusDirection::FullDuplexD2H, weight: 1.0 },
                    ],
                },
                // Link 1: HostRam(0) -> GPU(0), coupled to ddr0 + pcie0 (same component as Link 0)
                LinkSpec {
                    id: LinkId(1),
                    kind: LinkKind::CopyEngineH2D,
                    src: DomainId::HostRam(NumaNodeId(0)),
                    dst: DomainId::DeviceVram(DeviceId(0)),
                    curve: curve(26.0),
                    couplings: vec![
                        BusCoupling { bus: ddr0, direction: BusDirection::Read, weight: 1.0 },
                        BusCoupling { bus: pcie0, direction: BusDirection::FullDuplexH2D, weight: 1.0 },
                    ],
                },
                // Link 2: HostMemcpy on NUMA node 1 (independent component)
                LinkSpec {
                    id: LinkId(2),
                    kind: LinkKind::HostMemcpy,
                    src: DomainId::HostRam(NumaNodeId(1)),
                    dst: DomainId::HostRam(NumaNodeId(1)),
                    curve: curve(40.0),
                    couplings: vec![BusCoupling {
                        bus: ddr1,
                        direction: BusDirection::Bidirectional,
                        weight: 2.0,
                    }],
                },
            ],
        };

        let fabric = Fabric::build(spec).expect("valid fabric");
        assert_eq!(fabric.coupled_sets().len(), 2);
        assert_eq!(
            fabric.coupled_set_of_link(LinkId(0)),
            fabric.coupled_set_of_link(LinkId(1)),
            "NVMe and H2D sharing HostDdr(0)/PcieRoot(0) must share one CoupledSet"
        );
        assert_ne!(
            fabric.coupled_set_of_link(LinkId(0)),
            fabric.coupled_set_of_link(LinkId(2)),
            "NUMA node 1 link must live in a disjoint CoupledSet"
        );
    }

    #[test]
    fn route_handles_discrete_multihop_and_conditional_uma_collapse() {
        let soc = SharedBusId::UnifiedFabric(SocId(0));
        let ddr = SharedBusId::HostDdr(NumaNodeId(0));
        let pcie = SharedBusId::PcieRoot(RootComplexId(0));

        // UMA fabric
        let uma = Fabric::build(FabricSpec {
            revision: 1,
            buses: vec![SharedBusSpec { id: soc, peak_bps: 273e9, uncertainty_rel: 0.02 }],
            links: vec![LinkSpec {
                id: LinkId(10),
                kind: LinkKind::UnifiedDirect,
                src: DomainId::Storage(VolumeId(1)),
                dst: DomainId::Unified(DeviceId(1)),
                curve: curve(4.6),
                couplings: vec![BusCoupling { bus: soc, direction: BusDirection::Write, weight: 1.0 }],
            }],
        })
        .unwrap();

        let r0 = RepresentationId(0);
        let r1 = RepresentationId(1);
        let collapsed = uma
            .route(DomainId::Storage(VolumeId(1)), DomainId::Unified(DeviceId(1)), r0, r0, true)
            .unwrap();
        assert!(collapsed.uma_collapsed);
        assert_eq!(collapsed.stages.len(), 1);

        let not_collapsed = uma
            .route(DomainId::Storage(VolumeId(1)), DomainId::Unified(DeviceId(1)), r0, r0, false)
            .unwrap();
        assert!(!not_collapsed.uma_collapsed);

        let repack_uma = uma
            .route(DomainId::Storage(VolumeId(1)), DomainId::Unified(DeviceId(1)), r0, r1, true)
            .unwrap();
        assert!(!repack_uma.uma_collapsed);
        assert_eq!(repack_uma.stages.len(), 2);
        assert!(matches!(repack_uma.stages[1], RouteStage::Transform { .. }));

        // Discrete GPU fabric
        let discrete = Fabric::build(FabricSpec {
            revision: 2,
            buses: vec![
                SharedBusSpec { id: ddr, peak_bps: 60e9, uncertainty_rel: 0.03 },
                SharedBusSpec { id: pcie, peak_bps: 48e9, uncertainty_rel: 0.03 },
            ],
            links: vec![
                LinkSpec {
                    id: LinkId(0),
                    kind: LinkKind::NvmeDirect,
                    src: DomainId::Storage(VolumeId(0)),
                    dst: DomainId::HostRam(NumaNodeId(0)),
                    curve: curve(6.0),
                    couplings: vec![BusCoupling { bus: ddr, direction: BusDirection::Write, weight: 1.0 }],
                },
                LinkSpec {
                    id: LinkId(1),
                    kind: LinkKind::CopyEngineH2D,
                    src: DomainId::HostRam(NumaNodeId(0)),
                    dst: DomainId::DeviceVram(DeviceId(0)),
                    curve: curve(48.0),
                    couplings: vec![
                        BusCoupling { bus: ddr, direction: BusDirection::Read, weight: 1.0 },
                        BusCoupling { bus: pcie, direction: BusDirection::FullDuplexH2D, weight: 1.0 },
                    ],
                },
            ],
        })
        .unwrap();

        let raw = discrete
            .route(DomainId::Storage(VolumeId(0)), DomainId::DeviceVram(DeviceId(0)), r0, r0, false)
            .unwrap();
        assert_eq!(raw.stages.len(), 2);

        let repacked = discrete
            .route(DomainId::Storage(VolumeId(0)), DomainId::DeviceVram(DeviceId(0)), r0, r1, false)
            .unwrap();
        assert_eq!(repacked.stages.len(), 3);
        assert!(matches!(repacked.stages[1], RouteStage::Transform { .. }));
    }
}
