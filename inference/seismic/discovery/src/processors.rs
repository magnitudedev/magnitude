//! The host's processors as the OS reports them, thread/process pinning, and
//! PCI device NUMA node lookup.

pub use seismic_topology::{Core, HostNode, MemoryNode, PciLocation, Processors};

/// The processors as the OS reports them now: each reading from its
/// sources in order, or its documented value where none answers.
pub fn discover_processors() -> Processors {
    crate::host::processors(&crate::host::LiveSys)
}

/// Place the calling thread on `cpus` (numbers [`Processors`] reports, at
/// least one): it runs on those alone. macOS places no thread on a core it
/// names (its scheduler places every thread), so there the call places
/// nothing; elsewhere an OS that refuses is an error.
pub fn pin(cpus: &[usize]) -> Result<(), String> {
    if cpus.is_empty() {
        return Err("a thread placed on no CPU".into());
    }
    crate::host::pin(cpus)
}

/// Whether a [`ProcessPlacement`] is held: one at a time in a process.
static PLACED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The process's placement ([`place_process`]): while it is held, every
/// thread of the process but those [`pin`] placed runs where it placed
/// them; as it drops, every such thread is placed back on the CPUs the
/// process could run on as it was claimed, the threads started meanwhile
/// among them. One is held at a time in a process: its holder owns every
/// thread's CPUs, and what the process may run on reads as its placement
/// until it drops.
#[must_use = "the process's placement lasts as long as it is held"]
pub struct ProcessPlacement {
    every: Vec<usize>,
}

/// Restore the process's original CPU affinity `every`, falling back to its
/// intersection with currently online topology CPUs if direct restoration fails
/// (for instance, if a CPU was offlined while the placement was held).
fn restore_affinity(
    every: &[usize],
    mut apply: impl FnMut(&[usize]) -> Result<(), String>,
    online_nodes: impl FnOnce() -> Vec<MemoryNode>,
) {
    if apply(every).is_err() {
        let online: std::collections::BTreeSet<usize> = online_nodes()
            .iter()
            .flat_map(|node| &node.cores)
            .flat_map(|core| &core.threads)
            .copied()
            .collect();
        let restore: Vec<usize> = every
            .iter()
            .copied()
            .filter(|cpu| online.contains(cpu))
            .collect();
        if !restore.is_empty() {
            let _ = apply(&restore);
        }
    }
}

impl Drop for ProcessPlacement {
    fn drop(&mut self) {
        // Restore best-effort onto the originally allowed CPUs, falling back to
        // their intersection with currently online topology CPUs (never
        // `allowed()`, which reflects the narrowed affinity we are undoing).
        // Never panic inside `Drop`, and always clear `PLACED`.
        restore_affinity(
            &self.every,
            crate::host::place_process,
            || discover_processors().nodes().to_vec(),
        );
        PLACED.store(false, std::sync::atomic::Ordering::Release);
    }
}

/// Clears `PLACED` if setup in [`place_process`] returns early or panics.
struct PlacementClaim;

impl Drop for PlacementClaim {
    fn drop(&mut self) {
        PLACED.store(false, std::sync::atomic::Ordering::Release);
    }
}

/// Place every thread of the process now on `cpus` (numbers [`Processors`]
/// reports, at least one) but those [`pin`] placed, which keep their CPUs,
/// until the returned placement drops; a thread it starts later inherits
/// its starter's placement, so threads a placed thread starts run there
/// too, and one may then be placed elsewhere with [`pin`]. Linux places
/// each of the process's threads (`/proc/self/task`, `sched_setaffinity`),
/// again until a pass finds none it had not placed (one started meanwhile
/// by a thread not yet placed); macOS and Windows place nothing here
/// (macOS's scheduler places every thread; Windows' process mask would
/// bound [`pin`]).
///
/// # Panics
/// While another placement is held: two owners of every thread's CPUs in
/// one process would each size its work from what the other left it, and
/// restore what it found.
pub fn place_process(cpus: &[usize]) -> Result<ProcessPlacement, String> {
    if cpus.is_empty() {
        return Err("a process placed on no CPU".into());
    }
    assert!(
        !PLACED.swap(true, std::sync::atomic::Ordering::AcqRel),
        "the process is placed already: one placement owns every thread's CPUs at a time (a second serving executor in one process)"
    );
    let claim = PlacementClaim;
    let every = discover_processors().allowed().iter().copied().collect::<Vec<_>>();
    crate::host::place_process(cpus)?;
    std::mem::forget(claim);
    Ok(ProcessPlacement { every })
}

/// The memory node of the device at `location` ([`HostNode`]): Linux's
/// `/sys/bus/pci/devices/<location>.0/numa_node`, or [`HostNode::Any`].
pub fn device_node(location: PciLocation) -> HostNode {
    crate::host::device_node(&crate::host::LiveSys, location)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::num::NonZeroUsize;

    /// A quota of `cpus`.
    fn quota(cpus: usize) -> NonZeroUsize {
        NonZeroUsize::new(cpus).unwrap()
    }

    /// The pool is the performance cores the process may run on, capped by
    /// its quota where that is fewer, never raised by it.
    #[test]
    fn the_pool_is_the_performance_cores_under_the_quota() {
        let nodes = seismic_topology::flat_nodes(&(0..128).collect());
        let pool = |cap| {
            Processors::new(nodes.clone(), (0..128).collect(), quota(cap))
                .unwrap()
                .cores()
                .unwrap()
                .get()
        };
        assert_eq!(pool(20), 20);
        assert_eq!(pool(128), 128);
        let few = Processors::new(nodes, (0..8).collect(), quota(128)).unwrap();
        assert_eq!(few.cores().unwrap().get(), 8);
    }

    #[test]
    fn this_host_reports_its_processors() {
        let processors = discover_processors();
        assert!(processors.performance_cores() >= 1);
        let cores = processors.cores().unwrap().get();
        assert_eq!(cores, processors.performance_cores().min(processors.quota().get()));
        assert!(!processors.memory_cores().is_empty());
        let cpus = processors
            .memory_cores()
            .into_iter()
            .flat_map(|(_, cores)| cores)
            .flatten()
            .collect::<Vec<_>>();
        pin(&cpus[..1]).unwrap();
        pin(&cpus).unwrap();
        assert!(pin(&[]).is_err());
    }

    #[test]
    fn restore_affinity_restores_full_set_and_falls_back_to_online_topology() {
        // Direct restore succeeds with the original CPU set even when the
        // process's current affinity was narrowed; topology is not queried.
        let mut applied: Vec<Vec<usize>> = Vec::new();
        restore_affinity(
            &[0, 1, 2, 3],
            |cpus| {
                applied.push(cpus.to_vec());
                Ok(())
            },
            || panic!("online_nodes should not be queried when direct restore succeeds"),
        );
        assert_eq!(applied, vec![vec![0, 1, 2, 3]]);

        // If a CPU was offlined (direct restore fails), intersect with
        // currently online topology CPUs rather than narrowed process affinity.
        let mut applied: Vec<Vec<usize>> = Vec::new();
        restore_affinity(
            &[0, 1, 2, 3],
            |cpus| {
                applied.push(cpus.to_vec());
                if cpus.contains(&3) {
                    Err("CPU 3 offline".into())
                } else {
                    Ok(())
                }
            },
            || seismic_topology::flat_nodes(&[0, 1, 2].into_iter().collect()),
        );
        assert_eq!(applied, vec![vec![0, 1, 2, 3], vec![0, 1, 2]]);
    }
}
