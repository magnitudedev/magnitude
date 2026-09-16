//! Cached descriptive CPU topology, independent of scheduling capacity.
use std::sync::OnceLock;

pub(super) fn physical_cores() -> Option<usize> {
    static CORES: OnceLock<Option<usize>> = OnceLock::new();
    *CORES.get_or_init(|| read_physical_cores().filter(|count| *count > 0))
}

#[cfg(not(target_os = "linux"))]
fn read_physical_cores() -> Option<usize> {
    // sysctl on macOS and GetLogicalProcessorInformationEx on Windows.
    sysinfo::System::physical_core_count()
}

#[cfg(target_os = "linux")]
fn read_physical_cores() -> Option<usize> {
    use std::io::Read;
    const LIMIT: u64 = 16 * 1024 * 1024;
    let mut text = String::new();
    std::fs::File::open("/proc/cpuinfo")
        .ok()?
        .take(LIMIT + 1)
        .read_to_string(&mut text)
        .ok()?;
    if text.len() as u64 > LIMIT {
        return None;
    }
    parse_linux_physical_cores(&text)
}

#[cfg(any(target_os = "linux", test))]
fn parse_linux_physical_cores(text: &str) -> Option<usize> {
    let mut cores = std::collections::BTreeSet::new();
    for record in text.split("\n\n") {
        let mut processor = false;
        let mut package = None;
        let mut core = None;
        for line in record.lines() {
            let Some((key, value)) = line.split_once(':') else {
                continue;
            };
            match key.trim() {
                "processor" => processor = true,
                "physical id" => package = Some(value.trim().parse::<u32>().ok()?),
                "core id" => core = Some(value.trim().parse::<u32>().ok()?),
                _ => {}
            }
        }
        if processor {
            // ARM or virtualized /proc files may omit topology. Counting their
            // logical processor records would falsely imply physical cores.
            cores.insert((package?, core?));
        }
    }
    (!cores.is_empty()).then_some(cores.len())
}

#[cfg(test)]
mod tests {
    use super::parse_linux_physical_cores;
    #[test]
    fn separates_smt_threads_and_socket_local_core_ids() {
        let mut text = String::new();
        for socket in 0..2 {
            for core in 0..4 {
                for thread in 0..2 {
                    text.push_str(&format!(
                        "processor : {}\nphysical id : {socket}\ncore id : {core}\n\n",
                        socket * 8 + core * 2 + thread
                    ));
                }
            }
        }
        assert_eq!(parse_linux_physical_cores(&text), Some(8));
        assert_eq!(parse_linux_physical_cores(text.trim_end()), Some(8));
    }
    #[test]
    fn incomplete_topology_never_becomes_a_logical_core_guess() {
        for text in [
            "",
            "processor : 0\n",
            "processor : 0\ncore id : 0",
            "processor : 0\nphysical id : 0",
            "processor : 0\nphysical id : -1\ncore id : 0",
            "processor : 0\nphysical id : 0\ncore id : bad",
            "processor : 0\nphysical id : 0\ncore id : 0\n\nprocessor : 1",
        ] {
            assert_eq!(parse_linux_physical_cores(text), None);
        }
    }
}
