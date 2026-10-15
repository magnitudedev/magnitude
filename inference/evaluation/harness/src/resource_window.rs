//! Process and device resource snapshots outside the measured decode window.
use serde_json::{json, Value};

pub fn snapshot(device: &seismic::Device) -> Value {
    let usage = device.memory_usage();
    json!({
        "device_charged_bytes": usage.charged,
        "device_pool_charged_bytes": usage.pool_charged,
        "device_high_water_bytes_since_open": usage.high_water,
        "process": process(),
    })
}

#[cfg(target_os = "macos")]
fn process() -> Value {
    // Darwin's public rusage_info_v4, in SDK field order. This version
    // includes lifetime peak physical footprint and process disk counters.
    #[repr(C)]
    struct Usage {
        uuid: [u8; 16],
        user_time: u64, system_time: u64, pkg_idle_wkups: u64,
        interrupt_wkups: u64, pageins: u64, wired_size: u64,
        resident_size: u64, phys_footprint: u64,
        proc_start_abstime: u64, proc_exit_abstime: u64,
        child_user_time: u64, child_system_time: u64,
        child_pkg_idle_wkups: u64, child_interrupt_wkups: u64,
        child_pageins: u64, child_elapsed_abstime: u64,
        diskio_bytesread: u64, diskio_byteswritten: u64,
        cpu_time_qos_default: u64, cpu_time_qos_maintenance: u64,
        cpu_time_qos_background: u64, cpu_time_qos_utility: u64,
        cpu_time_qos_legacy: u64, cpu_time_qos_user_initiated: u64,
        cpu_time_qos_user_interactive: u64, billed_system_time: u64,
        serviced_system_time: u64, logical_writes: u64,
        lifetime_max_phys_footprint: u64, instructions: u64, cycles: u64,
        billed_energy: u64, serviced_energy: u64,
        interval_max_phys_footprint: u64, runnable_time: u64,
    }
    const _: () = assert!(std::mem::size_of::<Usage>() == 296);
    #[link(name = "proc")]
    unsafe extern "C" {
        fn proc_pid_rusage(pid: i32, flavor: i32, buffer: *mut Usage) -> i32;
    }
    let mut usage = std::mem::MaybeUninit::<Usage>::zeroed();
    // The syscall writes exactly this versioned public structure; use it
    // only after success. Failure is an observation error, never zero usage.
    let code = unsafe { proc_pid_rusage(std::process::id() as i32, 4, usage.as_mut_ptr()) };
    if code != 0 {
        return json!({"source":"proc_pid_rusage/RUSAGE_INFO_V4", "error":std::io::Error::last_os_error().to_string()});
    }
    let usage = unsafe { usage.assume_init() };
    json!({"source":"proc_pid_rusage/RUSAGE_INFO_V4",
        "physical_footprint_bytes":usage.phys_footprint,
        "lifetime_peak_physical_footprint_bytes":usage.lifetime_max_phys_footprint,
        "resident_bytes":usage.resident_size,"wired_bytes":usage.wired_size,
        "disk_read_bytes":usage.diskio_bytesread,"disk_write_bytes":usage.diskio_byteswritten,
        "pageins":usage.pageins,
        "disk_counter_boundary":"Darwin per-process disk I/O counters; logical source bytes are separate",
    })
}

#[cfg(not(target_os = "macos"))]
fn process() -> Value {
    json!({"error":"process disk/footprint observer requires Darwin"})
}
