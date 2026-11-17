//! Prints one host sample as JSON: the recording a fixture
//! (`seismic_discovery::fixtures`) is built from.

fn main() {
    seismic_threads::main("host-sample", start)
}

fn start() {
    let sample = seismic_discovery::host_sample();
    let wired = sample.wired.map(|wired| {
        serde_json::json!({
            "limit": match wired.limit {
                seismic_discovery::WiredLimit::OsDefault => serde_json::Value::Null,
                seismic_discovery::WiredLimit::Set(bytes) => bytes.into(),
            },
            "wire_count": wired.wire_count,
        })
    });
    let own_cold = match sample.own_cold {
        seismic_discovery::OwnCold::Bytes(bytes) => serde_json::Value::from(bytes),
        seismic_discovery::OwnCold::Unobserved => serde_json::Value::Null,
    };
    let harm = sample.harm.map(|harm| serde_json::json!({ "machine": harm.machine, "own": harm.own }));
    let recorded = serde_json::json!({
        "page_bytes": sample.page_bytes,
        "total": sample.total,
        "supply": {
            "free_beyond_target": sample.supply.free_beyond_target,
            "clean_cache": sample.supply.clean_cache,
            "cold_reclaimable": sample.supply.cold_reclaimable,
        },
        "counted_lag_ns": sample.counted.lag.as_nanos() as u64,
        "cold": sample.cold,
        "own_cold": own_cold,
        "own_reclaimed": sample.own_reclaimed.map(|own| serde_json::json!({ "bytes": own.bytes, "stored": own.stored, "exact": own.exact })),
        "wired": wired,
        "harm": harm,
        "pressure": sample.pressure.map(|level| format!("{level:?}")),
        "lag": { "release_ns": sample.lag.release.as_nanos() as u64, "growth_ns": sample.lag.growth.as_nanos() as u64 },
    });
    println!("{}", serde_json::to_string_pretty(&recorded).expect("a sample serializes"));
}
