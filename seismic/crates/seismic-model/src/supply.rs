//! Static and calibrated supply of a device, persisted in the tuning cache.

use std::path::PathBuf;

#[derive(Clone, Debug, Default)]
pub struct Supply {
    pub device: String,
    /// Calibrated streaming-read bandwidth, bytes per second.
    pub bandwidth: f64,
}

fn cache_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join(".seismic-cache")
}

fn key(device: &str) -> String {
    device.chars().map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_lowercase() } else { '-' }).collect()
}

impl Supply {
    pub fn load(device: &str) -> Option<Supply> {
        let path = cache_dir().join(format!("{}.supply", key(device)));
        let text = std::fs::read_to_string(path).ok()?;
        let mut s = Supply { device: device.to_string(), bandwidth: 0.0 };
        for line in text.lines() {
            if let Some((k, v)) = line.split_once('=') {
                if k.trim() == "bandwidth" {
                    s.bandwidth = v.trim().parse().ok()?;
                }
            }
        }
        if s.bandwidth > 0.0 { Some(s) } else { None }
    }

    pub fn store(&self) -> Result<(), String> {
        let dir = cache_dir();
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let path = dir.join(format!("{}.supply", key(&self.device)));
        std::fs::write(path, format!("device={}\nbandwidth={}\n", self.device, self.bandwidth)).map_err(|e| e.to_string())
    }
}
