use anyhow::Context;
use icn_contracts::bootstrap_protocol::{IcnParentCommand, IcnParentCommandType};
use std::io::{self, Read};
use tokio::sync::watch;

// Bounded framing on the lifetime thread: malformed input never disables parent-loss protection.
fn read_commands(mut input: impl Read, mut shutdown: impl FnMut()) -> io::Result<()> {
    let mut frame = Vec::with_capacity(256);
    let mut bytes = [0_u8; 64];
    loop {
        let length = match input.read(&mut bytes) {
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            result => result?,
        };
        if length == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "parent lifetime channel closed",
            ));
        }
        for byte in &bytes[..length] {
            if *byte == b'\n' {
                let command: IcnParentCommand = serde_json::from_slice(&frame).map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidData, "invalid parent command")
                })?;
                match command.command_type {
                    IcnParentCommandType::Shutdown => shutdown(),
                }
                frame.clear();
            } else {
                if frame.len() == 256 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "parent command too large",
                    ));
                }
                frame.push(*byte);
            }
        }
    }
}

pub fn install() -> anyhow::Result<watch::Receiver<bool>> {
    #[cfg(unix)]
    anyhow::ensure!(
        unsafe { libc::getpgrp() == libc::getpid() },
        "managed ICN must lead its owned process group"
    );
    let (shutdown, receiver) = watch::channel(false);
    std::thread::Builder::new()
        .name("icn-parent-channel-watchdog".to_owned())
        .spawn(move || {
            let _ = read_commands(std::io::stdin().lock(), || {
                shutdown.send_replace(true);
            });
            // This path is independent of Tokio and remains armed during orderly shutdown.
            #[cfg(unix)]
            unsafe {
                libc::kill(-libc::getpid(), libc::SIGKILL);
                libc::_exit(91);
            }
            #[cfg(windows)]
            std::process::exit(91);
        })
        .context("cannot start ICN parent-channel watchdog")?;
    Ok(receiver)
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Fragmented<'a>(&'a [u8]);
    impl Read for Fragmented<'_> {
        fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
            if self.0.is_empty() {
                return Ok(0);
            }
            output[0] = self.0[0];
            self.0 = &self.0[1..];
            Ok(1)
        }
    }
    #[test]
    fn fragmented_commands_preserve_eof_detection_after_shutdown() {
        let encoded = serde_json::to_vec(&IcnParentCommand {
            command_type: IcnParentCommandType::Shutdown,
        })
        .unwrap();
        let input = [encoded.as_slice(), b"\n", encoded.as_slice(), b"\n"].concat();
        let mut requests = 0;
        let result = read_commands(Fragmented(&input), || requests += 1);
        assert_eq!(requests, 2);
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::UnexpectedEof);
    }
    #[test]
    fn invalid_or_oversized_frames_cannot_request_shutdown() {
        for input in [
            b"{\"type\":\"other\"}\n".to_vec(),
            b"{\"type\":\"shutdown\",\"extra\":1}\n".to_vec(),
            vec![b'x'; 257],
            vec![0xff, b'\n'],
        ] {
            let result = read_commands(input.as_slice(), || panic!("invalid command dispatched"));
            assert_eq!(result.unwrap_err().kind(), io::ErrorKind::InvalidData);
        }
    }
}
