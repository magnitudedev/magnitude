//! A pool worker: one program formed under one target in a process of its
//! own. The pool writes one [`Request`] to standard input and reads one
//! [`seismic::coverage::Verdict`] from standard output; a worker that ends
//! any other way established nothing.

use seismic::coverage::{ProgramSource, Target};

#[derive(serde::Deserialize)]
pub struct Request {
    pub target: Target,
    pub program: ProgramSource,
}

/// Serve one request from standard input to standard output.
pub fn serve() -> anyhow::Result<()> {
    use std::io::{Read, Write};
    let mut bytes = Vec::new();
    std::io::stdin().read_to_end(&mut bytes)?;
    let request: Request = postcard::from_bytes(&bytes)?;
    let verdict = seismic::coverage::form(request.target, &request.program);
    let mut stdout = std::io::stdout().lock();
    stdout.write_all(&postcard::to_allocvec(&verdict)?)?;
    stdout.flush()?;
    Ok(())
}
