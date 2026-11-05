//! Demand reads of expert records from the model's files into staging.
//! Reads run uncached on the read engine
//! (`magnitude_artifacts::ArtifactReader`), which aligns, coalesces and
//! splits them into pieces its workers read in parallel, each landing
//! straight in its staging record.

use magnitude_artifacts::{ArtifactReader, CachePolicy, Ended, Job, Panic, Priority, Range};
use seismic::Tensor;
use std::ptr::NonNull;
use std::sync::Arc;
use std::time::Instant;

/// The records of a host-shared tensor that reads land in: `count` records
/// of `bytes` each, its whole extent, each one contiguous range where the
/// tensor lays it, checked once when they are made.
#[derive(Clone)]
pub(crate) struct Records {
    tensor: Tensor,
    base: NonNull<u8>,
    placement: seismic::RecordPlacement,
    count: u64,
}

// SAFETY: `base` addresses host-shared memory that `tensor`, held beside
// it, keeps alive; the host writes it only through the read engine's jobs
// (`ExpertSource::submit`) and record copies (`Records::copy_record`).
unsafe impl Send for Records {}

impl Records {
    /// `tensor` as `count` records of `bytes` each, its leading extent's,
    /// where the tensor lays them (`Tensor::record_placement`): refused
    /// unless they are its whole extent and the host addresses it.
    pub fn new(tensor: Tensor, bytes: u64, count: u64) -> Result<Self, String> {
        let placement = tensor
            .record_placement()
            .filter(|placement| {
                tensor.extents().first() == Some(&count) && placement.record() == bytes
            })
            .ok_or_else(|| format!("{} bytes are not {count} records of {bytes}", tensor.byte_len()))?;
        let base = tensor
            .shared_address()
            .ok_or_else(|| format!("{count} records of {bytes} bytes are no memory the host reads into"))?;
        Ok(Self {
            tensor,
            base,
            placement,
            count,
        })
    }

    /// Its records `start..end` as records of their own, which hold the
    /// tensor's storage only through the last of them
    /// (`Tensor::record_region`): a recommit of the tensor to `end` records
    /// or more never meets them.
    pub fn range(&self, start: u64, end: u64) -> Result<Self, String> {
        let tensor = self
            .tensor
            .record_region(start, end)
            .map_err(|error| format!("records {start}..{end} of {}: {error}", self.count))?;
        Self::new(tensor, self.bytes(), end - start)
    }

    /// The bytes of one record.
    pub fn bytes(&self) -> u64 {
        self.placement.record()
    }

    /// The records.
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Where its records lie.
    #[cfg(test)]
    pub fn placement(&self) -> seismic::RecordPlacement {
        self.placement
    }

    /// The tensor the records lie in.
    #[cfg(test)]
    pub fn tensor(&self) -> &Tensor {
        &self.tensor
    }

    /// Where record `record` lies, in bytes from the first's.
    pub fn offset(&self, record: u64) -> u64 {
        self.placement.offset(record)
    }

    /// The first `bytes` of record `record`, to read.
    ///
    /// # Safety
    ///
    /// No read lands in the record, and no device work writes it, while the
    /// slice lives.
    pub unsafe fn record(&self, record: u64, bytes: u64) -> &[u8] {
        assert!(record < self.count && bytes <= self.bytes(), "{bytes} bytes of record {record} of {}", self.count);
        // SAFETY: the record lies within the host-shared tensor `self`
        // keeps alive, and the caller excludes every writer.
        unsafe { std::slice::from_raw_parts(self.base.as_ptr().add(self.offset(record) as usize), bytes as usize) }
    }

    /// Where record `record` begins in host memory: what a writer of part
    /// of it (one piece of a landing's conversion) writes through.
    pub fn address(&self, record: u64) -> std::ptr::NonNull<u8> {
        assert!(record < self.count, "record {record} of {}", self.count);
        // SAFETY: the record lies within the host-shared tensor `self`
        // keeps alive.
        unsafe { self.base.add(self.offset(record) as usize) }
    }

    /// Copy the first `bytes` of record `source` of `from` into record
    /// `record`, where a read of the same bytes would land them.
    ///
    /// # Safety
    ///
    /// No read lands in either record, and no device work reads or writes
    /// them, until the copy returns.
    pub unsafe fn copy_record(&self, record: u64, from: &Records, source: u64, bytes: u64) {
        assert!(
            record < self.count && source < from.count && bytes <= self.bytes().min(from.bytes()),
            "a copy of {bytes} bytes from record {source} of {} into record {record} of {}",
            from.count,
            self.count
        );
        // SAFETY: both records lie within their host-shared tensors, which
        // `self` and `from` keep alive, and the caller excludes every other
        // access to them; `copy` allows the two to overlap.
        unsafe {
            std::ptr::copy(
                from.base.as_ptr().add(from.offset(source) as usize),
                self.base.as_ptr().add(self.offset(record) as usize),
                bytes as usize,
            );
        }
    }
}

/// One range of a model file read into a record: `bytes` at `offset` of
/// file `file` (an index of the source's files) into record `record` of
/// `into`.
pub(super) struct Read {
    file: usize,
    offset: u64,
    bytes: usize,
    into: Records,
    record: u64,
}

impl Read {
    /// A read of `bytes` at `offset` of file `file` into record `record` of
    /// `into`, which holds them: a record of `into` and no more than its
    /// bytes, as the watcher's geometry (checked at its start) gives.
    pub fn new(file: usize, offset: u64, bytes: u64, into: &Records, record: u64) -> Self {
        assert!(
            record < into.count && bytes <= into.bytes(),
            "a read of {bytes} bytes into record {record} of {} of {} bytes",
            into.count,
            into.bytes()
        );
        Self {
            file,
            offset,
            bytes: bytes as usize,
            into: into.clone(),
            record,
        }
    }

    pub fn bytes(&self) -> u64 {
        self.bytes as u64
    }
}

/// How a submitted job ended.
#[derive(Clone, Debug)]
pub(super) enum Outcome {
    /// Every read landed in staging.
    Read,
    /// A read failed with an I/O error from the source: why.
    Failed(String),
    /// A read worker panicked running the job: a defect, which the watcher
    /// raises as its own.
    Panicked(Panic),
}

/// A submitted job: one read engine job per model file it reads, and the
/// staging it lands in, held until every one has ended.
pub(super) struct Ticket {
    /// Each file's job, and the reads it serves in their order: (the read's
    /// index among the ticket's, file, offset, bytes).
    jobs: Vec<(Job, Vec<(usize, usize, u64, usize)>)>,
    _staging: Vec<Tensor>,
}

impl Drop for Ticket {
    /// What no worker has taken reads nothing ([`Ticket::withdraw`]).
    fn drop(&mut self) {
        self.withdraw();
    }
}

impl Ticket {
    /// The outcome, waiting for it until `deadline`; `None` while a read
    /// runs. The first failed read is its outcome.
    pub fn wait_until(&self, deadline: Instant) -> Option<Outcome> {
        self.outcome(|job| job.wait_until(deadline))
    }

    /// Whether a read of the ticket still runs: it has not ended.
    pub fn running(&self) -> bool {
        self.wait_until(Instant::now()).is_none()
    }

    /// The outcome, waiting for it.
    pub fn wait(&self) -> Outcome {
        self.outcome(|job| Some(job.wait())).expect("a waited job ends")
    }

    /// The outcome of the jobs as `ended` finds them, none while one runs.
    fn outcome(&self, ended: impl Fn(&Job) -> Option<Ended>) -> Option<Outcome> {
        let endeds = self.jobs.iter().map(|(job, _)| ended(job)).collect::<Option<Vec<_>>>()?;
        let reads = self.jobs.iter().map(|(_, reads)| reads.clone()).collect::<Vec<_>>();
        Some(outcome_of(&endeds, &reads).0)
    }

    /// Run `hook` with the outcome, and the indices of the ticket's reads
    /// that failed (every one after a panic), once every job has ended, on
    /// the thread that ends the last (a read worker), or at once where every
    /// one has: after every job's end is set, so whatever the hook wakes
    /// finds the ticket no longer running. It must not wait for a read.
    pub fn on_end(&self, hook: impl FnOnce(Outcome, &[usize]) + Send + 'static) {
        type Hook = Box<dyn FnOnce(Outcome, &[usize]) + Send>;
        struct Ending {
            ended: Vec<Option<Ended>>,
            hook: Option<Hook>,
        }
        if self.jobs.is_empty() {
            return hook(Outcome::Read, &[]);
        }
        let reads = std::sync::Arc::new(self.jobs.iter().map(|(_, reads)| reads.clone()).collect::<Vec<_>>());
        let ending = std::sync::Arc::new(std::sync::Mutex::new(Ending {
            ended: vec![None; self.jobs.len()],
            hook: Some(Box::new(hook)),
        }));
        for (index, (job, _)) in self.jobs.iter().enumerate() {
            let (ending, reads) = (ending.clone(), reads.clone());
            job.on_end(move |ended| {
                let mut ending = ending.lock().expect("a ticket's end");
                ending.ended[index] = Some(ended.clone());
                let Some(ended) = ending.ended.iter().cloned().collect::<Option<Vec<_>>>() else {
                    return;
                };
                let hook = ending.hook.take().expect("a ticket ends once");
                drop(ending);
                let (outcome, failed) = outcome_of(&ended, &reads);
                hook(outcome, &failed);
            });
        }
    }

    /// When a worker takes each job's first piece, each set then
    /// (`magnitude_artifacts::Job::started`): the ticket's reads start at
    /// the earliest.
    pub fn started(&self) -> Vec<std::sync::Arc<std::sync::OnceLock<Instant>>> {
        self.jobs.iter().map(|(job, _)| job.started()).collect()
    }

    /// Read every job no less urgently than `priority` (never demoted).
    pub fn promote(&self, priority: Priority) {
        for (job, _) in &self.jobs {
            job.promote(priority);
        }
    }

    /// Withdraw each job's pieces no worker has taken: they read nothing and
    /// the reads they serve fail as withdrawn
    /// (`magnitude_artifacts::Error::Withdrawn`); pieces in flight read on.
    pub fn withdraw(&self) {
        self.jobs.iter().for_each(|(job, _)| job.withdraw());
    }
}

/// The outcome of a ticket whose jobs ended as `ended`, each serving
/// `reads` (index among the ticket's, file, offset, bytes) in its order:
/// the first failed read in job order, or a panic; and the indices of the
/// ticket's reads that failed.
fn outcome_of(ended: &[Ended], reads: &[Vec<(usize, usize, u64, usize)>]) -> (Outcome, Vec<usize>) {
    let mut outcome = Outcome::Read;
    let mut failed_reads = Vec::new();
    for (ended, reads) in ended.iter().zip(reads) {
        match ended {
            Ended::Read { failed } => {
                if let (Outcome::Read, Some((index, error))) = (&outcome, failed.first()) {
                    let (_, file, offset, bytes) = reads[*index];
                    outcome = Outcome::Failed(format!(
                        "expert record at {offset} of {bytes} bytes in model file {file}: {error}"
                    ));
                }
                failed_reads.extend(failed.iter().map(|(index, _)| reads[*index].0));
            }
            Ended::Panicked(panic) => return (Outcome::Panicked(panic.clone()), reads.iter().map(|read| read.0).collect()),
        }
    }
    (outcome, failed_reads)
}

/// The model's files, opened for reads that bypass the file cache.
pub(super) struct ExpertSource {
    files: Vec<ArtifactReader>,
}

impl ExpertSource {
    /// The model's `files`, read for decode requests of `urgent`.
    pub fn open(files: &[std::path::PathBuf], urgent: &magnitude_artifacts::Urgent) -> Result<Self, String> {
        let files = files
            .iter()
            .map(|path| {
                ArtifactReader::open_urgent(path, CachePolicy::Uncached, urgent)
                    .map_err(|error| format!("expert source {}: {error}", path.display()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { files })
    }

    /// What the volumes its files lie on read and how long each was busy,
    /// each volume once ([`magnitude_artifacts::Delivery`]).
    pub fn deliveries(&self) -> Vec<Arc<magnitude_artifacts::Delivery>> {
        let mut deliveries = Vec::<Arc<magnitude_artifacts::Delivery>>::new();
        for file in &self.files {
            let delivery = file.delivery();
            if !deliveries.iter().any(|known| Arc::ptr_eq(known, &delivery)) {
                deliveries.push(delivery);
            }
        }
        deliveries
    }

    /// Its reads of the read engine now (whether any is submitted and not
    /// yet ended, and the engine's work), as a check that reads only the
    /// read engine's counts, never an owner's lock: the exchange's
    /// watchdog's (`crate::watched::HostClock`).
    pub(crate) fn reads(&self) -> impl Fn() -> crate::watched::Reads + Send + 'static {
        let outstanding = self.files.iter().map(ArtifactReader::outstanding).collect::<Vec<_>>();
        move || crate::watched::Reads {
            outstanding: outstanding.iter().any(|reader| reader.jobs() > 0),
            engine: magnitude_artifacts::engine_work(),
        }
    }

    /// Queue `reads` on the read engine at `priority`: [`Priority::Demand`]
    /// for a request's misses, [`Priority::Prefetch`] for the next layer's
    /// hints. Fails only when the engine takes no job.
    pub fn submit(&self, reads: Vec<Read>, priority: Priority) -> Result<Ticket, String> {
        assert!(reads.iter().all(|read| read.file < self.files.len()), "a read of a file the source has not open");
        let mut jobs = Vec::new();
        for (index, file) in self.files.iter().enumerate() {
            let mine = reads.iter().enumerate().filter(|(_, read)| read.file == index).collect::<Vec<_>>();
            if mine.is_empty() {
                continue;
            }
            let ranges = mine
                .iter()
                .map(|(_, read)| Range {
                    offset: read.offset,
                    length: read.bytes,
                    // SAFETY: within the records, which `Read::new` holds
                    // its record and bytes to.
                    sink: unsafe { read.into.base.add(read.into.offset(read.record) as usize) },
                })
                .collect();
            // SAFETY: each staging record is the request's alone while its
            // job runs: the device reads it only after the watcher publishes
            // it, once the job has ended, and the ticket holds the staging
            // until then.
            let job = unsafe { file.submit(ranges, priority) }
                .map_err(|error| format!("expert source: {error}"))?;
            let served = mine
                .iter()
                .map(|(at, read)| (*at, index, read.offset, read.bytes))
                .collect();
            jobs.push((job, served));
        }
        Ok(Ticket {
            jobs,
            _staging: reads.into_iter().map(|read| read.into.tensor).collect(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use seismic::{BackendName, Element};

    /// Each read lands its range in its staging record; a range outside
    /// the file fails its job, and the jobs after it read.
    #[test]
    fn reads_land_in_staging_and_a_failed_one_fails_its_job_alone() {
        let device = seismic::Devices::discover(&[])
            .unwrap()
            .open_backend(BackendName::Cpu)
            .unwrap();
        let path = std::env::temp_dir().join(format!("expert-source-{}", std::process::id()));
        let bytes = (0..64 * 1024).map(|index| (index % 251) as u8).collect::<Vec<_>>();
        std::fs::write(&path, &bytes).unwrap();
        let source = ExpertSource::open(std::slice::from_ref(&path), &magnitude_artifacts::Urgent { parts: vec![128], records: 2 }).unwrap();
        let staging = Tensor::shared(&device, Element::i32(), &[2, 32]).unwrap();
        let records = Records::new(staging.clone(), 128, 2).unwrap();
        let read = |offset: u64, record: u64| Read::new(0, offset, 128, &records, record);
        let demand = Priority::Demand;
        let failed = source.submit(vec![read(bytes.len() as u64 - 64, 0)], demand).unwrap();
        let landed = source.submit(vec![read(1000, 0), read(40_000, 1)], demand).unwrap();
        let deadline = Instant::now() + std::time::Duration::from_secs(10);
        assert!(matches!(failed.wait_until(deadline), Some(Outcome::Failed(_))));
        assert!(matches!(landed.wait_until(deadline), Some(Outcome::Read)));
        let mut host = vec![0u8; 256];
        unsafe { staging.read_shared(0, &mut host) }.unwrap();
        assert_eq!(&host[..128], &bytes[1000..1128]);
        assert_eq!(&host[128..], &bytes[40_000..40_128]);
        // Records that are not their tensor's whole extent are refused when
        // made, before anything is read.
        assert!(Records::new(staging.clone(), 128, 3).is_err());
        drop((failed, landed));
        std::fs::remove_file(path).unwrap();
    }

    /// Where a CUDA device reserves its device storage in host memory (GB10,
    /// whose memory is the host's), the uncached reads land in a bank's
    /// reserved records in place, as prompt groups read into the conveyor
    /// lent from them: two 2 MiB records of a 4 MiB file, each read whole,
    /// hold the file's bytes. Such a host's check, run on one by name with
    /// `--ignored`.
    #[test]
    #[ignore]
    fn uncached_reads_land_in_a_host_reserved_bank_in_place() {
        let device = seismic::Devices::discover(&[])
            .expect("devices are discovered")
            .open_backend(BackendName::Cuda)
            .expect("a CUDA device opens");
        const RECORD: u64 = 2 << 20;
        let (bank, _resizer) =
            crate::planning::BankAllocation::reserve(&device, crate::planning::BankTier::Bank.contract_on(&device), Element::i32(), &[2, RECORD / 4], 2, None)
                .unwrap();
        let records = Records::new(bank.clone(), RECORD, 2).expect("the host addresses the device's reserved storage");
        let path = std::env::temp_dir().join(format!("expert-source-reserved-{}", std::process::id()));
        let bytes = (0..2 * RECORD).map(|index| (index % 251) as u8 ^ (index >> 20) as u8).collect::<Vec<_>>();
        std::fs::write(&path, &bytes).unwrap();
        let source = ExpertSource::open(std::slice::from_ref(&path), &magnitude_artifacts::Urgent { parts: vec![128], records: 2 }).unwrap();
        // Reversed: record 0 takes the file's second half.
        let landed = source
            .submit(vec![Read::new(0, RECORD, RECORD, &records, 0), Read::new(0, 0, RECORD, &records, 1)], Priority::Demand)
            .unwrap();
        let deadline = Instant::now() + std::time::Duration::from_secs(30);
        assert!(matches!(landed.wait_until(deadline), Some(Outcome::Read)));
        let mut host = vec![0u8; 2 * RECORD as usize];
        unsafe { bank.read_shared(0, &mut host) }.unwrap();
        assert!(host[..RECORD as usize] == bytes[RECORD as usize..], "record 0 holds the file's second half");
        assert!(host[RECORD as usize..] == bytes[..RECORD as usize], "record 1 holds its first half");
        drop(landed);
        std::fs::remove_file(path).unwrap();
    }
}
