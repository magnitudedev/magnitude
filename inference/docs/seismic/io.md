# Seismic I/O

`seismic-io` (`inference/seismic/io`) is Seismic's I/O layer: positional reads of files on storage volumes, the scheduler every reader's jobs run on, and the accounting of what each volume did. It depends on no other Seismic crate, so a tool that only reads files needs no device runtime.

## What it owns

| Module | Owns |
| --- | --- |
| `destination` | The `Destination` contract: memory a read lands in. |
| `disk` | Each storage volume's identity (`Volume`), which accounting and in-flight reports name. Opening a file for reads (`OpenFile`) and every positional read of the process, uncached and cached reads per OS, each volume's probe (`probe_volume`), and the choice a load adopts (`adopt_volume`, `VolumeChoice`). Also `Reader`, through which callers submit jobs. |
| `scheduler` | The read engine. It holds jobs and cuts them into pieces in one queue, with the pool of workers and the priority classes Demand, Prefetch and Load. It applies each volume's windows and the narrowest urgent reader's bound, and handles withdrawal and promotion. |
| `accounting` | Each volume's counts since the process began. |

## Contracts

- **A destination outlives its reads by construction.**
  - A range names an `Arc<dyn Destination>` and an offset into it.
  - The job holds every destination until its last piece has ended, whatever becomes of the job's handle. A read into freed memory therefore can't be expressed.
  - A range past its destination's region is a defect and panics.
  - Submission stays `unsafe` for one remaining obligation: the bytes each range names are disjoint from every other read in flight and from what the caller touches while the job runs.
  - A read the caller waits out within a borrow (`Reader::range`, an `io::Read`) lands straight in the caller's buffer.
- **Uncached reads never fill the file cache.**
  - Linux uses `O_DIRECT`, macOS `F_NOCACHE`, and Windows `FILE_FLAG_NO_BUFFERING`.
  - Where a filesystem refuses direct I/O, the file is read through the cache from then on, and each read's pages are dropped (Linux `POSIX_FADV_DONTNEED`, macOS `MS_INVALIDATE`).
- **Cached reads have no read-ahead past the ranges named.**
- **Accounting is never consumed.**
  - Each volume keeps cumulative totals, and a reader of a span takes the difference of two looks (`Totals::since`). Readers therefore never take counts from one another.
  - Per volume:
    - busy time: any piece in flight;
    - depth: pieces in flight, integrated over time.
  - Per class:
    - bytes and requests ended;
    - busy time and depth;
    - each ended job's wait: device time, time behind speculative pieces, and time behind demand.
  - Rate while busy and mean depth while busy are derived from these totals.
  - The scheduler keeps the record, so it is the same on every backend. Each volume's book and each job's wait live in the scheduler's queue, under the lock it already holds wherever a piece is taken or ends, so counting takes no lock of its own. A look (`accounting::totals`) takes that lock once.
  - Liveness (`engine_work`) reads the pieces ended from it.
  - A streamed model's per-step record (`StepCost::storage`) is the span of each volume's totals over the step.

## What it doesn't own

- **Formats and sources.** Formats stay in `magnitude-artifacts`. Its `FileSource` holds an `OpenFile` and reads its cached ranges through it (`OpenFile::read_exact_at`), synchronously on the caller's thread and outside the scheduler, so they don't appear in any volume's accounting. Mmap windows serve resident loads.
- **Where storage measurements are stored**, and how assessment prices reads. The executor's assessment persists the probe's choice.
- **Policy over what to read.** Wants, residency and prediction belong to the engine.
- **The copy engine.** Host-to-device copies are a separate link owned by the runtime, and they get their own contract alongside the runtime's per-queue timeline.
- **Copies ordered by the compute stream**, which belong to the runtime's execution.
