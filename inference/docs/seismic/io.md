# Seismic I/O

`seismic-io` (`inference/seismic/io`) is Seismic's I/O layer: positional reads of files on storage volumes, the scheduler every reader's requests run on, and the accounting of what each volume did. It depends on no other Seismic crate, so a tool that only reads files needs no device runtime.

## What it owns

| Module | Owns |
| --- | --- |
| `destination` | The `Destination` contract: memory a read lands in. |
| `disk` | Each storage volume's identity (`Volume`), which accounting and in-flight reports name. Opening a file for reads (`OpenFile`) and every positional read of the process, uncached and cached reads per OS, each volume's probe (`probe_volume`), which keeps its whole curve (`Curve`: every point and lone time), and what a load adopts (`adopt_volume`, or the rule before's `EarlierChoice`). The link's depths and read sizes (`Shape`) are derived from the curve where they are used. Also `Reader`, through which callers make requests. |
| `scheduler` | The read engine. One link per volume, the only authority on what of it is in flight: it orders requests by `Want` (due before read ahead, then deadline, then value), starts due pieces, parked or streamed, up to the due depth, and pieces read ahead only while no parked piece waits and the parked bytes and those read ahead fit under the knee. A streamed want (a prompt group's conveyor) is due but holds back nothing read ahead, so reads ahead proceed beside a prompt; a parked one (a step waiting on its read) holds back everything read ahead. Only a step's own wait parks a want: a prediction's expected instant (a hint's) orders it among the reads ahead, and its passing parks nothing, so no parked read queues behind a prediction. A want's value is the step time it is expected to save, in seconds. A `Handle` owns a request (reprice; dropping it cancels what no worker took); its `Ending` observes the end. One pool of workers serves every link: each reader and probe claims workers for the pieces it submits, so the pool holds, on each volume, its link's due depth and knee in the smallest piece read there and one piece past them, never more than its claims have had queued or in flight at once; it shrinks as claims end, and no read the link admits waits for a thread. |
| `accounting` | Each volume's counts since the process began. |

## Contracts

- **A destination outlives its reads by construction.**
  - A range names an `Arc<dyn Destination>` and an offset into it.
  - The request holds every destination until its last piece has ended, whatever becomes of its handle. A read into freed memory therefore can't be expressed.
  - A range past its destination's region is a defect and panics.
  - A request stays `unsafe` for one remaining obligation: the bytes each range names are disjoint from every other read in flight and from what the caller touches until the request has ended (observed through its `Ending`; dropping the handle doesn't wait).
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
  - For due pieces and pieces read ahead (due or not when a worker took them):
    - bytes and requests ended;
    - busy time and depth;
    - each ended request's wait: device time, time behind pieces read ahead, and time behind due pieces.
  - Rate while busy and mean depth while busy are derived from these totals.
  - The scheduler keeps the record, so it is the same on every backend. Each link's book and each request's wait live in the scheduler's queue, under the lock it already holds wherever a piece is taken or ends, so counting takes no lock of its own. A look (`accounting::totals`) takes that lock once.
  - A piece ends at the instant its read returns. Its read lands at once, never waiting for the queue's lock: where another holds it, the piece is left for the lock's next holder, which ends every left piece before it looks at the queue.
  - Liveness (`engine_work`) reads the pieces ended from it.
  - A streamed model's per-step record (`StepCost::storage`) is the span of each volume's totals over the step.

## What it doesn't own

- **Formats and sources.** Formats stay in `magnitude-artifacts`. Its `FileSource` reads every range around the file cache as a request of its volume's link (`Reader::read_at`, due now: a header), and streams a whole stored range (`Reader::stream`: no deadline, half-knee requests three at a time, each chunk handed on as it lands) into an import's upload, so every read appears in its volume's accounting and no model byte stays in the cache. A host table's rows are the one model read through the cache (`Reader::read_at` of their own cached reader, one request a step), kept there for the steps that gather them again.
- **Where storage measurements are stored**, and how assessment prices reads. The executor's assessment persists the probe's curve.
- **Policy over what to read.** Why a unit moves, its deadline and value, residency and prediction belong to the engine; the link sees deadlines, values and bytes, never "hint" or "load".
- **The copy engine.** Host-to-device copies are a separate link owned by the runtime, and they get their own contract alongside the runtime's per-queue timeline.
- **Copies ordered by the compute stream**, which belong to the runtime's execution.
