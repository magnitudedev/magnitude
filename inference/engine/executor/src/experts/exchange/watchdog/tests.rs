//! A waited lane of each kind left unanswered trips the watchdog with its
//! report; an answered lane, a scratch request's own lane and a lane of a
//! step that ended never do. Each decision is a look at an instant the
//! test chooses ([`Watch::look`]), so the host's load is never its input;
//! one test runs the thread.

use super::*;
use magnitude_kernels::SlotTable;
use seismic::{BackendName, Devices, Element, Tensor};
use crate::assessment::ReadLine;
use crate::watched::VolumeBound;
use seismic_io::{EngineWork, Flight, Volume};
use std::collections::HashMap;

/// A piece in flight's bytes: any read size a volume reads in.
const PIECE: u64 = 128 << 10;
use std::path::Path;
use std::sync::mpsc;

fn slots(table: SlotTable) -> ExchangeTable {
    let device = Devices::discover(&[]).unwrap().open_backend(BackendName::Cpu).unwrap();
    ExchangeTable::new(Tensor::shared(&device, Element::i32(), &[table.words()]).unwrap(), table, &super::super::Pulse::new()).unwrap()
}

/// The reads of a load that reads nothing, of an engine that works on
/// nothing.
fn idle() -> Reads {
    Reads { outstanding: false, engine: EngineWork { ended: 0, reading: Vec::new(), last_ended: HashMap::new() } }
}

/// Each volume bounded at [`MULTIPLE`](crate::watched::MULTIPLE) times a
/// read line of `latency` seconds a piece of its window.
fn volumes(latency: f64) -> VolumeBound {
    VolumeBound(Some(ReadLine::new(latency, 0.0).unwrap()))
}

/// The bounds of a test's load: each lane and park at `limit`, each volume
/// by `volumes`.
fn bounds(limit: Duration, volumes: VolumeBound) -> Bounds {
    Bounds { lane: Bound { limit, derivation: "the test's".into() }, volumes }
}

/// A watch begun at `began` over `tables` and the park `parked`, whose
/// steps end as `ended` says and whose reads are as `reads` says,
/// reporting one witness.
fn watch(tables: Vec<Watched>, parked: ParkedAt, ended: Ended, reads: Box<dyn Fn() -> Reads + Send>, began: Instant) -> Watch {
    let witnesses = vec![Witness { name: "owner", report: Arc::new(|| "the owner's state".into()) }];
    Watch::new(tables, parked, ended, Arc::default(), reads, witnesses, began)
}

/// `watch`'s looks every tenth of `bounds`' lane bound from `began`, up to
/// `until` after it: the first that trips, its instant and report.
fn looks(watch: &mut Watch, began: Instant, bounds: &Bounds, until: Duration) -> Option<(Duration, String)> {
    let period = bounds.lane.limit / 10;
    (1..)
        .map(|look| period * look)
        .take_while(|&after| after <= until)
        .find_map(|after| watch.look(began + after, bounds).map(|report| (after, report)))
}

/// A load's first read, with no read line yet, on a volume that never
/// returns it: its progress stops once the volume has ended none of it for
/// its bound, and the lane waiting on it then trips within its own, the
/// volume in the report. A slow read that keeps ending pieces never does.
#[test]
fn a_read_that_never_ends_trips_with_no_read_line() {
    let began = Instant::now();
    let unpriced = Bounds::unpriced();
    let limit = unpriced.lane.limit;
    let flight = move |ended| {
        Box::new(move || {
            let reading = vec![Flight { path: Path::new("/models/experts.gguf").into(), volume: Volume(1), offset: 0, bytes: PIECE, window: PIECE, since: began }];
            Reads { outstanding: true, engine: EngineWork { ended, reading, last_ended: HashMap::new() } }
        }) as Box<dyn Fn() -> Reads + Send>
    };
    let unanswered = || {
        let slots = slots(SlotTable::file(4, 2));
        slots.request(6, 0);
        vec![Watched { kind: Kind::File, layer: 3, slots }]
    };
    let mut stuck = watch(unanswered(), Box::new(|| None), Box::new(|_| false), flight(7), began);
    let (after, report) = looks(&mut stuck, began, &unpriced, limit * 3).expect("a read that never ends trips");
    assert!(after > limit && after <= limit * 2 + limit / 5, "tripped after {after:?}");
    assert!(report.contains("past its bound") && report.contains("no read line yet"), "{report}");
    // Pieces ending between the looks: progress, however long the lane waits.
    let ended = std::sync::atomic::AtomicU64::new(0);
    let progressing = Box::new(move || Reads {
        outstanding: true,
        engine: EngineWork { ended: ended.fetch_add(1, std::sync::atomic::Ordering::Relaxed), reading: Vec::new(), last_ended: HashMap::new() },
    });
    let mut slow = watch(unanswered(), Box::new(|| None), Box::new(|_| false), progressing, began);
    assert!(looks(&mut slow, began, &unpriced, limit * 10).is_none(), "tripped while its reads progressed");
}

/// The watchdog's thread looks until a lane is overdue, hands its report
/// to its trip, and stops when dropped.
#[test]
fn the_watchdogs_thread_trips_with_its_report_and_stops_when_dropped() {
    let limit = Duration::from_millis(20);
    let start = |ended: Ended| {
        let slots = slots(SlotTable::file(4, 2));
        slots.request(6, 0);
        let (tripped, report) = mpsc::channel();
        let watchdog = Watchdog::start(
            vec![Watched { kind: Kind::File, layer: 3, slots }],
            Box::new(|| None),
            ended,
            Arc::default(),
            Box::new(idle),
            Box::new(move || bounds(limit, volumes(1e-3))),
            Vec::new(),
            Box::new(move |report| tripped.send(report).unwrap()),
        )
        .unwrap();
        (watchdog, report)
    };
    let began = Instant::now();
    let (_watchdog, report) = start(Box::new(|_| false));
    let report = report.recv_timeout(Duration::from_secs(10)).expect("never tripped");
    assert!(began.elapsed() >= limit, "tripped before its bound");
    assert!(report.contains("File 3 2 0: 6, 0x0"), "{report}");
    let (watchdog, _report) = start(Box::new(|step| step == 6));
    drop(watchdog);
}

#[test]
fn an_unanswered_lane_of_each_kind_trips_with_its_report() {
    let (limit, began) = (Duration::from_millis(50), Instant::now());
    let bounds = bounds(limit, volumes(1e-3));
    for (kind, table, lane) in [
        (Kind::File, SlotTable::file(4, 2), 0),
        (Kind::Site, SlotTable::site(2, 4), 0),
        (Kind::Scratch, SlotTable::scratch(4, 3), constants::SCRATCH_ROUND + 1),
    ] {
        let slots = slots(table);
        // Step 6 requests the lane and is never answered; step 5's request
        // of another slot is answered.
        slots.request(5, lane);
        slots.answer(5, lane, false);
        slots.request(6, lane);
        let mut watch = watch(vec![Watched { kind, layer: 3, slots }], Box::new(|| None), Box::new(|_| false), Box::new(idle), began);
        let (after, report) = looks(&mut watch, began, &bounds, limit * 2).unwrap_or_else(|| panic!("{kind:?}: never tripped"));
        assert!(after > limit, "{kind:?}: tripped before its bound");
        let slot = 6 % table.slots();
        assert!(report.contains(&format!("{kind:?} 3 {slot} {lane}: 6, 0x0")), "{kind:?}: {report}");
        assert!(!report.contains(": 5,"), "{kind:?}: an answered lane is reported: {report}");
        assert!(report.contains("no host owner progressed"), "{kind:?}: {report}");
        assert!(report.contains("owner: the owner's state"), "{kind:?}: {report}");
    }
}

/// A lane whose read has pieces in flight is being answered and never trips
/// while their volume ends one within its bound: neither while it is within
/// the bound ending none (the load's only reads on a saturated volume, far
/// past the lane's bound), nor while it holds one piece far past what the
/// line prices for its whole window as it ends the others (I/O the read
/// engine does not see, or pieces started after it), however many pass it.
/// A volume that ends none past its bound (one that stopped returning
/// reads) trips the lane the lane's bound later, the report naming it and
/// its pieces; so does a piece it holds to the end of what any read is given
/// while it ends others; and so does a read outstanding that the engine
/// neither reads nor ends a piece of.
#[test]
fn a_lane_whose_read_is_in_flight_trips_only_past_its_volumes_bound() {
    let (limit, began) = (Duration::from_millis(30), Instant::now());
    let period = limit / 10;
    let unanswered = || {
        let slots = slots(SlotTable::file(4, 2));
        slots.request(6, 0);
        vec![Watched { kind: Kind::File, layer: 3, slots }]
    };
    // A volume reading a piece in 2 ms: with a window of 16 pieces, its bound
    // is 320 ms.
    let bounds = bounds(limit, VolumeBound(Some(ReadLine::new(0.0, 2e-3 / PIECE as f64).unwrap())));
    let window = Duration::from_millis(320);
    let flight = |piece: u64, since| Flight { path: Path::new("/models/experts.gguf").into(), volume: Volume(1), offset: piece * PIECE, bytes: PIECE, window: 16 * PIECE, since };
    let flying = |count: u64, since: Instant| -> Box<dyn Fn() -> Reads + Send> {
        let reading = (0..count).map(|piece| flight(piece, since)).collect::<Vec<_>>();
        Box::new(move || Reads { outstanding: true, engine: EngineWork { ended: 7, reading: reading.clone(), last_ended: HashMap::new() } })
    };
    let reading = |reads| watch(unanswered(), Box::new(|| None), Box::new(|_| false), reads, began);
    // Sixteen pieces in flight ten lane bounds, none ending: within 320 ms.
    assert!(looks(&mut reading(flying(16, began)), began, &bounds, limit * 10).is_none(), "tripped while the volume was within its bound");
    // The last of the window, alone in flight, no sooner than the window's.
    assert!(looks(&mut reading(flying(1, began)), began, &bounds, limit * 10).is_none(), "tripped on the last of a window within the window's bound");
    // One piece held from the start as the volume ends another, started
    // after it, each look: past its window's bound many times, and passed
    // by more pieces than a window holds, as the observed failure was.
    let starved = || -> Box<dyn Fn() -> Reads + Send> {
        let ended = std::sync::atomic::AtomicU64::new(0);
        Box::new(move || {
            let ended = ended.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let since = began + period * u32::try_from(ended).unwrap();
            let reading = vec![flight(0, began), flight(ended + 1, since)];
            Reads { outstanding: true, engine: EngineWork { ended, reading, last_ended: HashMap::from([(Volume(1), since)]) } }
        })
    };
    let held = window * 20;
    assert!(looks(&mut reading(starved()), began, &bounds, held).is_none(), "tripped while one piece was held as others ended");
    // Held until it is stuck, however busy its volume: the lane's bound after.
    let (after, report) = looks(&mut reading(starved()), began, &bounds, UNPRICED_PROGRESS + limit * 2).expect("a piece stuck on a busy volume never tripped");
    let both = UNPRICED_PROGRESS + limit;
    assert!(after + period >= both && after <= both + period, "tripped {after:?} after, not within a look of {both:?}");
    assert!(report.contains("/models/experts.gguf at 0 (131072 bytes) for") && report.contains(", stuck"), "{report}");
    // Sixteen pieces the volume ends none of, past its bound from the start:
    // the lane's bound later.
    let (after, report) = looks(&mut reading(flying(16, began - window)), began, &bounds, limit * 2).expect("a volume that stopped returning reads never tripped");
    assert!(after > limit, "tripped before the lane's bound");
    assert!(report.contains("File 3 2 0: 6, 0x0"), "{report}");
    assert!(report.contains(&format!("of {window:?}, past its bound, in flight [/models/experts.gguf at 0 (131072 bytes) for")), "{report}");
    // A piece alone, its volume passing its bound 320 ms in: tripped the
    // lane's bound after. The look that first sees it past its bound credits
    // the period it ends, so within a look of both.
    let (after, _) = looks(&mut reading(flying(1, began)), began, &bounds, window + limit * 2).expect("a lone piece in flight forever never tripped");
    let both = window + limit;
    assert!(after + period >= both && after <= both + period, "tripped {after:?} after, not within a look of {both:?}");
    // Unserved: outstanding, nothing in flight and nothing ending.
    let (after, _) = looks(&mut reading(flying(0, began)), began, &bounds, limit * 2).expect("a read the engine does not serve never tripped");
    assert!(after > limit, "tripped before its bound");
}

/// A scratch request's own lane, answered only once its round 0 arrives,
/// is never timed; nor is a lane of a step that ended, nor an answered or
/// failed one.
#[test]
fn only_a_lane_the_device_waits_on_trips() {
    let (limit, began) = (Duration::from_millis(20), Instant::now());
    let scratch = slots(SlotTable::scratch(4, 2));
    scratch.request(1, constants::SCRATCH_ASK);
    let file = slots(SlotTable::file(4, 2));
    file.request(2, 0);
    file.answer(2, 0, true);
    file.request(3, 0);
    // Step 3 ended without its answer (a device error): nothing waits.
    let mut watch = watch(
        vec![Watched { kind: Kind::Scratch, layer: 0, slots: scratch }, Watched { kind: Kind::File, layer: 0, slots: file }],
        Box::new(|| None),
        Box::new(|step| step == 3),
        Box::new(idle),
        began,
    );
    assert!(looks(&mut watch, began, &bounds(limit, volumes(1e-3)), limit * 10).is_none(), "tripped on a lane nothing waits on");
}

/// A park the device reached and no answer released trips the watchdog
/// though no lane is requested (the keeper never wrote the round's arrival);
/// one of a step that ended does not.
#[test]
fn a_reached_unreleased_park_trips_whoever_writes_its_request() {
    let (limit, began) = (Duration::from_millis(30), Instant::now());
    let bounds = bounds(limit, volumes(1e-3));
    let parked = || -> ParkedAt { Box::new(|| Some(Parked { value: 853, step: 5, site: "layer 7 Round(2)".into() })) };
    let mut tripping = watch(Vec::new(), parked(), Box::new(|_| false), Box::new(idle), began);
    let (after, report) = looks(&mut tripping, began, &bounds, limit * 2).expect("a reached park never tripped");
    assert!(after > limit, "tripped before its bound");
    assert!(report.contains("parked at layer 7 Round(2) (value 853, step 5)"), "{report}");
    assert!(report.contains("owner: the owner's state"), "{report}");
    let mut ended = watch(Vec::new(), parked(), Box::new(|step| step == 5), Box::new(idle), began);
    assert!(looks(&mut ended, began, &bounds, limit * 10).is_none(), "tripped on an ended step's park");
}
