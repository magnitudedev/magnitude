//! Graph traces record once per topology key (`seismic::NativeGraphTraces`):
//! models of one key that miss at once, as concurrent assessments on a cold
//! catalog do, wait for one recording and are answered by it, while another
//! key records beside them. A recording that panics leaves its key whole:
//! the next model of the key records it again.

use seismic::{
    BackendName, ClassFact, Dim, Element, NativeGraphClassSlice, NativeGraphMetadata, NativeGraphMetadataError,
    NativeGraphOutline, NativeGraphResourceTemplate, NativeGraphTraces, Reals,
};
use seismic_native_tests::scale_rows;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Barrier;

/// A topology key: its width scale, which decides the graph's columns.
type Key = u64;

static TRACES: NativeGraphTraces<Key> = NativeGraphTraces::new();

/// The traces of the panicking recorder's key, apart from [`TRACES`]' count.
static PANICKED_TRACES: NativeGraphTraces<Key> = NativeGraphTraces::new();

/// Whether [`record_after_a_panic`] has panicked yet.
static PANICKED: AtomicBool = AtomicBool::new(false);

/// [`record`], panicking at its first recording as an invariant would.
fn record_after_a_panic(
    key: &Key,
    facts: &[Dim],
    reals: &Reals,
) -> Result<(NativeGraphResourceTemplate, ()), NativeGraphMetadataError> {
    assert!(PANICKED.swap(true, Ordering::SeqCst), "the first recording panics");
    record(key, facts, reals)
}

/// `rows` rows (fact 0) of `columns · key` columns (fact 1), scaled and
/// stored.
fn record(key: &Key, facts: &[Dim], _reals: &Reals) -> Result<(NativeGraphResourceTemplate, ()), NativeGraphMetadataError> {
    let (rows, columns) = (&facts[0], &facts[1] * *key);
    let mut template = NativeGraphMetadata::new_template(BackendName::Cpu);
    let input = template.port_with_class_extent(Element::f32(), &[rows, &columns], 0, "M")?;
    let landing = template.port_with_class_extent(Element::f32(), &[rows, &columns], 0, "M")?;
    let scaled = template
        .enqueue::<scale_rows::Entry>(
            &[],
            &[("M", rows.clone()), ("N", columns.clone())],
            scale_rows::WorkflowArgs { x: input.tensor().into(), factor: seismic::Argument::from(2.0) },
        )?
        .value;
    template.store(&scaled, &landing)?;
    Ok((template.seal_template()?, ()))
}

fn outline(key: Key) -> NativeGraphOutline {
    outline_of(&TRACES, key, record)
}

fn outline_of(
    traces: &NativeGraphTraces<Key>,
    key: Key,
    recorder: seismic::GraphRecorder<Key, (), NativeGraphMetadataError>,
) -> NativeGraphOutline {
    let slices = vec![NativeGraphClassSlice::new().dimension("M", [1, 2, 4])];
    let classes = [ClassFact { fact: 0, scope: None, dimension: "M" }];
    let [(outline, ())]: [_; 1] = traces
        .outline(&key, &[4, 8], &Reals::new(), &classes, &[slices], recorder)
        .unwrap()
        .try_into()
        .unwrap();
    outline
}

#[test]
fn concurrent_misses_of_one_key_record_it_once() {
    const THREADS: usize = 8;
    let barrier = Barrier::new(2 * THREADS);
    let outlines = std::thread::scope(|scope| {
        let threads = (0..2 * THREADS)
            .map(|at| {
                let barrier = &barrier;
                let key = 1 + (at % 2) as Key;
                scope.spawn(move || {
                    barrier.wait();
                    (key, outline(key))
                })
            })
            .collect::<Vec<_>>();
        threads.into_iter().map(|thread| thread.join().unwrap()).collect::<Vec<_>>()
    });
    assert_eq!(TRACES.recorded(), 2, "each key records once");
    for key in [1, 2] {
        let mut of_key = outlines.iter().filter(|(of, _)| *of == key).map(|(_, outline)| *outline);
        let first = of_key.next().unwrap();
        assert!(of_key.all(|outline| outline == first), "every model of key {key} gets the same outline");
    }
}

#[test]
fn a_panicked_recording_leaves_its_key_to_record_again() {
    let panicked = std::panic::catch_unwind(|| outline_of(&PANICKED_TRACES, 3, record_after_a_panic));
    assert!(panicked.is_err(), "the first recording panics");
    assert_eq!(PANICKED_TRACES.recorded(), 0, "a panicked recording holds no trace");
    let first = outline_of(&PANICKED_TRACES, 3, record_after_a_panic);
    assert_eq!(outline_of(&PANICKED_TRACES, 3, record_after_a_panic), first);
    assert_eq!(PANICKED_TRACES.recorded(), 1, "the key records again, once");
}
