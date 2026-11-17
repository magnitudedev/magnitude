//! A request's memory at the service owner, over a plain load of the
//! planning fixture's decoder on the CPU (`magnitude_executor::fixtures`),
//! its memory read as the fixture's alone: each request reserves the rows it is
//! about to write as it opens and grows its reservation ahead of its steps.
//! A growth short of memory preempts another request, which replays; a
//! request that alone needs more than the load can give ends with a
//! capacity error of its own.

use magnitude_executor::{
    fixtures::{loaded, opening_rows, reserved_bytes},
    TokenId,
};
use magnitude_family_contracts::PreparedModelInput;
use magnitude_generation::{EndOfGeneration, Generation, InputLayout, MethodChoice, Options, Sampling, Shaping};
use magnitude_scheduler::{
    owner::{Owner, Status},
    prefix_cache::PrefixRetention,
    protocol::{RoomAnswer, RoomRefusal},
    publication::RequestError,
    ServiceLimits,
};
use owner_host::{Host, Stream};
use std::collections::BTreeSet;
use std::task::{Context, Poll, Waker};
use std::time::Duration;

mod owner_host;

/// The served context: long enough that a request's whole context is many
/// times what it writes first.
const CONTEXT: usize = 512;
const PROMPT: usize = 8;

fn limits() -> ServiceLimits {
    ServiceLimits {
        prefill_tokens: 2,
        decode_tokens: 2,
        decode_share: 0.5,
        locality_seconds: 1.0,
        conversations: std::num::NonZeroUsize::MIN,
    }
}

/// An owner over a load at [`CONTEXT`], that load's bytes for a request
/// opening about to write `rows` rows, its vocabulary, and the directory
/// its GGUF was written to, for the test to remove once the owner is
/// dropped.
fn owner(test: &str) -> (Host, impl Fn(u64) -> u64, usize, std::path::PathBuf) {
    let (directory, definition, domain, bindings) = loaded::plain(test, CONTEXT as u64);
    let vocabulary = definition.decoder.vocabulary as usize;
    let sizes = (0..=CONTEXT as u64).map(|rows| reserved_bytes(&domain, opening_rows(&domain, rows))).collect::<Vec<_>>();
    let host = Host::new(|wakes| Owner::new(domain, bindings, limits(), wakes));
    (host, move |rows: u64| sizes[rows as usize], vocabulary, directory)
}

/// A request of [`PROMPT`] tokens with no `max_tokens`, as the chat layer
/// leaves it unset: every token to the end of the context.
fn unbounded(host: &mut Host, vocabulary: usize) -> Stream {
    let prompt = vec![TokenId(1); PROMPT];
    let generation = Generation::new(
        prompt.clone(),
        InputLayout::new(PROMPT, vec![]).unwrap(),
        Options {
            max_tokens: CONTEXT - PROMPT + 1,
            output_capacity: CONTEXT,
            context_limit: CONTEXT,
            vocabulary,
            stop_tokens: BTreeSet::new(),
            suppressed_tokens: BTreeSet::new(),
            sampling: Sampling::Greedy,
            shaping: Shaping { temperature: 0.0, ..Shaping::default() },
            seed: 0,
            forced_quantum: 0,
            method: MethodChoice::Plain,
            end_of_generation: EndOfGeneration::Suppress,
            reasoning_budget: None,
        },
        None,
    )
    .unwrap();
    host.admit(generation, PreparedModelInput::continuation_only(), PrefixRetention::Transient, CONTEXT).unwrap()
}

/// Leave `keep` bytes of the room the model's committed memory leaves
/// (what a claim past it is short of), the rest held by a measurement's
/// room.
fn leave_room(host: &mut Host, keep: u64) {
    const PAST: u64 = u64::MAX / 4;
    let poll = |grant: &mut magnitude_scheduler::protocol::RoomGrant| grant.poll_answer(&mut Context::from_waker(Waker::noop()));
    let room = match poll(&mut host.owner_mut().claim_measurement_room(PAST)) {
        Poll::Ready(RoomAnswer::Refused(RoomRefusal::Short { bytes })) => PAST - bytes,
        answer => panic!("a claim past the budget is refused, not {answer:?}"),
    };
    assert!(room >= keep, "the load leaves {room} bytes, not {keep} more");
    let mut grant = host.owner_mut().claim_measurement_room(room - keep);
    assert_eq!(poll(&mut grant), Poll::Ready(RoomAnswer::Made), "the room beside {keep} bytes is held");
}

/// A request writing past its opening grows its reservation at its steps'
/// boundaries; once that growth is short of memory, it preempts the other
/// request, never itself, and the other replays from its committed history
/// once room frees: both complete their whole contexts.
#[test]
fn a_growth_short_of_memory_preempts_another_request_which_replays() {
    let (mut host, bytes, vocabulary, directory) = owner("growth-preempts");
    let mut first = unbounded(&mut host, vocabulary);
    let mut second = unbounded(&mut host, vocabulary);
    // Room for one whole context beside the load's request slot: each
    // alone fits, both together run short before either context ends.
    leave_room(&mut host, bytes(CONTEXT as u64));
    let mut preempted = false;
    host.step_until(&mut [&mut first, &mut second], Duration::from_secs(300), |host, streams| {
        let status = |stream: &Stream| host.snapshot(stream.id).map(|snapshot| snapshot.status);
        preempted |= matches!(status(&streams[1]), Some(Status::Preempted));
        assert!(!matches!(status(&streams[0]), Some(Status::Preempted)), "the earliest request is never preempted");
        streams.iter().all(|stream| stream.finished())
    });
    assert!(preempted, "the first request's growth preempted the second");
    for stream in [&first, &second] {
        let _completed = stream.usage();
    }
    drop(host);
    std::fs::remove_dir_all(&directory).expect("the fixture's directory is removed");
}

/// A request alone, writing toward its whole context in less room than
/// that takes, with nothing else to free: it ends with a capacity error of
/// its own once its growth is short, and the owner serves on.
#[test]
fn a_request_alone_past_the_loads_memory_ends_with_its_own_capacity_error() {
    let (mut host, _, vocabulary, directory) = owner("alone-past-memory");
    let mut alone = unbounded(&mut host, vocabulary);
    // Only the load's request slot, one request's opening.
    leave_room(&mut host, 0);
    host.step_until(&mut [&mut alone], Duration::from_secs(300), |_, streams| streams[0].finished());
    assert!(
        matches!(alone.failure(), Some(RequestError::ContextExceedsMemory { required, available }) if required > available),
        "it ends for its context's memory, not {:?}",
        alone.terminal
    );
    host.owner_mut().release_measurement_room();
    let mut next = unbounded(&mut host, vocabulary);
    host.drive(&mut [&mut next], Duration::from_secs(300));
    drop(host);
    std::fs::remove_dir_all(&directory).expect("the fixture's directory is removed");
}
