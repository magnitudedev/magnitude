//! Drives a real [`Owner`] as its execution worker does: admission hands back
//! the request's publication stream, `run` performs every transition possible
//! now, and the host feeds the owner's recorded publication wakes (output
//! credit, receiver closure) back before each run. Flight completion is polled
//! by `run` itself, so the host only pauses briefly while a flight is on the
//! device.
//!
//! Each test binary uses a subset of these helpers.
#![allow(dead_code)]

use magnitude_executor::{CompletionWake, RequestId, TokenId};
use magnitude_family_contracts::PreparedModelInput;
use magnitude_generation::{DetailedUsage, Generation};
use magnitude_scheduler::{
    owner::{AdmissionError, Owner, Ran},
    prefix_cache::PrefixRetention,
    protocol::RequestSnapshot,
    publication::{Publication, PublicationReceiver, PublicationWake, RequestError},
    worker::Wakes,
};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};
use std::time::{Duration, Instant};

/// The clock advance of one [`Host::run`], as a worker's steps would take.
pub const TICK_NS: u64 = 10_000_000;

/// Records the owner's publication wakes, as the worker's reserved mailbox
/// path does. The owner polls its flight's completion in `run`.
#[derive(Default)]
struct RecordedWakes(Mutex<Vec<(RequestId, PublicationWake)>>);

impl Wakes for RecordedWakes {
    fn completion(&self) -> CompletionWake {
        CompletionWake::new(|| {})
    }

    fn publication(&self, request: RequestId, wake: PublicationWake) {
        self.0.lock().unwrap().push((request, wake));
    }
}

/// One admitted request's host side: its stream and what arrived on it.
pub struct Stream {
    pub id: RequestId,
    receiver: PublicationReceiver,
    pub output: Vec<TokenId>,
    pub terminal: Option<Publication>,
}

impl Stream {
    /// Take everything published so far. Draining grants the owner output
    /// credit through a recorded wake.
    pub fn drain(&mut self) {
        let mut context = Context::from_waker(Waker::noop());
        while let Poll::Ready(Some(publication)) = self.receiver.poll_next(&mut context) {
            match publication {
                Publication::Output(tokens) => {
                    self.output.extend(tokens.into_iter().map(|token| token.token))
                }
                terminal => {
                    assert!(self.terminal.is_none(), "a second terminal publication");
                    self.terminal = Some(terminal);
                }
            }
        }
    }

    pub fn finished(&self) -> bool {
        self.terminal.is_some()
    }

    /// The usage of a completed request; panics on any other outcome.
    pub fn usage(&self) -> &DetailedUsage {
        match &self.terminal {
            Some(Publication::Completed { usage, .. }) => usage,
            other => panic!("request {:?} did not complete: {other:?}", self.id),
        }
    }

    pub fn failure(&self) -> Option<&RequestError> {
        match &self.terminal {
            Some(Publication::Failed(error)) => Some(error),
            _ => None,
        }
    }
}

pub struct Host {
    /// Moved out only while [`Owner::run`] consumes it.
    owner: Option<Owner>,
    wakes: Arc<RecordedWakes>,
    now: u64,
    due: u64,
}

impl Host {
    pub fn new(build: impl FnOnce(Arc<dyn Wakes>) -> Result<Owner, String>) -> Self {
        let wakes = Arc::new(RecordedWakes::default());
        let owner = build(wakes.clone()).unwrap();
        Self {
            owner: Some(owner),
            wakes,
            now: 0,
            due: 0,
        }
    }

    pub fn owner(&self) -> &Owner {
        self.owner.as_ref().expect("the owner is between runs")
    }

    pub fn now(&self) -> u64 {
        self.now
    }

    pub fn snapshot(&self, request: RequestId) -> Option<RequestSnapshot> {
        self.owner().snapshot(request)
    }

    /// Logical admission at the host's current time.
    pub fn admit(
        &mut self,
        generation: Generation,
        input: PreparedModelInput,
        retention: PrefixRetention,
        output_capacity: usize,
    ) -> Result<Stream, AdmissionError> {
        let now = self.now;
        let (id, receiver) = self
            .owner
            .as_mut()
            .expect("the owner is between runs")
            .admit(generation, input, retention, output_capacity, now)?;
        Ok(Stream {
            id,
            receiver,
            output: Vec::new(),
            terminal: None,
        })
    }

    /// Feed the recorded publication wakes, then run every transition
    /// possible at `now`.
    pub fn run_at(&mut self, now: u64) -> Ran {
        assert!(now >= self.now, "the host clock moved backwards");
        self.now = now;
        let mut owner = self.owner.take().expect("the owner is between runs");
        let wakes = std::mem::take(&mut *self.wakes.0.lock().unwrap());
        for (request, wake) in wakes {
            owner.publication_wake(request, wake);
        }
        let (owner, ran) = owner
            .run(now)
            .unwrap_or_else(|error| panic!("execution stopped: {error}"));
        self.owner = Some(owner);
        self.due = ran.due;
        ran
    }

    /// Run one tick later.
    pub fn run(&mut self) -> Ran {
        self.run_at(self.now + TICK_NS)
    }

    /// Run no earlier than the owner's next memory observation, as the
    /// worker does when it wakes for one.
    pub fn observe(&mut self) -> Ran {
        self.run_at((self.now + TICK_NS).max(self.due))
    }

    /// Drain `streams`, run once, drain again, and pause briefly while a
    /// flight is on the device.
    pub fn step(&mut self, streams: &mut [&mut Stream]) -> Ran {
        streams.iter_mut().for_each(|stream| stream.drain());
        let ran = self.run();
        streams.iter_mut().for_each(|stream| stream.drain());
        if ran.in_flight {
            std::thread::sleep(Duration::from_micros(100));
        }
        ran
    }

    /// Step until every stream received its terminal publication, and
    /// require every one to have completed.
    pub fn drive(&mut self, streams: &mut [&mut Stream], timeout: Duration) {
        let deadline = Instant::now() + timeout;
        while !streams.iter().all(|stream| stream.finished()) {
            assert!(
                Instant::now() < deadline,
                "requests did not finish: {:?}",
                self.progress(streams)
            );
            self.step(streams);
        }
        for stream in streams.iter() {
            assert!(
                matches!(stream.terminal, Some(Publication::Completed { .. })),
                "request {:?} ended with {:?}",
                stream.id,
                stream.terminal
            );
        }
    }

    /// Step until `done` holds.
    pub fn step_until(
        &mut self,
        streams: &mut [&mut Stream],
        timeout: Duration,
        mut done: impl FnMut(&Self, &[&mut Stream]) -> bool,
    ) {
        let deadline = Instant::now() + timeout;
        while !done(self, streams) {
            assert!(
                Instant::now() < deadline,
                "condition not reached: {:?}",
                self.progress(streams)
            );
            self.step(streams);
        }
    }

    /// The host abandons `stream`: dropping its receiver cancels the request.
    /// Run until the owner has retired it.
    pub fn release(&mut self, stream: Stream, timeout: Duration) {
        let id = stream.id;
        drop(stream);
        let deadline = Instant::now() + timeout;
        loop {
            let ran = self.run();
            if self.snapshot(id).is_none() {
                return;
            }
            assert!(Instant::now() < deadline, "request {id:?} was not released");
            if ran.in_flight {
                std::thread::sleep(Duration::from_micros(100));
            }
        }
    }

    /// Run until no flight remains, then a few idle runs, so an owner with
    /// nothing live shrinks its idle state backing.
    pub fn settle(&mut self) {
        let deadline = Instant::now() + Duration::from_secs(20);
        while self.run().in_flight {
            assert!(Instant::now() < deadline, "the owner did not settle");
            std::thread::sleep(Duration::from_micros(100));
        }
        for _ in 0..4 {
            assert!(!self.run().in_flight, "an idle owner submitted work");
        }
    }

    fn progress(&self, streams: &[&mut Stream]) -> Vec<String> {
        streams
            .iter()
            .map(|stream| {
                format!(
                    "{:?}: output={} terminal={:?} snapshot={:?}",
                    stream.id,
                    stream.output.len(),
                    stream.terminal,
                    self.snapshot(stream.id)
                )
            })
            .collect()
    }
}
