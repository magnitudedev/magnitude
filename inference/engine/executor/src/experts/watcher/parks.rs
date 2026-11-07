//! The host's side of the streamed entries' parks where a park is a
//! submission boundary (Vulkan: `seismic::BackendName::parks_in_kernels` is
//! false), the one way the device waits for the host there: the Vulkan spec makes
//! no host write visible to running work, nor the device's writes to the
//! host before its submission ends.
//!
//! A graph places a park node (`seismic::timeline::Park`) before each
//! call that waits on a slot of the exchange (`experts::exchange`): a
//! decode layer's tier wait (`routed_tier_wait`, its file slot), a scratch
//! round's wait (`routed_scratch_wait`, its round's lane) and a sited
//! combine's sum (`routed_down_sited`, `routed_output_sited`, its site
//! slot); and a publication (`seismic::timeline::Publish`) right after a site
//! request (`routed_site_request`), so the CPU pool reads the request while
//! the device's hits run, before the sum's park. Every node binds the
//! load's one park timeline (`NativeGraphBindings::set_park`). Where
//! kernels wait in place (Metal, CUDA) a node binds nothing and takes no
//! value, nothing is registered, and the kernels spin on their slots.
//!
//! Where parks are boundaries, a run takes the timeline's next values when
//! it is queued, in the order of the device's one queue, so the device
//! reaches them in order, and each park and publication is registered here
//! with its value ([`Parks::register`], [`Parks::publish`]). At a park the
//! device stops before the call, its every write before visible to the
//! host, until the host raises the timeline past it. A registered park is
//! answered by the exchange's one rule, the kernels' own
//! (`ExchangeTable::answered`): once its slot is not requested for its step (the
//! call waits for nothing) or holds the step's answer or failure. A wait
//! whose request is its arrival (a scratch round's) stores it in its kernel
//! after the park, too late for the host: the keeper stores it as the device
//! reaches the park, its realization of the wait's first act. A park is
//! answered once the host can read its slot's request: once the device is
//! parked at it, or, for a sited combine's, once the device reached its
//! site request's publication ([`Parks::published`]), so the CPU's answer
//! (or a request for nothing) opens the park before the device reaches it
//! and the device runs on through it. The device's queue holds the one
//! submission behind the first shut gate (`seismic`'s Vulkan hold), which
//! starts as the gate is raised. Each answer is monotone by construction:
//! parks are answered strictly in value order, each only once its request
//! is the host's to read, so no answer opens a later park.
//!
//! Every answerer of a slot (a landing's file answer, the watcher's round
//! answers and failures, the CPU pool's site answers, which the pool
//! announces through its gate, `site::SiteGate`) settles the parks at once,
//! so an answer opens its park with no thread between. The keeper, a thread
//! of its own, settles what no answer does (a park whose slot was answered
//! before the device reached it, or one that waits for nothing). While the
//! head is unreached it looks again within the request service's rest
//! (`requests::IDLE`, rested precisely: `seismic_io::waited_on`), as the
//! service looks at the device's words where kernels wait in place: the one poll of serving,
//! which the hardware forces here too. A blocking wait on the timeline
//! (`vkWaitSemaphores`) returns 25 to 97 µs after the device reached the
//! value on GB10 (`seismic_vulkan`'s `the_keepers_wait_is_measured`), and a
//! tier wait's or a round's request is the host's only from then.
//! Otherwise the keeper waits on the parks' changes.
//!
//! While the device runs between parks its writes are not visible to the
//! host, so the request service reads the device's words only while the
//! device is parked ([`Parks::parked`]), which it learns from the load's
//! events: the device newly parked at a head no answer opens yet, and a
//! round's arrival, are each one event, so where parks are boundaries the
//! service never polls. A park answered as the device reaches it waits on
//! no host job and is no event, so a decode whose parks wait for nothing
//! wakes no other thread. The CPU
//! pool takes a site request only once the device reached its publication
//! ([`Parks::published`]), which the graph places right after the request
//! and before the combine's park: the one point the request is the host's
//! to read, and one the device reaches without the host.

use super::Step;
use super::scratch::round_lane;
use crate::experts::ExchangeTable;
use crate::experts::events::Events;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

/// Which park of a streamed layer: which slot of the exchange its call
/// waits on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ParkKind {
    /// The decode tier wait (`routed_tier_wait`): the file slot.
    TierWait,
    /// A scratch round's wait (`routed_scratch_wait`), by its round: its
    /// lane of the scratch slot, which its arrival requests.
    Round(u64),
    /// A sited combine's wait for the CPU's answer (`routed_down_sited`,
    /// `routed_output_sited`): the site slot.
    Site,
}

impl ParkKind {
    /// The kind's name in a trace of the work that waited at it
    /// (`seismic::ParkAnswer::park`).
    fn label(self) -> &'static str {
        match self {
            Self::TierWait => "tier wait",
            Self::Round(_) => "scratch round",
            Self::Site => "sited combine",
        }
    }
}

/// A park of a graph: its streamed layer (the bank's index) and kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ParkSite {
    pub layer: usize,
    pub kind: ParkKind,
}

/// A streamed layer's tables of the exchange its parks wait on: its file
/// and scratch tables, and its site's, where it has a site.
pub(in crate::experts) struct ParkLayer {
    pub file: ExchangeTable,
    pub scratch: ExchangeTable,
    pub site: Option<ExchangeTable>,
}

impl ParkLayer {
    /// The lane `kind` waits on, and whether its request is its arrival;
    /// none for a site park of a layer without a site, which waits for
    /// nothing.
    fn lane(&self, kind: ParkKind) -> Option<(&ExchangeTable, u64, bool)> {
        match kind {
            ParkKind::TierWait => Some((&self.file, 0, false)),
            ParkKind::Round(round) => Some((&self.scratch, round_lane(round), true)),
            ParkKind::Site => self.site.as_ref().map(|site| (site, 0, false)),
        }
    }

    /// Whether a wait of `step`'s at this layer ended on its still pulse,
    /// before its answer ([`ExchangeTable::expired`]): its tier wait's,
    /// any round's, or its sited combine's.
    fn expired(&self, step: u64) -> bool {
        let lanes = |table: &ExchangeTable| (0..table.table().lanes()).any(|lane| table.expired(step, lane));
        lanes(&self.file) || lanes(&self.scratch) || self.site.as_ref().is_some_and(lanes)
    }
}

/// A registered park: its value, its step and its site.
#[derive(Clone, Copy)]
struct Pending {
    value: u64,
    sequence: Step,
    site: ParkSite,
}

/// Where settling stopped: what the parks wait on next.
enum Stop {
    /// Nothing is registered.
    Empty,
    /// The head waits for the device to reach a value: the head's own, or
    /// its site request's publication's while that is unreached.
    Unreached,
    /// The device is parked at the head, which took this value, and its
    /// slot is unanswered.
    Unanswered(u64),
}

/// What one settling did: the parks it took to answer, as the value of the
/// last (the caller answers it, which opens every one before it) and the
/// site of the first (the one the device waits at, if any does), whether it
/// is an event for the load (`announces`), and where it stopped.
struct Settled {
    answer: Option<(u64, ParkSite)>,
    announces: bool,
    stop: Stop,
}

struct State {
    pending: VecDeque<Pending>,
    /// The value of the last head announced as parked unanswered: each such
    /// park is one event, however many settlings see it.
    announced: u64,
    /// Whether the load stopped: the keeper ends.
    stopped: bool,
}

/// The load's parks: its timeline, the registered parks not yet answered,
/// in the device's order, and each site slot's last publication.
pub(crate) struct Parks {
    /// Whether a park binds operations on the load's device: none where
    /// kernels wait in place, so nothing registers and no keeper runs.
    pub(super) kept: bool,
    timeline: seismic::Timeline<seismic::timeline::Parks>,
    layers: Vec<ParkLayer>,
    state: Mutex<State>,
    /// What the keeper waits on while the head is parked or none is
    /// registered: a registration, an answer, a retirement or the load
    /// stopping, each of which may change what it waits for.
    changed: Condvar,
    /// Per streamed layer and site slot, the step and value of the last
    /// publication registered of its site request.
    publications: Vec<Vec<[AtomicU64; 2]>>,
    /// The load's events: the device's words in place for the request
    /// service ([`Self::settled`]) is one.
    events: Arc<Events>,
    /// Why the parks can go no further: the device could not say where its
    /// timeline is, or refused an answer (it is lost). Serving ends with it
    /// ([`Self::lost`]), and the keeper ends. Set only under the parks'
    /// lock ([`Self::lose`]), read without it.
    loss: std::sync::OnceLock<String>,
}

impl Parks {
    /// Whether a wait of `step`'s, at any streamed layer, ended on its still
    /// pulse before its answer: the one way a live host's step can run on
    /// without an answer (`exchange::Pulse`), which the waiting kernel
    /// records. Read once the device is past the step.
    pub(super) fn expired(&self, step: u64) -> bool {
        self.layers.iter().any(|layer| layer.expired(step))
    }

    pub(in crate::experts) fn new(timeline: seismic::Timeline<seismic::timeline::Parks>, layers: Vec<ParkLayer>, events: Arc<Events>) -> Self {
        Self {
            events,
            kept: !timeline.parks_in_kernels(),
            timeline,
            publications: layers
                .iter()
                .map(|layer| {
                    let slots = layer.site.as_ref().map_or(0, |site| site.table().slots());
                    (0..slots).map(|_| [AtomicU64::new(0), AtomicU64::new(0)]).collect()
                })
                .collect(),
            layers,
            state: Mutex::new(State { pending: VecDeque::new(), announced: 0, stopped: false }),
            changed: Condvar::new(),
            loss: std::sync::OnceLock::new(),
        }
    }

    /// Why the device was lost under the parks, where it was: serving
    /// ends with it (`Shared::fault`).
    pub(super) fn lost(&self) -> Option<String> {
        self.loss.get().cloned()
    }

    /// Record the device's loss, the first only, under the parks' lock
    /// (`held`), and wake the keeper and the request service, which end
    /// with it. The keeper decides to wait under the same lock and looks at
    /// the loss last before it waits ([`Self::look`]), so no loss is missed
    /// between the two.
    fn lose(&self, _held: &State, reason: String) {
        if self.loss.set(reason).is_ok() {
            self.changed.notify_all();
            self.events.notify();
        }
    }

    /// What the device said of its timeline (`said`): never yes where it
    /// cannot say, its loss recorded under `held` ([`Self::lose`]), so
    /// nothing it wrote is read.
    fn device_says(&self, held: &State, said: Result<bool, seismic::ExecutionError>) -> bool {
        said.unwrap_or_else(|error| {
            self.lose(held, format!("the device lost the parks' timeline: {error}"));
            false
        })
    }

    /// Whether the device reached `value`, as [`Self::device_says`].
    fn device_reached(&self, held: &State, value: u64) -> bool {
        self.device_says(held, self.timeline.reached(value))
    }

    fn state(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().expect("parks")
    }

    /// The timeline every park of the load binds.
    pub(crate) fn timeline(&self) -> &seismic::Timeline<seismic::timeline::Parks> {
        &self.timeline
    }

    /// Register the park that took `value` (`NativeGraphOutputs::park`), of
    /// step `sequence`, once its run is queued.
    pub(crate) fn register(&self, value: u64, sequence: Step, site: ParkSite) {
        self.state().pending.push_back(Pending { value, sequence, site });
        self.changed.notify_all();
    }

    /// The publication's place of `layer`'s site request of `step`.
    fn publication(&self, layer: usize, step: Step) -> &[AtomicU64; 2] {
        let slots = &self.publications[layer];
        &slots[(step % slots.len() as u64) as usize]
    }

    /// Register the publication that took `value` of `layer`'s site request
    /// of step `sequence`, once its run is queued: its value first, then
    /// its step, which [`Self::published`] reads first.
    pub(crate) fn publish(&self, value: u64, layer: usize, step: Step) {
        let [registered, at] = self.publication(layer, step);
        at.store(value, Ordering::Release);
        registered.store(step, Ordering::Release);
    }

    /// Whether `layer`'s site request of `step` is the host's to read: where
    /// parks are boundaries, once the device reached its publication, every
    /// write before it visible; where kernels wait in place, always (the
    /// slot's request, stored last with system-scope release, orders the
    /// rest).
    pub(in crate::experts) fn published(&self, layer: usize, step: u64) -> bool {
        if !self.kept {
            return true;
        }
        let [registered, at] = self.publication(layer, step);
        // The lock is taken only to record a loss: a read is lock-free.
        registered.load(Ordering::Acquire) == step
            && match self.timeline.reached(at.load(Ordering::Acquire)) {
                Ok(reached) => reached,
                failed @ Err(_) => self.device_says(&self.state(), failed),
            }
    }

    /// A step's issued work has retired: its registrations not yet answered
    /// (a failed step's tail, never reached) leave the queue.
    pub(super) fn retire(&self, sequence: Step) {
        self.state().pending.retain(|park| park.sequence != sequence);
        self.changed.notify_all();
    }

    /// Whether the device's words are in place for the host to read: no
    /// registered park is pending, or the device is parked at the first.
    /// "None pending" is not "the device is past every park": a sited
    /// combine's park is answered and leaves the queue once the device
    /// reached its publication, possibly before the device reaches the
    /// park, so with none pending the device may still run toward the last
    /// park answered. Only the words visible at the last boundary it
    /// reached are in place then, as when it runs past its last park.
    pub(super) fn settled(&self) -> bool {
        let state = self.state();
        state.pending.front().is_none_or(|head| self.device_reached(&state, head.value))
    }

    /// Whether the device's requests are the host's to read: it is parked
    /// at the first registered park. Unlike [`Self::settled`], none pending
    /// is not enough: a request is written in a run the device may be
    /// running, whose park after it is not registered yet (`run_fill`
    /// submits a front before it queues the fill), and its word can reach
    /// the host before the words it names. Every request has a park after
    /// it, so the device parks there, an event, and the host reads it then.
    /// Where kernels wait in place nothing is registered, and every word is
    /// the host's to read as it lands.
    pub(super) fn parked(&self) -> bool {
        let state = self.state();
        match state.pending.front() {
            Some(head) => self.device_reached(&state, head.value),
            None => !self.kept,
        }
    }

    /// Answer every registered park, in order, whose request is the host's
    /// to read and whose slot the exchange's rule answers (the module
    /// documentation). True when it answered any.
    pub(in crate::experts) fn settle(&self) -> bool {
        let settled = self.settle_held(&mut self.state());
        if settled.announces {
            self.events.notify();
        }
        // An answerer settles as its answer lands.
        self.answer(settled.answer, seismic::AnsweredBy::Ready)
    }

    /// Raise the timeline past `answer`, the last park a settling took,
    /// outside the parks' lock (the timeline orders its signals), `by` an
    /// answerer or the keeper; true when there is one and the device took
    /// it (a refusal is its loss, [`Self::lose`]). Its answer opens every
    /// park before it.
    fn answer(&self, answer: Option<(u64, ParkSite)>, by: seismic::AnsweredBy) -> bool {
        let Some((value, first)) = answer else {
            return false;
        };
        let answered = self.timeline.answer(value, seismic::ParkAnswer { by, park: first.kind.label(), layer: first.layer });
        if let Err(error) = answered {
            self.lose(&self.state(), format!("the device refused a park's answer: {error}"));
            return false;
        }
        self.changed.notify_all();
        true
    }

    /// Whether the device has yet to reach the value before which the host
    /// may not read `head`'s request: the head's own, but a sited
    /// combine's site request's publication's once registered, as the
    /// request is the host's to read there (`Self::published`); never for a
    /// park that waits for nothing (a layer without a site), whose answer
    /// reads nothing.
    fn unread(&self, held: &State, head: &Pending) -> bool {
        let layer = &self.layers[head.site.layer];
        let value = match head.site.kind {
            ParkKind::Site if layer.site.is_none() => return false,
            ParkKind::Site => {
                let [registered, at] = self.publication(head.site.layer, head.sequence);
                match registered.load(Ordering::Acquire) == head.sequence {
                    true => at.load(Ordering::Acquire),
                    false => head.value,
                }
            }
            ParkKind::TierWait | ParkKind::Round(_) => head.value,
        };
        !self.device_reached(held, value)
    }

    /// [`Self::settle`] under the parks' lock: the parks it takes to
    /// answer, removed (the caller answers, outside the lock), whether it
    /// is an event, and where it stopped, which is what the parks wait on
    /// next. What stays under the lock is the queue and what the settling
    /// decides from it: the head's pops, its arrival's request, and the
    /// head last announced. Its events are the two things a host job waits
    /// for from the parks: a round's arrival it requested (the request
    /// service learns the device arrived only from it), and the device
    /// newly parked at an unanswered head, whose words are now the host's to
    /// read (`Self::settled`). A park it answers at once waits on no host
    /// job, so it is no event, nor is a look that changed nothing.
    fn settle_held(&self, state: &mut State) -> Settled {
        let (mut answer, mut arrived) = (None, false);
        let stop = loop {
            let Some(&head) = state.pending.front() else {
                break Stop::Empty;
            };
            if self.unread(state, &head) {
                break Stop::Unreached;
            }
            if let Some((slots, lane, arrives)) = self.layers[head.site.layer].lane(head.site.kind) {
                if arrives && !slots.requested(head.sequence, lane) {
                    slots.request(head.sequence, lane);
                    arrived = true;
                }
                if !slots.answered(head.sequence, lane) {
                    break match self.device_reached(state, head.value) {
                        true => Stop::Unanswered(head.value),
                        false => Stop::Unreached,
                    };
                }
            }
            state.pending.pop_front();
            answer = Some((head.value, answer.map_or(head.site, |(_, first)| first)));
        };
        let newly_parked = match stop {
            Stop::Unanswered(value) => std::mem::replace(&mut state.announced, value) != value,
            Stop::Empty | Stop::Unreached => false,
        };
        Settled { answer, announces: arrived || newly_parked, stop }
    }

    /// One look of the keeper: settle, then answer what it took, or else
    /// wait for what may settle the parks further, decided under the same
    /// lock as the settling, so no arrival between the two is missed: the
    /// device reaching the value the head waits for while it is unreached
    /// (a rest of `requests::IDLE`, after which it looks again); else a
    /// registration, an answer, a retirement or the load stopping (an
    /// answerer settles what its answer opens). False once the load
    /// stopped or the device was lost ([`Self::lose`]): a loss is recorded
    /// under the same lock, here last before the wait, so the keeper either
    /// sees it or is waiting when it is notified.
    fn look(&self) -> bool {
        let mut state = self.state();
        if state.stopped || self.loss.get().is_some() {
            return false;
        }
        let Settled { answer, announces, stop } = self.settle_held(&mut state);
        if announces {
            self.events.notify();
        }
        if answer.is_some() {
            drop(state);
            // The keeper answers what was ready before: it saw the device
            // arrive, or what a park reads become the host's.
            self.answer(answer, seismic::AnsweredBy::Arrival);
            return true;
        }
        match stop {
            Stop::Unreached => {
                drop(state);
                std::thread::sleep(super::requests::IDLE);
            }
            // Its own settling may have lost the device.
            Stop::Unanswered(_) | Stop::Empty if self.loss.get().is_some() => return false,
            Stop::Unanswered(_) | Stop::Empty => drop(self.changed.wait(state).expect("parks")),
        }
        true
    }

    /// The park the device has reached and no answer has released, for the
    /// exchange's watchdog (`exchange::Parked`): the head once the device
    /// is parked there (an answer removes it), whoever writes its lane's
    /// request. None where kernels wait in place, while nothing is reached,
    /// or while another thread holds the parks (the keeper, briefly).
    pub(in crate::experts) fn reached(&self) -> Option<crate::experts::exchange::Parked> {
        let state = self.state.try_lock().ok()?;
        let head = state.pending.front().filter(|head| self.device_reached(&state, head.value))?;
        Some(crate::experts::exchange::Parked {
            value: head.value,
            step: head.sequence,
            site: format!("layer {} {:?}", head.site.layer, head.site.kind),
        })
    }

    /// The parks' state, for the exchange's watchdog: the timeline's value,
    /// and each registered park pending, in order, the head's whether the
    /// device is parked there and whether its slot answers it. None while
    /// another thread holds the parks (a stuck holder).
    pub(in crate::experts) fn describe(&self) -> Option<String> {
        if !self.kept {
            return Some("kernels wait in place: none kept".into());
        }
        let state = self.state.try_lock().ok()?;
        let head = state.pending.front().map(|head| {
            let answered = self.layers[head.site.layer].lane(head.site.kind).is_none_or(|(slots, lane, _)| slots.answered(head.sequence, lane));
            format!("parked {:?}, its slot answered {answered}", self.timeline.reached(head.value))
        });
        let pending = state
            .pending
            .iter()
            .map(|park| format!("{} step {} layer {} {:?}", park.value, park.sequence, park.site.layer, park.site.kind))
            .collect::<Vec<_>>();
        Some(format!(
            "timeline at {:?}; head {}; pending (value step layer kind) [{}]; stopped {}",
            self.timeline.current(),
            head.unwrap_or_else(|| "none".into()),
            pending.join(", "),
            state.stopped
        ))
    }

    /// The load stopped: the keeper ends.
    pub(super) fn stop(&self) {
        self.state().stopped = true;
        self.changed.notify_all();
    }

    /// The keeper: looks at the parks ([`Self::look`]) until the load
    /// stops. A look is an event only where it found one
    /// ([`Self::settle_held`]), so a keeper with nothing to act on wakes no
    /// host job.
    pub(super) fn keep(parks: Arc<Self>) -> Result<JoinHandle<()>, String> {
        std::thread::Builder::new()
            .name("expert-parks".into())
            .spawn(move || {
                seismic_io::waited_on();
                while parks.look() {}
            })
            .map_err(|error| format!("expert parks' keeper: {error}"))
    }
}

impl crate::experts::site::SiteGate for Parks {
    fn published(&self, layer: usize, step: u64) -> bool {
        Parks::published(self, layer, step)
    }

    /// The pool's answer settles the parks it may open: a sited combine's
    /// opens at its answer.
    fn answered(&self) {
        self.settle();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use magnitude_kernels::{SlotTable, constants};
    use seismic::{BackendName, Devices, Element, Tensor};

    /// Far past a rest and a look on a loaded test host: the bound a
    /// keeper's notice is held to here.
    const NOTICE: std::time::Duration = std::time::Duration::from_millis(50);

    /// Whether the host answered the park that took `value` within
    /// `timeout`: the keeper answers on its own thread, and an answer is the
    /// host's, never anything the device raises.
    fn answered_within(timeline: &seismic::Timeline<seismic::timeline::Parks>, value: u64, timeout: std::time::Duration) -> bool {
        let deadline = std::time::Instant::now() + timeout;
        while !timeline.answered(value).unwrap() {
            if std::time::Instant::now() >= deadline {
                return false;
            }
            std::thread::yield_now();
        }
        true
    }

    /// A device whose parks are submission boundaries (Vulkan). The tests
    /// taking one are a Vulkan host's checks, run on one by name with
    /// `--ignored`.
    fn boundary_device() -> seismic::Device {
        let catalog = Devices::discover(&[]).expect("devices are discovered");
        crate::test_support::open_backend(&catalog, BackendName::Vulkan).expect("a Vulkan device opens")
    }

    fn slots(device: &seismic::Device, table: SlotTable) -> ExchangeTable {
        ExchangeTable::new(Tensor::shared(device, Element::i32(), &[table.words()]).unwrap(), table, &crate::experts::exchange::Pulse::new()).unwrap()
    }

    fn parks(device: &seismic::Device) -> Parks {
        let layer = ParkLayer {
            file: slots(device, SlotTable::file(4, 4)),
            scratch: slots(device, SlotTable::scratch(4, 2)),
            site: Some(slots(device, SlotTable::site(2, 4))),
        };
        Parks::new(seismic::Timeline::new(device).unwrap(), vec![layer], Arc::default())
    }

    /// Each registered park is answered once the device is parked at it and
    /// its slot's rule holds, in order: a tier wait without a request at
    /// once, one with a request once its slot is answered, a round once its
    /// lane is answered, its arrival requested as the device reaches it.
    #[test]
    #[ignore]
    fn a_parked_park_is_answered_by_the_exchanges_rule() {
        let device = boundary_device();
        let parks = parks(&device);
        assert!(parks.kept);
        assert!(!parks.parked(), "with no park registered, a request may be in flight");
        let wait = ParkSite { layer: 0, kind: ParkKind::TierWait };
        let round = ParkSite { layer: 0, kind: ParkKind::Round(1) };
        // The values runs take when queued, each park two.
        let values = [1, 3, 5];
        parks.register(values[0], 7, wait);
        parks.register(values[1], 8, wait);
        parks.register(values[2], 8, round);
        // Not parked: the device's words are not in place.
        assert!(!parks.settled());
        assert!(!parks.settle());
        // The device reaches the first: step 7 requested nothing.
        parks.timeline.reach(values[0]).unwrap();
        assert!(parks.settled());
        assert!(parks.parked());
        assert!(parks.settle());
        assert!(parks.timeline.answered(values[0]).unwrap());
        // Step 8 requested its file slot: answered once the slot is.
        parks.timeline.reach(values[1]).unwrap();
        let file = &parks.layers[0].file;
        file.request(8, 0);
        assert!(!parks.settle());
        file.answer(4, 0, false);
        assert!(!parks.settle(), "the slot's earlier step answers nothing");
        file.answer(8, 0, false);
        assert!(parks.settle());
        // Its round 1: requested as reached, answered once its lane is.
        parks.timeline.reach(values[2]).unwrap();
        assert!(!parks.settle());
        let scratch = &parks.layers[0].scratch;
        assert!(scratch.requested(8, round_lane(1)), "the keeper requests a round as the device arrives");
        scratch.answer(8, round_lane(1), false);
        assert!(parks.settle());
        assert!(parks.timeline.answered(values[2]).unwrap());
        assert!(parks.settled());
    }

    /// The site path through parks: the pool takes a step's request only
    /// once the device reached its publication, right after the request,
    /// and never a later step's in the slot; the combine's park after it is
    /// answered once the pool answered the slot, a step that requested
    /// nothing at once. Run on an RTX 3090 (Vulkan) by
    /// `the_site_path_through_parks_publishes_then_answers`.
    #[test]
    #[ignore]
    fn the_site_path_through_parks_publishes_then_answers() {
        let device = boundary_device();
        let parks = parks(&device);
        let site = parks.layers[0].site.clone().unwrap();
        let combine = ParkSite { layer: 0, kind: ParkKind::Site };
        // Step 7 requests the CPU: its publication takes value 1, its
        // combine's park 3, as the run is queued.
        site.store(7, 0, constants::SITE_COUNT, 1);
        site.request(7, 0);
        assert!(!parks.published(0, 7), "nothing registered");
        parks.publish(1, 0, 7);
        parks.register(3, 7, combine);
        assert!(!parks.published(0, 7), "not reached");
        parks.timeline.reach(1).unwrap();
        assert!(parks.published(0, 7), "the device reached the publication");
        assert!(!parks.published(0, 7 + site.table().slots()), "the slot's next step");
        // The device parks at the combine: answered once the pool answers.
        parks.timeline.reach(3).unwrap();
        assert!(!parks.settle());
        site.answer(7, 0, false);
        assert!(parks.settle());
        assert!(parks.timeline.answered(3).unwrap());
        // Step 8 requests nothing of the CPU: its combine's park opens as
        // reached.
        parks.register(5, 8, combine);
        parks.timeline.reach(5).unwrap();
        assert!(parks.settle());
        assert!(parks.timeline.answered(5).unwrap());
    }

    /// A sited combine's park is answered before the device reaches it,
    /// once the device reached its request's publication: at once where
    /// the step requested nothing of the CPU, else once the pool answers.
    /// Neither answer reads as the device having reached the park, and a
    /// tier wait, whose request the host reads only at the park, waits for
    /// the device. The keeper answers the first as the device reaches the
    /// publication.
    #[test]
    #[ignore]
    fn a_sited_park_is_answered_before_the_device_reaches_it() {
        let device = boundary_device();
        let parks = Arc::new(parks(&device));
        let site = parks.layers[0].site.clone().unwrap();
        let combine = ParkSite { layer: 0, kind: ParkKind::Site };
        // Step 7 requests the CPU: publication 1, combine 3.
        site.store(7, 0, constants::SITE_COUNT, 1);
        site.request(7, 0);
        parks.publish(1, 0, 7);
        parks.register(3, 7, combine);
        assert!(!parks.settle(), "the request is not the host's to read before its publication");
        parks.timeline.reach(1).unwrap();
        assert!(!parks.settle(), "the pool has not answered");
        site.answer(7, 0, false);
        assert!(parks.settle());
        assert!(parks.timeline.answered(3).unwrap() && !parks.timeline.reached(3).unwrap(), "the pool's answer opens the park first");
        // Step 8's tier wait reads a request the device has not made
        // visible: it waits for the device to park.
        parks.register(5, 8, ParkSite { layer: 0, kind: ParkKind::TierWait });
        assert!(!parks.settle());
        parks.timeline.reach(5).unwrap();
        assert!(parks.settle());
        // Step 9 requests nothing of the CPU: the keeper answers its
        // combine's park as the device reaches the publication.
        let keeper = Parks::keep(parks.clone()).unwrap();
        parks.publish(7, 0, 9);
        parks.register(9, 9, combine);
        parks.timeline.reach(7).unwrap();
        let began = std::time::Instant::now();
        while !parks.timeline.answered(9).unwrap() {
            assert!(began.elapsed() < NOTICE, "the keeper left a published park unanswered");
            std::hint::spin_loop();
        }
        assert!(!parks.timeline.reached(9).unwrap(), "an answer reads as no arrival");
        parks.stop();
        keeper.join().unwrap();
    }

    /// While the head is unreached the keeper looks again within a rest,
    /// so it answers a park that waits for nothing as the device reaches
    /// it, not a blocking wait's wake later.
    #[test]
    #[ignore]
    fn the_keeper_answers_a_reached_head_within_a_rest() {
        let device = boundary_device();
        let parks = Arc::new(parks(&device));
        let keeper = Parks::keep(parks.clone()).unwrap();
        parks.register(1, 7, ParkSite { layer: 0, kind: ParkKind::TierWait });
        std::thread::sleep(std::time::Duration::from_millis(5));
        assert!(!parks.timeline.answered(1).unwrap(), "a park answered before the device reached it");
        parks.timeline.reach(1).unwrap();
        let began = std::time::Instant::now();
        while !parks.timeline.answered(1).unwrap() {
            assert!(began.elapsed() < NOTICE, "the keeper did not answer the reached head");
            std::hint::spin_loop();
        }
        parks.stop();
        keeper.join().unwrap();
    }

    /// With nothing registered the keeper waits for a change, not a time:
    /// a registration ends its wait, and once the load stopped it ends.
    #[test]
    fn the_keeper_waits_for_a_change_and_ends_once_stopped() {
        let device = crate::test_support::open_backend(&Devices::discover(&[]).unwrap(), BackendName::Cpu).unwrap();
        let parks = Arc::new(parks(&device));
        let register = {
            let parks = parks.clone();
            std::thread::spawn(move || {
                std::thread::sleep(std::time::Duration::from_millis(5));
                parks.register(1, 7, ParkSite { layer: 0, kind: ParkKind::TierWait });
            })
        };
        assert!(parks.look(), "a registration ends the wait");
        register.join().unwrap();
        assert_eq!(parks.state().pending.len(), 1);
        parks.stop();
        assert!(!parks.look(), "stopped, the keeper ends");
    }

    /// The keeper answers every park whatever moment the device reaches it,
    /// the moment it looks included: a park reached between its settling
    /// and its wait was left waiting for a change that never came
    /// (a Vulkan prompt-group hang: the head parked, its lane never
    /// requested).
    #[test]
    fn the_keeper_answers_a_park_reached_at_any_moment() {
        let device = crate::test_support::open_backend(&Devices::discover(&[]).unwrap(), BackendName::Cpu).unwrap();
        let parks = Arc::new(parks(&device));
        let keeper = Parks::keep(parks.clone()).unwrap();
        let wait = ParkSite { layer: 0, kind: ParkKind::TierWait };
        for (index, value) in (1..4000u64).step_by(2).enumerate() {
            parks.register(value, 7, wait);
            // The device arrives after a varying delay, which sweeps the
            // keeper's look.
            for _ in 0..index % 97 {
                std::hint::spin_loop();
            }
            parks.timeline.reach(value).unwrap();
            assert!(answered_within(&parks.timeline, value, std::time::Duration::from_secs(1)), "park {value} reached and never answered");
        }
        parks.stop();
        keeper.join().unwrap();
    }

    /// A round's arrival, which the keeper requests as the device reaches
    /// its park, is an event: the request service, which looks at the
    /// device's words only on the load's events, sees each arrival whenever
    /// the device reaches the park, and its answer opens the park.
    #[test]
    fn the_keeper_announces_a_rounds_arrival() {
        let device = crate::test_support::open_backend(&Devices::discover(&[]).unwrap(), BackendName::Cpu).unwrap();
        let parks = Arc::new(parks(&device));
        let keeper = Parks::keep(parks.clone()).unwrap();
        let round = ParkSite { layer: 0, kind: ParkKind::Round(0) };
        let scratch = &parks.layers[0].scratch;
        for (index, value) in (1..800u64).step_by(2).enumerate() {
            let step = 8 + index as u64;
            parks.register(value, step, round);
            for _ in 0..index % 97 {
                std::hint::spin_loop();
            }
            parks.timeline.reach(value).unwrap();
            // The request service: it looks at the round's lane on each event.
            loop {
                let seen = parks.events.seen();
                if scratch.requested(step, round_lane(0)) {
                    break;
                }
                parks.events.wait_at_most(seen, std::time::Duration::from_secs(1));
                assert_ne!(parks.events.seen(), seen, "step {step}'s round arrived and no event announced it");
            }
            scratch.answer(step, round_lane(0), false);
            parks.settle();
            assert!(answered_within(&parks.timeline, value, std::time::Duration::from_secs(1)), "park {value} answered and never opened");
        }
        parks.stop();
        keeper.join().unwrap();
    }

    /// The keeper wakes no host job while nothing it acts on changes:
    /// parks answered as the device reaches them (a decode's all-hit
    /// layers) are no event, a keeper waiting on an unreached head or on
    /// nothing is none, and a head the device parks at unanswered is
    /// exactly one, however often it is looked at, until it opens.
    #[test]
    fn the_keeper_announces_only_what_a_host_job_waits_for() {
        let device = crate::test_support::open_backend(&Devices::discover(&[]).unwrap(), BackendName::Cpu).unwrap();
        let parks = Arc::new(parks(&device));
        let keeper = Parks::keep(parks.clone()).unwrap();
        let wait = ParkSite { layer: 0, kind: ParkKind::TierWait };
        let quiet = parks.events.seen();
        // A decode's parks that wait for nothing: none requested its slot.
        for value in (1..400u64).step_by(2) {
            parks.register(value, 7, wait);
            parks.timeline.reach(value).unwrap();
            assert!(answered_within(&parks.timeline, value, std::time::Duration::from_secs(1)), "park {value} never opened");
        }
        assert_eq!(parks.events.seen(), quiet, "parks answered as reached woke the load");
        // Waiting on nothing, then on a head the device has not reached.
        std::thread::sleep(std::time::Duration::from_millis(20));
        parks.register(401, 8, wait);
        std::thread::sleep(NOTICE * 3);
        assert_eq!(parks.events.seen(), quiet, "an idle keeper woke the load");
        // The device parks at a head whose slot it requested: one event.
        let file = &parks.layers[0].file;
        file.request(8, 0);
        parks.timeline.reach(401).unwrap();
        while parks.events.seen() == quiet {
            std::hint::spin_loop();
        }
        for _ in 0..50 {
            assert!(!parks.settle(), "the head is unanswered");
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
        assert_eq!(parks.events.seen(), quiet + 1, "one parked head announced more than once");
        file.answer(8, 0, false);
        assert!(parks.settle());
        assert!(parks.timeline.answered(401).unwrap());
        parks.stop();
        keeper.join().unwrap();
    }

    /// A retired step's registrations leave the queue; the others keep
    /// their order.
    #[test]
    fn retiring_a_step_removes_its_parks() {
        let device = crate::test_support::open_backend(&Devices::discover(&[]).unwrap(), BackendName::Cpu).unwrap();
        let parks = parks(&device);
        let site = ParkSite { layer: 0, kind: ParkKind::TierWait };
        for (value, sequence) in [(1, 3), (3, 4), (5, 3), (7, 5)] {
            parks.register(value, sequence, site);
        }
        parks.retire(3);
        assert!(parks.state().pending.iter().map(|park| park.value).eq([3, 7]));
        parks.retire(4);
        parks.retire(5);
        assert!(parks.state().pending.is_empty());
    }
}
