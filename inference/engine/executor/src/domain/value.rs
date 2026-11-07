//! One priced heap: every memory holding of a load valued by what its bytes
//! save, on one scale, for the plan's commitment and for every grant and
//! release while serving.
//!
//! Value is a rate, [`Rate`]: seconds of serving a holding saves per second
//! of serving (a bank slot's step saving times decode steps a second, a
//! prompt pass size's prefill seconds saved times prefills a second, a
//! prefix's recompute seconds times its hits a second). Each holding gives
//! its curve as steps ([`Curve`]): those it holds, the last granted first,
//! and those it would take next, the next first. A curve need not be
//! concave (a bank's last slot saves the streamed forms' whole cost, a
//! slot before it may save nothing a miss model resolves): the settle
//! moves each curve by the runs of steps its hull joins ([`Run`]), each
//! worth its steps together, so choosing by rate per byte across curves is
//! the optimum of the knapsack the heap's room poses, at load from the
//! mandatory part and while serving from what is held.
//!
//! What a claim of the load's own needs now (a live sequence's state) is
//! owed ([`Supply::owed`]): releases cover it whatever they lose.
//!
//! The room ([`Supply`]) is priced by where it comes from: memory the host
//! gives without displacing anyone is free; memory it gives by compressing
//! other programs' data is borrowed, at the price of giving it back; a
//! device's hard limit (Metal's working set) is a ceiling no price moves.
//! A step held is exchanged for one offered only where the rate gained over
//! the memory's measured horizon pays both moves.

use std::time::Duration;

/// A holding the heap prices: each owned where it lives, each giving its
/// [`Curve`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Holding {
    /// The expert bank's slots, its host tier's records, or a stored
    /// table's slots (`experts/bank/served.rs`).
    Bank(super::BankHolding),
    /// A prompt group's places beyond the first: the class's graph slots,
    /// group buffers and table stagings (`planning::class_slots`).
    Places,
    /// Rows per merged-expert prompt pass beyond the least that is a pass:
    /// the target binding's pass buffers.
    PassRows,
    /// Cached prefixes' retained state.
    Prefixes,
    /// An optional component: the head (drafter) or vision.
    Optional(super::OptionalComponent),
}

/// Seconds of serving time saved per second of serving: the one unit every
/// holding is valued in.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub(crate) struct Rate(f64);

impl Rate {
    /// # Panics
    /// Where `seconds_per_second` is negative or not finite: a value is a
    /// saving, measured or derived, and a holding that costs time is not
    /// one.
    pub(crate) fn new(seconds_per_second: f64) -> Self {
        assert!(
            seconds_per_second.is_finite() && seconds_per_second >= 0.0,
            "a holding's value is a finite saving, not {seconds_per_second}"
        );
        Self(seconds_per_second)
    }

    pub(crate) fn get(self) -> f64 {
        self.0
    }
}

/// One step of a holding's curve: its bytes, what holding them saves, and
/// what moving it (granting or releasing it) costs the load, in seconds, as
/// measured (a resize's thread time); zero where nothing measured it yet.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Step {
    pub(crate) bytes: u64,
    pub(crate) worth: Rate,
    pub(crate) move_seconds: f64,
}

impl Step {
    /// Its rate per byte: what orders every grant and release.
    pub(crate) fn per_byte(&self) -> f64 {
        self.worth.get() / self.bytes.max(1) as f64
    }
}

/// The rate per byte of `steps` moved together: their worth over their
/// bytes, as a settle's run of them is ordered.
pub(crate) fn per_byte(steps: &[Step]) -> f64 {
    let (worth, bytes) = steps.iter().fold((0.0, 0u64), |(worth, bytes), step| (worth + step.worth.get(), bytes + step.bytes));
    worth / bytes.max(1) as f64
}

/// A holding's curve at its current size.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Curve<H> {
    pub(crate) holding: H,
    /// Whether its grants may take borrowed room: only memory whose
    /// reading counts what it holds there (the one store's expert slots,
    /// `BankRoom`) may.
    pub(crate) borrows: bool,
    /// The steps it holds that may be released, the last granted first.
    pub(crate) held: Vec<Step>,
    /// The steps it may take next, the next first.
    pub(crate) offered: Vec<Step>,
}

/// The room a settle grants into.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Supply {
    /// Bytes the host gives without displacing anyone, above the derived
    /// reserve.
    pub(crate) free: u64,
    /// Bytes owed now: claims beyond the free room (an ask, a fall below
    /// the reserve), which releases must cover.
    pub(crate) owed: u64,
    /// Bytes the host gives beyond the free room by compressing other
    /// programs' data.
    pub(crate) borrowable: u64,
    /// What a borrowed byte costs to hold: the rate per byte a step must
    /// beat to take it.
    pub(crate) borrow_price: f64,
    /// Bytes the device's hard limit still allows (`u64::MAX` where none
    /// binds).
    pub(crate) ceiling: u64,
    /// How long the memory is expected to stay as read: what an exchange's
    /// gain is earned over.
    pub(crate) horizon: Duration,
}

/// What a settle decided: per holding, the steps released (from its held,
/// in order) and granted (from its offered, in order), and the bytes of
/// borrowed room its grants take.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Settlement<H> {
    pub(crate) released: Vec<(H, usize)>,
    pub(crate) granted: Vec<(H, usize)>,
    pub(crate) borrowed: Vec<(H, u64)>,
}

/// A run of a curve's steps the settle moves together: its steps, and their
/// bytes, worth and moves summed. A curve's runs are its hull's: offered,
/// each worth no more per byte than the run before it, so a step worth
/// little that leads to one worth more is taken with it; held, each worth no
/// less per byte than the run released before it, so a step worth much is not
/// released for the little one behind it.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Run {
    steps: usize,
    step: Step,
}

/// `steps` in order as runs, each run joined with the one before while
/// `joins(before, run)` by their rates per byte.
fn hull(steps: &[Step], joins: fn(f64, f64) -> bool) -> Vec<Run> {
    let mut runs: Vec<Run> = Vec::with_capacity(steps.len());
    for &step in steps {
        let mut run = Run { steps: 1, step };
        while let Some(before) = runs.last().filter(|before| joins(before.step.per_byte(), run.step.per_byte())) {
            run = Run {
                steps: before.steps + run.steps,
                step: Step {
                    bytes: before.step.bytes + run.step.bytes,
                    worth: Rate::new(before.step.worth.get() + run.step.worth.get()),
                    // A run moves in one resize of its holding.
                    move_seconds: before.step.move_seconds.max(run.step.move_seconds),
                },
            };
            runs.pop();
        }
        runs.push(run);
    }
    runs
}

/// A curve as the settle moves it: its held and offered runs.
struct Runs {
    held: Vec<Run>,
    offered: Vec<Run>,
}

impl Runs {
    fn of<H>(curve: &Curve<H>) -> Self {
        Self {
            held: hull(&curve.held, |before, run| before > run),
            offered: hull(&curve.offered, |before, run| before < run),
        }
    }
}

/// A curve's progress through a settle, in runs.
struct Cursor {
    released: usize,
    granted: usize,
    /// Borrowed bytes its grants take.
    borrowed: u64,
    /// No more of its offered steps may be granted this settle: the next
    /// found no room, and the steps after it need it first.
    blocked: bool,
}

/// Settle `curves` into `supply`: cover what is owed by releasing the
/// lowest rate per byte first, then grant offered steps that save something
/// by rate per byte, a curve ending at its first that saves nothing,
/// from free room, then borrowed room where the step beats the borrow price
/// and its holding may hold borrowed room ([`Curve::borrows`]), then by
/// exchange where the gain pays both moves.
pub(crate) fn settle<H: Clone>(curves: &[Curve<H>], supply: Supply) -> Settlement<H> {
    let runs: Vec<Runs> = curves.iter().map(Runs::of).collect();
    let mut cursors: Vec<Cursor> =
        curves.iter().map(|_| Cursor { released: 0, granted: 0, borrowed: 0, blocked: false }).collect();
    let mut free = supply.free as i128 - supply.owed as i128;
    let mut ceiling = supply.ceiling as i128;
    let mut borrow_left = supply.borrowable;

    // Cover what is owed, the lowest rate per byte first: by borrowed room
    // while what it would release beats the borrow price (it keeps what it
    // holds there), by a release otherwise; the device's ceiling only by a
    // release. What nothing covers is the release ladder's.
    while free < 0 || ceiling < 0 {
        let lowest = runs
            .iter()
            .enumerate()
            .filter_map(|(index, runs)| runs.held.get(cursors[index].released).map(|run| (index, run.step.per_byte())))
            .min_by(|a, b| a.1.total_cmp(&b.1));
        if ceiling >= 0 && borrow_left > 0 && lowest.is_none_or(|(_, per_byte)| per_byte > supply.borrow_price) {
            let taken = (borrow_left as i128).min(-free);
            borrow_left -= taken as u64;
            free += taken;
            continue;
        }
        let Some((index, _)) = lowest else { break };
        let bytes = runs[index].held[cursors[index].released].step.bytes as i128;
        cursors[index].released += 1;
        free += bytes;
        ceiling += bytes;
    }

    // Grow by rate per byte. A run that saves nothing takes no room: room
    // nothing prices stays free, for the host and the next claim. A
    // holding's runs granted in one settle move in one resize: its first
    // pays the move, which its saving over the horizon must cover, as an
    // exchange's gain must cover its moves.
    loop {
        let best = runs
            .iter()
            .enumerate()
            .filter(|(index, _)| !cursors[*index].blocked)
            .filter_map(|(index, runs)| runs.offered.get(cursors[index].granted).map(|run| (index, run.step.per_byte(), run.step)))
            .filter(|(_, _, step)| step.worth.get() > 0.0)
            .max_by(|a, b| a.1.total_cmp(&b.1));
        let Some((index, per_byte, step)) = best else { break };
        let step = Step { move_seconds: if cursors[index].granted == 0 { step.move_seconds } else { 0.0 }, ..step };
        let bytes = step.bytes as i128;
        let fits_ceiling = bytes <= ceiling;
        let borrows = curves[index].borrows && per_byte > supply.borrow_price;
        let pays = step.worth.get() * supply.horizon.as_secs_f64() >= step.move_seconds;
        if !pays {
            cursors[index].blocked = true;
            continue;
        }
        if fits_ceiling && bytes <= free {
            free -= bytes;
        } else if fits_ceiling && borrows && bytes <= free.max(0) + borrow_left as i128 {
            let excess = (bytes - free.max(0)) as u64;
            borrow_left -= excess;
            cursors[index].borrowed += excess;
            free = free.min(0);
        } else if let Some(taken) = exchange(&runs, &cursors, index, &step, per_byte, free, ceiling, supply.horizon) {
            for (other, released) in taken {
                for _ in 0..released {
                    let bytes = runs[other].held[cursors[other].released].step.bytes as i128;
                    cursors[other].released += 1;
                    free += bytes;
                    ceiling += bytes;
                }
            }
            free -= bytes;
        } else {
            cursors[index].blocked = true;
            continue;
        }
        ceiling -= bytes;
        cursors[index].granted += 1;
    }

    // Each curve's runs moved, as the steps they hold.
    let collect = |count: &dyn Fn(usize) -> u64| {
        curves
            .iter()
            .enumerate()
            .map(|(index, curve)| (curve.holding.clone(), count(index)))
            .filter(|(_, count)| *count > 0)
            .collect::<Vec<_>>()
    };
    let steps = |of: fn(&Runs) -> &[Run], moved: fn(&Cursor) -> usize| {
        collect(&|index| of(&runs[index])[..moved(&cursors[index])].iter().map(|run| run.steps as u64).sum())
            .into_iter()
            .map(|(holding, steps)| (holding, steps as usize))
            .collect()
    };
    Settlement {
        released: steps(|runs| &runs.held, |cursor| cursor.released),
        granted: steps(|runs| &runs.offered, |cursor| cursor.granted),
        borrowed: collect(&|index| cursors[index].borrowed),
    }
}

/// The held runs of curves other than `index` an exchange for `step` would
/// release, as (curve, runs) pairs, the lowest rate per byte first: only
/// runs worth less per byte than `step`, only enough to fit it beside
/// `free` and within `ceiling`, and only where the rate it gains over what
/// they lose, earned over `horizon`, pays its move and theirs.
#[allow(clippy::too_many_arguments)]
fn exchange(
    runs: &[Runs],
    cursors: &[Cursor],
    index: usize,
    step: &Step,
    per_byte: f64,
    free: i128,
    ceiling: i128,
    horizon: Duration,
) -> Option<Vec<(usize, usize)>> {
    let mut released: Vec<usize> = cursors.iter().map(|cursor| cursor.released).collect();
    let mut taken: Vec<(usize, usize)> = Vec::new();
    let (mut room, mut limit) = (free, ceiling);
    let (mut lost, mut moves) = (0.0, step.move_seconds);
    let bytes = step.bytes as i128;
    while room < bytes || limit < bytes {
        let (other, held) = runs
            .iter()
            .enumerate()
            .filter(|(other, _)| *other != index)
            .filter_map(|(other, runs)| {
                let held = runs.held.get(released[other])?.step;
                let value = held.per_byte();
                (value < per_byte).then_some(((other, held), value))
            })
            .min_by(|a, b| a.1.total_cmp(&b.1))?
            .0;
        released[other] += 1;
        match taken.last_mut() {
            Some((last, steps)) if *last == other => *steps += 1,
            _ => taken.push((other, 1)),
        }
        room += held.bytes as i128;
        limit += held.bytes as i128;
        lost += held.worth.get();
        // A holding's runs released in one settle move in one resize: its
        // first released here pays it.
        if released[other] == cursors[other].released + 1 {
            moves += held.move_seconds;
        }
    }
    ((step.worth.get() - lost) * horizon.as_secs_f64() > moves).then_some(taken)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rate(bytes: u64, seconds_per_second: f64) -> Step {
        Step { bytes, worth: Rate::new(seconds_per_second), move_seconds: 0.0 }
    }

    fn supply(free: u64) -> Supply {
        Supply { free, owed: 0, borrowable: 0, borrow_price: 0.0, ceiling: u64::MAX, horizon: Duration::from_secs(60) }
    }

    fn curve(holding: &'static str, held: Vec<Step>, offered: Vec<Step>) -> Curve<&'static str> {
        Curve { holding, borrows: true, held, offered }
    }

    /// Free room goes to the steps that save the most per byte, across
    /// holdings, each holding's steps in its own order: a bank whose early
    /// slots save most takes them before a pass size that saves less, and
    /// the pass size's first step before the bank's flat tail.
    #[test]
    fn free_room_goes_to_the_most_value_per_byte_across_holdings() {
        let bank = curve("bank", vec![], vec![rate(10, 1.0), rate(10, 0.5), rate(10, 0.01)]);
        let pass = curve("pass", vec![], vec![rate(10, 0.2), rate(10, 0.001)]);
        let settled = settle(&[bank, pass], supply(30));
        assert_eq!(settled.granted, vec![("bank", 2), ("pass", 1)]);
        assert_eq!(settled.borrowed, vec![]);
    }

    /// What is owed is covered by releasing the lowest rate per byte first,
    /// across holdings.
    #[test]
    fn owed_room_releases_the_lowest_value_per_byte_first() {
        let bank = curve("bank", vec![rate(10, 0.05), rate(10, 0.5)], vec![]);
        let prefix = curve("prefix", vec![rate(10, 0.01)], vec![]);
        let surplus = curve("surplus", vec![rate(10, 0.0)], vec![]);
        let settled = settle(&[bank, prefix, surplus], Supply { owed: 25, ..supply(0) });
        assert_eq!(settled.released, vec![("bank", 1), ("prefix", 1), ("surplus", 1)]);
    }

    /// Borrowed room goes only to a step that beats its price, of a
    /// holding that may hold borrowed room, and is counted against it.
    #[test]
    fn borrowed_room_goes_only_to_steps_that_beat_its_price() {
        let bank = curve("bank", vec![], vec![rate(10, 1.0), rate(10, 0.05)]);
        let room = Supply { borrowable: 20, borrow_price: 0.01, ..supply(5) };
        let settled = settle(&[bank.clone()], room);
        assert_eq!(settled.granted, vec![("bank", 1)]);
        assert_eq!(settled.borrowed, vec![("bank", 5)]);
        let places = Curve { borrows: false, ..curve("places", vec![], vec![rate(10, 1.0)]) };
        let settled = settle(&[places], room);
        assert_eq!((settled.granted, settled.borrowed), (vec![], vec![]));
    }

    /// What is owed is held on borrowed room only while what it would
    /// release beats the borrow price: a slot worth more per byte than
    /// borrowing costs stays on borrowed room, one worth less is released.
    #[test]
    fn owed_room_is_borrowed_only_above_the_borrow_price() {
        let bank = || curve("bank", vec![rate(10, 0.5)], vec![]);
        let owed = |borrow_price| Supply { owed: 10, borrowable: 10, borrow_price, ..supply(0) };
        let kept = settle(&[bank()], owed(0.01));
        assert_eq!(kept.released, vec![]);
        let given = settle(&[bank()], owed(0.1));
        assert_eq!(given.released, vec![("bank", 1)]);
    }

    /// A held step is exchanged for an offered one worth more per byte only
    /// where the gain over the horizon pays both moves.
    #[test]
    fn an_exchange_happens_only_where_its_gain_pays_both_moves() {
        let held = |move_seconds| Step { move_seconds, ..rate(10, 0.1) };
        let offered = |move_seconds| Step { move_seconds, ..rate(10, 0.3) };
        let pays = settle(&[curve("pass", vec![held(1.0)], vec![]), curve("bank", vec![], vec![offered(1.0)])], supply(0));
        assert_eq!((pays.released, pays.granted), (vec![("pass", 1)], vec![("bank", 1)]));
        // 0.2 s/s over 60 s is 12 s of gain, against 20 s of moves.
        let costly = settle(&[curve("pass", vec![held(10.0)], vec![]), curve("bank", vec![], vec![offered(10.0)])], supply(0));
        assert_eq!((costly.released, costly.granted), (vec![], vec![]));
    }

    /// A grant from free room pays its move like an exchange: a holding's
    /// runs granted in one settle move in one resize, which what they save
    /// over the horizon must cover.
    #[test]
    fn a_grant_pays_its_one_move() {
        let moving = |move_seconds| Step { move_seconds, ..rate(10, 0.1) };
        // 0.1 s/s over 60 s is 6 s: a 5 s move is paid, once for both steps.
        let paid = settle(&[curve("bank", vec![], vec![moving(5.0), moving(5.0)])], supply(20));
        assert_eq!(paid.granted, vec![("bank", 2)]);
        let unpaid = settle(&[curve("bank", vec![], vec![moving(7.0)])], supply(20));
        assert_eq!(unpaid.granted, vec![]);
    }

    /// A step that saves nothing takes no room, even room nothing else can
    /// use: a pass size worth nothing stays unheld beside a bank whose next
    /// slot does not fit, and the room stays free.
    #[test]
    fn a_step_worth_nothing_takes_no_room() {
        let bank = curve("bank", vec![], vec![rate(40, 2.0), rate(40, 1.0), rate(40, 0.5)]);
        let pass = curve("pass", vec![], vec![rate(10, 0.0), rate(10, 0.0)]);
        let settled = settle(&[bank, pass], supply(100));
        assert_eq!(settled.granted, vec![("bank", 2)]);
    }

    /// A curve moves by its hull: a step that saves nothing before one that
    /// saves much is taken with it, ahead of another holding's step worth
    /// less per byte than the two together; and a held step worth much is
    /// not released ahead of the little one granted before it.
    #[test]
    fn a_step_worth_little_moves_with_the_step_it_leads_to() {
        let bank = curve("bank", vec![], vec![rate(10, 1.0), rate(10, 0.0), rate(10, 0.6)]);
        let pass = curve("pass", vec![], vec![rate(10, 0.2)]);
        let settled = settle(&[bank, pass], supply(30));
        assert_eq!(settled.granted, vec![("bank", 3)]);
        // Held, the last granted first: releasing the one worth 0.6 frees
        // the room of the one worth nothing too, so the two go together.
        let held = curve("bank", vec![rate(10, 0.6), rate(10, 0.0), rate(10, 1.0)], vec![]);
        let other = curve("prefix", vec![rate(10, 0.25)], vec![]);
        let owed = settle(&[held, other], Supply { owed: 10, ..supply(0) });
        assert_eq!(owed.released, vec![("prefix", 1)]);
        let more = settle(
            &[curve("bank", vec![rate(10, 0.6), rate(10, 0.0), rate(10, 1.0)], vec![]), curve("prefix", vec![rate(10, 0.35)], vec![])],
            Supply { owed: 10, ..supply(0) },
        );
        assert_eq!(more.released, vec![("bank", 2)]);
    }

    /// The device's hard limit binds whatever the host would give.
    #[test]
    fn the_device_ceiling_binds_beside_free_room() {
        let bank = curve("bank", vec![], vec![rate(10, 1.0), rate(10, 0.5)]);
        let settled = settle(&[bank], Supply { ceiling: 15, ..supply(100) });
        assert_eq!(settled.granted, vec![("bank", 1)]);
    }

    /// A holding whose next step finds no room offers nothing beyond it,
    /// while another's smaller step may still fit.
    #[test]
    fn a_step_that_finds_no_room_blocks_only_its_own_curve() {
        let bank = curve("bank", vec![], vec![rate(20, 3.0), rate(1, 1.0)]);
        let prefix = curve("prefix", vec![], vec![rate(5, 0.5)]);
        let settled = settle(&[bank, prefix], supply(10));
        assert_eq!(settled.granted, vec![("prefix", 1)]);
    }
}
