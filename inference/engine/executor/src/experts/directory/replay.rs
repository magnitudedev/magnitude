use super::{Directory, Ending, HostTier, Point};
use crate::experts::reuse::Reuse;
use crate::experts::watcher::{Step, Steps};
use crate::experts::watcher::low_word;
use super::stack::RoutedRead;

/// How a replay ranks a group's held records for eviction (least first).
#[cfg(any(test, feature = "expert-routing"))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Recency {
    /// Reuse as the replay's reads measure it, beside each record's last
    /// read: the directory's own ranking ([`Reuse`]).
    Reuse,
    /// The farthest next read first: Bélády's bound, which knows the
    /// future, so what any ranking could still gain.
    NextRead,
}

/// What a replay found: each read's misses, and the experts each layer
/// holds after the last read.
#[cfg(any(test, feature = "expert-routing"))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Replayed {
    pub misses: Vec<u64>,
    pub held: Vec<Vec<usize>>,
}

/// What routing costs a bank: each read's misses when `reads` (in the
/// device's order) go through the directory, from a bank of groups of
/// `records` holding each layer's `held` experts (read before any of them),
/// `layers` naming each layer's group, every miss placed and committed,
/// evicting by `recency`.
#[cfg(any(test, feature = "expert-routing"))]
pub fn replay(
    records: &[usize],
    layers: &[usize],
    experts: usize,
    held: &[Vec<usize>],
    reads: &[RoutedRead],
    recency: Recency,
) -> Result<Replayed, String> {
    #[derive(Clone, Copy, Default)]
    struct Reads {
        last: Step,
        owner: usize,
        expert: usize,
    }
    let mut directory = Directory::new(records, layers, &vec![experts; layers.len()], HostTier::replay(vec![0; records.len()]));
    // A replay's steps, which begin none: its seeds commit unmapped.
    let steps = Steps::new(crate::planning::FlightDepth::SERIAL, 1, super::exchange::Pulse::new());
    let mut stamps = records.iter().map(|&count| vec![Reads::default(); count]).collect::<Vec<_>>();
    let mut commits = records.iter().map(|&count| vec![0; count]).collect::<Vec<_>>();
    // Each layer's expert's reads, by position, for the bound.
    let mut uses = vec![vec![Vec::new(); experts]; layers.len()];
    for (position, read) in reads.iter().enumerate() {
        let uses = uses
            .get_mut(read.layer)
            .ok_or_else(|| format!("read {position} names layer {} of {}", read.layer, layers.len()))?;
        for &expert in &read.experts {
            uses.get_mut(expert)
                .ok_or_else(|| format!("read {position} names expert {expert} of {experts}"))?
                .push(position);
        }
    }
    if held.len() != layers.len() {
        return Err(format!("{} layers hold experts, of {}", held.len(), layers.len()));
    }
    // What the bank held before the first read: placed, committed and read
    // at no step, at group 0, before every read's (each read at position p
    // is group p + 1).
    for (layer, experts) in held.iter().enumerate() {
        let point = Point::whole(0, layer, 0);
        let placed = directory
            .place(point, layer, &[], experts, |_| 0)
            .ok_or_else(|| format!("layer {layer}'s group holds fewer records than it held"))?;
        for (&expert, &record) in experts.iter().zip(&placed.records) {
            stamps[layers[layer]][record.index()] = Reads {
                owner: layer,
                expert,
                ..Reads::default()
            };
        }
    }
    directory.retire(Point::whole(0, usize::MAX, Step::MAX), |_, _| 0, |_| Ending::Ended, &steps);
    // Each read is a point of its own in the device's order, each record
    // it reads last read at its step's sequence.
    let mut misses = vec![0; reads.len()];
    for (position, read) in reads.iter().enumerate() {
        let (layer, sequence, group) = (read.layer, read.sequence, layers[read.layer]);
        let point = Point::whole(position as Step + 1, layer, sequence);
        directory.retire(point, |group, record| commits[group][record.index()], |_| Ending::Ended, &steps);
        let (held, absent): (Vec<usize>, Vec<usize>) =
            read.experts.iter().partition(|&&expert| directory.held(layer, expert).is_some());
        for &expert in &held {
            stamps[group][directory.held(layer, expert).expect("held")].last = sequence;
        }
        // Bélády's bound ranks by the future alone: it logs no read.
        if recency == Recency::Reuse {
            Reuse::of(&directory.ranking).read(read, &vec![1; read.experts.len()]);
        }
        if absent.is_empty() {
            continue;
        }
        misses[position] = absent.len() as u64;
        // Higher ranks are kept: Bélády's keeps the soonest next read.
        let rank = |record: usize| -> Step {
            let read = stamps[group][record];
            match recency {
                Recency::Reuse => read.last,
                Recency::NextRead => {
                    let next = uses[read.owner][read.expert].iter().find(|&&at| at > position);
                    next.map_or(0, |&at| Step::MAX - at as Step)
                }
            }
        };
        let placed = directory
            .place(point, layer, &held, &absent, rank)
            .ok_or_else(|| format!("layer {layer}'s group holds fewer records than read {position} misses"))?;
        for (&expert, &record) in absent.iter().zip(&placed.records) {
            commits[group][record.index()] = low_word(sequence);
            stamps[group][record.index()] = Reads {
                last: sequence,
                owner: layer,
                expert,
            };
        }
    }
    directory.retire(Point::whole(reads.len() as Step + 1, usize::MAX, Step::MAX), |group, record| commits[group][record.index()], |_| Ending::Ended, &steps);
    let held = (0..layers.len())
        .map(|layer| (0..experts).filter(|&expert| directory.held(layer, expert).is_some()).collect())
        .collect();
    Ok(Replayed { misses, held })
}
