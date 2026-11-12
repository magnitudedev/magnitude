use super::table::stored_rows;
use crate::error::{PlanError, UnstreamedExperts};
use crate::operators::routed::fused_graph::DECODE_ROWS;
use crate::planning::weights::{Holding, WeightPlan, representation_bytes};
use magnitude_family_contracts::{FeedForwardUp, ModelDefinition, Operator, RoutedFfn, WeightKind, WeightRole, WeightScope};
use seismic::Element;
use std::num::NonZeroU64;

/// Each streamed layer's home, of `layers` in their order, each its group
/// and its experts: the map word (and record) of its expert 0 in its
/// group's map, past every earlier member's experts, expert `e`'s `e` past
/// it. The one rule the bank's records, its map, the device's
/// `policy[POLICY_HOME]` and the host's directory place experts by.
pub(crate) fn layer_homes(layers: impl IntoIterator<Item = (usize, u64)>) -> Vec<u64> {
    let mut next = Vec::<u64>::new();
    layers
        .into_iter()
        .map(|(group, experts)| {
            if next.len() <= group {
                next.resize(group + 1, 0);
            }
            let home = next[group];
            next[group] += experts;
            home
        })
        .collect()
}

/// Every routed layer of `definition`, in decoder order: a routed
/// sublayer, or a parallel block's routed branch, and the scope of its
/// weights.
pub(super) fn routed_layers(
    definition: &ModelDefinition,
) -> impl Iterator<Item = (magnitude_family_contracts::SublayerIndex, WeightScope, &RoutedFfn)> {
    definition.decoder.sublayers().filter_map(|(sublayer, layer)| match &layer.op {
        Operator::RoutedFfn(routed) => Some((sublayer, WeightScope::TargetSublayer(sublayer), &**routed)),
        Operator::Parallel(branches) => (0u32..).zip(branches).find_map(|(branch, each)| match &each.op {
            Operator::RoutedFfn(routed) => Some((sublayer, WeightScope::TargetBranch { sublayer, branch }, &**routed)),
            _ => None,
        }),
        _ => None,
    })
}

/// The most slots a bank of `definition`'s routed layers takes: the widest
/// layer's experts, every expert; a model that routes no experts streams
/// none. Its least is its plan's ([`ExpertBankPlan::floor`]).
pub fn routed_experts(definition: &ModelDefinition) -> Result<u64, PlanError> {
    routed_layers(definition)
        .map(|(_, _, routed)| routed.experts)
        .max()
        .ok_or(PlanError::Unstreamed(UnstreamedExperts::Unrouted))
}

/// The choices of the decode form's widest launch of a model whose rows
/// each select `selected` experts, `DECODE_ROWS · K`: the most a decode
/// request names at a layer.
pub(crate) fn decode_choices(selected: u64) -> u64 {
    DECODE_ROWS * selected
}

/// A routed tensor of an expert record, in the order the fill converts them
/// ([`Self::ALL`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RoutedTensor {
    Gate,
    Up,
    Down,
}

impl RoutedTensor {
    pub const ALL: [Self; 3] = [Self::Gate, Self::Up, Self::Down];

    /// The weight kind of its expert stack.
    pub fn kind(self) -> WeightKind {
        match self {
            Self::Gate => WeightKind::ExpertGate,
            Self::Up => WeightKind::ExpertUp,
            Self::Down => WeightKind::ExpertDown,
        }
    }

    /// The routed tensor of an expert stack of `kind`, if it is one.
    pub fn of(kind: WeightKind) -> Option<Self> {
        Self::ALL.into_iter().find(|tensor| tensor.kind() == kind)
    }
}

/// Which routed tensors an expert's record holds: gate, up and down for
/// gated experts, up and down for up-only ones.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RecordForm {
    Gated,
    Up,
}

impl RecordForm {
    /// The form of `up`'s experts.
    pub fn of(up: &FeedForwardUp) -> Self {
        match up {
            FeedForwardUp::Gated { .. } => Self::Gated,
            _ => Self::Up,
        }
    }

    /// Its routed tensors, in fill order.
    pub fn tensors(self) -> &'static [RoutedTensor] {
        match self {
            Self::Gated => &RoutedTensor::ALL,
            Self::Up => &RoutedTensor::ALL[1..],
        }
    }

    /// The routed tensor whose lent records carry routed tensor `kind`'s
    /// conveyor: the next of the form's, so a round's convert of `kind`'s
    /// records into the scratch reads a conveyor in another
    /// tensor than it writes, even where the conveyor lies in the bank's own
    /// records (unified memory).
    pub fn carrier(self, kind: RoutedTensor) -> RoutedTensor {
        let tensors = self.tensors();
        let at = tensors.iter().position(|&tensor| tensor == kind).expect("a routed tensor of the form");
        tensors[(at + 1) % tensors.len()]
    }

    /// The record of what `make` makes of each of its routed tensors, or the
    /// first error.
    pub fn each<T, E>(self, mut make: impl FnMut(RoutedTensor) -> Result<T, E>) -> Result<RecordTensors<T>, E> {
        Ok(RecordTensors {
            gate: match self {
                Self::Gated => Some(make(RoutedTensor::Gate)?),
                Self::Up => None,
            },
            up: make(RoutedTensor::Up)?,
            down: make(RoutedTensor::Down)?,
        })
    }
}

/// One value for each routed tensor of an expert record: the gate exactly
/// when its form is gated ([`RecordForm`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct RecordTensors<T> {
    pub gate: Option<T>,
    pub up: T,
    pub down: T,
}

impl<T> RecordTensors<T> {
    pub fn form(&self) -> RecordForm {
        match self.gate {
            Some(_) => RecordForm::Gated,
            None => RecordForm::Up,
        }
    }

    /// The value of routed tensor `kind`, where the record holds it.
    pub fn get(&self, kind: RoutedTensor) -> Option<&T> {
        match kind {
            RoutedTensor::Gate => self.gate.as_ref(),
            RoutedTensor::Up => Some(&self.up),
            RoutedTensor::Down => Some(&self.down),
        }
    }

    /// Each routed tensor's value, in fill order.
    pub fn iter(&self) -> impl Iterator<Item = (RoutedTensor, &T)> {
        self.gate
            .iter()
            .map(|gate| (RoutedTensor::Gate, gate))
            .chain([(RoutedTensor::Up, &self.up), (RoutedTensor::Down, &self.down)])
    }

    /// Each routed tensor's value, mutably, in fill order.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (RoutedTensor, &mut T)> {
        self.gate
            .iter_mut()
            .map(|gate| (RoutedTensor::Gate, gate))
            .chain([(RoutedTensor::Up, &mut self.up), (RoutedTensor::Down, &mut self.down)])
    }

    /// Each routed tensor and its value, in fill order.
    pub fn into_tensors(self) -> impl Iterator<Item = (RoutedTensor, T)> {
        self.gate
            .map(|gate| (RoutedTensor::Gate, gate))
            .into_iter()
            .chain([(RoutedTensor::Up, self.up), (RoutedTensor::Down, self.down)])
    }

    pub fn as_ref(&self) -> RecordTensors<&T> {
        RecordTensors {
            gate: self.gate.as_ref(),
            up: &self.up,
            down: &self.down,
        }
    }

    pub fn as_mut(&mut self) -> RecordTensors<&mut T> {
        RecordTensors {
            gate: self.gate.as_mut(),
            up: &mut self.up,
            down: &mut self.down,
        }
    }

    /// Each routed tensor's value of this record and of `other`, of one
    /// form: a gate where both hold one.
    pub fn zip<U>(self, other: RecordTensors<U>) -> RecordTensors<(T, U)> {
        RecordTensors {
            gate: self.gate.zip(other.gate),
            up: (self.up, other.up),
            down: (self.down, other.down),
        }
    }

    /// What `make` makes of each routed tensor's value.
    pub fn map<U>(self, mut make: impl FnMut(RoutedTensor, T) -> U) -> RecordTensors<U> {
        RecordTensors {
            gate: self.gate.map(|gate| make(RoutedTensor::Gate, gate)),
            up: make(RoutedTensor::Up, self.up),
            down: make(RoutedTensor::Down, self.down),
        }
    }

    /// What `make` makes of each routed tensor's value, or the first error.
    pub fn try_map<U, E>(self, mut make: impl FnMut(RoutedTensor, T) -> Result<U, E>) -> Result<RecordTensors<U>, E> {
        Ok(RecordTensors {
            gate: self.gate.map(|gate| make(RoutedTensor::Gate, gate)).transpose()?,
            up: make(RoutedTensor::Up, self.up)?,
            down: make(RoutedTensor::Down, self.down)?,
        })
    }
}

/// A record's values, in fill order.
impl<T> IntoIterator for RecordTensors<T> {
    type Item = T;
    type IntoIter = std::iter::Chain<std::option::IntoIter<T>, std::array::IntoIter<T, 2>>;

    fn into_iter(self) -> Self::IntoIter {
        self.gate.into_iter().chain([self.up, self.down])
    }
}

/// One routed tensor of a bank group's records.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct RecordTensor {
    /// As the model file stores it.
    pub source: Element,
    /// As the resident entries read it.
    pub resident: Element,
    /// One expert's matrix `[N, K]`.
    pub matrix: [u64; 2],
}

impl RecordTensor {
    /// One expert as stored: a record of the staging it is read into, its
    /// parts in the order they lie in it (`Element::record_parts`).
    pub fn stored_parts(&self) -> Result<Vec<u64>, String> {
        let [rows, columns] = self.matrix;
        self.source
            .record_parts(&[1, rows, columns])
            .ok_or_else(|| format!("{} is not one matrix's parts of {rows}×{columns}", self.source.name()))
    }

    /// One expert as stored: a record of the staging it is read into.
    pub fn stored_bytes(&self) -> Result<u64, String> {
        record_bytes(self.source, self.matrix)
    }

    /// One expert as resident: a record of its bank group's tensor.
    pub fn record_bytes(&self) -> Result<u64, String> {
        record_bytes(self.resident, self.matrix)
    }

    /// `records` records as stored: the element and extents of the staging
    /// and conveyor tensors that hold them, wherever one is laid.
    pub fn stored_tensor(&self, records: u64) -> (Element, [u64; 3]) {
        (self.source, [records, self.matrix[0], self.matrix[1]])
    }

    /// `records` records as resident: the element and extents of the
    /// tensors of a bank group's records, its scratch and host tier, wherever
    /// one is laid.
    pub fn resident_tensor(&self, records: u64) -> (Element, [u64; 3]) {
        (self.resident, [records, self.matrix[0], self.matrix[1]])
    }
}

/// One expert's `[N, K]` `matrix` in `element`: a record's bytes.
pub(super) fn record_bytes(element: Element, matrix: [u64; 2]) -> Result<u64, String> {
    representation_bytes(element, &[1, matrix[0], matrix[1]])
}

/// `bytes` as the record a reservation lays out
/// (`seismic::Reservation::of_records`), which a record of none is not.
pub(super) fn reserved_record(bytes: u64) -> Result<NonZeroU64, String> {
    NonZeroU64::new(bytes).ok_or_else(|| "a reserved record has no bytes".to_owned())
}

/// The record form of an expert stack planned resident: refused when its
/// bytes are uploaded other than as stored, or it does not hold each of its
/// `experts` experts as one range of its stored experts' bytes
/// ([`stored_rows`]) and as one contiguous range resident.
pub(super) fn record_tensor(plan: &WeightPlan, experts: u64) -> Result<RecordTensor, PlanError> {
    let refuse = |reason: fn(WeightRole) -> UnstreamedExperts| {
        PlanError::Unstreamed(reason(plan.role))
    };
    let [count, rows, columns] = plan.shape[..] else {
        return Err(refuse(UnstreamedExperts::Geometry));
    };
    if count != experts {
        return Err(refuse(UnstreamedExperts::Geometry));
    }
    // A record is read from the stored bytes as they lie: uploaded as
    // stored, at most a range of each expert's rows taken.
    if plan.upload != plan.source || stored_rows(&plan.descriptor).is_none() || plan.holding != Holding::Whole {
        return Err(refuse(UnstreamedExperts::Record));
    }
    let tensor = RecordTensor {
        source: plan.source,
        resident: plan.resident,
        matrix: [rows, columns],
    };
    // One expert's slice of a stack whose bytes are sized is itself sized.
    let contiguous = |record: Result<u64, String>, whole: u64| {
        record.expect("an expert of a sized stack is sized").checked_mul(experts) == Some(whole)
    };
    if !contiguous(tensor.stored_bytes(), plan.source_bytes)
        || !contiguous(tensor.record_bytes(), plan.resident_bytes)
    {
        return Err(refuse(UnstreamedExperts::Record));
    }
    Ok(tensor)
}
