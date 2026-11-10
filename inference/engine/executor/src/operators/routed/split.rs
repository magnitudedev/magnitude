//! How a block's prompt rows split at its routed layer: the graphs a block
//! seals ([`Stage`]), what a layer's front passes its back and its merge
//! set's pass ([`Crossing`]), what a pass leaves its backs ([`Merged`]), the
//! ports of each graph, and the buffers a load keeps of those values
//! ([`SplitValues`]).

use crate::programs::graph::facts::{Numbers, constant};
use crate::programs::graph::{GraphError, draft::GraphDraft};
use magnitude_family_contracts::{Decoder, RoutedFfn};
use seismic::{Dim, Element, NativePort};

/// What decides a routed layer's crossed and merged values beside their
/// numbers ([`SplitLayer`]): the element of the experts' input (the
/// decoder's activations), and whether the layer is of the fused form,
/// which crosses its shared expert's coefficient.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct SplitForm {
    pub activation: Element,
    pub fused: bool,
}

/// The numbers of a routed layer's crossed and merged values: its experts,
/// its selected experts and the width its experts act in.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct SplitLayer<D = Dim> {
    pub experts: D,
    pub selected: D,
    pub width: D,
}

impl<D> Numbers<D> for SplitLayer<D> {
    type Of<E> = SplitLayer<E>;
    fn map<E>(self, f: &mut impl FnMut(D) -> E) -> SplitLayer<E> {
        SplitLayer { experts: f(self.experts), selected: f(self.selected), width: f(self.width) }
    }
}

/// The form and numbers of `routed`'s split values in `geometry`: the only
/// reading of the model they make.
pub(crate) fn split(routed: &RoutedFfn, geometry: &Decoder) -> (SplitForm, SplitLayer<u64>) {
    (
        SplitForm {
            activation: crate::programs::native_target_graph::activation(geometry),
            fused: super::is_fused(routed),
        },
        SplitLayer {
            experts: routed.experts,
            selected: routed.selected,
            width: routed.expert_hidden(geometry.hidden),
        },
    )
}

/// The part of a block a sealed graph holds: the whole block, or, of a
/// block whose prompt rows split at its routed layer, its front (its mixer,
/// route and, where its experts are scratch-served, request) and its back,
/// the combine from its merge set's pass and what follows it. The pass is a
/// graph of its own over a merge set's crossed rows, sealed once at each
/// length a group's pass takes; a streamed layer's fill, a scratch-served
/// step's rounds and seeding, which precede its set's pass, is sealed once
/// at the longest pass's rows. A prompt group runs
/// every step's front of a layer before any step's back, so every request
/// of the layer is published before its experts are served, and every row
/// is crossed before a pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum Stage {
    /// A run's graph, at its launch's row class.
    Run(RunStage),
    /// A graph sealed apart from the launches' row classes, at a pass's
    /// rows, reading no rows of a run's own.
    Once(OnceStage),
}

/// The part of a block a run's graph holds: the whole block, or a split
/// block's front or back. A whole decode block whose streamed layer's bank
/// can hold every expert (`ExpertBankPlan::reaches_whole`) has a second
/// whole graph, its resident form, which reads the layer's experts in place
/// at their home records, as a resident layer reads its stacks: a decode run
/// takes it in place of the whole block's while its bank is whole
/// (`ExpertBank::begin`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum RunStage {
    Whole,
    Resident,
    Front,
    Combine,
}

/// A graph sealed once: a streamed layer's fill, or a merge set's pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum OnceStage {
    Fill,
    Pass,
}

impl RunStage {
    /// Which of a block's draft taps, at its entry, before its
    /// feed-forward and at its output, a graph of this stage writes.
    pub(crate) fn taps(self) -> [bool; 3] {
        let mixes = matches!(self, Self::Whole | Self::Resident | Self::Front);
        [mixes, mixes, matches!(self, Self::Whole | Self::Resident | Self::Combine)]
    }

    /// The stages of `rows` rows of a block whose prompt rows `split`, their
    /// experts run in their merge sets' passes.
    pub(crate) fn served(split: bool, rows: u64) -> &'static [Self] {
        match split {
            _ if super::fused_graph::decodes(rows) => &[Self::Whole],
            true => &[Self::Front, Self::Combine],
            false => &[Self::Whole],
        }
    }

    /// The stages a block's graphs of `rows` rows are sealed in: those its
    /// runs take ([`Self::served`]), and beside a decode block's whole graph
    /// its resident form, where its streamed layer's bank can be whole
    /// (`resident`).
    pub(crate) fn sealed(split: bool, resident: bool, rows: u64) -> &'static [Self] {
        match resident && super::fused_graph::decodes(rows) {
            true => &[Self::Whole, Self::Resident],
            false => Self::served(split, rows),
        }
    }

    /// The stage whose place in a run this one's graph takes: the resident
    /// form takes its whole block's.
    pub(crate) fn place(self) -> Self {
        match self {
            Self::Resident => Self::Whole,
            stage => stage,
        }
    }
}

impl OnceStage {
    /// The stages sealed once of a block whose prompt rows `split` and
    /// whose experts are `streamed`.
    pub(crate) fn sealed(streamed: bool, split: bool) -> &'static [Self] {
        match (split, streamed) {
            (true, true) => &[Self::Fill, Self::Pass],
            (true, false) => &[Self::Pass],
            (false, _) => &[],
        }
    }
}

/// What a split layer's front passes its back beside the residual
/// ([`Stage`]): its rows' routes, their weights (the fused form's scores),
/// the experts' input (the normalized rows, a latent operator's projected
/// down) and the fused form's shared-expert coefficient. Each lies in the
/// run's chain place's rows of a view of one of the load's crossing
/// buffers at the layer's extents ([`SplitValues`]), which the front
/// writes and the back reads.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Crossing<T> {
    pub routes: T,
    pub weights: T,
    pub normalized: T,
    pub coefficient: Option<T>,
}

/// What a merge set's pass reads of its rows' crossing ([`Crossing::pass`]):
/// their routes and the experts' input.
#[derive(Clone)]
pub(crate) struct PassCrossing<T> {
    pub routes: T,
    pub normalized: T,
}

/// What a back reads of its rows' crossing ([`Crossing::back`]): their
/// weights, and the fused form's shared expert's input.
#[derive(Clone)]
pub(crate) struct BackCrossing<T> {
    pub weights: T,
    pub shared: Option<SharedInput<T>>,
}

/// The fused form's shared expert's input, which its back combines: the
/// normalized rows and their coefficient.
#[derive(Clone)]
pub(crate) struct SharedInput<T> {
    pub normalized: T,
    pub coefficient: T,
}

/// What a group's pass at a layer leaves its places' backs, in views of the
/// load's merged buffers at the layer's extents ([`SplitValues`]): the
/// experts' rows of the blocks its grouping filled, `[B, T, H]` in the
/// experts' width for the pass's rows (`grouped_blocks`), each grouped
/// choice's position in them, `[rows, K]`, a place's in its chain region's
/// rows, and the row table, `[rows + 1]`, entry `i` holding `i`, of which a
/// pass reads its count and first row ([`PassPorts`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Merged<T> {
    pub experts: T,
    pub inverse: T,
    pub table: T,
}

impl Merged<Extents<Dim>> {
    /// The merged values of a layer of `form` and `layer` over a pass of
    /// `rows` rows.
    pub(crate) fn at(form: &SplitForm, layer: &SplitLayer, rows: &Dim) -> Result<Self, String> {
        let blocks = super::fused_graph::grouped_blocks(rows, &layer.experts, &layer.selected)?;
        Ok(Merged {
            experts: (form.activation, vec![blocks, Dim::from(super::fused_graph::TILE_ROWS), layer.width.clone()]),
            inverse: (Element::i32(), vec![rows.clone(), layer.selected.clone()]),
            table: (Element::i32(), vec![rows + 1]),
        })
    }

    /// The expert tiles the merged experts hold.
    fn blocks(&self) -> Dim {
        self.experts.1[0].clone()
    }
}

impl Merged<Extents> {
    /// The merged values of `routed` over a pass of `rows` rows
    /// ([`Merged::at`]).
    pub(crate) fn of(routed: &RoutedFfn, geometry: &Decoder, rows: u64) -> Result<Self, String> {
        let (form, layer) = split(routed, geometry);
        let merged = Merged::at(&form, &constant(layer), &Dim::from(rows))?;
        Ok(Merged { experts: fixed(merged.experts), inverse: fixed(merged.inverse), table: fixed(merged.table) })
    }
}

/// Constant extents' values.
fn fixed((element, extents): Extents<Dim>) -> Extents {
    (element, extents.iter().map(Dim::fixed).collect())
}

/// The ports of a merge set's pass (`OnceStage::Pass`): the crossed rows it
/// groups, over the pass's rows; the merged experts it writes and its rows'
/// positions in them; and the count and first row it groups, one entry of
/// the row table each. The merged experts' tiles are its graph's
/// ([`PassPorts::of`]): ports outlive their graph's dimensions.
#[derive(Clone)]
pub(crate) struct PassPorts {
    pub crossing: PassCrossing<NativePort>,
    pub experts: NativePort,
    pub inverse: NativePort,
    pub rows: NativePort,
    pub first: NativePort,
}

/// The ports of a back after its merge set's pass (`RunStage::Combine`): what
/// it reads of its rows' crossing; the merged experts; and its own rows'
/// positions in them.
#[derive(Clone)]
pub(crate) struct BackPorts {
    pub crossing: BackCrossing<NativePort>,
    pub experts: NativePort,
    pub inverse: NativePort,
}

impl BackPorts {
    /// The pass the back combines from, of `blocks` expert tiles.
    pub(crate) fn pass(&self, blocks: &Dim) -> super::grouped::ExpertsPass {
        super::grouped::ExpertsPass {
            output: self.experts.tensor().clone(),
            inverse: self.inverse.tensor().clone(),
            blocks: blocks.clone(),
        }
    }
}

/// The ports of a split layer's front or back ([`Stage`]): a front writes
/// every crossed value.
#[derive(Clone)]
pub(crate) enum SplitPorts {
    Front(Crossing<NativePort>),
    Back(BackPorts),
}

/// A split layer's graph ([`Stage`]): its front, which routes into the
/// crossing's ports and requests where scratch-served; or its back after
/// its merge set's pass (`fused_graph::group_pass`, `graph::general_pass`),
/// which combines its rows from the merged ports, of the expert tiles its
/// graph's merged experts hold.
pub(crate) enum Split<'c> {
    Front(&'c mut Crossing<NativePort>),
    Combine(&'c BackPorts, &'c Dim),
}

/// A split layer's ports as its graph is built: its stage's ports
/// ([`SplitPorts`], which the graph keeps), with a back's merged experts'
/// tiles.
pub(crate) enum SplitBuild {
    Front(Crossing<NativePort>),
    Back(BackPorts, Dim),
}

impl SplitBuild {
    pub(crate) fn split(&mut self) -> Split<'_> {
        match self {
            Self::Front(crossing) => Split::Front(crossing),
            Self::Back(back, blocks) => Split::Combine(back, blocks),
        }
    }

    /// The ports its graph keeps.
    pub(crate) fn into_ports(self) -> SplitPorts {
        match self {
            Self::Front(crossing) => SplitPorts::Front(crossing),
            Self::Back(back, _) => SplitPorts::Back(back),
        }
    }
}

/// A crossed value's port in `graph`, its rows the graph class's.
fn crossed<G: GraphDraft>(graph: &mut G, (element, extents): &Extents<Dim>) -> Result<NativePort, GraphError> {
    graph.port_with_class_extent(*element, extents, 0, "M")
}

impl SplitPorts {
    /// The ports in `graph` of a run's graph of `stage` over `rows` rows of a
    /// block with a routed layer of `split` ([`split`]), whose passes merge
    /// `capacity` rows: none of a whole block's.
    pub(crate) fn of<G: GraphDraft>(
        graph: &mut G,
        stage: RunStage,
        split: Option<(&SplitForm, &SplitLayer)>,
        rows: &Dim,
        capacity: &Dim,
    ) -> Result<Option<SplitBuild>, GraphError> {
        let routed = || split.ok_or("a split block without a routed layer");
        Ok(match stage {
            RunStage::Whole | RunStage::Resident => None,
            RunStage::Front => {
                let (form, layer) = routed()?;
                let crossing = Crossing::at(form, layer, rows).try_map(|value| crossed(graph, value))?;
                Some(SplitBuild::Front(crossing))
            }
            RunStage::Combine => {
                let (form, layer) = routed()?;
                let crossing = Crossing::at(form, layer, rows).back().try_map(|value| crossed(graph, value))?;
                let merged = Merged::at(form, layer, capacity)?;
                let experts = graph.port(merged.experts.0, &merged.experts.1)?;
                let (element, selected) = (merged.inverse.0, merged.inverse.1[1].clone());
                let inverse = graph.port_with_class_extent(element, &[rows.clone(), selected], 0, "M")?;
                Some(SplitBuild::Back(BackPorts { crossing, experts, inverse }, merged.blocks()))
            }
        })
    }
}

impl PassPorts {
    /// The ports in `graph` of a merge set's pass over `rows` rows of a
    /// routed layer of `form` and `layer` ([`split`]), merging `capacity`
    /// rows, and the expert tiles their merged experts hold.
    pub(crate) fn of<G: GraphDraft>(
        graph: &mut G,
        form: &SplitForm,
        layer: &SplitLayer,
        rows: &Dim,
        capacity: &Dim,
    ) -> Result<(Self, Dim), GraphError> {
        let crossing = Crossing::at(form, layer, rows).pass().try_map(|value| crossed(graph, value))?;
        let merged = Merged::at(form, layer, capacity)?;
        let ports = Self {
            crossing,
            experts: graph.port(merged.experts.0, &merged.experts.1)?,
            inverse: graph.port(merged.inverse.0, &merged.inverse.1)?,
            rows: graph.port(Element::i32(), &[1u64])?,
            first: graph.port(Element::i32(), &[1u64])?,
        };
        Ok((ports, merged.blocks()))
    }
}

impl<T> Crossing<T> {
    pub(crate) fn iter(&self) -> impl Iterator<Item = &T> {
        [&self.routes, &self.weights, &self.normalized]
            .into_iter()
            .chain(&self.coefficient)
    }

    pub(crate) fn try_map<U, E>(&self, mut map: impl FnMut(&T) -> Result<U, E>) -> Result<Crossing<U>, E> {
        Ok(Crossing {
            routes: map(&self.routes)?,
            weights: map(&self.weights)?,
            normalized: map(&self.normalized)?,
            coefficient: self.coefficient.as_ref().map(map).transpose()?,
        })
    }

    /// What a pass reads of these values.
    pub(crate) fn pass(&self) -> PassCrossing<&T> {
        PassCrossing { routes: &self.routes, normalized: &self.normalized }
    }

    /// What a back reads of these values: the fused form's (which crosses a
    /// coefficient) also its shared expert's input.
    pub(crate) fn back(&self) -> BackCrossing<&T> {
        BackCrossing {
            weights: &self.weights,
            shared: self
                .coefficient
                .as_ref()
                .map(|coefficient| SharedInput { normalized: &self.normalized, coefficient }),
        }
    }
}

impl<T> PassCrossing<T> {
    pub(crate) fn iter(&self) -> impl Iterator<Item = &T> {
        [&self.routes, &self.normalized].into_iter()
    }

    pub(crate) fn try_map<U, E>(&self, mut map: impl FnMut(&T) -> Result<U, E>) -> Result<PassCrossing<U>, E> {
        Ok(PassCrossing { routes: map(&self.routes)?, normalized: map(&self.normalized)? })
    }
}

impl<T> BackCrossing<T> {
    pub(crate) fn iter(&self) -> impl Iterator<Item = &T> {
        std::iter::once(&self.weights)
            .chain(self.shared.iter().flat_map(|shared| [&shared.normalized, &shared.coefficient]))
    }

    pub(crate) fn try_map<U, E>(&self, mut map: impl FnMut(&T) -> Result<U, E>) -> Result<BackCrossing<U>, E> {
        Ok(BackCrossing {
            weights: map(&self.weights)?,
            shared: self
                .shared
                .as_ref()
                .map(|shared| Ok(SharedInput { normalized: map(&shared.normalized)?, coefficient: map(&shared.coefficient)? }))
                .transpose()?,
        })
    }
}

/// A value's element and extents.
pub(crate) type Extents<D = u64> = (Element, Vec<D>);

/// What a routed layer's prompt rows cross and merge ([`Crossing`],
/// [`Merged`]). A group's passes run one layer after another, so no two
/// layers' values are live at once: the load keeps one buffer of each
/// value, as long as the longest any routed layer takes
/// ([`SplitValues::widest`]), and each layer's stages bind its leading
/// elements at the layer's own extents ([`SplitValues::view`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SplitValues<T> {
    pub crossing: Crossing<T>,
    pub merged: Merged<T>,
}

impl SplitValues<Extents> {
    /// The values `routed` crosses over a chain of `chain_rows` rows and
    /// merges by passes of `pass_rows`: each pass's backs combine from the
    /// merged experts' rows before the next pass writes them, and each row's
    /// positions stay in its chain region's rows.
    pub(crate) fn of(routed: &RoutedFfn, geometry: &Decoder, chain_rows: u64, pass_rows: u64) -> Result<Self, String> {
        let crossing = Crossing::of(routed, geometry, chain_rows);
        let merged = Merged::of(routed, geometry, pass_rows)?;
        Ok(Self {
            merged: Merged { inverse: (merged.inverse.0, vec![chain_rows, routed.selected]), ..merged },
            crossing,
        })
    }

    /// These values over a group of `rows` rows: the crossing and the merged
    /// positions are a group's, a row each; the merged experts and row table
    /// are the pass's (`of`).
    pub(crate) fn at_rows(&self, rows: u64) -> Self {
        let at = |(element, extents): &Extents| {
            (*element, std::iter::once(rows).chain(extents[1..].iter().copied()).collect())
        };
        Self {
            crossing: Crossing {
                routes: at(&self.crossing.routes),
                weights: at(&self.crossing.weights),
                normalized: at(&self.crossing.normalized),
                coefficient: self.crossing.coefficient.as_ref().map(at),
            },
            merged: Merged {
                inverse: at(&self.merged.inverse),
                ..self.merged.clone()
            },
        }
    }

    /// The load's buffers of the values of `first` and `rest`, its routed
    /// layers: each value's element, in one dimension as long as the
    /// longest any layer's.
    pub(crate) fn widest(first: &Self, rest: &[Self]) -> Self {
        let length = |(element, extents): &Extents| (*element, extents.iter().product::<u64>());
        let widest = |value: fn(&Self) -> &Extents| {
            let (element, longest) = rest
                .iter()
                .map(|layer| length(value(layer)))
                .fold(length(value(first)), |(element, longest), (_, length)| (element, longest.max(length)));
            (element, vec![longest])
        };
        let coefficient = std::iter::once(first)
            .chain(rest)
            .filter_map(|layer| layer.crossing.coefficient.as_ref().map(length))
            .reduce(|(element, longest), (_, length)| (element, longest.max(length)))
            .map(|(element, longest)| (element, vec![longest]));
        Self {
            crossing: Crossing {
                routes: widest(|layer| &layer.crossing.routes),
                weights: widest(|layer| &layer.crossing.weights),
                normalized: widest(|layer| &layer.crossing.normalized),
                coefficient,
            },
            merged: Merged {
                experts: widest(|layer| &layer.merged.experts),
                inverse: widest(|layer| &layer.merged.inverse),
                table: widest(|layer| &layer.merged.table),
            },
        }
    }
}

impl SplitValues<seismic::Tensor> {
    /// A layer's views of the load's buffers (`widest`): each buffer's
    /// leading elements at the layer's extents.
    pub(crate) fn view(&self, layer: &SplitValues<Extents>) -> Result<Self, String> {
        let view = |buffer: &seismic::Tensor, (_, extents): &Extents| {
            buffer
                .slice_leading(0, extents.iter().product())
                .and_then(|leading| leading.reshape(extents))
                .map_err(|error| format!("a layer's view of a split buffer: {error}"))
        };
        let optional = |buffer: &Option<seismic::Tensor>, value: &Option<Extents>| {
            value
                .as_ref()
                .map(|value| view(buffer.as_ref().ok_or("a layer's split value has no buffer")?, value))
                .transpose()
        };
        Ok(Self {
            crossing: Crossing {
                routes: view(&self.crossing.routes, &layer.crossing.routes)?,
                weights: view(&self.crossing.weights, &layer.crossing.weights)?,
                normalized: view(&self.crossing.normalized, &layer.crossing.normalized)?,
                coefficient: optional(&self.crossing.coefficient, &layer.crossing.coefficient)?,
            },
            merged: Merged {
                experts: view(&self.merged.experts, &layer.merged.experts)?,
                inverse: view(&self.merged.inverse, &layer.merged.inverse)?,
                table: view(&self.merged.table, &layer.merged.table)?,
            },
        })
    }
}

impl Crossing<Extents<Dim>> {
    /// The element and extents of each value a layer of `form` and `layer`
    /// crosses over `rows` rows: `[rows, K]` routes and weights, the
    /// experts' `[rows, H]` input in the decoder's activation (in the latent
    /// width where latent), and the fused form's `[rows]` coefficient.
    pub(crate) fn at(form: &SplitForm, layer: &SplitLayer, rows: &Dim) -> Self {
        let choices = vec![rows.clone(), layer.selected.clone()];
        Crossing {
            routes: (Element::i32(), choices.clone()),
            weights: (Element::f32(), choices),
            normalized: (form.activation, vec![rows.clone(), layer.width.clone()]),
            coefficient: form.fused.then(|| (Element::f32(), vec![rows.clone()])),
        }
    }
}

impl Crossing<Extents> {
    /// The values `routed` crosses over `rows` rows ([`Crossing::at`]).
    pub(crate) fn of(routed: &RoutedFfn, geometry: &Decoder, rows: u64) -> Self {
        let (form, layer) = split(routed, geometry);
        let crossing = Crossing::at(&form, &constant(layer), &Dim::from(rows));
        Crossing {
            routes: fixed(crossing.routes),
            weights: fixed(crossing.weights),
            normalized: fixed(crossing.normalized),
            coefficient: crossing.coefficient.map(fixed),
        }
    }
}
