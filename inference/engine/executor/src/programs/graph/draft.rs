//! One graph topology can be built from prepared entries or checked metadata.
//! The entry binding is the only route-specific input; Seismic owns shapes,
//! storage layout and interval placement in either route.

use super::GraphError;
use super::sizing::{AxisProbeDraft, SizedCall, SizingDraft, SizingGroupDraft, SizingOutline};
use crate::native::RangedKernels;
use seismic::{
    BackendName, Dim, Element, EntryContract, KernelRequest, LaunchEncoding, NativeGraph,
    NativeGraphMetadata, NativeGraphPlan, NativeGraphStorageBytes, NativeKernel, NativePort,
    NativeTimelinePort, OperandCarry, OperandForms, WorkflowTensor,
};

/// How a block's graph takes its launch's per-step controls (coordinates,
/// visible spans, fresh rows and destinations; slot and bank words).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum ControlSource {
    /// As inputs it uploads itself: a graph that alone reads them in its
    /// launch.
    Uploaded,
    /// Bound to the launch's one upload of them, which every block graph of
    /// the launch binds.
    Bound,
}

pub(crate) trait GraphDraft: Sized {
    type Plan;
    type Binding<'a, E: EntryContract + 'a>: Copy
    where
        Self: 'a;
    /// An entry's bindings by the range of rows a node's launch takes
    /// ([`RangedKernels`]).
    type Ranged<'a, E: EntryContract + 'a>: Copy
    where
        Self: 'a;
    /// The forms the graph names its operand carries in
    /// (`seismic::OperandForms`), those their readers' kernels take: a
    /// graph of kernels names them, and its seal writes them; a graph built
    /// to be sized names none, a carry's ports and everything a graph
    /// reserves for it being the same in each.
    type Forms: Copy;

    /// The binding of the range `rows` lie in.
    fn pick<'a, E: EntryContract + 'a>(ranged: Self::Ranged<'a, E>, rows: &Dim) -> Self::Binding<'a, E>;
    /// The binding of the lowest range, whose layout every range shares.
    fn first<'a, E: EntryContract + 'a>(ranged: Self::Ranged<'a, E>) -> Self::Binding<'a, E>;

    fn port<D: Clone>(&mut self, element: Element, extents: &[D]) -> Result<NativePort, GraphError>
    where
        Dim: From<D>;
    /// Declare an external weight value port (tagged for assessment's decode
    /// demand census). Default delegates to [`Self::port`].
    fn weight_port<D: Clone>(&mut self, element: Element, extents: &[D]) -> Result<NativePort, GraphError>
    where
        Dim: From<D>,
    {
        self.port(element, extents)
    }
    fn local_for<'a, E: EntryContract + 'a>(
        &mut self,
        entry: Self::Binding<'a, E>,
        parameter: &str,
        dimensions: &[(&str, impl Clone + Into<Dim>)],
    ) -> Result<NativePort, GraphError>;
    fn input_for<'a, E: EntryContract + 'a>(
        &mut self,
        entry: Self::Binding<'a, E>,
        parameter: &str,
        dimensions: &[(&str, impl Clone + Into<Dim>)],
    ) -> Result<NativePort, GraphError>;
    /// An external port in the layout `entry`'s `parameter` takes at
    /// `dimensions`, which each run binds.
    fn port_for<'a, E: EntryContract + 'a>(
        &mut self,
        entry: Self::Binding<'a, E>,
        parameter: &str,
        dimensions: &[(&str, impl Clone + Into<Dim>)],
    ) -> Result<NativePort, GraphError>;
    /// A launch control `entry`'s `parameter` reads at `dimensions`, as
    /// `source` gives it.
    fn control_for<'a, E: EntryContract + 'a>(
        &mut self,
        source: ControlSource,
        entry: Self::Binding<'a, E>,
        parameter: &str,
        dimensions: &[(&str, impl Clone + Into<Dim>)],
    ) -> Result<NativePort, GraphError> {
        match source {
            ControlSource::Uploaded => self.input_for(entry, parameter, dimensions),
            ControlSource::Bound => self.port_for(entry, parameter, dimensions),
        }
    }
    fn enqueue<'a, E: EntryContract + 'a>(
        &mut self,
        entry: Self::Binding<'a, E>,
        dimensions: &[(&str, impl Clone + Into<Dim>)],
        args: E::WorkflowArgs<'_>,
    ) -> Result<E::WorkflowResults, GraphError>;
    /// [`Self::enqueue`] under word `index` of the host-written port `port`
    /// (a graph-scope tensor no kernel of the node need bind): its launches
    /// are encoded only while the word is nonzero (`WhileSet`) or zero
    /// (`WhileClear`), as the host reads it when a run is queued. Planning
    /// holds the node whether or not a run encodes it.
    fn enqueue_under<'a, E: EntryContract + 'a>(
        &mut self,
        entry: Self::Binding<'a, E>,
        dimensions: &[(&str, impl Clone + Into<Dim>)],
        args: E::WorkflowArgs<'_>,
        port: &NativePort,
        index: u64,
        encoded: LaunchEncoding,
    ) -> Result<E::WorkflowResults, GraphError>;
    fn export(&mut self, result: &WorkflowTensor) -> Result<(), GraphError>;
    /// Place node result `other` where node result `primary` lies, one
    /// storage for both, readers reading `primary`: their nodes are
    /// conditioned on one word under opposite encodings
    /// ([`Self::enqueue_under`]), which sealing proves.
    fn share(&mut self, other: &WorkflowTensor, primary: &WorkflowTensor) -> Result<(), GraphError>;
    /// Write the node result `result` into the tensor each run binds to the
    /// external port `port`, which describes exactly that result.
    fn store(&mut self, result: &WorkflowTensor, port: &NativePort) -> Result<(), GraphError>;
    /// The ports of an operand carry between two graphs
    /// (`seismic::OperandCarry`): a formed operand of `rows` rows of
    /// `width` values, rounded to `activation`, which each run binds.
    fn operand_carry(&mut self, rows: u64, width: &Dim, activation: Element) -> Result<OperandCarry, GraphError>;
    /// The producer of `result` emits its normed operand into `carry`'s
    /// ports, in `forms`.
    fn carry_out(
        &mut self,
        result: &WorkflowTensor,
        carry: &OperandCarry,
        forms: Self::Forms,
    ) -> Result<(), GraphError>;
    /// The consumers of the tensor bound to `hidden` that form their
    /// operand under the norm bound to `norm` at `epsilon` read it from
    /// `carry`, which holds it in `forms`, their formation left out.
    fn carry_in(
        &mut self,
        hidden: &NativePort,
        carry: &OperandCarry,
        forms: Self::Forms,
        norm: &NativePort,
        epsilon: f32,
    ) -> Result<(), GraphError>;
    /// A timeline node after every node enqueued so far: a host gate or a
    /// signal to the host (`seismic::Timeline`).
    fn timeline<N: seismic::timeline::TimelineNodeKind>(&mut self, node: N) -> NativeTimelinePort<N::Role>;
    /// A span mark after every node enqueued so far, its ordinal returned:
    /// where a sampled run stamps the device's time (`seismic::NativeSpans`),
    /// dividing no graph. Marked alike on a graph and its metadata, so every
    /// plan of a layout marks alike.
    fn span_mark(&mut self) -> usize;
    fn seal(self) -> Result<Self::Plan, GraphError>;
}

/// A prepared graph's failure: its kernels are formed, so any error is an
/// engine defect.
fn invalid(error: impl std::fmt::Display) -> GraphError {
    GraphError::Invalid(error.to_string())
}

/// A prepared graph's extents: constants, as no recording reaches a
/// prepared graph ([`Dim::fixed`]).
fn fixed_extents<D: Clone>(extents: &[D]) -> Vec<u64>
where
    Dim: From<D>,
{
    extents.iter().map(|extent| Dim::from(extent.clone()).fixed()).collect()
}

/// A prepared graph's named dimensions: constants ([`fixed_extents`]).
fn fixed_dimensions<'n>(dimensions: &[(&'n str, impl Clone + Into<Dim>)]) -> Vec<(&'n str, u64)> {
    dimensions
        .iter()
        .map(|(name, dimension)| (*name, dimension.clone().into().fixed()))
        .collect()
}

impl GraphDraft for NativeGraph {
    type Plan = NativeGraphPlan;
    type Binding<'a, E: EntryContract + 'a> = &'a NativeKernel<E>;
    type Ranged<'a, E: EntryContract + 'a> = &'a RangedKernels<E>;
    type Forms = OperandForms;

    fn pick<'a, E: EntryContract + 'a>(ranged: Self::Ranged<'a, E>, rows: &Dim) -> Self::Binding<'a, E> {
        ranged.for_rows(rows)
    }
    fn first<'a, E: EntryContract + 'a>(ranged: Self::Ranged<'a, E>) -> Self::Binding<'a, E> {
        ranged.first()
    }

    fn port<D: Clone>(&mut self, element: Element, extents: &[D]) -> Result<NativePort, GraphError>
    where
        Dim: From<D>,
    {
        self.port(element, &fixed_extents(extents)).map_err(invalid)
    }
    fn local_for<'a, E: EntryContract + 'a>(
        &mut self,
        entry: Self::Binding<'a, E>,
        parameter: &str,
        dimensions: &[(&str, impl Clone + Into<Dim>)],
    ) -> Result<NativePort, GraphError> {
        self.local_for(entry, parameter, &fixed_dimensions(dimensions))
            .map_err(invalid)
    }
    fn input_for<'a, E: EntryContract + 'a>(
        &mut self,
        entry: Self::Binding<'a, E>,
        parameter: &str,
        dimensions: &[(&str, impl Clone + Into<Dim>)],
    ) -> Result<NativePort, GraphError> {
        self.input_for(entry, parameter, &fixed_dimensions(dimensions))
            .map_err(invalid)
    }
    fn port_for<'a, E: EntryContract + 'a>(
        &mut self,
        entry: Self::Binding<'a, E>,
        parameter: &str,
        dimensions: &[(&str, impl Clone + Into<Dim>)],
    ) -> Result<NativePort, GraphError> {
        self.port_for(entry, parameter, &fixed_dimensions(dimensions))
            .map_err(invalid)
    }
    fn enqueue<'a, E: EntryContract + 'a>(
        &mut self,
        entry: Self::Binding<'a, E>,
        _dimensions: &[(&str, impl Clone + Into<Dim>)],
        args: E::WorkflowArgs<'_>,
    ) -> Result<E::WorkflowResults, GraphError> {
        self.enqueue(entry, args).map_err(invalid)
    }
    fn enqueue_under<'a, E: EntryContract + 'a>(
        &mut self,
        entry: Self::Binding<'a, E>,
        _dimensions: &[(&str, impl Clone + Into<Dim>)],
        args: E::WorkflowArgs<'_>,
        port: &NativePort,
        index: u64,
        encoded: LaunchEncoding,
    ) -> Result<E::WorkflowResults, GraphError> {
        self.enqueue_under(entry, args, port, index, encoded).map_err(invalid)
    }
    fn export(&mut self, result: &WorkflowTensor) -> Result<(), GraphError> {
        self.export(result).map_err(invalid)
    }
    fn share(&mut self, other: &WorkflowTensor, primary: &WorkflowTensor) -> Result<(), GraphError> {
        self.share(other, primary).map_err(invalid)
    }
    fn store(&mut self, result: &WorkflowTensor, port: &NativePort) -> Result<(), GraphError> {
        self.store(result, port).map_err(invalid)
    }
    fn operand_carry(&mut self, rows: u64, width: &Dim, activation: Element) -> Result<OperandCarry, GraphError> {
        self.operand_carry(rows, width.fixed(), activation).map_err(invalid)
    }
    fn carry_out(
        &mut self,
        result: &WorkflowTensor,
        carry: &OperandCarry,
        forms: OperandForms,
    ) -> Result<(), GraphError> {
        self.carry_out(result, carry, forms).map_err(invalid)
    }
    fn carry_in(
        &mut self,
        hidden: &NativePort,
        carry: &OperandCarry,
        forms: OperandForms,
        norm: &NativePort,
        epsilon: f32,
    ) -> Result<(), GraphError> {
        self.carry_in(hidden, carry, forms, norm, epsilon).map_err(invalid)
    }
    fn timeline<N: seismic::timeline::TimelineNodeKind>(&mut self, node: N) -> NativeTimelinePort<N::Role> {
        self.timeline(node)
    }
    fn span_mark(&mut self) -> usize {
        self.span_mark()
    }
    fn seal(self) -> Result<Self::Plan, GraphError> {
        self.seal().map_err(invalid)
    }
}

impl GraphDraft for NativeGraphMetadata {
    type Plan = NativeGraphStorageBytes;
    type Binding<'a, E: EntryContract + 'a> = &'a [(&'a str, Element)];
    type Ranged<'a, E: EntryContract + 'a> = &'a [(&'a str, Element)];
    type Forms = ();

    fn pick<'a, E: EntryContract + 'a>(ranged: Self::Ranged<'a, E>, _rows: &Dim) -> Self::Binding<'a, E> {
        ranged
    }
    fn first<'a, E: EntryContract + 'a>(ranged: Self::Ranged<'a, E>) -> Self::Binding<'a, E> {
        ranged
    }

    fn port<D: Clone>(&mut self, element: Element, extents: &[D]) -> Result<NativePort, GraphError>
    where
        Dim: From<D>,
    {
        Ok(self.port(element, extents)?)
    }
    fn local_for<'a, E: EntryContract + 'a>(
        &mut self,
        entry: Self::Binding<'a, E>,
        parameter: &str,
        dimensions: &[(&str, impl Clone + Into<Dim>)],
    ) -> Result<NativePort, GraphError> {
        Ok(self.local_for::<E>(entry, parameter, dimensions)?)
    }
    fn input_for<'a, E: EntryContract + 'a>(
        &mut self,
        entry: Self::Binding<'a, E>,
        parameter: &str,
        dimensions: &[(&str, impl Clone + Into<Dim>)],
    ) -> Result<NativePort, GraphError> {
        Ok(self.input_for::<E>(entry, parameter, dimensions)?)
    }
    fn port_for<'a, E: EntryContract + 'a>(
        &mut self,
        entry: Self::Binding<'a, E>,
        parameter: &str,
        dimensions: &[(&str, impl Clone + Into<Dim>)],
    ) -> Result<NativePort, GraphError> {
        Ok(self.port_for::<E>(entry, parameter, dimensions)?)
    }
    fn enqueue<'a, E: EntryContract + 'a>(
        &mut self,
        entry: Self::Binding<'a, E>,
        dimensions: &[(&str, impl Clone + Into<Dim>)],
        args: E::WorkflowArgs<'_>,
    ) -> Result<E::WorkflowResults, GraphError> {
        Ok(self.enqueue::<E>(entry, dimensions, args)?)
    }
    /// Planning holds a conditioned node as any other (its storage is the
    /// graph's whether or not a run encodes it), its condition kept for the
    /// proof that results may share storage.
    fn enqueue_under<'a, E: EntryContract + 'a>(
        &mut self,
        entry: Self::Binding<'a, E>,
        dimensions: &[(&str, impl Clone + Into<Dim>)],
        args: E::WorkflowArgs<'_>,
        port: &NativePort,
        index: u64,
        encoded: LaunchEncoding,
    ) -> Result<E::WorkflowResults, GraphError> {
        Ok(self.enqueue_under::<E>(entry, dimensions, args, port, index, encoded)?)
    }
    fn export(&mut self, result: &WorkflowTensor) -> Result<(), GraphError> {
        Ok(self.export(result)?)
    }
    fn share(&mut self, other: &WorkflowTensor, primary: &WorkflowTensor) -> Result<(), GraphError> {
        Ok(self.share(other, primary)?)
    }
    fn store(&mut self, result: &WorkflowTensor, port: &NativePort) -> Result<(), GraphError> {
        Ok(self.store(result, port)?)
    }
    fn operand_carry(&mut self, rows: u64, width: &Dim, activation: Element) -> Result<OperandCarry, GraphError> {
        Ok(self.operand_carry(Dim::from(rows), width.clone(), activation)?)
    }
    fn carry_out(&mut self, result: &WorkflowTensor, carry: &OperandCarry, (): ()) -> Result<(), GraphError> {
        Ok(self.carry_out(result, carry)?)
    }
    fn carry_in(
        &mut self,
        hidden: &NativePort,
        carry: &OperandCarry,
        (): (),
        norm: &NativePort,
        epsilon: f32,
    ) -> Result<(), GraphError> {
        Ok(self.carry_in(hidden, carry, norm, epsilon)?)
    }
    fn timeline<N: seismic::timeline::TimelineNodeKind>(&mut self, node: N) -> NativeTimelinePort<N::Role> {
        self.timeline(node)
    }
    fn span_mark(&mut self) -> usize {
        self.span_mark()
    }
    fn seal(self) -> Result<Self::Plan, GraphError> {
        Ok(self.seal()?)
    }
}

/// A [`GraphDraft`] whose entry bindings are element slices (`&[(&str, Element)]`)
/// and whose operand carry forms are `()`: implemented by both
/// [`NativeGraphMetadata`] (load layout certification) and [`SizingDraft`]
/// (direct contract sizing for assessment).
pub(crate) trait ElementBoundDraft: GraphDraft<Forms = ()> {
    fn bind<'a, E: EntryContract + 'a>(
        elements: &'a [(&'a str, Element)],
    ) -> Self::Binding<'a, E>;
    fn bind_ranged<'a, E: EntryContract + 'a>(
        elements: &'a [(&'a str, Element)],
    ) -> Self::Ranged<'a, E>;
}

impl ElementBoundDraft for NativeGraphMetadata {
    fn bind<'a, E: EntryContract + 'a>(
        elements: &'a [(&'a str, Element)],
    ) -> Self::Binding<'a, E> {
        elements
    }
    fn bind_ranged<'a, E: EntryContract + 'a>(
        elements: &'a [(&'a str, Element)],
    ) -> Self::Ranged<'a, E> {
        elements
    }
}

impl ElementBoundDraft for SizingDraft {
    fn bind<'a, E: EntryContract + 'a>(
        elements: &'a [(&'a str, Element)],
    ) -> Self::Binding<'a, E> {
        elements
    }
    fn bind_ranged<'a, E: EntryContract + 'a>(
        elements: &'a [(&'a str, Element)],
    ) -> Self::Ranged<'a, E> {
        elements
    }
}

impl ElementBoundDraft for SizingGroupDraft {
    fn bind<'a, E: EntryContract + 'a>(
        elements: &'a [(&'a str, Element)],
    ) -> Self::Binding<'a, E> {
        elements
    }
    fn bind_ranged<'a, E: EntryContract + 'a>(
        elements: &'a [(&'a str, Element)],
    ) -> Self::Ranged<'a, E> {
        elements
    }
}

impl ElementBoundDraft for AxisProbeDraft {
    fn bind<'a, E: EntryContract + 'a>(
        elements: &'a [(&'a str, Element)],
    ) -> Self::Binding<'a, E> {
        elements
    }
    fn bind_ranged<'a, E: EntryContract + 'a>(
        elements: &'a [(&'a str, Element)],
    ) -> Self::Ranged<'a, E> {
        elements
    }
}

/// What a structural regime's concrete classes certify to: the layout a load
/// seals them into ([`seismic::NativeGraphLayout`]), or the direct
/// contract-sized outline ([`SizingOutline`]) that assessment charges. Every
/// family certifies a group by evaluating its concrete classes through one
/// builder closure and taking the element-wise maximum over buffers and active
/// launches — no scope or slice declarations exist.
pub(crate) trait Certified: Clone {
    type Draft: ElementBoundDraft + 'static;

    /// Certify a single layout covering every concrete class in `classes`.
    /// `build` is called once per class; the first class's auxiliary value is
    /// returned alongside the certified layout. Topology divergence within the
    /// group is rejected as an invariant error.
    fn certify_group<C: Copy, Aux>(
        backend: BackendName,
        classes: &[C],
        build: impl FnMut(Self::Draft, C) -> Result<(Self::Draft, Aux), GraphError>,
    ) -> Result<(Self, Vec<Aux>), GraphError>;
    fn storage_bytes(&self) -> NativeGraphStorageBytes;
    fn plan_heap(&self) -> seismic::HostHeap;
    fn topology_heap(&self) -> seismic::HostHeap;
    fn layout_identity(&self) -> u64;
    fn bound_heap(&self) -> seismic::HostHeap;
    fn sizing_census(&self) -> Option<(&[SizedCall], &[KernelRequest], usize)> {
        None
    }
}

impl Certified for seismic::NativeGraphLayout {
    type Draft = NativeGraphMetadata;

    fn certify_group<C: Copy, Aux>(
        backend: BackendName,
        classes: &[C],
        mut build: impl FnMut(Self::Draft, C) -> Result<(Self::Draft, Aux), GraphError>,
    ) -> Result<(Self, Vec<Aux>), GraphError> {
        if classes.is_empty() {
            return Err(GraphError::Invalid("graph group has no concrete classes".to_owned()));
        }
        let mut drafts = Vec::with_capacity(classes.len());
        let mut auxes = Vec::with_capacity(classes.len());
        let device = super::sizing::current_sizing_device();
        for &class in classes {
            let (draft, aux) = build(NativeGraphMetadata::new_for_device(backend, device), class)?;
            drafts.push(draft);
            auxes.push(aux);
        }
        let layout = NativeGraphMetadata::certify_group(drafts)?;
        Ok((layout, auxes))
    }
    fn storage_bytes(&self) -> NativeGraphStorageBytes {
        self.storage_bytes()
    }
    fn plan_heap(&self) -> seismic::HostHeap {
        self.plan_heap()
    }
    fn topology_heap(&self) -> seismic::HostHeap {
        self.topology_heap()
    }
    fn layout_identity(&self) -> u64 {
        self.identity()
    }
    fn bound_heap(&self) -> seismic::HostHeap {
        self.bound_heap()
    }
}

impl Certified for SizingOutline {
    type Draft = SizingGroupDraft;

    fn certify_group<C: Copy, Aux>(
        backend: BackendName,
        classes: &[C],
        mut build: impl FnMut(Self::Draft, C) -> Result<(Self::Draft, Aux), GraphError>,
    ) -> Result<(Self, Vec<Aux>), GraphError> {
        let Some((&first, rest)) = classes.split_first() else {
            return Err(GraphError::Invalid("graph group has no concrete classes".to_owned()));
        };
        let _scope = super::sizing::SizingScope::enter();
        let mut auxes = Vec::with_capacity(classes.len());
        let (first_draft, first_aux) = build(SizingGroupDraft::Full(SizingDraft::new(backend)), first)?;
        auxes.push(first_aux);
        let SizingGroupDraft::Full(first_draft) = first_draft else {
            unreachable!("first class always builds a Full SizingDraft");
        };
        let mut group = first_draft.begin_group()?;
        for &class in rest {
            let (bounds_draft, aux) = build(SizingGroupDraft::Bounds(group.next_bounds()), class)?;
            auxes.push(aux);
            let SizingGroupDraft::Bounds(bounds) = bounds_draft else {
                unreachable!("subsequent classes always build a Bounds draft");
            };
            group = bounds.into_group()?;
        }
        Ok((group.finish()?, auxes))
    }
    fn storage_bytes(&self) -> NativeGraphStorageBytes {
        self.storage_bytes()
    }
    fn plan_heap(&self) -> seismic::HostHeap {
        self.plan_heap()
    }
    fn topology_heap(&self) -> seismic::HostHeap {
        self.topology_heap()
    }
    fn layout_identity(&self) -> u64 {
        self.identity()
    }
    fn bound_heap(&self) -> seismic::HostHeap {
        self.bound_heap()
    }
    fn sizing_census(&self) -> Option<(&[SizedCall], &[KernelRequest], usize)> {
        Some((self.launches(), self.kernels(), self.timeline_count()))
    }
}
