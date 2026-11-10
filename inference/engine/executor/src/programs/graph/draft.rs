//! One graph topology can be built from prepared entries or checked metadata.
//! The entry binding is the only route-specific input; Seismic owns shapes,
//! storage layout and interval placement in either route.

use super::GraphError;
use seismic::{
    Dim, Element, Entry, LaunchEncoding, NativeGraph, NativeGraphMetadata, NativeGraphPlan, NativeGraphStorageBytes,
    NativeKernel, NativePort, NativeTimelinePort, WorkflowTensor,
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
    type Binding<'a, E: Entry + 'a>: Copy
    where
        Self: 'a;

    fn port<D: Clone>(&mut self, element: Element, extents: &[D]) -> Result<NativePort, GraphError>
    where
        Dim: From<D>;
    fn set_class_scope(&mut self, _scope: Option<&'static str>) {}
    fn port_with_class_extent<D: Clone>(
        &mut self,
        element: Element,
        extents: &[D],
        _extent_axis: usize,
        _class_dimension: &'static str,
    ) -> Result<NativePort, GraphError>
    where
        Dim: From<D>,
    {
        self.port(element, extents)
    }
    fn local_for<'a, E: Entry + 'a>(
        &mut self,
        entry: Self::Binding<'a, E>,
        parameter: &str,
        dimensions: &[(&str, impl Clone + Into<Dim>)],
    ) -> Result<NativePort, GraphError>;
    fn input_for<'a, E: Entry + 'a>(
        &mut self,
        entry: Self::Binding<'a, E>,
        parameter: &str,
        dimensions: &[(&str, impl Clone + Into<Dim>)],
    ) -> Result<NativePort, GraphError>;
    /// An external port in the layout `entry`'s `parameter` takes at
    /// `dimensions`, which each run binds.
    fn port_for<'a, E: Entry + 'a>(
        &mut self,
        entry: Self::Binding<'a, E>,
        parameter: &str,
        dimensions: &[(&str, impl Clone + Into<Dim>)],
    ) -> Result<NativePort, GraphError>;
    /// A launch control `entry`'s `parameter` reads at `dimensions`, as
    /// `source` gives it.
    fn control_for<'a, E: Entry + 'a>(
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
    fn enqueue<'a, E: Entry + 'a>(
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
    fn enqueue_under<'a, E: Entry + 'a>(
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
    type Binding<'a, E: Entry + 'a> = &'a NativeKernel<E>;

    fn port<D: Clone>(&mut self, element: Element, extents: &[D]) -> Result<NativePort, GraphError>
    where
        Dim: From<D>,
    {
        self.port(element, &fixed_extents(extents)).map_err(invalid)
    }
    fn local_for<'a, E: Entry + 'a>(
        &mut self,
        entry: Self::Binding<'a, E>,
        parameter: &str,
        dimensions: &[(&str, impl Clone + Into<Dim>)],
    ) -> Result<NativePort, GraphError> {
        self.local_for(entry, parameter, &fixed_dimensions(dimensions))
            .map_err(invalid)
    }
    fn input_for<'a, E: Entry + 'a>(
        &mut self,
        entry: Self::Binding<'a, E>,
        parameter: &str,
        dimensions: &[(&str, impl Clone + Into<Dim>)],
    ) -> Result<NativePort, GraphError> {
        self.input_for(entry, parameter, &fixed_dimensions(dimensions))
            .map_err(invalid)
    }
    fn port_for<'a, E: Entry + 'a>(
        &mut self,
        entry: Self::Binding<'a, E>,
        parameter: &str,
        dimensions: &[(&str, impl Clone + Into<Dim>)],
    ) -> Result<NativePort, GraphError> {
        self.port_for(entry, parameter, &fixed_dimensions(dimensions))
            .map_err(invalid)
    }
    fn enqueue<'a, E: Entry + 'a>(
        &mut self,
        entry: Self::Binding<'a, E>,
        _dimensions: &[(&str, impl Clone + Into<Dim>)],
        args: E::WorkflowArgs<'_>,
    ) -> Result<E::WorkflowResults, GraphError> {
        self.enqueue(entry, args).map_err(invalid)
    }
    fn enqueue_under<'a, E: Entry + 'a>(
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
    type Binding<'a, E: Entry + 'a> = &'a [(&'a str, Element)];

    fn port<D: Clone>(&mut self, element: Element, extents: &[D]) -> Result<NativePort, GraphError>
    where
        Dim: From<D>,
    {
        Ok(self.port(element, extents)?)
    }
    fn set_class_scope(&mut self, scope: Option<&'static str>) {
        self.set_class_scope(scope);
    }
    fn port_with_class_extent<D: Clone>(
        &mut self,
        element: Element,
        extents: &[D],
        extent_axis: usize,
        class_dimension: &'static str,
    ) -> Result<NativePort, GraphError>
    where
        Dim: From<D>,
    {
        Ok(self.port_with_class_extent(element, extents, extent_axis, class_dimension)?)
    }
    fn local_for<'a, E: Entry + 'a>(
        &mut self,
        entry: Self::Binding<'a, E>,
        parameter: &str,
        dimensions: &[(&str, impl Clone + Into<Dim>)],
    ) -> Result<NativePort, GraphError> {
        Ok(self.local_for::<E>(entry, parameter, dimensions)?)
    }
    fn input_for<'a, E: Entry + 'a>(
        &mut self,
        entry: Self::Binding<'a, E>,
        parameter: &str,
        dimensions: &[(&str, impl Clone + Into<Dim>)],
    ) -> Result<NativePort, GraphError> {
        Ok(self.input_for::<E>(entry, parameter, dimensions)?)
    }
    fn port_for<'a, E: Entry + 'a>(
        &mut self,
        entry: Self::Binding<'a, E>,
        parameter: &str,
        dimensions: &[(&str, impl Clone + Into<Dim>)],
    ) -> Result<NativePort, GraphError> {
        Ok(self.port_for::<E>(entry, parameter, dimensions)?)
    }
    fn enqueue<'a, E: Entry + 'a>(
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
    fn enqueue_under<'a, E: Entry + 'a>(
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

/// What a structural regime's classes certify to: the layout a load seals
/// them into ([`NativeGraphLayout`]), or exactly what that layout holds,
/// without building it ([`seismic::NativeGraphOutline`]), which the
/// assessment charges. Every family's enumeration builds either from the
/// same templates and classes.
///
/// A family on graph traces (`seismic::trace`) certifies through
/// [`Self::traced_each`] (or [`Self::traced`] for one slice set): one recorder, recorded once at the model's facts
/// for the load's layouts, and looked up in the family's traces for
/// assessment's outlines.
pub(crate) trait Certified: Clone {
    fn of(
        template: &seismic::NativeGraphResourceTemplate,
        slices: &[seismic::NativeGraphClassSlice],
    ) -> Result<Self, seismic::NativeGraphMetadataError>;
    /// What `record`'s template of topology `key` at `facts` (a model's
    /// split: its facts and its reals) certifies to over each slice set of
    /// `slice_sets`, with the builder's output at `facts`: one recording for the load, one trace lookup per set for
    /// assessment. `classes` names the facts the slices vary
    /// (`seismic::ClassFact`).
    fn traced_each<K: Clone + Eq + std::hash::Hash, X: seismic::TraceOutput>(
        traces: &seismic::NativeGraphTraces<K, X>,
        key: &K,
        facts: &super::facts::FactSink,
        classes: &[seismic::ClassFact],
        slice_sets: &[Vec<seismic::NativeGraphClassSlice>],
        record: seismic::GraphRecorder<K, X, GraphError>,
    ) -> Result<Vec<(Self, X::Value)>, GraphError>
    where
        X::Value: Clone;
    /// [`Self::traced_each`] over the one slice set `slices`.
    ///
    /// # Panics
    ///
    /// Where `traced_each` does not answer one set with one result.
    fn traced<K: Clone + Eq + std::hash::Hash, X: seismic::TraceOutput>(
        traces: &seismic::NativeGraphTraces<K, X>,
        key: &K,
        facts: &super::facts::FactSink,
        classes: &[seismic::ClassFact],
        slices: Vec<seismic::NativeGraphClassSlice>,
        record: seismic::GraphRecorder<K, X, GraphError>,
    ) -> Result<(Self, X::Value), GraphError>
    where
        X::Value: Clone,
    {
        let [certified]: [_; 1] = Self::traced_each(traces, key, facts, classes, &[slices], record)?
            .try_into()
            .unwrap_or_else(|results: Vec<_>| panic!("one slice set certified to {} results", results.len()));
        Ok(certified)
    }
    fn storage_bytes(&self) -> NativeGraphStorageBytes;
    fn plan_heap(&self) -> seismic::HostHeap;
    fn bound_heap(&self) -> seismic::HostHeap;
}

impl Certified for seismic::NativeGraphLayout {
    fn of(
        template: &seismic::NativeGraphResourceTemplate,
        slices: &[seismic::NativeGraphClassSlice],
    ) -> Result<Self, seismic::NativeGraphMetadataError> {
        template.certify(slices)
    }
    /// One recording at the model's facts: a load's graphs are its own
    /// model's.
    fn traced_each<K: Clone + Eq + std::hash::Hash, X: seismic::TraceOutput>(
        _traces: &seismic::NativeGraphTraces<K, X>,
        key: &K,
        facts: &super::facts::FactSink,
        classes: &[seismic::ClassFact],
        slice_sets: &[Vec<seismic::NativeGraphClassSlice>],
        record: seismic::GraphRecorder<K, X, GraphError>,
    ) -> Result<Vec<(Self, X::Value)>, GraphError>
    where
        X::Value: Clone,
    {
        let (layouts, output) =
            seismic::NativeGraphTraces::<K, X>::certify(
                key,
                facts.values(),
                facts.reals(),
                classes,
                slice_sets,
                record,
            )?;
        Ok(layouts.into_iter().map(|layout| (layout, output.clone())).collect())
    }
    fn storage_bytes(&self) -> NativeGraphStorageBytes {
        self.storage_bytes()
    }
    fn plan_heap(&self) -> seismic::HostHeap {
        self.plan_heap()
    }
    fn bound_heap(&self) -> seismic::HostHeap {
        self.bound_heap()
    }
}

impl Certified for seismic::NativeGraphOutline {
    fn of(
        template: &seismic::NativeGraphResourceTemplate,
        slices: &[seismic::NativeGraphClassSlice],
    ) -> Result<Self, seismic::NativeGraphMetadataError> {
        template.outline(slices)
    }
    fn traced_each<K: Clone + Eq + std::hash::Hash, X: seismic::TraceOutput>(
        traces: &seismic::NativeGraphTraces<K, X>,
        key: &K,
        facts: &super::facts::FactSink,
        classes: &[seismic::ClassFact],
        slice_sets: &[Vec<seismic::NativeGraphClassSlice>],
        record: seismic::GraphRecorder<K, X, GraphError>,
    ) -> Result<Vec<(Self, X::Value)>, GraphError>
    where
        X::Value: Clone,
    {
        traces.outline(key, facts.values(), facts.reals(), classes, slice_sets, record)
    }
    fn storage_bytes(&self) -> NativeGraphStorageBytes {
        self.storage_bytes()
    }
    fn plan_heap(&self) -> seismic::HostHeap {
        self.plan_heap()
    }
    fn bound_heap(&self) -> seismic::HostHeap {
        self.bound_heap()
    }
}
