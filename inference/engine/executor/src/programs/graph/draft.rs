//! One graph topology can be built from prepared entries or checked metadata.
//! The entry binding is the only route-specific input; Seismic owns shapes,
//! storage layout and interval placement in either route.

use super::GraphError;
use seismic::{
    Element, Entry, LaunchEncoding, NativeGraph, NativeGraphMetadata, NativeGraphPlan, NativeGraphStorageBytes,
    NativeKernel, NativePort, NativeTimelinePort, WorkflowTensor,
};

/// How a block's graph takes its launch's per-step controls (coordinates,
/// visible spans, fresh rows and destinations; slot and bank words).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

    fn port(&mut self, element: Element, extents: &[u64]) -> Result<NativePort, GraphError>;
    fn set_class_scope(&mut self, _scope: Option<&'static str>) {}
    fn port_with_class_extent(
        &mut self,
        element: Element,
        extents: &[u64],
        _extent_axis: usize,
        _class_dimension: &'static str,
    ) -> Result<NativePort, GraphError> {
        self.port(element, extents)
    }
    fn local_for<'a, E: Entry + 'a>(
        &mut self,
        entry: Self::Binding<'a, E>,
        parameter: &str,
        dimensions: &[(&str, u64)],
    ) -> Result<NativePort, GraphError>;
    fn input_for<'a, E: Entry + 'a>(
        &mut self,
        entry: Self::Binding<'a, E>,
        parameter: &str,
        dimensions: &[(&str, u64)],
    ) -> Result<NativePort, GraphError>;
    /// An external port in the layout `entry`'s `parameter` takes at
    /// `dimensions`, which each run binds.
    fn port_for<'a, E: Entry + 'a>(
        &mut self,
        entry: Self::Binding<'a, E>,
        parameter: &str,
        dimensions: &[(&str, u64)],
    ) -> Result<NativePort, GraphError>;
    /// A launch control `entry`'s `parameter` reads at `dimensions`, as
    /// `source` gives it.
    fn control_for<'a, E: Entry + 'a>(
        &mut self,
        source: ControlSource,
        entry: Self::Binding<'a, E>,
        parameter: &str,
        dimensions: &[(&str, u64)],
    ) -> Result<NativePort, GraphError> {
        match source {
            ControlSource::Uploaded => self.input_for(entry, parameter, dimensions),
            ControlSource::Bound => self.port_for(entry, parameter, dimensions),
        }
    }
    fn enqueue<'a, E: Entry + 'a>(
        &mut self,
        entry: Self::Binding<'a, E>,
        dimensions: &[(&str, u64)],
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
        dimensions: &[(&str, u64)],
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

impl GraphDraft for NativeGraph {
    type Plan = NativeGraphPlan;
    type Binding<'a, E: Entry + 'a> = &'a NativeKernel<E>;

    fn port(&mut self, element: Element, extents: &[u64]) -> Result<NativePort, GraphError> {
        self.port(element, extents).map_err(invalid)
    }
    fn local_for<'a, E: Entry + 'a>(
        &mut self,
        entry: Self::Binding<'a, E>,
        parameter: &str,
        dimensions: &[(&str, u64)],
    ) -> Result<NativePort, GraphError> {
        self.local_for(entry, parameter, dimensions)
            .map_err(invalid)
    }
    fn input_for<'a, E: Entry + 'a>(
        &mut self,
        entry: Self::Binding<'a, E>,
        parameter: &str,
        dimensions: &[(&str, u64)],
    ) -> Result<NativePort, GraphError> {
        self.input_for(entry, parameter, dimensions)
            .map_err(invalid)
    }
    fn port_for<'a, E: Entry + 'a>(
        &mut self,
        entry: Self::Binding<'a, E>,
        parameter: &str,
        dimensions: &[(&str, u64)],
    ) -> Result<NativePort, GraphError> {
        self.port_for(entry, parameter, dimensions)
            .map_err(invalid)
    }
    fn enqueue<'a, E: Entry + 'a>(
        &mut self,
        entry: Self::Binding<'a, E>,
        _dimensions: &[(&str, u64)],
        args: E::WorkflowArgs<'_>,
    ) -> Result<E::WorkflowResults, GraphError> {
        self.enqueue(entry, args).map_err(invalid)
    }
    fn enqueue_under<'a, E: Entry + 'a>(
        &mut self,
        entry: Self::Binding<'a, E>,
        _dimensions: &[(&str, u64)],
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

    fn port(&mut self, element: Element, extents: &[u64]) -> Result<NativePort, GraphError> {
        Ok(self.port(element, extents)?)
    }
    fn set_class_scope(&mut self, scope: Option<&'static str>) {
        self.set_class_scope(scope);
    }
    fn port_with_class_extent(
        &mut self,
        element: Element,
        extents: &[u64],
        extent_axis: usize,
        class_dimension: &'static str,
    ) -> Result<NativePort, GraphError> {
        Ok(self.port_with_class_extent(element, extents, extent_axis, class_dimension)?)
    }
    fn local_for<'a, E: Entry + 'a>(
        &mut self,
        entry: Self::Binding<'a, E>,
        parameter: &str,
        dimensions: &[(&str, u64)],
    ) -> Result<NativePort, GraphError> {
        Ok(self.local_for::<E>(entry, parameter, dimensions)?)
    }
    fn input_for<'a, E: Entry + 'a>(
        &mut self,
        entry: Self::Binding<'a, E>,
        parameter: &str,
        dimensions: &[(&str, u64)],
    ) -> Result<NativePort, GraphError> {
        Ok(self.input_for::<E>(entry, parameter, dimensions)?)
    }
    fn port_for<'a, E: Entry + 'a>(
        &mut self,
        entry: Self::Binding<'a, E>,
        parameter: &str,
        dimensions: &[(&str, u64)],
    ) -> Result<NativePort, GraphError> {
        Ok(self.port_for::<E>(entry, parameter, dimensions)?)
    }
    fn enqueue<'a, E: Entry + 'a>(
        &mut self,
        entry: Self::Binding<'a, E>,
        dimensions: &[(&str, u64)],
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
        dimensions: &[(&str, u64)],
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
