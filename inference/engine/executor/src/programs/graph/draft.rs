//! One graph topology can be built from prepared entries or checked metadata.
//! The entry binding is the only route-specific input; Seismic owns shapes,
//! storage layout and interval placement in either route.

use super::GraphError;
use seismic::{
    Element, Entry, NativeGraph, NativeGraphMetadata, NativeGraphPlan, NativeGraphStorageBytes,
    NativeKernel, NativePort, WorkflowTensor,
};

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
    fn enqueue<'a, E: Entry + 'a>(
        &mut self,
        entry: Self::Binding<'a, E>,
        dimensions: &[(&str, u64)],
        args: E::WorkflowArgs<'_>,
    ) -> Result<E::WorkflowResults, GraphError>;
    fn export(&mut self, result: &WorkflowTensor) -> Result<(), GraphError>;
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
    fn enqueue<'a, E: Entry + 'a>(
        &mut self,
        entry: Self::Binding<'a, E>,
        _dimensions: &[(&str, u64)],
        args: E::WorkflowArgs<'_>,
    ) -> Result<E::WorkflowResults, GraphError> {
        self.enqueue(entry, args).map_err(invalid)
    }
    fn export(&mut self, result: &WorkflowTensor) -> Result<(), GraphError> {
        self.export(result).map_err(invalid)
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
    fn enqueue<'a, E: Entry + 'a>(
        &mut self,
        entry: Self::Binding<'a, E>,
        dimensions: &[(&str, u64)],
        args: E::WorkflowArgs<'_>,
    ) -> Result<E::WorkflowResults, GraphError> {
        Ok(self.enqueue::<E>(entry, dimensions, args)?)
    }
    fn export(&mut self, result: &WorkflowTensor) -> Result<(), GraphError> {
        Ok(self.export(result)?)
    }
    fn seal(self) -> Result<Self::Plan, GraphError> {
        Ok(self.seal()?)
    }
}
