//! One graph topology can be built from prepared entries or checked metadata.
//! The entry binding is the only route-specific input; Seismic owns shapes,
//! storage layout and interval placement in either route.

use seismic::{
    Element, Entry, NativeGraph, NativeGraphMetadata, NativeGraphPlan, NativeGraphStorageBytes,
    NativeKernel, NativePort, WorkflowTensor,
};

pub(crate) trait GraphDraft: Sized {
    type Plan;
    type Binding<'a, E: Entry + 'a>: Copy
    where
        Self: 'a;

    fn port(&mut self, element: Element, extents: &[u64]) -> Result<NativePort, String>;
    fn local_for<'a, E: Entry + 'a>(
        &mut self,
        entry: Self::Binding<'a, E>,
        parameter: &str,
        dimensions: &[(&str, u64)],
    ) -> Result<NativePort, String>;
    fn input_for<'a, E: Entry + 'a>(
        &mut self,
        entry: Self::Binding<'a, E>,
        parameter: &str,
        dimensions: &[(&str, u64)],
    ) -> Result<NativePort, String>;
    fn enqueue<'a, E: Entry + 'a>(
        &mut self,
        entry: Self::Binding<'a, E>,
        dimensions: &[(&str, u64)],
        args: E::WorkflowArgs<'_>,
    ) -> Result<E::WorkflowResults, String>;
    fn export(&mut self, result: &WorkflowTensor) -> Result<(), String>;
    fn seal(self) -> Result<Self::Plan, String>;
}

impl GraphDraft for NativeGraph {
    type Plan = NativeGraphPlan;
    type Binding<'a, E: Entry + 'a> = &'a NativeKernel<E>;

    fn port(&mut self, element: Element, extents: &[u64]) -> Result<NativePort, String> {
        self.port(element, extents)
            .map_err(|error| error.to_string())
    }
    fn local_for<'a, E: Entry + 'a>(
        &mut self,
        entry: Self::Binding<'a, E>,
        parameter: &str,
        dimensions: &[(&str, u64)],
    ) -> Result<NativePort, String> {
        self.local_for(entry, parameter, dimensions)
            .map_err(|error| error.to_string())
    }
    fn input_for<'a, E: Entry + 'a>(
        &mut self,
        entry: Self::Binding<'a, E>,
        parameter: &str,
        dimensions: &[(&str, u64)],
    ) -> Result<NativePort, String> {
        self.input_for(entry, parameter, dimensions)
            .map_err(|error| error.to_string())
    }
    fn enqueue<'a, E: Entry + 'a>(
        &mut self,
        entry: Self::Binding<'a, E>,
        _dimensions: &[(&str, u64)],
        args: E::WorkflowArgs<'_>,
    ) -> Result<E::WorkflowResults, String> {
        self.enqueue(entry, args).map_err(|error| error.to_string())
    }
    fn export(&mut self, result: &WorkflowTensor) -> Result<(), String> {
        self.export(result).map_err(|error| error.to_string())
    }
    fn seal(self) -> Result<Self::Plan, String> {
        self.seal().map_err(|error| error.to_string())
    }
}

impl GraphDraft for NativeGraphMetadata {
    type Plan = NativeGraphStorageBytes;
    type Binding<'a, E: Entry + 'a> = &'a [(&'a str, Element)];

    fn port(&mut self, element: Element, extents: &[u64]) -> Result<NativePort, String> {
        self.port(element, extents)
            .map_err(|error| error.to_string())
    }
    fn local_for<'a, E: Entry + 'a>(
        &mut self,
        entry: Self::Binding<'a, E>,
        parameter: &str,
        dimensions: &[(&str, u64)],
    ) -> Result<NativePort, String> {
        self.local_for::<E>(entry, parameter, dimensions)
            .map_err(|error| error.to_string())
    }
    fn input_for<'a, E: Entry + 'a>(
        &mut self,
        entry: Self::Binding<'a, E>,
        parameter: &str,
        dimensions: &[(&str, u64)],
    ) -> Result<NativePort, String> {
        self.input_for::<E>(entry, parameter, dimensions)
            .map_err(|error| error.to_string())
    }
    fn enqueue<'a, E: Entry + 'a>(
        &mut self,
        entry: Self::Binding<'a, E>,
        dimensions: &[(&str, u64)],
        args: E::WorkflowArgs<'_>,
    ) -> Result<E::WorkflowResults, String> {
        self.enqueue::<E>(entry, dimensions, args)
            .map_err(|error| error.to_string())
    }
    fn export(&mut self, result: &WorkflowTensor) -> Result<(), String> {
        self.export(result).map_err(|error| error.to_string())
    }
    fn seal(self) -> Result<Self::Plan, String> {
        self.seal().map_err(|error| error.to_string())
    }
}
