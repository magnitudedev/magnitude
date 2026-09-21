use seismic_compiler::target::{Backend, NativeKernel, NativeKernelCandidate};

fn consume_reflected<B: Backend>(_kernel: &NativeKernel<B>) {}

fn bypass_reflection<B: Backend>(candidate: &NativeKernelCandidate<B>) {
    consume_reflected(candidate);
}

fn main() {}
