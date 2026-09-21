use seismic_compiler::target::{Backend, NativeKernelCandidate};

fn fabricate<B: Backend>() -> NativeKernelCandidate<B> {
    NativeKernelCandidate {
        raw: panic!(),
        layout: panic!(),
        abi: panic!(),
        compatibility: panic!(),
        services: panic!(),
    }
}

fn main() {}
