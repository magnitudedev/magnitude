use seismic_native_target::{NativeCompiler, NativeKernelCandidate, TargetFamily};

fn fabricate<T, C>() -> NativeKernelCandidate<'static, T, C::Candidate>
where
    T: TargetFamily,
    C: NativeCompiler<T>,
{
    NativeKernelCandidate {
        raw: panic!(),
        target: panic!(),
        kernel: panic!(),
        layout: panic!(),
        abi: panic!(),
        compatibility: panic!(),
        _target: panic!(),
    }
}

fn main() {}
