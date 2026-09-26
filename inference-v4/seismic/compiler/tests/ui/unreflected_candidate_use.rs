use seismic_native_target::{NativeCompiler, NativeKernel, NativeKernelCandidate, TargetFamily};

fn consume_reflected<T: TargetFamily, H>(_kernel: &NativeKernel<T, H>) {}

fn bypass_reflection<T, C>(candidate: &NativeKernelCandidate<'_, T, C::Candidate>)
where
    T: TargetFamily,
    C: NativeCompiler<T>,
{
    consume_reflected(candidate);
}

fn main() {}
