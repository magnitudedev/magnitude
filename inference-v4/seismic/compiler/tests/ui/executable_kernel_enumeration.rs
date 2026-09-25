use seismic_compiler::executable::ExecutableVariant;
use seismic_native_target::TargetFamily;

fn enumerate<T: TargetFamily, H>(variant: &ExecutableVariant<T, H>) {
    let _ = variant.kernels();
}

fn main() {}
