use seismic_compiler::executable::ExecutableVariant;
use seismic_target::TargetFamily;

fn inspect<T: TargetFamily, H>(variant: &ExecutableVariant<T, H>) {
    let _ = variant.schedule();
}

fn main() {}
