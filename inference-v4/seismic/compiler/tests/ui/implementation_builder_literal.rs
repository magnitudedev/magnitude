use seismic_compiler::implementation::ImplementationBuilder;
use seismic_target::TargetFamily;

fn fabricate<'a, T: TargetFamily>() -> ImplementationBuilder<'a, T> {
    ImplementationBuilder {}
}

fn main() {}
