use seismic_compiler::implementation::ImplementationBuilder;
use seismic_compiler::target::Backend;

fn fabricate<'a, B: Backend>() -> ImplementationBuilder<'a, B> {
    ImplementationBuilder { inner: panic!() }
}

fn main() {}
