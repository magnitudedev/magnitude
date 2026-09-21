use seismic_lang::checked::CheckedModule;

fn main() {
    let _ = CheckedModule { inner: panic!() };
}
