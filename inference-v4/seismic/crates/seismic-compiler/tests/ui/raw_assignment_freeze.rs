use seismic_compiler::frozen::freeze;
use seismic_compiler::solve::RawAssignment;

fn main() {
    let _ = freeze::<()>;
    let _ = std::mem::size_of::<RawAssignment>();
}
