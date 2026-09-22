fn forge<T: seismic_target::TargetFamily>() {
    let _ = seismic_compiler::SelectionPolicy::<T> {
        candidates: todo!(),
        selection_function: todo!(),
    };
}

fn main() {}
