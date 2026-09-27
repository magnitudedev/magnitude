use seismic_ir::storage::AnyBufferView;
fn change(mut view: AnyBufferView) {
    view.index = 99;
}
fn main() {}
