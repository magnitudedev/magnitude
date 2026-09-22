use seismic_ir::kernel::ScalarId;
use seismic_ir::repr::U32;

fn main() {
    let _ = ScalarId::<U32> {
        owner: panic!(),
        kernel: 0,
        block: panic!(),
        index: 0,
        marker: std::marker::PhantomData,
    };
}
