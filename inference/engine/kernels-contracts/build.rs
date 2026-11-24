fn main() {
    seismic_build::Build::new("kernels")
        .source("../kernels/kernels")
        .mode(seismic_build::EmitMode::Contracts)
        .run()
        .unwrap_or_else(|error| {
            panic!("checking engine Seismic sources and generating contract helpers failed: {error}")
        });
}
