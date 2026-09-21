fn main() {
    seismic_build::Build::new("kernels")
        .source("lib")
        .run()
        .unwrap_or_else(|error| {
            panic!("checking engine Seismic sources and generating bindings failed: {error}")
        });
}
