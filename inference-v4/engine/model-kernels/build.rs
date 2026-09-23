fn main() {
    seismic_build::Build::new("kernels")
        .source("kernels")
        // Keep newly introduced stage roots explicit so Cargo notices their
        // first addition; the recursive directory source deduplicates them.
        .source("kernels/head_rows.seismic")
        .source("kernels/vision.seismic")
        .source("kernels/recurrent_stages.seismic")
        .run()
        .unwrap_or_else(|error| {
            panic!("checking engine Seismic sources and generating bindings failed: {error}")
        });
}
