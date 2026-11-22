fn main() {
    seismic_build::Build::new("line")
        .source("line")
        .run()
        .unwrap_or_else(|error| {
            panic!("checking machine line Seismic sources and generating bindings failed: {error}")
        });
}
