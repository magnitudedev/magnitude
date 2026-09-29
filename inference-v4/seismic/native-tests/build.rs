fn main() {
    seismic_build::Build::new("fixtures")
        .source("fixtures")
        .std(false)
        .run()
        .unwrap_or_else(|error| panic!("checking native route fixtures failed: {error}"));
}
