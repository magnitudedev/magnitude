fn main() {
    seismic_build::Build::new("kernels")
        .source("../elementwise.seismic")
        .std(false)
        .run()
        .expect("generate native example bindings");
}

