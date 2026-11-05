//! Host-shared words (`Tensor::load_acquire`, `Tensor::store_release`): the
//! host and a running Metal kernel exchange a word both ways through a
//! `Tensor::shared` view, each seeing what the other wrote before publishing.

#[cfg(target_os = "macos")]
#[test]
fn a_running_kernel_and_the_host_exchange_words() {
    use seismic::{BackendName, Devices, Element, NativeSpecialization, Tensor};
    use seismic_native_tests::word_exchange;
    use std::time::{Duration, Instant};

    let device = Devices::discover(&[])
        .unwrap()
        .open_backend(BackendName::Metal)
        .unwrap();
    let kernel = word_exchange::native_for_device(&device, &NativeSpecialization::new()).unwrap();
    let mut graph = device.native_graph();
    let mut exchange = graph.port(Element::i32(), &[4]).unwrap();
    graph
        .enqueue(
            &kernel,
            word_exchange::WorkflowArgs {
                exchange: exchange.tensor_mut().into(),
            },
        )
        .unwrap();
    let plan = graph.seal().unwrap();
    let mut slot = plan.new_slot().unwrap();
    let words = Tensor::shared(&device, Element::i32(), &[4]).unwrap();
    // SAFETY: no device work uses the tensor yet.
    unsafe { words.write_shared(0, &[0; 16]).unwrap() };
    let mut bindings = plan.bindings();
    bindings.set(&exchange, &words).unwrap();
    let (_, completion) = slot
        .attach(bindings, plan.new_outputs().unwrap())
        .unwrap()
        .submit()
        .unwrap();

    // Device to host: once word 0 is published, what the kernel wrote
    // before it (word 3) is visible.
    let deadline = Instant::now() + Duration::from_secs(10);
    while words.load_acquire(0).unwrap() != 1 {
        assert!(Instant::now() < deadline, "the kernel never published word 0");
        std::hint::spin_loop();
    }
    assert_eq!(words.load_acquire(12).unwrap(), 42);

    // Host to device: the parked kernel sees the host's word 1 and echoes it.
    words.store_release(4, 7).unwrap();
    completion.wait().unwrap();
    assert_eq!(words.load_acquire(8).unwrap(), 7);
}
