use seismic::{BackendName, Element, KernelRequest};
use seismic::coverage::RequestFormer;
fn request(entry: &'static str, elements: &[(&str, Element)], statics: &[(&str,u64)]) -> KernelRequest {
    KernelRequest { backend: BackendName::Vulkan, entry,
        elements: elements.iter().map(|(k,v)|(k.to_string(),*v)).collect(),
        statics: statics.iter().map(|(k,v)|(k.to_string(),*v)).collect() }
}
#[test]
fn vulkan_priming_abi_forms() {
    let module = magnitude_kernels::module().unwrap();
    let former = RequestFormer::open(BackendName::Vulkan).unwrap();
    let mut requests = Vec::new();
    // Qwen 4B fixture geometry; cover all normalization/presence branches.
    for entry in ["attention_append_dense", "attention_append_k8v4"] {
        for f in [0,1] { for n in [0,1] { for nv in [0,1] {
            requests.push(request(entry, &[("A",Element::bf16())],
                &[("KV",4),("P",32),("S",192),("F",f),("N",n),("NV",nv)]));
        }}}
    }
    // Full and static-zero query forms use identical K/V projection geometry.
    for q in [0,8192] {
        for weight in [Element::bf16(),Element::stored("q8g32s", seismic_lang::registry::Layout::Rows16).unwrap()] {
            requests.push(request("attention_project", &[("A",Element::bf16()),("NW",Element::bf16()),
                ("QW",weight),("GW",weight),("KW",weight),("VW",weight)],
                &[("D",2560),("Q",q),("GR",0),("K",1024),("V",1024)]));
        }
    }
    let results = former.form_all(module, &requests);
    for (request,result) in requests.iter().zip(results) {
        eprintln!("{request}: {result:?}");
        assert!(result.is_ok(), "{request}: {result:?}");
    }
    eprintln!("FORMED {} exact native Vulkan requests", requests.len());
}
