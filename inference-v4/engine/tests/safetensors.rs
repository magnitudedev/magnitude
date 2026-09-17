use seismic_engine::weights::safetensors;
use seismic_lang::types::DType;
use std::io::Cursor;
fn file(header: &str, bytes: usize) -> Vec<u8> {
    let mut data = (header.len() as u64).to_le_bytes().to_vec();
    data.extend_from_slice(header.as_bytes());
    data.resize(data.len() + bytes, 0);
    data
}
fn read(
    header: &str,
    bytes: usize,
) -> Result<safetensors::Directory, seismic_engine::weights::Error> {
    safetensors::read_directory(&mut Cursor::new(file(header, bytes)))
}
#[test]
fn preserves_scalar_shapes_stored_dtypes_and_physical_offsets() {
    let h = r#"{"second":{"dtype":"BF16","shape":[2,3],"data_offsets":[4,16]},"scalar":{"dtype":"F32","shape":[],"data_offsets":[0,4]},"__metadata__":{"format":"mlx"}}"#;
    let d = read(h, 16).unwrap();
    assert_eq!(d.tensors[0].name, "second");
    assert_eq!(d.tensors[0].dtype, DType::BF16);
    assert_eq!(d.tensors[0].shape, vec![2, 3]);
    assert_eq!(d.tensors[0].offset, 12 + h.len() as u64);
    assert_eq!(d.tensors[1].nbytes, 4);
    assert_eq!(d.metadata.unwrap()["format"], "mlx");
}
#[test]
fn rejects_unsupported_dtype_invalid_shapes_ranges_and_overlap() {
    for entry in [
        r#"{"dtype":"F64","shape":[1],"data_offsets":[0,8]}"#,
        r#"{"dtype":"F32","shape":[true],"data_offsets":[0,4]}"#,
        r#"{"dtype":"F32","shape":[0],"data_offsets":[0,0]}"#,
        r#"{"dtype":"F32","shape":[-1],"data_offsets":[0,4]}"#,
        r#"{"dtype":"F32","shape":[1.0],"data_offsets":[0,4]}"#,
        r#"{"dtype":"F32","shape":[18446744073709551615],"data_offsets":[0,4]}"#,
        r#"{"dtype":"F32","shape":[1],"data_offsets":[4,0]}"#,
        r#"{"dtype":"F32","shape":[1],"data_offsets":[0,3]}"#,
        r#"{"dtype":"F32","shape":[1],"data_offsets":[8,12]}"#,
    ] {
        assert!(read(&format!("{{\"x\":{entry}}}"), 8).is_err(), "{entry}");
    }
    let entry = r#"{"dtype":"F32","shape":[1],"data_offsets":[0,4]}"#;
    assert!(read(&format!("{{\"x\":{entry},\"y\":{entry}}}"), 8).is_err());
    assert!(read(&format!("{{\"x\":{entry},\"x\":{entry}}}"), 8).is_err());
}
#[test]
fn rejects_bad_header_lengths_and_non_object_json() {
    for h in ["[]", "null", "{", "{}garbage"] {
        assert!(read(h, 0).is_err());
    }
    assert!(safetensors::read_directory(&mut Cursor::new(u64::MAX.to_le_bytes())).is_err());
    let mut bytes = file("{}", 0);
    bytes[0] = 3;
    assert!(safetensors::read_directory(&mut Cursor::new(bytes)).is_err());
}
