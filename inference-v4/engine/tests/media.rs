use seismic_engine::inputs::media::{DType, PreparedMedia, PreparedTensor};
fn pixels(shape: Vec<usize>) -> PreparedTensor {
    PreparedTensor::new(
        "pixels".into(),
        DType::F32,
        shape,
        [1.25f32, -4.5]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect(),
    )
    .unwrap()
}
#[test]
fn binary_payload_and_identity_match_v3_across_clones() {
    for (shape, expected) in [
        (
            vec![1, 2],
            "3f5ef0ae71f63b9cecb7f38365c5891e4eca53d1735b577f881641e95dbb572e",
        ),
        (
            vec![2],
            "920cbb23c7d0743eb3bb76404c77b01bcfd696ae2610c2b99e110b0b57e3e561",
        ),
    ] {
        let media = PreparedMedia::new("a".repeat(64), vec![pixels(shape.clone())]).unwrap();
        assert_eq!(media.identity(), expected);
        assert_eq!(media.nbytes(), 8);
        assert_eq!(media.tensors()[0].shape(), shape);
        assert_eq!(media.tensors()[0].dtype(), DType::F32);
        let copy = media.clone();
        assert_eq!(
            media.tensors()[0].data().as_ptr(),
            copy.tensors()[0].data().as_ptr()
        );
        drop(media);
        assert_eq!(
            copy.tensors()[0]
                .data()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect::<Vec<_>>(),
            vec![1.25, -4.5]
        );
        assert_ne!(
            copy.identity(),
            PreparedMedia::new("b".repeat(64), vec![pixels(shape)])
                .unwrap()
                .identity()
        );
    }
}
#[test]
fn metadata_geometry_and_tensor_set_limits_fail_before_publication() {
    for shape in [
        vec![],
        vec![0],
        vec![1; 6],
        vec![1usize << 31],
        vec![1usize << 30; 5],
    ] {
        assert!(PreparedTensor::new("pixels".into(), DType::F32, shape, vec![]).is_err());
    }
    assert!(PreparedTensor::new("pixels".into(), DType::I64, vec![2], vec![0; 8]).is_err());
    assert!(PreparedTensor::new("".into(), DType::U8, vec![1], vec![0]).is_err());
    for processor in ["a".repeat(63), "A".repeat(64), "g".repeat(64)] {
        assert!(PreparedMedia::new(processor, vec![pixels(vec![2])]).is_err());
    }
    assert!(PreparedMedia::new("a".repeat(64), vec![]).is_err());
    assert!(PreparedMedia::new("a".repeat(64), vec![pixels(vec![2]), pixels(vec![2])]).is_err());
    let tensors = (0..17)
        .map(|i| PreparedTensor::new(i.to_string(), DType::U8, vec![1], vec![0]).unwrap())
        .collect();
    assert!(PreparedMedia::new("a".repeat(64), tensors).is_err());
}
#[test]
fn dtype_name_shape_bytes_and_tensor_order_all_affect_identity() {
    let tensor = |name: &str, dtype, shape, bytes| {
        PreparedTensor::new(name.into(), dtype, shape, bytes).unwrap()
    };
    let identity = |tensors| {
        PreparedMedia::new("a".repeat(64), tensors)
            .unwrap()
            .identity()
            .to_string()
    };
    let base = tensor("pixels", DType::I32, vec![2], vec![0; 8]);
    let original = identity(vec![base.clone()]);
    for other in [
        tensor("other", DType::I32, vec![2], vec![0; 8]),
        tensor("pixels", DType::F32, vec![2], vec![0; 8]),
        tensor("pixels", DType::I32, vec![1, 2], vec![0; 8]),
        tensor("pixels", DType::I32, vec![2], vec![1; 8]),
    ] {
        assert_ne!(original, identity(vec![other]));
    }
    let second = tensor("coordinates", DType::I32, vec![2], vec![0; 8]);
    assert_ne!(
        identity(vec![base.clone(), second.clone()]),
        identity(vec![second, base])
    );
}
