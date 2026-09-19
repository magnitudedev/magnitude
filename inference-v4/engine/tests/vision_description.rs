use seismic_engine::{models::qwen35::vision, weights::mlx::MlxArtifact};
use serde_json::{json, Value};
struct Fixture(std::path::PathBuf);
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn config() -> Value {
    json!({"quantization":{"group_size":64,"bits":4,"mode":"affine"}, "image_token_id":99,"vision_start_token_id":98,"vision_end_token_id":100,
        "text_config":{"hidden_size":8}, "vision_config":{"in_channels":3,"temporal_patch_size":1,"patch_size":2,"spatial_merge_size":2,
        "hidden_size":8,"intermediate_size":12,"out_hidden_size":8,"num_heads":2,"depth":1,"num_position_embeddings":4,
        "hidden_act":"gelu_pytorch_tanh","deepstack_visual_indexes":[]}})
}
fn fixture(config: &Value, malformed_patch: bool, omit: Option<&str>) -> Fixture {
    let path = std::env::temp_dir().join(format!(
        "seismic-vision-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&path).unwrap();
    let fixture = Fixture(path);
    std::fs::write(
        fixture.0.join("config.json"),
        serde_json::to_vec(config).unwrap(),
    )
    .unwrap();
    let mut header = serde_json::Map::new();
    let mut offset = 0usize;
    let mut tensor = |name: &str, shape: Vec<usize>| {
        if omit == Some(name) {
            return;
        }
        let bytes = shape.iter().product::<usize>() * 2;
        header.insert(
            format!("vision_tower.{name}"),
            json!({"dtype":"BF16","shape":shape,"data_offsets":[offset,offset+bytes]}),
        );
        offset += bytes;
    };
    tensor(
        "patch_embed.proj.weight",
        if malformed_patch {
            vec![8, 3, 1, 2, 2]
        } else {
            vec![8, 1, 2, 2, 3]
        },
    );
    tensor("patch_embed.proj.bias", vec![8]);
    tensor("pos_embed.weight", vec![4, 8]);
    for (name, out, input) in [
        ("blocks.0.norm1", 8, None),
        ("blocks.0.attn.qkv", 24, Some(8)),
        ("blocks.0.attn.proj", 8, Some(8)),
        ("blocks.0.norm2", 8, None),
        ("blocks.0.mlp.linear_fc1", 12, Some(8)),
        ("blocks.0.mlp.linear_fc2", 8, Some(12)),
        ("merger.norm", 8, None),
        ("merger.linear_fc1", 32, Some(32)),
        ("merger.linear_fc2", 8, Some(32)),
    ] {
        tensor(
            &format!("{name}.weight"),
            input.map_or(vec![out], |n| vec![out, n]),
        );
        tensor(&format!("{name}.bias"), vec![out]);
    }
    let mut metadata = serde_json::to_vec(&header).unwrap();
    metadata.resize(metadata.len().div_ceil(8) * 8, b' ');
    let mut bytes = (metadata.len() as u64).to_le_bytes().to_vec();
    bytes.extend(metadata);
    bytes.resize(bytes.len() + offset, 0);
    std::fs::write(fixture.0.join("model.safetensors"), bytes).unwrap();
    fixture
}
#[test]
fn image_tower_roles_preserve_patch_order_widths_and_merger_geometry() {
    let fixture = fixture(&config(), false, None);
    let artifact = MlxArtifact::open(&fixture.0).unwrap();
    let description = vision::describe(&artifact).unwrap();
    assert_eq!(description.geometry.image.patch_width().unwrap(), 12);
    assert_eq!(description.geometry.table_side, 2);
    assert_eq!(description.patch.weight.shape, vec![8, 12]);
    assert_eq!(description.blocks.len(), 1);
    assert_eq!(description.blocks[0].qkv.weight.shape, vec![24, 8]);
    assert_eq!(description.blocks[0].up.weight.shape, vec![12, 8]);
    assert_eq!(description.blocks[0].down.weight.shape, vec![8, 12]);
    assert_eq!(description.merger_up.weight.shape, vec![32, 32]);
    assert_eq!(description.merger_down.weight.shape, vec![8, 32]);
    assert_eq!(description.geometry.output, 8);
}
#[test]
fn unsupported_geometry_and_incomplete_roles_fail_before_loading() {
    for (key, value) in [
        ("hidden_act", json!("gelu")),
        ("deepstack_visual_indexes", json!([0])),
        ("num_position_embeddings", json!(5)),
        ("num_heads", json!(3)),
        ("depth", json!(1000000)),
        ("spatial_merge_size", json!(0)),
        ("out_hidden_size", json!(16)),
    ] {
        let mut config = config();
        config["vision_config"][key] = value;
        let fixture = fixture(&config, false, None);
        assert!(
            vision::describe(&MlxArtifact::open(&fixture.0).unwrap()).is_err(),
            "{key}"
        );
    }
    for (malformed, omit) in [
        (true, None),
        (false, Some("blocks.0.attn.qkv.bias")),
        (false, Some("merger.linear_fc2.weight")),
    ] {
        let fixture = fixture(&config(), malformed, omit);
        assert!(vision::describe(&MlxArtifact::open(&fixture.0).unwrap()).is_err());
    }
}

fn import_only_settings() -> seismic_runtime::plan::Settings {
    use seismic_accounting::{
        execution_model::{ScalarHardware, Scope},
        schedule::Timebase,
    };
    seismic_runtime::plan::Settings {
        hardware: seismic_runtime::tuner::Hardware::Cpu(ScalarHardware {
            identity: "no numerical execution in import-lifetime test".into(),
            scope: Scope::HypotheticalDirectScalarV1,
            timebase: Timebase {
                seconds_numerator: 1,
                seconds_denominator: 1,
            },
            resources: vec![],
            timings: vec![],
        }),
        form: seismic_runtime::tuner::Form::CpuScalar,
        derivation_limits: seismic_accounting::workload::DerivationLimits {
            instructions: 1,
            operations: 1,
        },
        search: seismic_runtime::tuner::Settings { limits: seismic_runtime::tuner::Limits { work: 1, ..Default::default() }, ..Default::default() },
    }
}

#[test]
fn encoder_loading_retains_weights_and_failed_preparation_releases_allocations() {
    use seismic_engine::{
        inputs::media::{DType, PreparedTensor},
        models::qwen35::{preparation::ImagePatches, vision_runtime::Encoder},
        weights::residency::Importer,
    };
    use seismic_runtime::{Device, Error};
    use std::rc::Rc;
    let fixture = fixture(&config(), false, None);
    let artifact = MlxArtifact::open(&fixture.0).unwrap();
    let description = vision::describe(&artifact).unwrap();
    let device = Rc::new(Device::cpu());
    // Only direct BF16 import and host validation run; there are no numerical
    // timing entries and no native selection or performance claim.
    let settings = import_only_settings();
    let mut importer = Importer::new(device.clone(), settings.clone()).unwrap();
    let mut imported = 0;
    let encoder = Encoder::load(device.clone(), &description, settings, |role, dtype| {
        imported += 1;
        importer
            .import(role, &artifact.stored(role).unwrap(), dtype)
            .map_err(|e| e.to_string())
    })
    .unwrap();
    assert_eq!(imported, 21);
    drop(importer);
    let retained = device.memory_usage().charged;
    assert!(retained > 0);
    let mut image = ImagePatches {
        identity: "fixture".into(),
        grid: [1, 2, 2],
        pixels: PreparedTensor::new(
            "pixel_values".into(),
            DType::F32,
            vec![4, 12],
            vec![0; 4 * 12 * 4],
        )
        .unwrap(),
    };
    image.grid = [1, 2, 4];
    assert!(encoder.prepare(&image).is_err());
    assert_eq!(device.memory_usage().charged, retained);
    image.grid = [1, 2, 2];
    // Admit pixels but fail on coordinates, before any selection or execution.
    device
        .set_memory_limit(Some(retained + image.pixels.data().len()))
        .unwrap();
    assert!(matches!(
        encoder.prepare(&image),
        Err(Error::Capacity { .. })
    ));
    assert_eq!(device.memory_usage().charged, retained);
    drop(encoder);
    assert_eq!(device.memory_usage().charged, 0);
}
