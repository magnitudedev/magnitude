//! The weight elements a CPU weight operand can be bound to and their
//! components, from the registry: every floating dense element, and every
//! packed element of a CPU row layout (`rows16`, `rows8`) that is its own
//! storage, decoded by the components its decode axes name
//! (`registry::decode_axes`). Elements of one decode share one component,
//! whatever their layout. Writes `$OUT_DIR/elements.rs`: `COMPONENT_OF`,
//! per registered element (in `registry::representations()` order) the index
//! of its component, and `with_components!`, which passes the component types
//! in index order to a macro.

use seismic_lang::registry::{
    self, CodeInterpretation, CoefficientAxes, Layout, LocalCodes, MinimumSign, RepresentationKind,
    ScaleEncoding,
};
use seismic_lang::types::DType;
use std::fmt::Write as _;

/// The `codes` interpretation type of `interpretation`.
fn interpretation(interpretation: &CodeInterpretation) -> String {
    match interpretation {
        CodeInterpretation::Unsigned => "codes::Unsigned".to_owned(),
        CodeInterpretation::TwosComplement => "codes::Twos".to_owned(),
        CodeInterpretation::Offset(offset) => format!("codes::Offset<{offset}>"),
        CodeInterpretation::Odd => "codes::Odd".to_owned(),
        CodeInterpretation::Table(codebook) => format!("codes::Table<{}>", codebook_id(codebook)),
        CodeInterpretation::Grid(_) => panic!("a grid is a code component, not an interpretation"),
    }
}

/// A codebook's position in `tables::CODEBOOKS`.
fn codebook_id(codebook: &magnitude_ggml::tables::Codebook) -> usize {
    magnitude_ggml::tables::CODEBOOKS
        .iter()
        .position(|known| *known == codebook)
        .expect("a ggml codebook")
}

fn scale(scale: ScaleEncoding) -> &'static str {
    match scale {
        ScaleEncoding::F16 => "scales::F16",
        ScaleEncoding::BF16 => "scales::Bf16",
        ScaleEncoding::F32 => "scales::F32",
        ScaleEncoding::E8M0 => "scales::E8m0",
        ScaleEncoding::UE4M3 => "scales::Ue4m3",
    }
}

/// The component type of packed storage `id` of `STORAGE`-value groups.
fn packed(id: seismic_lang::ids::RepresentationId, storage: u32) -> String {
    let axes = registry::decode_axes(id).expect("packed storage has decode axes");
    let codes = match &axes.code {
        CodeInterpretation::Grid(grid) => format!(
            "Grid<{}, {}, {}>",
            magnitude_ggml::tables::GRIDS
                .iter()
                .position(|known| *known == grid.grid)
                .expect("a ggml grid"),
            grid.index_bits(),
            axes.low / 8
        ),
        code => format!(
            "Planes<{}, {}, {}>",
            axes.low,
            axes.high,
            interpretation(code)
        ),
    };
    let coefficients = match &axes.coefficients {
        CoefficientAxes::Direct {
            scale: encoding,
            bias,
            group,
        } => format!(
            "Direct<{}, {group}, {bias}, {}>",
            scale(*encoding),
            axes.code_exponent
        ),
        // The factor group is the storage group, which `PackedRows` carries.
        CoefficientAxes::Hierarchical {
            factor,
            factor_group,
            exponent,
            group,
            locals,
        } => {
            assert_eq!(
                *factor_group, storage,
                "a factor group is one storage group"
            );
            let locals = match locals {
                LocalCodes::Carried {
                    bit,
                    bits,
                    interpretation: local,
                } => format!(
                    "Carried<{bit}, {}, {}, {bits}>",
                    axes.low / 8,
                    self::interpretation(local)
                ),
                LocalCodes::Plane {
                    bits,
                    interpretation: local,
                    minimum,
                } => format!(
                    "Scales<{}, {bits}, {}>",
                    self::interpretation(local),
                    minimum.map_or(0, MinimumSign::value)
                ),
            };
            format!(
                "Hier<{}, {exponent}, {}, {locals}, {group}>",
                scale(*factor),
                axes.code_exponent
            )
        }
    };
    format!("PackedRows<{codes}, {coefficients}, {storage}>")
}

fn main() {
    let mut components: Vec<String> = Vec::new();
    let mut component_of: Vec<Option<usize>> = Vec::new();
    for info in registry::representations() {
        let component = match &info.kind {
            RepresentationKind::Dense(DType::F32) => "DenseRows<crate::element::F32>".to_owned(),
            RepresentationKind::Dense(DType::BF16) => "DenseRows<crate::element::Bf16>".to_owned(),
            RepresentationKind::Dense(DType::F16) => "DenseRows<crate::element::F16>".to_owned(),
            RepresentationKind::PackedRows(rows)
                if matches!(info.layout, Layout::Rows16 | Layout::Rows8)
                    && registry::storage_of(info.id) == info.id =>
            {
                packed(info.id, rows.group())
            }
            _ => {
                component_of.push(None);
                continue;
            }
        };
        let index = components
            .iter()
            .position(|known| *known == component)
            .unwrap_or_else(|| {
                components.push(component);
                components.len() - 1
            });
        component_of.push(Some(index));
    }
    let mut text = String::from(
        "// The CPU weight elements and their components, generated from the\n// registry by seismic/native-cpu/build.rs.\n\n",
    );
    writeln!(
        text,
        "/// Components: the length of `with_components!`'s list.\npub(crate) const COMPONENTS: usize = {};\n",
        components.len()
    )
    .unwrap();
    writeln!(
        text,
        "/// Per registered element, in `registry::representations()` order, the\n/// index of its component, if it has one.\npub(crate) const COMPONENT_OF: [Option<usize>; {}] = [",
        component_of.len()
    )
    .unwrap();
    for index in &component_of {
        writeln!(text, "    {index:?},").unwrap();
    }
    text += "];\n\n/// `$m![component types]`, in component index order.\nmacro_rules! with_components {\n    ($m:ident) => {\n        $m![\n";
    for component in &components {
        writeln!(text, "            {component},").unwrap();
    }
    text += "        ]\n    };\n}\n";
    let out = std::path::PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR"));
    std::fs::write(out.join("elements.rs"), text).expect("elements");
}
