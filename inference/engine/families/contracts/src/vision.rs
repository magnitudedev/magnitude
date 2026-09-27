//! The vision component: a projector described as data (plan §3.11).
//!
//! A projector is a stem that turns patch rows into the tower's F32 residual
//! stream, a list of transformer blocks and a merger that turns the tower rows
//! into image features of the decoder's width. Every form a catalog projector
//! uses is a value here; the executor composes the vision operators from it.
//!
//! Rotary layout: a head row of width W = 4P rotates column i < 2P with
//! column i + 2P, at the coordinate of axis i / P and the frequency
//! base^(−(i mod P) / P). A projector stored in another layout reaches this
//! one through a row permutation of its query and key projections (and their
//! biases and head norms) at import.

use crate::{
    checked_product, expect_shape, ActivationDType, DefinitionError, RmsNorm, UnweightedRms,
    WeightDescriptor, WeightKind, WeightRole, WeightScope,
};
use magnitude_artifacts::{ImageProcessorConfig, ImageResize, Resampling};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VisionActivation {
    /// `0.5 x (1 + tanh(√(2/π) (x + 0.044715 x³)))`.
    GeluTanh,
    /// `0.5 x (1 + erf(x / √2))`.
    GeluErf,
    /// `x σ(1.702 x)`.
    GeluQuick,
}

/// How an image is resized before it is cut into patches.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VisionResize {
    /// Sides rounded half to even to multiples of patch · merge, then scaled
    /// into the pixel bounds (Qwen3-VL).
    PixelBounds { min_pixels: u64, max_pixels: u64 },
    /// Aspect preserved: the largest sides that are multiples of
    /// patch · merge with at most `max_patches` patches (Gemma 4).
    PatchBudget { max_patches: u64 },
    /// The grid of merged cells (patch · merge pixels) closest to the image's
    /// aspect with at most `max_cells` cells (Muse Glimmer).
    CellBudget { max_cells: u64 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VisionResampling {
    Bicubic,
    Lanczos,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VisionPreprocessing {
    pub resize: VisionResize,
    pub resampling: VisionResampling,
    pub mean: [f64; 3],
    pub std: [f64; 3],
    pub channels: u64,
    /// Side of one patch in pixels.
    pub patch: u64,
    /// Side of one merged (or pooled) cell in patches. Patch rows arrive cell
    /// by cell in raster order of the cells, each cell's patches in raster
    /// order.
    pub merge: u64,
}

/// How a patch row finds its rows of a learned position table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PositionSampling {
    /// A square table of `side`² rows, bilinearly resampled to the patch grid
    /// with aligned corners (Qwen3-VL).
    AlignedCorners { side: u64 },
    /// A square table of `side`² rows, bilinearly sampled at the patch
    /// centres with zero padding outside the table (Muse Glimmer's
    /// `grid_sample`).
    PixelCenters { side: u64 },
    /// Two tables `[2, length, H]`: row `column` of the first plus row `row`
    /// of the second (Gemma 4).
    Axes { length: u64 },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VisionPositions {
    /// `[side², H]` for a square table, `[2, length, H]` for axis tables.
    pub table: WeightDescriptor,
    pub sampling: PositionSampling,
}

impl VisionPositions {
    /// Rows of the table as the stem addresses them.
    pub fn rows(&self) -> u64 {
        match self.sampling {
            PositionSampling::AlignedCorners { side } | PositionSampling::PixelCenters { side } => {
                side * side
            }
            PositionSampling::Axes { length } => 2 * length,
        }
    }

    fn validate(&self, hidden: u64) -> Result<(), DefinitionError> {
        match self.sampling {
            PositionSampling::AlignedCorners { side } | PositionSampling::PixelCenters { side } => {
                if side == 0 {
                    return Err(DefinitionError::new("vision position table has no rows"));
                }
                expect_shape(
                    &self.table,
                    &[checked_product(&[side, side])?, hidden],
                    "vision position table",
                )
            }
            PositionSampling::Axes { length } => {
                if length == 0 {
                    return Err(DefinitionError::new("vision position table has no rows"));
                }
                expect_shape(&self.table, &[2, length, hidden], "vision position table")
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum VisionNorm {
    /// Centered statistics, weight and bias.
    Layer {
        weight: WeightDescriptor,
        bias: WeightDescriptor,
        epsilon: f64,
    },
    /// Root-mean-square statistics, an optional weight, no bias.
    Rms {
        weight: Option<WeightDescriptor>,
        epsilon: f64,
    },
}

impl VisionNorm {
    pub fn epsilon(&self) -> f64 {
        match self {
            Self::Layer { epsilon, .. } | Self::Rms { epsilon, .. } => *epsilon,
        }
    }

    fn validate(&self, width: u64, role: &str) -> Result<(), DefinitionError> {
        let epsilon = self.epsilon();
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(DefinitionError::new(format!(
                "{role} epsilon must be finite and positive"
            )));
        }
        match self {
            Self::Layer { weight, bias, .. } => {
                expect_shape(weight, &[width], role)?;
                expect_shape(bias, &[width], role)
            }
            Self::Rms { weight, .. } => weight
                .as_ref()
                .map_or(Ok(()), |weight| expect_shape(weight, &[width], role)),
        }
    }

    fn weights<'a>(&'a self, scope: WeightScope, site: VisionNormSite, out: &mut Roles<'a>) {
        let role = |part| role(scope, VisionWeight::Norm { site, part });
        match self {
            Self::Layer { weight, bias, .. } => {
                out.push((role(NormPart::Weight), weight));
                out.push((role(NormPart::Bias), bias));
            }
            Self::Rms { weight, .. } => {
                if let Some(weight) = weight {
                    out.push((role(NormPart::Weight), weight));
                }
            }
        }
    }
}

/// The bounds of a clippable linear: its input is clamped to
/// `[input_minimum, input_maximum]`, its output to
/// `[output_minimum, output_maximum]` (each a stored `[1]` F32 value).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LinearClamp {
    pub input_minimum: WeightDescriptor,
    pub input_maximum: WeightDescriptor,
    pub output_minimum: WeightDescriptor,
    pub output_maximum: WeightDescriptor,
}

/// `y = x · weightᵀ (+ bias)`, optionally clamped on both sides.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VisionLinear {
    /// `[outputs, inputs]`.
    pub weight: WeightDescriptor,
    pub bias: Option<WeightDescriptor>,
    pub clamp: Option<LinearClamp>,
}

impl VisionLinear {
    pub fn outputs(&self) -> u64 {
        self.weight.shape.first().copied().unwrap_or(0)
    }

    fn validate(&self, outputs: u64, inputs: u64, role: &str) -> Result<(), DefinitionError> {
        expect_shape(&self.weight, &[outputs, inputs], role)?;
        if let Some(bias) = &self.bias {
            expect_shape(bias, &[outputs], role)?;
        }
        if let Some(clamp) = &self.clamp {
            for bound in [
                &clamp.input_minimum,
                &clamp.input_maximum,
                &clamp.output_minimum,
                &clamp.output_maximum,
            ] {
                expect_shape(bound, &[1], role)?;
            }
        }
        Ok(())
    }

    fn weights<'a>(&'a self, scope: WeightScope, site: VisionLinearSite, out: &mut Roles<'a>) {
        let role = |part| role(scope, VisionWeight::Linear { site, part });
        out.push((role(LinearPart::Weight), &self.weight));
        if let Some(bias) = &self.bias {
            out.push((role(LinearPart::Bias), bias));
        }
        if let Some(clamp) = &self.clamp {
            out.push((role(LinearPart::InputMinimum), &clamp.input_minimum));
            out.push((role(LinearPart::InputMaximum), &clamp.input_maximum));
            out.push((role(LinearPart::OutputMinimum), &clamp.output_minimum));
            out.push((role(LinearPart::OutputMaximum), &clamp.output_maximum));
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum VisionStem {
    /// A convolution of each frame's pixels (Qwen3-VL: two frames of one
    /// still image), an optional bias, the position table, then an optional
    /// norm of the result (Muse Glimmer's pre-norm).
    Patch {
        /// `[H, C, patch, patch]` per frame.
        frames: Vec<WeightDescriptor>,
        bias: Option<WeightDescriptor>,
        positions: VisionPositions,
        norm: Option<VisionNorm>,
    },
    /// A norm of the flattened patch pixels, a linear to the tower width, a
    /// norm, the position table and a norm (Gemma 4 unified).
    NormalizedPatch {
        input_norm: VisionNorm,
        projection: VisionLinear,
        projection_norm: VisionNorm,
        positions: VisionPositions,
        position_norm: VisionNorm,
    },
}

impl VisionStem {
    /// Copies of each patch's pixels in one patch row.
    pub fn frames(&self) -> u64 {
        match self {
            Self::Patch { frames, .. } => frames.len() as u64,
            Self::NormalizedPatch { .. } => 1,
        }
    }

    pub fn positions(&self) -> &VisionPositions {
        match self {
            Self::Patch { positions, .. } | Self::NormalizedPatch { positions, .. } => positions,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VisionAttentionScale {
    /// `1 / √W`.
    InverseSqrtWidth,
    Unit,
}

/// The keys a patch row attends to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VisionAttentionSpan {
    /// Every patch row of the image.
    Full,
    /// The patch rows of the row's own window (the tower's `window`).
    Window,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VisionAttention {
    pub heads: u64,
    /// Head width W, a multiple of 4.
    pub width: u64,
    pub query: VisionLinear,
    pub key: VisionLinear,
    pub value: VisionLinear,
    /// Weighted RMS norms of each query and key head, both or neither.
    pub query_norm: Option<RmsNorm>,
    pub key_norm: Option<RmsNorm>,
    /// Weightless RMS norm of each value head.
    pub value_norm: Option<UnweightedRms>,
    pub rotary_base: f64,
    pub scale: VisionAttentionScale,
    pub span: VisionAttentionSpan,
    pub output: VisionLinear,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum VisionUp {
    Plain(VisionLinear),
    /// `activation(gate(x)) ⊙ up(x)`.
    Gated { gate: VisionLinear, up: VisionLinear },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VisionFeedForward {
    pub up: VisionUp,
    pub activation: VisionActivation,
    pub down: VisionLinear,
}

impl VisionFeedForward {
    pub fn intermediate(&self) -> u64 {
        match &self.up {
            VisionUp::Plain(up) | VisionUp::Gated { up, .. } => up.outputs(),
        }
    }
}

/// One transformer block: `x += post(attention(norm(x)))`, then
/// `x += post(feed_forward(norm(x)))`; an absent post-norm adds the
/// sublayer's output directly.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VisionBlock {
    pub attention_norm: VisionNorm,
    pub attention: VisionAttention,
    pub attention_post_norm: Option<VisionNorm>,
    pub feedforward_norm: VisionNorm,
    pub feedforward: VisionFeedForward,
    pub feedforward_post_norm: Option<VisionNorm>,
}

/// How the G = `merge`² tower rows of one cell become one row.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CellReduction {
    /// The rows side by side in the cell's raster order: value `c` of row
    /// `s` lands at `s · H + c`.
    Concatenate,
    /// The rows interleaved channel by channel (a channel-outer pixel
    /// shuffle): value `c` of row `s` lands at `c · G + s`.
    Interleave,
    /// The mean of the rows times `scale`.
    Average { scale: f64 },
}

/// `(x − bias) ⊙ scale`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Standardize {
    pub bias: WeightDescriptor,
    pub scale: WeightDescriptor,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MergerStage {
    pub linear: VisionLinear,
    pub activation: Option<VisionActivation>,
}

/// Tower rows to image features: an optional norm of each tower row, the
/// cell reduction, an optional standardization, an optional norm, the linear
/// stages, then an optional norm of the feature row. Features enter the
/// decoder as they are: the decoder's entry transform applies to text rows
/// only.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VisionMerger {
    pub norm: Option<VisionNorm>,
    pub reduction: CellReduction,
    pub standardize: Option<Standardize>,
    pub projection_norm: Option<VisionNorm>,
    pub stages: Vec<MergerStage>,
    pub output_norm: Option<VisionNorm>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VisionDescription {
    /// Element of the tower's published intermediates. The residual stream
    /// and the image features are F32.
    pub activation_dtype: ActivationDType,
    /// Width of the tower's residual stream.
    pub hidden: u64,
    /// Width of the image features (the decoder's hidden width).
    pub output_hidden: u64,
    pub preprocessing: VisionPreprocessing,
    pub stem: VisionStem,
    /// Side, in patches, of the windows `Window` attention spans. The tower
    /// then holds its rows window by window; the merger gathers them back
    /// into cell order.
    pub window: Option<u64>,
    pub blocks: Vec<VisionBlock>,
    pub merger: VisionMerger,
}

/// A structural site of a vision norm.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VisionNormSite {
    Stem,
    StemProjection,
    StemPosition,
    AttentionInput,
    AttentionOutput,
    FeedForwardInput,
    FeedForwardOutput,
    Merger,
    MergerProjection,
    Output,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NormPart {
    Weight,
    Bias,
}

/// A structural site of a vision linear.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VisionLinearSite {
    Patch,
    Query,
    Key,
    Value,
    AttentionOutput,
    Gate,
    Up,
    Down,
    /// A merger stage (scope `VisionMerger(stage)`).
    Merger,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LinearPart {
    Weight,
    Bias,
    InputMinimum,
    InputMaximum,
    OutputMinimum,
    OutputMaximum,
}

/// The semantic role of a vision weight within its scope (`Vision`,
/// `VisionPatch(frame)`, `VisionBlock(block)` or `VisionMerger(stage)`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VisionWeight {
    /// One frame's patch convolution (scope `VisionPatch(frame)`).
    PatchFrame,
    /// The patch convolution's bias.
    PatchBias,
    Position,
    Norm {
        site: VisionNormSite,
        part: NormPart,
    },
    Linear {
        site: VisionLinearSite,
        part: LinearPart,
    },
    QueryNorm,
    KeyNorm,
    StandardizeBias,
    StandardizeScale,
}

type Roles<'a> = Vec<(WeightRole, &'a WeightDescriptor)>;

fn role(scope: WeightScope, weight: VisionWeight) -> WeightRole {
    WeightRole {
        scope,
        kind: WeightKind::Vision(weight),
    }
}

fn u32_index(index: usize) -> u32 {
    u32::try_from(index).expect("vision component counts fit u32")
}

impl VisionDescription {
    /// The patch row width the processor produces: `frames · C · patch²`.
    pub fn patch_row_width(&self) -> Result<u64, DefinitionError> {
        let p = &self.preprocessing;
        checked_product(&[self.stem.frames(), p.channels, p.patch, p.patch])
    }

    /// Tower rows of one merged cell.
    pub fn cell_rows(&self) -> u64 {
        self.preprocessing.merge * self.preprocessing.merge
    }

    pub fn image_processor_config(&self) -> Result<ImageProcessorConfig, DefinitionError> {
        self.validate()?;
        let p = &self.preprocessing;
        let size = |value: u64, what: &str| {
            usize::try_from(value)
                .map_err(|_| DefinitionError::new(format!("image {what} exceeds host domain")))
        };
        let resize = match p.resize {
            VisionResize::PixelBounds {
                min_pixels,
                max_pixels,
            } => ImageResize::PixelBounds {
                min_pixels: size(min_pixels, "minimum pixels")?,
                max_pixels: size(max_pixels, "maximum pixels")?,
            },
            VisionResize::PatchBudget { max_patches } => ImageResize::PatchBudget {
                max_patches: size(max_patches, "patch budget")?,
            },
            VisionResize::CellBudget { max_cells } => ImageResize::CellBudget {
                max_cells: size(max_cells, "cell budget")?,
            },
        };
        let config = ImageProcessorConfig {
            resize,
            resampling: match p.resampling {
                VisionResampling::Bicubic => Resampling::Bicubic,
                VisionResampling::Lanczos => Resampling::Lanczos,
            },
            patch: size(p.patch, "patch size")?,
            merge: size(p.merge, "merge size")?,
            frames: size(self.stem.frames(), "frame count")?,
            mean: p.mean.map(|value| value as f32),
            std: p.std.map(|value| value as f32),
        };
        config.validate().map_err(DefinitionError::new)?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), DefinitionError> {
        let p = &self.preprocessing;
        if self.hidden == 0
            || self.output_hidden == 0
            || p.channels != 3
            || p.patch == 0
            || p.merge == 0
            || p.mean.iter().any(|value| !value.is_finite())
            || p.std.iter().any(|value| !value.is_finite() || *value <= 0.0)
            || match p.resize {
                VisionResize::PixelBounds {
                    min_pixels,
                    max_pixels,
                } => min_pixels == 0 || min_pixels > max_pixels,
                VisionResize::PatchBudget { max_patches } => max_patches == 0,
                VisionResize::CellBudget { max_cells } => max_cells == 0,
            }
            || self.window == Some(0)
        {
            return Err(DefinitionError::new("invalid vision preprocessing"));
        }
        let hidden = self.hidden;
        let patch_pixels = checked_product(&[p.channels, p.patch, p.patch])?;
        match &self.stem {
            VisionStem::Patch {
                frames,
                bias,
                positions,
                norm,
            } => {
                if frames.is_empty() || frames.len() > 2 {
                    return Err(DefinitionError::new(
                        "a vision patch stem convolves one or two frames",
                    ));
                }
                for frame in frames {
                    expect_shape(
                        frame,
                        &[hidden, p.channels, p.patch, p.patch],
                        "vision patch convolution",
                    )?;
                }
                if let Some(bias) = bias {
                    expect_shape(bias, &[hidden], "vision patch bias")?;
                }
                positions.validate(hidden)?;
                if let Some(norm) = norm {
                    norm.validate(hidden, "vision stem norm")?;
                }
            }
            VisionStem::NormalizedPatch {
                input_norm,
                projection,
                projection_norm,
                positions,
                position_norm,
            } => {
                input_norm.validate(patch_pixels, "vision patch norm")?;
                projection.validate(hidden, patch_pixels, "vision patch projection")?;
                projection_norm.validate(hidden, "vision patch projection norm")?;
                positions.validate(hidden)?;
                position_norm.validate(hidden, "vision position norm")?;
            }
        }
        for block in &self.blocks {
            self.validate_block(block)?;
        }
        let merger = &self.merger;
        if let Some(norm) = &merger.norm {
            norm.validate(hidden, "vision merger norm")?;
        }
        let mut width = match merger.reduction {
            CellReduction::Concatenate | CellReduction::Interleave => {
                checked_product(&[self.cell_rows(), hidden])?
            }
            CellReduction::Average { scale } => {
                if !scale.is_finite() || scale == 0.0 {
                    return Err(DefinitionError::new("invalid vision cell average scale"));
                }
                hidden
            }
        };
        if let Some(standardize) = &merger.standardize {
            expect_shape(&standardize.bias, &[width], "vision standardization")?;
            expect_shape(&standardize.scale, &[width], "vision standardization")?;
        }
        if let Some(norm) = &merger.projection_norm {
            norm.validate(width, "vision merger projection norm")?;
        }
        let Some(last) = merger.stages.last() else {
            return Err(DefinitionError::new("a vision merger needs a linear stage"));
        };
        if last.linear.outputs() != self.output_hidden {
            return Err(DefinitionError::new(
                "the last vision merger stage must produce the decoder width",
            ));
        }
        for stage in &merger.stages {
            stage
                .linear
                .validate(stage.linear.outputs(), width, "vision merger stage")?;
            width = stage.linear.outputs();
        }
        if let Some(norm) = &merger.output_norm {
            norm.validate(width, "vision feature norm")?;
        }
        Ok(())
    }

    fn validate_block(&self, block: &VisionBlock) -> Result<(), DefinitionError> {
        let hidden = self.hidden;
        let attention = &block.attention;
        let inner = checked_product(&[attention.heads, attention.width])?;
        if attention.heads == 0
            || attention.width == 0
            || attention.width % 4 != 0
            || !attention.rotary_base.is_finite()
            || attention.rotary_base <= 1.0
            || attention.query_norm.is_some() != attention.key_norm.is_some()
            || (attention.span == VisionAttentionSpan::Window && self.window.is_none())
        {
            return Err(DefinitionError::new("invalid vision attention"));
        }
        block
            .attention_norm
            .validate(hidden, "vision attention norm")?;
        for (linear, role) in [
            (&attention.query, "vision query"),
            (&attention.key, "vision key"),
            (&attention.value, "vision value"),
        ] {
            linear.validate(inner, hidden, role)?;
        }
        for norm in [&attention.query_norm, &attention.key_norm]
            .into_iter()
            .flatten()
        {
            crate::decoder::validate_rms(norm, attention.width, "vision head norm")?;
        }
        if let Some(norm) = &attention.value_norm {
            if !norm.epsilon.is_finite() || norm.epsilon <= 0.0 {
                return Err(DefinitionError::new("invalid vision value norm"));
            }
        }
        attention
            .output
            .validate(hidden, inner, "vision attention output")?;
        if let Some(norm) = &block.attention_post_norm {
            norm.validate(hidden, "vision attention post-norm")?;
        }
        block
            .feedforward_norm
            .validate(hidden, "vision feed-forward norm")?;
        let feedforward = &block.feedforward;
        let intermediate = feedforward.intermediate();
        if intermediate == 0 {
            return Err(DefinitionError::new("invalid vision feed-forward width"));
        }
        match &feedforward.up {
            VisionUp::Plain(up) => up.validate(intermediate, hidden, "vision up projection")?,
            VisionUp::Gated { gate, up } => {
                gate.validate(intermediate, hidden, "vision gate projection")?;
                up.validate(intermediate, hidden, "vision up projection")?;
            }
        }
        feedforward
            .down
            .validate(hidden, intermediate, "vision down projection")?;
        if let Some(norm) = &block.feedforward_post_norm {
            norm.validate(hidden, "vision feed-forward post-norm")?;
        }
        Ok(())
    }

    /// Every weight of the projector with its structural role, in model
    /// order.
    pub fn weights(&self) -> Vec<(WeightRole, &WeightDescriptor)> {
        let mut out = Vec::new();
        let vision = WeightScope::Vision;
        match &self.stem {
            VisionStem::Patch {
                frames,
                bias,
                positions,
                norm,
            } => {
                for (index, frame) in frames.iter().enumerate() {
                    out.push((
                        role(
                            WeightScope::VisionPatch(u32_index(index)),
                            VisionWeight::PatchFrame,
                        ),
                        frame,
                    ));
                }
                if let Some(bias) = bias {
                    out.push((role(vision, VisionWeight::PatchBias), bias));
                }
                out.push((role(vision, VisionWeight::Position), &positions.table));
                if let Some(norm) = norm {
                    norm.weights(vision, VisionNormSite::Stem, &mut out);
                }
            }
            VisionStem::NormalizedPatch {
                input_norm,
                projection,
                projection_norm,
                positions,
                position_norm,
            } => {
                input_norm.weights(vision, VisionNormSite::Stem, &mut out);
                projection.weights(vision, VisionLinearSite::Patch, &mut out);
                projection_norm.weights(vision, VisionNormSite::StemProjection, &mut out);
                out.push((role(vision, VisionWeight::Position), &positions.table));
                position_norm.weights(vision, VisionNormSite::StemPosition, &mut out);
            }
        }
        for (index, block) in self.blocks.iter().enumerate() {
            let scope = WeightScope::VisionBlock(u32_index(index));
            let attention = &block.attention;
            block
                .attention_norm
                .weights(scope, VisionNormSite::AttentionInput, &mut out);
            attention
                .query
                .weights(scope, VisionLinearSite::Query, &mut out);
            attention.key.weights(scope, VisionLinearSite::Key, &mut out);
            attention
                .value
                .weights(scope, VisionLinearSite::Value, &mut out);
            if let Some(norm) = &attention.query_norm {
                out.push((role(scope, VisionWeight::QueryNorm), &norm.weight));
            }
            if let Some(norm) = &attention.key_norm {
                out.push((role(scope, VisionWeight::KeyNorm), &norm.weight));
            }
            attention
                .output
                .weights(scope, VisionLinearSite::AttentionOutput, &mut out);
            if let Some(norm) = &block.attention_post_norm {
                norm.weights(scope, VisionNormSite::AttentionOutput, &mut out);
            }
            block
                .feedforward_norm
                .weights(scope, VisionNormSite::FeedForwardInput, &mut out);
            match &block.feedforward.up {
                VisionUp::Plain(up) => up.weights(scope, VisionLinearSite::Up, &mut out),
                VisionUp::Gated { gate, up } => {
                    gate.weights(scope, VisionLinearSite::Gate, &mut out);
                    up.weights(scope, VisionLinearSite::Up, &mut out);
                }
            }
            block
                .feedforward
                .down
                .weights(scope, VisionLinearSite::Down, &mut out);
            if let Some(norm) = &block.feedforward_post_norm {
                norm.weights(scope, VisionNormSite::FeedForwardOutput, &mut out);
            }
        }
        let merger = &self.merger;
        if let Some(norm) = &merger.norm {
            norm.weights(vision, VisionNormSite::Merger, &mut out);
        }
        if let Some(standardize) = &merger.standardize {
            out.push((role(vision, VisionWeight::StandardizeBias), &standardize.bias));
            out.push((role(vision, VisionWeight::StandardizeScale), &standardize.scale));
        }
        if let Some(norm) = &merger.projection_norm {
            norm.weights(vision, VisionNormSite::MergerProjection, &mut out);
        }
        for (index, stage) in merger.stages.iter().enumerate() {
            stage.linear.weights(
                WeightScope::VisionMerger(u32_index(index)),
                VisionLinearSite::Merger,
                &mut out,
            );
        }
        if let Some(norm) = &merger.output_norm {
            norm.weights(vision, VisionNormSite::Output, &mut out);
        }
        out
    }
}
