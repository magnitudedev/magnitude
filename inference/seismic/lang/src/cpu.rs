//! The CPU's facts: the layout of its quantized activations and the most
//! participants a worker pool holds. The checker reads them as the named
//! constants of the CPU's native declarations (`forms::constants`), the
//! runtime and the API size the operand carry by them, and the CPU crates
//! (`seismic-native-cpu`'s `quant`, `seismic-cpu`'s `workers`) are held to
//! them.

/// Values of one CPU INT8 activation block (`quant::Q8Block`, the k-quant
/// super block): its `i8` codes, then the `i16` sums of each 16, then its
/// `f32` scale.
pub const CPU_Q8_VALUES: u64 = 256;
/// The alignment of a CPU INT8 activation block: the widest vector of any
/// CPU tier (`isa::Tier::vector_bytes`, the X86V4 tiers' 512-bit), so no
/// vector load of its codes straddles a cache line, wherever a block lies.
pub const CPU_Q8_ALIGNMENT: u64 = 64;
/// Bytes of one CPU INT8 activation block, its fields padded to its
/// alignment.
pub const CPU_Q8_BYTES: u64 = (CPU_Q8_VALUES + CPU_Q8_VALUES / 16 * 2 + 4).next_multiple_of(CPU_Q8_ALIGNMENT);

/// Activation rows of one CPU INT8 tile (`quant::Q8Tile`), the tile
/// kernel's strip: several rows quantized for one projection lie as tiles
/// of this many, each tile at most this many blocks' bytes, so `ceil_div(M,
/// Q8_ROWS) * Q8_ROWS` blocks a row hold either form.
pub const CPU_Q8_ROWS: u64 = 8;

/// Most participants a CPU worker pool holds (`seismic_cpu::workers::Workers`),
/// which bounds every `device.units` partition of a scratch buffer.
pub const CPU_MAX_UNITS: u64 = 256;
