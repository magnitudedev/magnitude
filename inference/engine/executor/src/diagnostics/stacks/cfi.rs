//! A stack unwound through its images' unwind tables (`.eh_frame`, found by
//! `.eh_frame_hdr`'s sorted table), from the registers its thread left:
//! for x86-64 Linux, whose code keeps no frame records by default, so a
//! walk of records stops at the first function that keeps none. Unwinding
//! with the tables needs no frame pointer, at no cost to the code
//! (`-C force-frame-pointers` would cost every function a register).
//!
//! Nothing here allocates, locks or loads, as the report requires
//! (`super`). The images are the loader's own list (`_r_debug`'s link
//! maps), read as the report begins, while the loader says it is
//! consistent: always current, a `dlopen`ed library's included, with no
//! hook to install and no `dl_iterate_phdr` (which takes the loader's
//! lock). The tables are read where the loader mapped them; the stacks,
//! and every header of the list, through the probe (`super::platform::copy`),
//! which fails rather than faults.

use super::platform::copy;
use super::FRAMES;
use gimli::{
    BaseAddresses, CfaRule, EhFrame, EhFrameHdr, LittleEndian, Pointer, Register, RegisterRule, UnwindContext,
    UnwindContextStorage, UnwindSection, UnwindTableRow, X86_64,
};

/// The most images, and executable segments of each, the report reads.
const IMAGES: usize = 512;
const SEGMENTS: usize = 8;

/// `PT_LOAD`, `PT_PHDR`, `PT_GNU_EH_FRAME` and `PF_X`.
const LOAD: u32 = 1;
const PHDR: u32 = 6;
const GNU_EH_FRAME: u32 = 0x6474_e550;
const EXECUTABLE: u32 = 1;

/// One image's segments and unwind tables, as mapped.
#[derive(Clone, Copy, Default)]
struct Image {
    /// Each loaded segment: its start and end, and whether it is code.
    segments: [(u64, u64, bool); SEGMENTS],
    count: usize,
    /// `.eh_frame_hdr` and its length.
    header: (u64, u64),
}

impl Image {
    fn code(&self, address: u64) -> bool {
        self.segments[..self.count].iter().any(|&(start, end, code)| code && (start..end).contains(&address))
    }

    /// The end of the loaded segment `address` lies in.
    fn segment_end(&self, address: u64) -> Option<u64> {
        self.segments[..self.count].iter().find(|&&(start, end, _)| (start..end).contains(&address)).map(|&(_, end, _)| end)
    }
}

/// glibc's `struct r_debug`.
#[repr(C)]
struct Debug {
    _version: i32,
    map: u64,
    _brk: u64,
    state: i32,
    _base: u64,
}

unsafe extern "C" {
    static _r_debug: Debug;
}

/// `RT_CONSISTENT`: no `dlopen` or `dlclose` is changing the list.
const CONSISTENT: i32 = 0;

/// The unwinding rules' storage, inline: no allocation.
struct Inline;

impl<T: gimli::ReaderOffset> UnwindContextStorage<T> for Inline {
    type Rules = [(Register, RegisterRule<T>); 32];
    type Stack = [UnwindTableRow<T, Self>; 4];
}

/// The loaded images and an unwinding context, on the reporter's stack.
pub(super) struct Images {
    images: [Image; IMAGES],
    count: usize,
    context: UnwindContext<usize, Inline>,
}

/// `count` bytes at `address` as a little-endian integer.
fn field(bytes: &[u8], at: usize, count: usize) -> u64 {
    bytes[at..at + count].iter().rev().fold(0, |value, &byte| value << 8 | u64::from(byte))
}

impl Images {
    pub(super) fn new() -> Self {
        Self { images: [Image::default(); IMAGES], count: 0, context: UnwindContext::new_in() }
    }

    /// Read the loader's list of images and each one's segments and tables:
    /// why not, where the loader is changing it.
    pub(super) fn read(&mut self) -> Result<(), &'static str> {
        // SAFETY: the loader's own structure, which lives for the process;
        // read field by field, volatile, as the loader changes it.
        let (state, mut map) = unsafe {
            let debug = std::ptr::addr_of!(_r_debug);
            (std::ptr::addr_of!((*debug).state).read_volatile(), std::ptr::addr_of!((*debug).map).read_volatile())
        };
        if state != CONSISTENT {
            return Err("the loader is changing its images");
        }
        let mut first = true;
        while map != 0 && self.count < IMAGES {
            // `struct link_map`: l_addr, l_name, l_ld, l_next, l_prev.
            let mut link = [0u8; 40];
            if !copy(map, &mut link) {
                break;
            }
            let bias = field(&link, 0, 8);
            if let Some(image) = image(bias, first) {
                self.images[self.count] = image;
                self.count += 1;
            }
            first = false;
            map = field(&link, 24, 8);
        }
        Ok(())
    }

    fn containing(&self, address: u64) -> Option<&Image> {
        self.images[..self.count].iter().find(|image| image.code(address))
    }

    /// The stack whose innermost frame is at `pc`, with stack pointer `sp`
    /// and frame pointer `fp`, unwound into `frames`: how many.
    pub(super) fn unwind(&mut self, mut pc: u64, mut sp: u64, mut fp: u64, frames: &mut [u64; FRAMES]) -> usize {
        let mut count = 0;
        while count < FRAMES && pc != 0 {
            frames[count] = pc;
            count += 1;
            // A return address is one past its call: the call's own row.
            let at = if count == 1 { pc } else { pc - 1 };
            let Some(image) = self.containing(at).copied() else { break };
            let Some((cfa, returned, frame)) = self.row(&image, at, sp, fp) else { break };
            if returned == 0 || cfa <= sp {
                break;
            }
            (pc, sp, fp) = (returned, cfa, frame);
        }
        count
    }

    /// At `pc` of `image`, the canonical frame address, the return address
    /// and the caller's frame pointer.
    fn row(&mut self, image: &Image, pc: u64, sp: u64, fp: u64) -> Option<(u64, u64, u64)> {
        let (header, length) = image.header;
        // SAFETY: an image the loader holds mapped while its list is
        // consistent; its table header lies in a loaded segment.
        let header_bytes = unsafe { std::slice::from_raw_parts(header as *const u8, length as usize) };
        let bases = BaseAddresses::default().set_eh_frame_hdr(header);
        let parsed = EhFrameHdr::new(header_bytes, LittleEndian).parse(&bases, 8).ok()?;
        let Pointer::Direct(eh_frame) = parsed.eh_frame_ptr() else { return None };
        let end = image.segment_end(eh_frame)?;
        // SAFETY: the rest of the loaded segment `.eh_frame` starts in.
        let eh_frame_bytes = unsafe { std::slice::from_raw_parts(eh_frame as *const u8, (end - eh_frame) as usize) };
        let section = EhFrame::new(eh_frame_bytes, LittleEndian);
        let bases = bases.set_eh_frame(eh_frame);
        let table = parsed.table()?;
        let row = table.unwind_info_for_address(&section, &bases, &mut self.context, pc, |section, bases, offset| section.cie_from_offset(bases, offset)).ok()?;
        let register = |register: Register| match register {
            X86_64::RSP => Some(sp),
            X86_64::RBP => Some(fp),
            _ => None,
        };
        let cfa = match row.cfa() {
            CfaRule::RegisterAndOffset { register: base, offset } => register(*base)?.wrapping_add_signed(*offset),
            CfaRule::Expression(_) => return None,
        };
        let saved = |rule: RegisterRule<usize>, current: u64| match rule {
            RegisterRule::Offset(offset) => {
                let mut word = [0u8; 8];
                copy(cfa.wrapping_add_signed(offset), &mut word).then(|| u64::from_le_bytes(word))
            }
            RegisterRule::SameValue | RegisterRule::Undefined => Some(current),
            _ => None,
        };
        let returned = saved(row.register(X86_64::RA), 0)?;
        let frame = saved(row.register(X86_64::RBP), fp).unwrap_or(fp);
        Some((cfa, returned, frame))
    }
}

/// The image whose load bias is `bias`: its program headers, from the
/// ELF header the loader mapped at its bias (a shared object), or the
/// process's own (`AT_PHDR`, the first image, the program); none where
/// they cannot be read or it has no unwind table.
fn image(bias: u64, program: bool) -> Option<Image> {
    let (headers, count) = if program {
        // SAFETY: reads the auxiliary vector; takes nothing.
        unsafe { (libc::getauxval(libc::AT_PHDR), libc::getauxval(libc::AT_PHNUM)) }
    } else {
        let mut elf = [0u8; 64];
        if !copy(bias, &mut elf) || elf[..4] != *b"\x7fELF" {
            return None;
        }
        (bias + field(&elf, 32, 8), field(&elf, 56, 2))
    };
    let mut image = Image::default();
    // The program's bias is where its headers are less where they say
    // they are (`PT_PHDR`); a shared object's is the loader's.
    let mut bias = bias;
    for index in 0..count.min(64) {
        let mut header = [0u8; 56];
        if !copy(headers + index * 56, &mut header) {
            return None;
        }
        let (kind, flags, address, size) = (field(&header, 0, 4) as u32, field(&header, 4, 4) as u32, field(&header, 16, 8), field(&header, 40, 8));
        match kind {
            PHDR if program => bias = headers - address,
            LOAD if image.count < SEGMENTS => {
                image.segments[image.count] = (address, address + size, flags & EXECUTABLE != 0);
                image.count += 1;
            }
            GNU_EH_FRAME => image.header = (address, size),
            _ => {}
        }
    }
    if image.header.1 == 0 {
        return None;
    }
    for segment in &mut image.segments[..image.count] {
        (segment.0, segment.1) = (segment.0 + bias, segment.1 + bias);
    }
    image.header.0 += bias;
    Some(image)
}
