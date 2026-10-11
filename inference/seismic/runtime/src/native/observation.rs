//! Capturing what one native call published: its returned values and the
//! observed regions of its writable state, as the subjects a comparison
//! measures. Nothing here compares or judges.
use super::tune::{Exclusion, WrittenRows};
use crate::api::kernel::{DecodedValue, EncodedArgs};
use seismic_compiler::numerics::{input_subject, result_subject};
use seismic_compiler::prepared::ArgumentValue;
use seismic_lang::{
    entry::LogicalEntry,
    ids::RepresentationId,
    registry::{self, RepresentationKind},
};
use sha2::{Digest, Sha256};
use std::{collections::HashMap, sync::Arc};

/// One thing a call publishes, as the comparison names it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Subject {
    pub name: String,
    /// The subject is outside what the entry may change: any difference
    /// from the default is a write outside the declared written rows.
    pub unwritten: bool,
}

/// What of one `&mut` parameter a point observes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct StateRegion {
    pub ordinal: usize,
    /// The rows the entry writes; the whole tensor when none are declared.
    pub written: Option<WrittenRows>,
}

/// The subjects of a call of `logical` observing `states`, in the order
/// [`observe`] captures them: every result, then per state its written
/// region and, where observed, its other rows.
pub(super) fn subjects(logical: &LogicalEntry, states: &[StateRegion]) -> Vec<Subject> {
    let results = logical.schema().results().iter().map(|result| Subject {
        name: result_subject(&result.path),
        unwritten: false,
    });
    let states = states.iter().flat_map(|state| {
        let name = input_subject(state.ordinal);
        let rest = state
            .written
            .as_ref()
            .filter(|written| written.observe_rest)
            .map(|_| Subject {
                name: format!("{name}:unwritten"),
                unwritten: true,
            });
        std::iter::once(Subject {
            name,
            unwritten: false,
        })
        .chain(rest)
    });
    results.chain(states).collect()
}

#[derive(Clone, Debug)]
pub(super) enum Observation {
    Tensor {
        representation: RepresentationId,
        shape: Vec<u64>,
        bytes: Vec<Page>,
    },
    Scalar(ArgumentValue),
}

// State views overlap across row classes and layer rotations. Retain identical
// immutable pages once, while still observing and comparing every byte of every case.
pub(super) const OBSERVATION_PAGE: usize = 64 * 1024;

/// Host memory of the pages one capture adds, apart from the process's
/// heap: taken from the system when the capture is made and given back to
/// it, whole and at once, when the last of its pages is released. An
/// observation is as large as the tensor it observes and lives as long as
/// its unit's point is resident; held in the heap, its pages were the
/// allocator's to keep once freed, and what a search had observed stayed in
/// the process for its life.
struct Region {
    pointer: std::ptr::NonNull<u8>,
    bytes: usize,
}

// SAFETY: the region's bytes are written once, before it is shared, and
// only read after; it owns its mapping.
unsafe impl Send for Region {}
unsafe impl Sync for Region {}

impl Region {
    /// A region holding `parts` one after another. `parts` hold a byte.
    fn holding(parts: &[&[u8]]) -> std::io::Result<Arc<Self>> {
        let bytes = parts.iter().map(|part| part.len()).sum::<usize>();
        let region = Self {
            pointer: system::map(bytes)?,
            bytes,
        };
        let mut written = 0;
        for part in parts {
            // SAFETY: the mapping is `bytes` long, `written + part.len()`
            // is within it, and nothing else refers to it yet.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    part.as_ptr(),
                    region.pointer.as_ptr().add(written),
                    part.len(),
                );
            }
            written += part.len();
        }
        Ok(Arc::new(region))
    }
}

impl Drop for Region {
    fn drop(&mut self) {
        // SAFETY: the mapping `map` returned for exactly these bytes, which
        // no page refers to any more.
        unsafe { system::unmap(self.pointer, self.bytes) };
    }
}

/// Memory mapped from the system and returned to it, outside the heap.
mod system {
    use std::ptr::NonNull;

    #[cfg(unix)]
    pub(super) fn map(bytes: usize) -> std::io::Result<NonNull<u8>> {
        // SAFETY: an anonymous private mapping at an address the system
        // chooses refers to no existing memory.
        let pointer = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                bytes,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            )
        };
        if pointer == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error());
        }
        Ok(NonNull::new(pointer.cast()).expect("a mapping has an address"))
    }

    /// # Safety
    /// `pointer` is a mapping of `bytes` bytes from [`map`], not yet
    /// unmapped, that nothing refers to.
    #[cfg(unix)]
    pub(super) unsafe fn unmap(pointer: NonNull<u8>, bytes: usize) {
        // SAFETY: the caller's contract.
        let unmapped = unsafe { libc::munmap(pointer.as_ptr().cast(), bytes) };
        debug_assert_eq!(unmapped, 0, "a whole mapping is unmapped");
    }

    #[cfg(windows)]
    pub(super) fn map(bytes: usize) -> std::io::Result<NonNull<u8>> {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc,
        };
        // SAFETY: a new committed region at an address the system chooses
        // refers to no existing memory.
        let pointer = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                bytes,
                MEM_COMMIT | MEM_RESERVE,
                PAGE_READWRITE,
            )
        };
        NonNull::new(pointer.cast()).ok_or_else(std::io::Error::last_os_error)
    }

    /// # Safety
    /// `pointer` is a region from [`map`], not yet released, that nothing
    /// refers to.
    #[cfg(windows)]
    pub(super) unsafe fn unmap(pointer: NonNull<u8>, _bytes: usize) {
        use windows_sys::Win32::System::Memory::{MEM_RELEASE, VirtualFree};
        // SAFETY: the caller's contract; a whole region is released by its
        // base address.
        let released = unsafe { VirtualFree(pointer.as_ptr().cast(), 0, MEM_RELEASE) };
        debug_assert_ne!(released, 0, "a whole region is released");
    }
}

/// One immutable page of an observation: at most [`OBSERVATION_PAGE`] bytes
/// of a [`Region`]. Clones share the page.
#[derive(Clone)]
pub(super) struct Page(Arc<Span>);

struct Span {
    region: Arc<Region>,
    start: usize,
    len: usize,
}

impl Page {
    /// The pages of `region` from `start` on, `len` bytes of it.
    fn of(region: &Arc<Region>, start: usize, len: usize) -> Self {
        Self(Arc::new(Span {
            region: region.clone(),
            start,
            len,
        }))
    }

    /// Whether anything but this handle holds the page.
    fn shared(&self) -> bool {
        Arc::strong_count(&self.0) > 1
    }

    /// Whether the two are one page, not merely equal bytes.
    #[cfg(test)]
    fn same(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl std::ops::Deref for Page {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        // SAFETY: the span lies within its region, which the page keeps
        // mapped and nothing writes once it is shared.
        unsafe {
            std::slice::from_raw_parts(
                self.0.region.pointer.as_ptr().add(self.0.start),
                self.0.len,
            )
        }
    }
}

impl PartialEq for Page {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0) || **self == **other
    }
}

impl std::fmt::Debug for Page {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("Page").field("len", &self.0.len).finish()
    }
}

/// The pages of `bytes`, in a region of their own.
pub(super) fn pages_of(bytes: &[u8]) -> std::io::Result<Vec<Page>> {
    if bytes.is_empty() {
        return Ok(Vec::new());
    }
    let region = Region::holding(&[bytes])?;
    Ok((0..bytes.len())
        .step_by(OBSERVATION_PAGE)
        .map(|start| Page::of(&region, start, OBSERVATION_PAGE.min(bytes.len() - start)))
        .collect())
}

/// The reference's retained pages, each distinct page held once. They are
/// host memory the unit holds while the observations that share them are
/// resident, and they are counted ([`ReferencePages::retained`]): what a
/// unit's work holds is bounded where its work is admitted, on its whole
/// footprint, not here. The pages a capture adds are one [`Region`]: it
/// leaves the process when the observations that share its pages are
/// released, and every region has left when the unit has.
#[derive(Default)]
pub(super) struct ReferencePages {
    pages: HashMap<[u8; 32], Vec<Page>>,
    retained: u64,
    /// The most `retained` has been since the mark was last restarted.
    high_water: u64,
}

impl ReferencePages {
    fn capture(&mut self, bytes: &[u8]) -> std::io::Result<Vec<Page>> {
        /// A page of the capture: one already held, or the capture's own
        /// `n`th new page.
        enum Captured {
            Held(Page),
            New(usize),
        }
        let mut new = Vec::<([u8; 32], &[u8])>::new();
        let captured = bytes
            .chunks(OBSERVATION_PAGE)
            .map(|page| {
                let key: [u8; 32] = Sha256::digest(page).into();
                let held = self
                    .pages
                    .get(&key)
                    .and_then(|matches| matches.iter().find(|held| ***held == *page));
                if let Some(held) = held {
                    return Captured::Held(held.clone());
                }
                let position = new
                    .iter()
                    .position(|(seen, bytes)| *seen == key && *bytes == page);
                Captured::New(position.unwrap_or_else(|| {
                    new.push((key, page));
                    new.len() - 1
                }))
            })
            .collect::<Vec<_>>();
        let added = if new.is_empty() {
            Vec::new()
        } else {
            let region =
                Region::holding(&new.iter().map(|(_, page)| *page).collect::<Vec<_>>())?;
            let mut start = 0;
            new.iter()
                .map(|(key, bytes)| {
                    let page = Page::of(&region, start, bytes.len());
                    start += bytes.len();
                    self.retained += bytes.len() as u64;
                    self.pages.entry(*key).or_default().push(page.clone());
                    page
                })
                .collect::<Vec<_>>()
        };
        self.high_water = self.high_water.max(self.retained);
        Ok(captured
            .into_iter()
            .map(|page| match page {
                Captured::Held(page) => page,
                Captured::New(position) => added[position].clone(),
            })
            .collect())
    }

    /// Host bytes of the distinct pages held.
    pub(super) fn retained(&self) -> u64 {
        self.retained
    }

    /// The most [`ReferencePages::retained`] has been since the mark was
    /// last restarted.
    pub(super) fn high_water(&self) -> u64 {
        self.high_water
    }

    pub(super) fn restart_high_water(&mut self) {
        self.high_water = self.retained;
    }

    /// Give up every page no observation shares any more: the
    /// observations that held them were released.
    pub(super) fn release_unshared(&mut self) {
        let retained = &mut self.retained;
        self.pages.retain(|_, matches| {
            matches.retain(|page| {
                let shared = page.shared();
                if !shared {
                    *retained -= page.len() as u64;
                }
                shared
            });
            !matches.is_empty()
        });
    }
}

fn pages(bytes: &[u8], retain: Option<&mut ReferencePages>) -> Result<Vec<Page>, Exclusion> {
    match retain {
        Some(retain) => retain.capture(bytes),
        None => pages_of(bytes),
    }
    .map_err(|error| Exclusion::Execution(format!("host memory for an observation: {error}")))
}

/// Capture the subjects of one completed call: its returned `values`, and
/// the observed regions of the `&mut` tensors `args` binds. `retain` holds
/// the reference's pages.
pub(super) fn observe(
    values: Vec<DecodedValue>,
    args: &EncodedArgs,
    states: &[StateRegion],
    mut retain: Option<&mut ReferencePages>,
) -> Result<Vec<Observation>, Exclusion> {
    let execution = |error: &dyn std::fmt::Display| Exclusion::Execution(error.to_string());
    let mut observed = Vec::new();
    for value in values {
        observed.push(match value {
            DecodedValue::Tensor(tensor) => Observation::Tensor {
                representation: tensor.representation(),
                shape: tensor.descriptor().extents,
                bytes: pages(
                    &tensor.read_to_host().map_err(|e| execution(&e))?,
                    retain.as_deref_mut(),
                )?,
            },
            DecodedValue::Scalar(value) => Observation::Scalar(value),
        });
    }
    for state in states {
        let tensor = args.tensor(state.ordinal).expect("checked writable tensor");
        let representation = tensor.representation();
        let mut shape = tensor.descriptor().extents;
        match &state.written {
            // The written rows alone are read.
            Some(written) if !written.observe_rest => {
                let slice = tensor
                    .slice_leading(written.rows.start, written.rows.end)
                    .map_err(|e| execution(&e))?;
                observed.push(Observation::Tensor {
                    representation,
                    shape: slice.descriptor().extents,
                    bytes: pages(
                        &slice.read_to_host().map_err(|e| execution(&e))?,
                        retain.as_deref_mut(),
                    )?,
                });
            }
            // Packed storage admits no difference anywhere, so its other
            // rows are held exact as part of the whole tensor.
            Some(written)
                if !matches!(
                    registry::representation_info(representation).kind,
                    RepresentationKind::Dense(_)
                ) =>
            {
                let slice = tensor
                    .slice_leading(written.rows.start, written.rows.end)
                    .map_err(|e| execution(&e))?;
                for (tensor, shape) in [(&slice, slice.descriptor().extents), (&**tensor, shape)] {
                    observed.push(Observation::Tensor {
                        representation,
                        shape,
                        bytes: pages(
                            &tensor.read_to_host().map_err(|e| execution(&e))?,
                            retain.as_deref_mut(),
                        )?,
                    });
                }
            }
            // The whole dense tensor is read and divided at the written rows.
            Some(written) => {
                let bytes = tensor.read_to_host().map_err(|e| execution(&e))?;
                let rows = shape[0];
                let row = (bytes.len() as u64)
                    .checked_div(rows)
                    .filter(|row| row * rows == bytes.len() as u64 && written.rows.end <= rows);
                let Some(row) = row else {
                    return Err(Exclusion::Execution(format!(
                        "the rows of `&mut` parameter {} outside {:?} cannot be told from its {} bytes",
                        state.ordinal,
                        written.rows,
                        bytes.len()
                    )));
                };
                let (start, end) = (
                    (written.rows.start * row) as usize,
                    (written.rows.end * row) as usize,
                );
                shape[0] = written.rows.end - written.rows.start;
                observed.push(Observation::Tensor {
                    representation,
                    shape: shape.clone(),
                    bytes: pages(&bytes[start..end], retain.as_deref_mut())?,
                });
                shape[0] = rows - shape[0];
                observed.push(Observation::Tensor {
                    representation,
                    shape,
                    bytes: pages(
                        &[&bytes[..start], &bytes[end..]].concat(),
                        retain.as_deref_mut(),
                    )?,
                });
            }
            None => observed.push(Observation::Tensor {
                representation,
                shape,
                bytes: pages(
                    &tensor.read_to_host().map_err(|e| execution(&e))?,
                    retain.as_deref_mut(),
                )?,
            }),
        }
    }
    Ok(observed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_pages_share_identical_prefixes_but_keep_distinct_tails() {
        let mut pages = ReferencePages::default();
        let mut bytes = vec![3; OBSERVATION_PAGE * 2];
        let first = pages.capture(&bytes).unwrap();
        assert!(first[0].same(&first[1]));
        bytes[OBSERVATION_PAGE + 5] = 4;
        let second = pages.capture(&bytes).unwrap();
        assert!(first[0].same(&second[0]));
        assert!(!first[1].same(&second[1]));
        assert_eq!(pages.retained(), 2 * OBSERVATION_PAGE as u64);
        assert_eq!(first[1][5], 3, "captured bytes remain immutable");
        assert_eq!(second[1][5], 4);
    }

    /// A page is held exactly while an observation shares it: the bytes
    /// counted are those resident, however many were ever captured.
    #[test]
    fn released_observations_give_their_pages_back() {
        let mut pages = ReferencePages::default();
        let kept = pages.capture(&vec![1; OBSERVATION_PAGE]).unwrap();
        let dropped = pages.capture(&vec![2; 3 * OBSERVATION_PAGE]).unwrap();
        // Three identical pages are one page held.
        assert_eq!(pages.retained(), 2 * OBSERVATION_PAGE as u64);
        drop(dropped);
        pages.release_unshared();
        assert_eq!(pages.retained(), OBSERVATION_PAGE as u64);
        // The page still shared is found again, not captured twice.
        let again = pages.capture(&vec![1; OBSERVATION_PAGE]).unwrap();
        assert!(kept[0].same(&again[0]));
        drop((kept, again));
        pages.release_unshared();
        assert_eq!(pages.retained(), 0);
    }

    /// A capture's new pages are one region outside the heap, and the
    /// region is the system's again when the last of its pages is released:
    /// what a unit observed leaves the process with the unit.
    #[test]
    fn a_captures_region_is_returned_with_its_last_page() {
        let mut pages = ReferencePages::default();
        // The last page of a capture is as long as what is left of it.
        let mut bytes = vec![7u8; 2 * OBSERVATION_PAGE + 100];
        bytes[OBSERVATION_PAGE] = 8;
        let captured = pages.capture(&bytes).unwrap();
        assert_eq!(
            captured.iter().map(|page| page.len()).collect::<Vec<_>>(),
            [OBSERVATION_PAGE, OBSERVATION_PAGE, 100]
        );
        assert_eq!(captured.iter().flat_map(|page| page.iter().copied()).collect::<Vec<_>>(), bytes);
        let region = Arc::downgrade(&captured[0].0.region);
        // The three pages share the one region the capture mapped.
        assert!(captured.iter().all(|page| Arc::ptr_eq(&page.0.region, &captured[0].0.region)));
        // A later capture of the same first page keeps the region mapped
        // after the first capture's observation is gone.
        let later = pages.capture(&bytes[..OBSERVATION_PAGE]).unwrap();
        drop(captured);
        pages.release_unshared();
        assert_eq!(pages.retained(), OBSERVATION_PAGE as u64);
        assert!(region.upgrade().is_some());
        assert_eq!(later[0][0], 7);
        drop(later);
        pages.release_unshared();
        assert_eq!(pages.retained(), 0);
        assert!(region.upgrade().is_none(), "the region is unmapped");
        // An observation of nothing holds nothing.
        assert!(pages.capture(&[]).unwrap().is_empty());
        assert!(pages_of(&[]).unwrap().is_empty());
        // Pages kept by no reference are a region of their own.
        let transient = pages_of(&bytes).unwrap();
        assert_eq!(transient.len(), 3);
        assert_eq!(transient[2].len(), 100);
        assert_eq!(pages.retained(), 0);
    }
}
