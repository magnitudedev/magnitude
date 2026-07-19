use std::collections::BTreeMap;

use llama_cpp_2::token::LlamaToken;

pub(crate) type RadixNodeId = usize;

#[derive(Debug)]
struct RadixNode<Page> {
    parent: Option<RadixNodeId>,
    children: BTreeMap<Box<[LlamaToken]>, RadixNodeId>,
    page: Option<Page>,
    depth_tokens: usize,
    lock_refs: u32,
    hits: u64,
    last_access: u64,
}

impl<Page> RadixNode<Page> {
    fn root() -> Self {
        Self {
            parent: None,
            children: BTreeMap::new(),
            page: None,
            depth_tokens: 0,
            lock_refs: 1,
            hits: 0,
            last_access: 0,
        }
    }
}

/// Result of an exact page-aligned prefix lookup.
#[cfg(test)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RadixMatch<Page> {
    pub(crate) pages: Vec<Page>,
    pub(crate) matched_tokens: usize,
    pub(crate) terminal: RadixNodeId,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RadixPath {
    pub(crate) nodes: Vec<RadixNodeId>,
    pub(crate) matched_tokens: usize,
    pub(crate) terminal: RadixNodeId,
}

/// A page-granular token radix whose values are independently retained native KV pages.
///
/// SGLang stores compressed variable-length token edges. ICN deliberately uses one fixed-size page
/// per edge at the physical boundary: it makes every node independently attachable/evictable and
/// gives lookup O(number of pages) behavior without hashing collisions. Metadata is tiny relative
/// to one K/V page, and a later path-compressed lookup index can be layered over the same page
/// ownership protocol without changing native handles.
#[derive(Debug)]
pub(crate) struct PagedRadixCache<Page> {
    page_size: usize,
    nodes: Vec<Option<RadixNode<Page>>>,
    clock: u64,
    page_count: usize,
}

impl<Page: Clone> PagedRadixCache<Page> {
    pub(crate) fn new(page_size: usize) -> Self {
        assert!(page_size > 0, "radix page size must be non-zero");
        Self {
            page_size,
            nodes: vec![Some(RadixNode::root())],
            clock: 0,
            page_count: 0,
        }
    }

    pub(crate) fn page_size(&self) -> usize {
        self.page_size
    }

    #[cfg(test)]
    pub(crate) fn page_count(&self) -> usize {
        self.page_count
    }

    #[cfg(test)]
    pub(crate) fn match_prefix(&mut self, tokens: &[LlamaToken]) -> RadixMatch<Page> {
        let path = self.match_path(tokens);
        let pages = path
            .nodes
            .iter()
            .map(|node_id| {
                self.node(*node_id)
                    .page
                    .as_ref()
                    .expect("non-root radix node owns a page")
                    .clone()
            })
            .collect();

        RadixMatch {
            pages,
            matched_tokens: path.matched_tokens,
            terminal: path.terminal,
        }
    }

    pub(crate) fn match_path(&mut self, tokens: &[LlamaToken]) -> RadixPath {
        let aligned = tokens.len() / self.page_size * self.page_size;
        let mut node_id = 0;
        let mut nodes = Vec::with_capacity(aligned / self.page_size);

        for chunk in tokens[..aligned].chunks_exact(self.page_size) {
            let child = self.node(node_id).children.get(chunk).copied();
            let Some(child) = child else {
                break;
            };
            node_id = child;
            let now = self.tick();
            let node = self.node_mut(node_id);
            node.hits = node.hits.saturating_add(1);
            node.last_access = now;
            nodes.push(node_id);
        }

        RadixPath {
            matched_tokens: nodes.len() * self.page_size,
            nodes,
            terminal: node_id,
        }
    }

    pub(crate) fn page(&self, node_id: RadixNodeId) -> &Page {
        self.node(node_id)
            .page
            .as_ref()
            .expect("non-root radix node owns a page")
    }

    pub(crate) fn page_mut(&mut self, node_id: RadixNodeId) -> &mut Page {
        self.node_mut(node_id)
            .page
            .as_mut()
            .expect("non-root radix node owns a page")
    }

    /// Select the coldest unlocked resident leaf. A resident leaf has no descendant accepted by
    /// `is_resident`; this lets a structural radix remain intact while device pages cascade to
    /// host and disk from suffix to prefix.
    pub(crate) fn resident_leaf(
        &self,
        mut is_resident: impl FnMut(&Page) -> bool,
    ) -> Option<RadixNodeId> {
        let mut resident_descendant = vec![false; self.nodes.len()];
        for node_id in (1..self.nodes.len()).rev() {
            let Some(node) = self.nodes[node_id].as_ref() else {
                continue;
            };
            let resident =
                is_resident(node.page.as_ref().expect("non-root radix node owns a page"));
            if resident || resident_descendant[node_id] {
                if let Some(parent) = node.parent {
                    resident_descendant[parent] = true;
                }
            }
        }

        self.nodes
            .iter()
            .enumerate()
            .skip(1)
            .filter_map(|(id, node)| node.as_ref().map(|node| (id, node)))
            .filter(|(id, node)| {
                node.lock_refs == 0
                    && is_resident(node.page.as_ref().expect("non-root radix node owns a page"))
                    && !node.children.iter().any(|(_, child)| {
                        let child = *child;
                        self.nodes[child].is_some()
                            && (resident_descendant[child] || is_resident(self.page(child)))
                    })
                    && !resident_descendant[*id]
            })
            .min_by_key(|(_, node)| (u8::from(node.hits > 1), node.last_access, node.depth_tokens))
            .map(|(id, _)| id)
    }

    pub(crate) fn remove_subtree(&mut self, victim: RadixNodeId) -> Option<Vec<Page>> {
        if victim == 0 || self.node(victim).lock_refs != 0 {
            return None;
        }
        let mut stack = vec![victim];
        let mut ids = Vec::new();
        while let Some(node_id) = stack.pop() {
            let node = self.node(node_id);
            if node.lock_refs != 0 {
                return None;
            }
            stack.extend(node.children.values().copied());
            ids.push(node_id);
        }

        let parent = self.node(victim).parent.expect("non-root node has parent");
        self.node_mut(parent)
            .children
            .retain(|_, child| *child != victim);
        let mut pages = Vec::with_capacity(ids.len());
        for node_id in ids.into_iter().rev() {
            if let Some(mut node) = self.nodes[node_id].take()
                && let Some(page) = node.page.take()
            {
                pages.push(page);
                self.page_count -= 1;
            }
        }
        Some(pages)
    }

    /// Insert missing page edges, constructing a native page only after proving the edge is new.
    /// A constructor failure leaves the already-committed prefix intact and useful.
    pub(crate) fn insert_missing<E>(
        &mut self,
        tokens: &[LlamaToken],
        mut make_page: impl FnMut(usize, usize) -> Result<Page, E>,
    ) -> Result<RadixNodeId, E> {
        let aligned = tokens.len() / self.page_size * self.page_size;
        let mut node_id = 0;

        for (page_index, chunk) in tokens[..aligned].chunks_exact(self.page_size).enumerate() {
            if let Some(child) = self.node(node_id).children.get(chunk).copied() {
                node_id = child;
                continue;
            }

            let start = page_index * self.page_size;
            let end = start + self.page_size;
            let page = make_page(start, end)?;
            let now = self.tick();
            let child_id = self.nodes.len();
            self.nodes.push(Some(RadixNode {
                parent: Some(node_id),
                children: BTreeMap::new(),
                page: Some(page),
                depth_tokens: end,
                lock_refs: 0,
                hits: 0,
                last_access: now,
            }));
            self.node_mut(node_id)
                .children
                .insert(chunk.into(), child_id);
            self.page_count += 1;
            node_id = child_id;
        }
        Ok(node_id)
    }

    /// Protect a matched path from cache eviction while a request depends on it.
    pub(crate) fn lock_path(&mut self, terminal: RadixNodeId) {
        let mut current = Some(terminal);
        while let Some(node_id) = current {
            let node = self.node_mut(node_id);
            node.lock_refs = node
                .lock_refs
                .checked_add(1)
                .expect("radix lock reference overflow");
            current = node.parent;
        }
    }

    pub(crate) fn unlock_path(&mut self, terminal: RadixNodeId) {
        let mut current = Some(terminal);
        while let Some(node_id) = current {
            let node = self.node_mut(node_id);
            assert!(node.lock_refs > 0, "radix path unlock underflow");
            node.lock_refs -= 1;
            current = node.parent;
        }
    }

    /// Evict one unlocked leaf using segmented LRU (one-hit probationary before reused pages).
    /// Native release runs before metadata removal so a failed release leaves the tree coherent.
    #[cfg(test)]
    pub(crate) fn evict_one(&mut self, mut release: impl FnMut(&Page) -> bool) -> bool {
        let victim = self
            .nodes
            .iter()
            .enumerate()
            .skip(1)
            .filter_map(|(id, node)| node.as_ref().map(|node| (id, node)))
            .filter(|(_, node)| node.lock_refs == 0 && node.children.is_empty())
            .min_by_key(|(_, node)| (u8::from(node.hits > 1), node.last_access, node.depth_tokens))
            .map(|(id, _)| id);
        let Some(victim) = victim else {
            return false;
        };
        let page = self
            .node(victim)
            .page
            .as_ref()
            .expect("evictable radix node owns a page")
            .clone();
        if !release(&page) {
            return false;
        }

        let parent = self.node(victim).parent.expect("non-root node has parent");
        self.node_mut(parent)
            .children
            .retain(|_, child| *child != victim);
        self.nodes[victim] = None;
        self.page_count -= 1;
        true
    }

    fn tick(&mut self) -> u64 {
        self.clock = self.clock.saturating_add(1);
        self.clock
    }

    fn node(&self, id: RadixNodeId) -> &RadixNode<Page> {
        self.nodes
            .get(id)
            .and_then(Option::as_ref)
            .expect("radix node id is live")
    }

    fn node_mut(&mut self, id: RadixNodeId) -> &mut RadixNode<Page> {
        self.nodes
            .get_mut(id)
            .and_then(Option::as_mut)
            .expect("radix node id is live")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens(values: &[i32]) -> Vec<LlamaToken> {
        values.iter().copied().map(LlamaToken::new).collect()
    }

    #[test]
    fn branches_and_matches_only_complete_pages() {
        let mut cache = PagedRadixCache::new(2);
        let first = tokens(&[1, 2, 3, 4, 5]);
        cache
            .insert_missing(&first, |start, _| Ok::<_, ()>(start / 2 + 10))
            .unwrap();

        assert_eq!(cache.match_prefix(&first).matched_tokens, 4);
        assert_eq!(cache.match_prefix(&tokens(&[1, 2, 9, 9])).matched_tokens, 2);
        assert_eq!(cache.match_prefix(&tokens(&[1])).matched_tokens, 0);

        let branch = tokens(&[1, 2, 9, 9]);
        cache
            .insert_missing(&branch, |start, _| Ok::<_, ()>(start / 2 + 20))
            .unwrap();
        assert_eq!(cache.page_count(), 3);
        assert_eq!(cache.match_prefix(&branch).pages, vec![10, 21]);
    }

    #[test]
    fn insertion_never_reconstructs_existing_pages() {
        let mut cache = PagedRadixCache::new(2);
        let key = tokens(&[1, 2, 3, 4]);
        let mut creates = 0;
        cache
            .insert_missing(&key, |_, _| {
                creates += 1;
                Ok::<_, ()>(creates)
            })
            .unwrap();
        cache
            .insert_missing(&key, |_, _| -> Result<usize, ()> {
                panic!("existing page was reconstructed")
            })
            .unwrap();
        assert_eq!(creates, 2);
    }

    #[test]
    fn locked_paths_survive_leaf_eviction() {
        let mut cache = PagedRadixCache::new(1);
        let left = tokens(&[1, 2]);
        let right = tokens(&[1, 3]);
        cache
            .insert_missing(&left, |start, _| Ok::<_, ()>(start + 1))
            .unwrap();
        cache
            .insert_missing(&right, |start, _| Ok::<_, ()>(start + 10))
            .unwrap();
        let locked = cache.match_prefix(&left).terminal;
        cache.lock_path(locked);

        let mut released = Vec::new();
        assert!(cache.evict_one(|page| {
            released.push(*page);
            true
        }));
        assert_eq!(cache.match_prefix(&left).matched_tokens, 2);
        assert_eq!(cache.match_prefix(&right).matched_tokens, 1);

        cache.unlock_path(locked);
        assert!(cache.evict_one(|page| {
            released.push(*page);
            true
        }));
        assert_eq!(cache.match_prefix(&left).matched_tokens, 1);
    }

    #[test]
    fn randomized_lookup_matches_a_linear_page_oracle() {
        let mut cache = PagedRadixCache::new(3);
        let mut inserted = Vec::<Vec<LlamaToken>>::new();
        let mut state = 0x1234_5678_u64;
        for id in 0..250_u64 {
            let len = 3 + (next(&mut state) % 24) as usize;
            let key = (0..len)
                .map(|_| LlamaToken::new((next(&mut state) % 11) as i32))
                .collect::<Vec<_>>();
            cache
                .insert_missing(&key, |start, _| Ok::<_, ()>((id, start)))
                .unwrap();
            inserted.push(key.clone());

            let got = cache.match_prefix(&key).matched_tokens;
            let expected = inserted
                .iter()
                .map(|candidate| {
                    candidate
                        .iter()
                        .zip(&key)
                        .take_while(|(a, b)| a == b)
                        .count()
                        / 3
                        * 3
                })
                .max()
                .unwrap_or(0);
            assert_eq!(got, expected);
        }
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum TierPage {
        Device(usize),
        Host(usize),
    }

    #[test]
    fn resident_leaves_cascade_without_deleting_the_radix() {
        let mut cache = PagedRadixCache::new(1);
        let key = tokens(&[1, 2, 3]);
        cache
            .insert_missing(&key, |start, _| Ok::<_, ()>(TierPage::Device(start)))
            .unwrap();
        let path = cache.match_path(&key);

        let suffix = cache
            .resident_leaf(|page| matches!(page, TierPage::Device(_)))
            .unwrap();
        assert_eq!(suffix, path.nodes[2]);
        *cache.page_mut(suffix) = TierPage::Host(2);
        let next = cache
            .resident_leaf(|page| matches!(page, TierPage::Device(_)))
            .unwrap();
        assert_eq!(next, path.nodes[1]);
        assert_eq!(cache.match_prefix(&key).matched_tokens, 3);
    }

    #[test]
    fn subtree_removal_is_atomic_with_respect_to_path_locks() {
        let mut cache = PagedRadixCache::new(1);
        let key = tokens(&[1, 2, 3]);
        cache
            .insert_missing(&key, |start, _| Ok::<_, ()>(start))
            .unwrap();
        let path = cache.match_path(&key);
        cache.lock_path(path.terminal);
        assert!(cache.remove_subtree(path.nodes[1]).is_none());
        assert_eq!(cache.match_prefix(&key).matched_tokens, 3);
        cache.unlock_path(path.terminal);
        assert_eq!(cache.remove_subtree(path.nodes[1]).unwrap().len(), 2);
        assert_eq!(cache.match_prefix(&key).matched_tokens, 1);
    }

    fn next(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }
}
