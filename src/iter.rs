use std::collections::VecDeque;
use std::hash::Hash;

use fnv::FnvHashSet;

/// Like `Map`, but allows the use of 1 common value
/// without capturing.
#[derive(Clone, Debug)]
pub struct MapWith<C, I, F> {
    iter: I,
    common: C,
    combine: F,
}

impl<C: Clone, O, I: Iterator, F: FnMut(C, I::Item) -> O> Iterator for MapWith<C, I, F> {
    type Item = O;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter
            .next()
            .map(|t| (self.combine)(self.common.clone(), t))
    }
}

/// Like `FlatMap`, but allows the use of 1 common value
/// without capturing.
#[derive(Clone, Debug)]
pub struct FlatMapWith<C, I, U: IntoIterator, F> {
    iter: I,
    inner_iter: Option<U::IntoIter>,
    common: C,
    combine: F,
}

impl<C: Clone, I: Iterator, U: IntoIterator, F: FnMut(C, I::Item) -> U> Iterator for FlatMapWith<C, I, U, F> {
    type Item = U::Item;

    fn next(&mut self) -> Option<U::Item> {
        loop {
            if let Some(item) = self.inner_iter.as_mut().and_then(|iter| iter.next()) {
                return Some(item);
            } else {
                self.inner_iter = self.iter.next().map(|t| (self.combine)(self.common.clone(), t).into_iter());
                if self.inner_iter.is_none() {
                    return None;
                }
            }
        }
    }
}

pub trait IteratorExt: Iterator {
    fn map_with<C: Clone, O, F: FnMut(C, Self::Item) -> O>(
        self,
        common: C,
        combine: F,
    ) -> MapWith<C, Self, F>
    where
        Self: Sized,
    {
        MapWith {
            iter: self,
            common,
            combine,
        }
    }

    fn flat_map_with<C: Clone, U: IntoIterator, F: FnMut(C, Self::Item) -> U>(
        self,
        common: C,
        combine: F,
    ) -> FlatMapWith<C, Self, U, F>
    where
        Self: Sized,
    {
        FlatMapWith {
            iter: self,
            inner_iter: None,
            common,
            combine,
        }
    }
}

impl<T: Iterator> IteratorExt for T {}

/// Structure returned by a BFS function
#[derive(Clone)]
pub struct Bfs<V, NF, SF> {
    next_fn: NF,
    search_pred: SF,
    to_search: VecDeque<V>,
    searched: FnvHashSet<V>,
}

impl<V: Clone + Eq + Hash, NI: IntoIterator<Item = V>, NF: FnMut(&V) -> NI, SF: FnMut(&V) -> bool> Iterator for Bfs<V, NF, SF> {
    type Item = V;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(node) = self.to_search.pop_front() {
            if !self.searched.contains(&node) {
                self.searched.insert(node.clone());

                if (self.search_pred)(&node) {
                    self.to_search.extend((self.next_fn)(&node));
                    return Some(node)
                }
            }
        }

        None
    }
}

/// Performs a BFS iteration over some nodes.
/// The BFS starts at the nodes returned by `start`.
/// A node is not returned or searched if `search_pred` returns false for that node.
///
/// `search_pred` is separated from `neighbors` to
/// avoid calling the search condition multiple times per node.
pub(crate) fn bfs<V, FI: IntoIterator<Item = V>, NI: IntoIterator<Item = V>, NF: FnMut(&V) -> NI, SF: FnMut(&V) -> bool>(
    start: FI, neighbors: NF, search_pred: SF
) -> Bfs<V, NF, SF> {
    Bfs {
        next_fn: neighbors,
        search_pred,
        to_search: start.into_iter().collect::<VecDeque<_>>(),
        searched: FnvHashSet::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bfs_empty() {
        let bfs = bfs(
            vec![0, 4],
            |i| vec![i + 1, i + 2],
            |i| *i < 0
        );
        assert_eq!(bfs.collect::<Vec<_>>(), vec![]);
    }

    #[test]
    fn test_bfs() {
        let bfs = bfs(
            vec![0, 4],
            |i| vec![i + 1, i + 2],
            |i| *i < 8
        );
        assert_eq!(bfs.collect::<Vec<_>>(), vec![0, 4, 1, 2, 5, 6, 7]);
    }
}