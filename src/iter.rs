use std::marker::PhantomData;

/// Like `Map`, but allows the use of 1 common value
/// without capturing.
#[derive(Clone, Debug)]
pub struct MapWith<C: Clone, O, I: Iterator, F: FnMut(C, I::Item) -> O> {
    iter: I,
    common: C,
    combine: F,
    marker: PhantomData<fn() -> O>,
}

impl<C: Clone, O, I: Iterator, F: FnMut(C, I::Item) -> O> Iterator for MapWith<C, O, I, F> {
    type Item = O;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|t| (self.combine)(self.common.clone(), t))
    }
}

pub trait IteratorExt: Iterator {
    fn map_with<C: Clone, O, F: FnMut(C, Self::Item) -> O>(self, common: C, combine: F) -> MapWith<C, O, Self, F> where Self: Sized {
        MapWith {
            iter: self,
            common,
            combine,
            marker: PhantomData,
        }
    }
}

impl<T: Iterator> IteratorExt for T {}