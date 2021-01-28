use idmap::OrderedIdMap;
use std::iter::{Extend, FromIterator, IntoIterator, Map};
use typenum::B0;
use nalgebra::dimension::{U2, U3};

use crate::{ComboMesh1, ComboMesh2, ComboMesh3, private::{Key, Lock}};
use crate::vertex::{HasVertices, IdType, Vertex as VertexIntr, VertexId};
use crate::PtN;

use internal::Vertex;

/// A combinatorial simplicial 0-complex, containing only vertices.
/// Basically a vertex list. Also known as a vertex mesh.
/// Each vertex stores a value of type `V`.
#[derive(Clone, Debug)]
pub struct ComboMesh0<V> {
    vertices: OrderedIdMap<VertexId, Vertex<V>>,
    next_vertex_id: IdType,
    default_v: fn() -> V,
}

crate::impl_index_vertex!(ComboMesh0<V>);
crate::impl_with_eft!(ComboMesh0<V>: <V, E> ComboMesh1<V, E>, <V, E, F> ComboMesh2<V, E, F>, <V, E, F, T> ComboMesh3<V, E, F, T>);

impl<V> HasVertices for ComboMesh0<V> {
    crate::impl_has_vertices!(Vertex<V>, Higher = B0);

    fn remove_vertex_higher<L: Lock>(&mut self, _: VertexId) {}

    fn clear_vertices_higher<L: Lock>(&mut self) {}
}

impl<V: Default> Default for ComboMesh0<V> {
    fn default() -> Self {
        Self {
            vertices: OrderedIdMap::default(),
            next_vertex_id: 0,
            default_v: Default::default,
        }
    }
}

impl<V> ComboMesh0<V> {
    /// Creates an empty vertex mesh.
    pub fn new() -> Self
    where
        V: Default,
    {
        Self::default()
    }

    /// Creates an empty vertex mesh with default values for elements.
    pub fn with_defaults(vertex: fn() -> V) -> Self {
        Self {
            vertices: OrderedIdMap::default(),
            next_vertex_id: 0,
            default_v: vertex,
        }
    }
}

impl<V> IntoIterator for ComboMesh0<V> {
    type IntoIter = Map<
        <OrderedIdMap<VertexId, Vertex<V>> as IntoIterator>::IntoIter,
        fn((VertexId, Vertex<V>)) -> (VertexId, V),
    >;
    type Item = (VertexId, V);

    /// Converts this into an iterator of vertex values.
    fn into_iter(self) -> Self::IntoIter {
        self.vertices
            .into_iter()
            .map(|(id, v)| (id, v.to_value::<Key>()))
    }
}

impl<V: Default> FromIterator<(VertexId, V)> for ComboMesh0<V> {
    fn from_iter<T: IntoIterator<Item = (VertexId, V)>>(iter: T) -> Self {
        let mut mesh = Self {
            vertices: iter
                .into_iter()
                .map(|(id, v)| (id, VertexIntr::new::<Key>(id, v)))
                .collect(),
            next_vertex_id: 0,
            default_v: Default::default,
        };
        mesh.next_vertex_id = mesh.vertices.len() as IdType;
        mesh
    }
}

impl<V> Extend<(VertexId, V)> for ComboMesh0<V> {
    fn extend<T: IntoIterator<Item = (VertexId, V)>>(&mut self, iter: T) {
        self.vertices.extend(
            iter.into_iter()
                .map(|(id, v)| (id, VertexIntr::new::<Key>(id, v))),
        )
    }
}

/// A position-containing vertex mesh
pub type Mesh0<V, D> = ComboMesh0<(PtN<D>, V)>;

/// A 2D-position-containing vertex mesh
pub type Mesh02<V> = Mesh0<V, U2>;

/// A 3D-position-containing vertex mesh
pub type Mesh03<V> = Mesh0<V, U3>;

mod internal {

    /// Vertex storage
    #[derive(Clone, Debug)]
    pub struct Vertex<V> {
        value: V,
    }
    #[rustfmt::skip]
    crate::impl_vertex!(Vertex<V>, new |_id, value| Vertex { value });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vertex::HasPosition;
    use nalgebra::Point2;

    #[test]
    fn test_bounding_box() {
        let mut mesh = ComboMesh0::with_defaults(|| Point2::origin());
        mesh.extend_vertices(vec![
            Point2::new(0.0, 1.0),
            Point2::new(-1.0, 2.0),
            Point2::new(5.0, 3.0),
        ]);

        assert_eq!(
            mesh.bounding_box(),
            Some([Point2::new(-1.0, 1.0), Point2::new(5.0, 3.0)])
        );
    }
}
