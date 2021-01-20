use idmap::OrderedIdMap;
use typenum::{U2, U3};
use std::iter::{FromIterator, IntoIterator, Extend, Map};
#[cfg(feature = "serde_")]
use serde::{Serialize, Deserialize};

use crate::VecN;
use crate::vertex::{HasVertices, VertexId};
use crate::vertex::internal::Vertex as VertexIntr;

use internal::Vertex;

/// A combinatorial simplicial 0-complex, containing only vertices.
/// Basically a vertex list. Also known as a vertex mesh.
/// Each vertex stores a value of type `V`.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde_", derive(Serialize, Deserialize))]
pub struct ComboMesh0<V> {
    vertices: OrderedIdMap<VertexId, Vertex<V>>,
    next_vertex_id: u64,
}

crate::impl_has_vertices!(ComboMesh0<V>, Vertex);
crate::impl_index_vertex!(ComboMesh0<V>);

impl<V> HasVertices for ComboMesh0<V> {}

impl<V> Default for ComboMesh0<V> {
    fn default() -> Self {
        Self {
            vertices: OrderedIdMap::default(),
            next_vertex_id: 0,
        }
    }
}

impl<V> ComboMesh0<V> {
    /// Creates an empty vertex mesh.
    pub fn new() -> Self {
        Self::default()
    }
}

impl<V> IntoIterator for ComboMesh0<V> {
    type IntoIter = Map<<OrderedIdMap<VertexId, Vertex<V>> as IntoIterator>::IntoIter,
        fn((VertexId, Vertex<V>)) -> (VertexId, V)>;
    type Item = (VertexId, V);

    /// Converts this into an iterator of vertex values.
    fn into_iter(self) -> Self::IntoIter {
        self.vertices.into_iter().map(|(id, v)| (id, v.to_value()))
    }
}

impl<V> FromIterator<(VertexId, V)> for ComboMesh0<V> {
    fn from_iter<T: IntoIterator<Item = (VertexId, V)>>(iter: T) -> Self {
        let mut mesh = Self {
            vertices: iter.into_iter().map(|(id, v)| (id, Vertex::new(id, v))).collect(),
            next_vertex_id: 0
        };
        mesh.next_vertex_id = mesh.vertices.len() as u64;
        mesh
    }
}

impl<V> Extend<(VertexId, V)> for ComboMesh0<V> {
    fn extend<T: IntoIterator<Item = (VertexId, V)>>(&mut self, iter: T) {
        self.vertices.extend(iter.into_iter().map(|(id, v)| (id, Vertex::new(id, v))))
    }
}

/// A position-containing vertex mesh
pub type Mesh0<V, D> = ComboMesh0<(VecN<D>, V)>;

/// A 2D-position-containing vertex mesh
pub type Mesh02<V> = Mesh0<V, U2>;

/// A 3D-position-containing vertex mesh
pub type Mesh03<V> = Mesh0<V, U3>;

mod internal {
    use crate::vertex::internal::{RemoveVertexHigher, ClearVerticesHigher};
    use crate::vertex::VertexId;
    use super::ComboMesh0;

    /// Vertex storage
    #[derive(Clone, Debug)]
    #[cfg_attr(feature = "serde_", derive(Serialize, Deserialize))]
    pub struct Vertex<V> {
        value: V,
    }
    crate::impl_vertex!(Vertex<V>, new |_id, value| Vertex { value });

    impl<V> RemoveVertexHigher for ComboMesh0<V> {
        fn remove_vertex_higher(&mut self, _: VertexId) {}
    }

    impl<V> ClearVerticesHigher for ComboMesh0<V> {
        fn clear_vertices_higher(&mut self) {}
    }
}