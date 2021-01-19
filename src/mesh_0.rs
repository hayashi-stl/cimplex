use idmap::OrderedIdMap;
use idmap::table::DenseEntryTable;
use typenum::{U2, U3};
use std::ops::{Index, IndexMut};
use std::iter::{FromIterator, IntoIterator, Extend};
#[cfg(feature = "serde_")]
use serde::{Serialize, Deserialize};

use crate::{VecN, impl_integer_id};

/// An index to a vertex of a vertex mesh.
/// Will not be invalidated unless the vertex gets removed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde_", derive(Serialize, Deserialize))]
pub struct VertexId(u64);
impl_integer_id!(VertexId);

pub type Vertices<'a, V> = idmap::Iter<'a, VertexId, V, DenseEntryTable<VertexId, V>>;
pub type VerticesMut<'a, V> = idmap::IterMut<'a, VertexId, V, DenseEntryTable<VertexId, V>>;

/// A combinatorial simplicial 0-complex, containing only vertices.
/// Basically a vertex list. Also known as a vertex mesh.
/// Each vertex stores a value of type `V`.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde_", derive(Serialize, Deserialize))]
pub struct ComboMesh0<V> {
    vertices: OrderedIdMap<VertexId, V>,
    next_vertex_id: u64,
}

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

    /// Gets the number of vertices.
    pub fn num_vertices(&self) -> usize {
        self.vertices.len()
    }

    /// Iterates over the vertices of this vertex mesh.
    /// Gives (id, vertex) pairs
    pub fn vertices(&self) -> Vertices<V> {
        self.vertices.iter()
    }

    /// Iterates mutably over the vertices of this vertex mesh.
    /// Gives (id, vertex) pairs
    pub fn vertices_mut(&mut self) -> VerticesMut<V> {
        self.vertices.iter_mut()
    }

    /// Gets the value of the vertex at a specific id.
    /// Returns None if not found.
    pub fn vertex(&self, id: VertexId) -> Option<&V> {
        self.vertices.get(id)
    }

    /// Gets the value of the vertex at a specific id mutably.
    /// Returns None if not found.
    pub fn vertex_mut(&mut self, id: VertexId) -> Option<&mut V> {
        self.vertices.get_mut(id)
    }

    /// Adds a vertex to the mesh and returns the id.
    pub fn add_vertex(&mut self, value: V) -> VertexId {
        let id = VertexId(self.next_vertex_id);
        self.next_vertex_id += 1;
        debug_assert!(self.vertices.insert(id, value).is_none());
        id
    }

    /// Extends the vertex list with an iterator and returns a `Vec`
    /// of the vertex ids that are created in order.
    pub fn extend_vertices<I: IntoIterator<Item = V>>(&mut self, iter: I) -> Vec<VertexId> {
        iter.into_iter().map(|value| self.add_vertex(value)).collect()
    }

    /// Removes a vertex from the mesh.
    /// Returns the vertex that was there or None if none was there.
    pub fn remove_vertex(&mut self, id: VertexId) -> Option<V> {
        self.vertices.remove(id)
    }

    /// Removes all vertices from the mesh.
    pub fn clear_vertices(&mut self) {
        self.vertices.clear()
    }
}

impl<V> Index<VertexId> for ComboMesh0<V> {
    type Output = V;

    fn index(&self, index: VertexId) -> &Self::Output {
        &self.vertices[index]
    }
}

impl<V> IndexMut<VertexId> for ComboMesh0<V> {
    fn index_mut(&mut self, index: VertexId) -> &mut Self::Output {
        &mut self.vertices[index]
    }
}

impl<V> IntoIterator for ComboMesh0<V> {
    type IntoIter = <OrderedIdMap<VertexId, V> as IntoIterator>::IntoIter;
    type Item = (VertexId, V);

    /// Converts this into an iterator of vertex values.
    fn into_iter(self) -> Self::IntoIter {
        self.vertices.into_iter()
    }
}

impl<V> FromIterator<(VertexId, V)> for ComboMesh0<V> {
    fn from_iter<T: IntoIterator<Item = (VertexId, V)>>(iter: T) -> Self {
        let mut mesh = Self {
            vertices: iter.into_iter().collect(),
            next_vertex_id: 0
        };
        mesh.next_vertex_id = mesh.vertices.len() as u64;
        mesh
    }
}

impl<V> Extend<(VertexId, V)> for ComboMesh0<V> {
    fn extend<T: IntoIterator<Item = (VertexId, V)>>(&mut self, iter: T) {
        self.vertices.extend(iter)
    }
}

/// A position-containing vertex mesh
pub type Mesh0<V, D> = ComboMesh0<(VecN<D>, V)>;

/// A 2D-position-containing vertex mesh
pub type Mesh02<V> = Mesh0<V, U2>;

/// A 3D-position-containing vertex mesh
pub type Mesh03<V> = Mesh0<V, U3>;