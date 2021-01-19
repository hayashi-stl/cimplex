use idmap::OrderedIdMap;
use idmap::table::DenseEntryTable;
use typenum::{U2, U3};
use std::ops::{Index, IndexMut};
use std::iter::{Map, Filter};
use std::collections::hash_map;
use fnv::{FnvHashMap, FnvHashSet};
#[cfg(feature = "serde_")]
use serde::{Serialize, Deserialize};

use crate::{VecN, impl_integer_id};

/// An index to a vertex of an edge mesh.
/// Will not be invalidated unless the vertex gets removed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde_", derive(Serialize, Deserialize))]
pub struct VertexId(u64);
impl_integer_id!(VertexId);

/// An edge id is just the edge's vertices in order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde_", derive(Serialize, Deserialize))]
pub struct EdgeId([VertexId; 2]);

impl From<[VertexId; 2]> for EdgeId {
    fn from(vertices: [VertexId; 2]) -> Self {
        EdgeId(vertices)
    }
}

impl EdgeId {
    /// Gets the vertices that this edge id is made of
    pub fn vertices(self) -> [VertexId; 2] {
        self.0
    }

    /// Reverses the vertices of this edge id to get the one for the twin edge
    fn twin(self) -> EdgeId {
        EdgeId([self.0[1], self.0[0]])
    }
}

pub type Vertices<'a, V> = Map<idmap::Iter<'a, VertexId, Vertex<V>, DenseEntryTable<VertexId, Vertex<V>>>,
    for<'b> fn((&'b VertexId, &'b Vertex<V>)) -> (&'b VertexId, &'b V)>;
pub type VerticesMut<'a, V> = Map<idmap::IterMut<'a, VertexId, Vertex<V>, DenseEntryTable<VertexId, Vertex<V>>>,
    for<'b> fn((&'b VertexId, &'b mut Vertex<V>)) -> (&'b VertexId, &'b mut V)>;

type EdgeFilterFn<'a, E> = for<'b> fn(&'b (&'a EdgeId, &'a Edge<E>)) -> bool;
type EdgeMapFn<'a, E> = fn((&'a EdgeId, &'a Edge<E>)) -> (&'a EdgeId, &'a E);
pub type Edges<'a, E> = Map<Filter<hash_map::Iter<'a, EdgeId, Edge<E>>, EdgeFilterFn<'a, E>>, EdgeMapFn<'a, E>>;
type EdgeFilterFnMut<'a, E> = for<'b> fn(&'b (&'a EdgeId, &'a mut Edge<E>)) -> bool;
type EdgeMapFnMut<'a, E> = fn((&'a EdgeId, &'a mut Edge<E>)) -> (&'a EdgeId, &'a mut E);
pub type EdgesMut<'a, E> = Map<Filter<hash_map::IterMut<'a, EdgeId, Edge<E>>, EdgeFilterFnMut<'a, E>>, EdgeMapFnMut<'a, E>>;

//// A vertex of an edge mesh
#[derive(Clone, Debug)]
#[doc(hidden)]
#[cfg_attr(feature = "serde_", derive(Serialize, Deserialize))]
pub struct Vertex<V> {
    /// `target` is this vertex's id if there is no target
    target: VertexId,
    value: V,
}

/// An edge of an edge mesh
#[derive(Clone, Debug)]
#[doc(hidden)]
#[cfg_attr(feature = "serde_", derive(Serialize, Deserialize))]
pub struct Edge<E> {
    /// Previous outgoing target from the same vertex, whether the edge actually exists or not
    prev_target: VertexId,
    /// Next outgoing target from the same vertex, whether the edge actually exists or not
    next_target: VertexId,
    /// The edge does not actually exist if the value is None;
    /// it is just there for the structural purpose of
    /// ensuring that every edge has a twin.
    value: Option<E>,
}

/// A combinatorial simplicial 1-complex, containing only vertices and (oriented) edges.
/// Also known as an edge mesh.
/// Each vertex stores a value of type `V`.
/// Each edge stores its vertices and a value of type `E`.
/// The edge manipulation methods can either be called with an array of 2 `VertexId`s
/// or an `EdgeId`.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde_", derive(Serialize, Deserialize))]
pub struct ComboMesh1<V, E> {
    vertices: OrderedIdMap<VertexId, Vertex<V>>,
    edges: FnvHashMap<EdgeId, Edge<E>>,
    next_vertex_id: u64,
}

impl<V, E> Default for ComboMesh1<V, E> {
    fn default() -> Self {
        ComboMesh1 {
            vertices: OrderedIdMap::default(),
            edges: FnvHashMap::default(),
            next_vertex_id: 0,
        }
    }
}

impl<V, E> ComboMesh1<V, E> {
    /// Creates an empty edge mesh.
    pub fn new() -> Self {
        Self::default()
    }

    /// Gets the number of vertices.
    pub fn num_vertices(&self) -> usize {
        self.vertices.len()
    }

    /// Iterates over the vertices of this mesh.
    /// Gives (id, value) pairs
    pub fn vertices(&self) -> Vertices<V> {
        self.vertices.iter().map(|(id, v)| (id, &v.value))
    }

    /// Iterates mutably over the vertices of this mesh.
    /// Gives (id, value) pairs
    pub fn vertices_mut(&mut self) -> VerticesMut<V> {
        self.vertices.iter_mut().map(|(id, v)| (id, &mut v.value))
    }

    /// Iterates over the edges of this mesh.
    /// Gives (id, value) pairs
    pub fn edges(&self) -> Edges<E> {
        self.edges.iter()
            .filter::<EdgeFilterFn<E>>(|(_, e)| e.value.is_some())
            .map::<_, EdgeMapFn<E>>(|(id, e)| (id, e.value.as_ref().unwrap()))
    }

    /// Iterates mutably over the edges of this mesh.
    /// Gives (id, value) pairs
    pub fn edges_mut(&mut self) -> EdgesMut<E> {
        self.edges.iter_mut()
            .filter::<EdgeFilterFnMut<E>>(|(_, e)| e.value.is_some())
            .map::<_, EdgeMapFnMut<E>>(|(id, e)| (id, e.value.as_mut().unwrap()))
    }

    /// Gets the value of the vertex at a specific id.
    /// Returns None if not found.
    pub fn vertex(&self, id: VertexId) -> Option<&V> {
        self.vertices.get(id).map(|v| &v.value)
    }

    /// Gets the value of the vertex at a specific id mutably.
    /// Returns None if not found.
    pub fn vertex_mut(&mut self, id: VertexId) -> Option<&mut V> {
        self.vertices.get_mut(id).map(|v| &mut v.value)
    }

    /// Gets the value of the edge at a specific id.
    /// Returns None if not found.
    pub fn edge<EI: Into<EdgeId>>(&self, id: EI) -> Option<&E> {
        self.edges.get(&id.into()).and_then(|e| e.value.as_ref())
    }

    /// Gets the value of the edge at a specific id mutably.
    /// Returns None if not found.
    pub fn edge_mut<EI: Into<EdgeId>>(&mut self, id: EI) -> Option<&mut E> {
        self.edges.get_mut(&id.into()).and_then(|e| e.value.as_mut())
    }
    
    /// Adds a vertex to the mesh and returns the id.
    pub fn add_vertex(&mut self, value: V) -> VertexId {
        let id = VertexId(self.next_vertex_id);
        self.next_vertex_id += 1;
        debug_assert!(self.vertices.insert(id, Vertex {
            target: id,
            value
        }).is_none());
        id
    }

    /// Extends the vertex list with an iterator and returns a `Vec`
    /// of the vertex ids that are created in order.
    pub fn extend_vertices<I: IntoIterator<Item = V>>(&mut self, iter: I) -> Vec<VertexId> {
        iter.into_iter().map(|value| self.add_vertex(value)).collect()
    }

    /// Removes a vertex from the mesh.
    /// Returns the value of the vertex that was there or None if none was there,
    /// along with the values of all the edges that were removed as a result.
    pub fn remove_vertex(&mut self, id: VertexId) -> (Option<V>, Vec<E>) {
        let vertex = &self.vertices[id];
        if vertex.target == id {
            (self.vertices.remove(id).map(|v| v.value), vec![])
        } else {
            // Get edges to remove
            let mut targets = vec![];
            let mut target = vertex.target;
            while targets.last() != Some(&vertex.target) {
                target = self.edges[&[id, target].into()].next_target;
                targets.push(target);
            }

            println!("Source: {:?}", id);
            println!("Targets: {:?}", targets);

            // Remove edges
            let edge_values = targets.into_iter().flat_map(|target|
                vec![self.remove_edge([id, target]), self.remove_edge([target, id])].into_iter().flatten())
                .collect();
            (self.vertices.remove(id).map(|v| v.value), edge_values)
        }
    }

    /// Removes all vertices from the mesh.
    /// Removes all edges as a side-effect.
    pub fn clear_vertices(&mut self) {
        self.edges.clear();
        self.vertices.clear();
    }

    /// Adds an edge to the mesh. Vertex order is important!
    /// If the edge was already there, this replaces the value.
    /// Returns the previous value of the edge, if there was one.
    ///
    /// # Panics
    /// Panics if either vertex doesn't exist or if the vertices are the same
    pub fn add_edge<EI: Into<EdgeId>>(&mut self, vertices: EI, value: E) -> Option<E> {
        let id = vertices.into();
        assert!(id.0[0] != id.0[1], "vertices of edge may not equal");

        // Can't use entry().or_insert() because that would cause a
        // mutable borrow and an immutable borrow to exist at the same time
        if let Some(edge) = self.edges.get_mut(&id) {
            let old = edge.value.take();
            edge.value = Some(value);
            old
        } else {
            let mut insert_edge = |id: EdgeId, value: Option<E>| {
                let target = self.vertices[id.0[0]].target;

                let (prev, next) = if target == id.0[0] { // First edge from vertex
                    self.vertices[id.0[0]].target = id.0[1];
                    (id.0[1], id.0[1])
                } else {
                    let prev = self.edges[&[id.0[0], target].into()].prev_target;
                    let next = target;
                    self.edges.get_mut(&[id.0[0], prev].into()).unwrap().next_target = id.0[1];
                    self.edges.get_mut(&[id.0[0], next].into()).unwrap().prev_target = id.0[1];
                    (prev, next)
                };

                self.edges.insert(id, Edge {
                    prev_target: prev,
                    next_target: next,
                    value,
                });
            };

            insert_edge(id, Some(value));
            insert_edge(id.twin(), None);
            None
        }
    }

    /// Extends the edge list with an iterator.
    ///
    /// # Panics
    /// Panics if either vertex doesn't exist or if the vertices are the same
    /// in any of the edges.
    pub fn extend_edges<EI: Into<EdgeId>, I: IntoIterator<Item = (EI, E)>>(&mut self, iter: I) {
        iter.into_iter().for_each(|(id, value)| { self.add_edge(id, value); })
    }

    /// Removes an edge from the mesh and returns the value that was there,
    /// or None if there was nothing there
    pub fn remove_edge<EI: Into<EdgeId>>(&mut self, id: EI) -> Option<E> {
        let id = id.into();

        match self.edges.get(&id.twin()).map(|e| e.value.as_ref()) {
            // Twin actually exists; just set value to None
            Some(Some(_)) => {
                self.edges.get_mut(&id).unwrap().value.take()
            }

            // Twin is phantom, so remove both edge and twin from map
            Some(None) => {
                let mut delete_edge = |id: EdgeId| {
                    let edge = &self.edges[&id];
                    let prev = edge.prev_target;
                    let next = edge.next_target;
                    let source = &mut self.vertices[id.0[0]];
                    self.edges.get_mut(&[id.0[0], prev].into()).unwrap().next_target = next;
                    self.edges.get_mut(&[id.0[0], next].into()).unwrap().prev_target = prev;

                    if id.0[1] == next { // this was the last edge from the vertex
                        source.target = id.0[0];
                    } else if id.0[1] == source.target {
                        source.target = next;
                    }

                    self.edges.remove(&id).and_then(|e| e.value)
                };
                
                delete_edge(id.twin());
                delete_edge(id)
            }

            // Twin isn't in map, and neither is the edge to remove
            None => None,
        }
    }

    /// Removes all edges from the mesh.
    pub fn clear_edges(&mut self) {
        self.edges.clear();
        // Fix vertex-target links
        for (id, vertex) in &mut self.vertices {
            vertex.target = *id;
        }
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt::Debug;
    use std::hash::Hash;

    fn assert_vertices<V: Clone + Debug + Eq + Hash, E, I: IntoIterator<Item = (VertexId, V)>>(mesh: &ComboMesh1<V, E>, vertices: I) {
        let result = mesh.vertices().map(|(id, v)| (*id, v.clone()))
            .collect::<FnvHashSet<_>>();
        let expect = vertices.into_iter().collect::<FnvHashSet<_>>();

        assert_eq!(result, expect);
    }

    fn assert_edges<V, E: Clone + Debug + Eq + Hash, EI: Into<EdgeId>, I: IntoIterator<Item = (EI, E)>>(mesh: &ComboMesh1<V, E>, edges: I) {
        let result = mesh.edges().map(|(id, e)| (*id, e.clone()))
            .collect::<FnvHashSet<_>>();
        let expect = edges.into_iter()
            .map(|(vertices, e)| (vertices.into(), e))
            .collect::<FnvHashSet<_>>();

        assert_eq!(result, expect);
    }

    #[test]
    fn test_default() {
        let mesh = ComboMesh1::<(), ()>::default();
        assert!(mesh.vertices.is_empty());
        assert!(mesh.edges.is_empty());
    }

    #[test]
    fn test_add_vertex() {
        let mut mesh = ComboMesh1::<usize, usize>::default();
        let id = mesh.add_vertex(3);
        assert_eq!(mesh.vertex(id), Some(&3));

        let id2 = mesh.add_vertex(9);
        assert_eq!(mesh.vertex(id), Some(&3));
        assert_eq!(mesh.vertex(id2), Some(&9));
    }

    #[test]
    fn test_extend_vertices() {
        let mut mesh = ComboMesh1::<usize, usize>::default();
        let ids = mesh.extend_vertices(vec![3, 6, 9, 2]);
        assert_eq!(mesh.vertex(ids[0]), Some(&3));
        assert_eq!(mesh.vertex(ids[1]), Some(&6));
        assert_eq!(mesh.vertex(ids[2]), Some(&9));
        assert_eq!(mesh.vertex(ids[3]), Some(&2));

        let ids2 = mesh.extend_vertices(vec![5, 8]);
        assert_vertices(&mesh, vec![
            (ids[0], 3),
            (ids[1], 6),
            (ids[2], 9),
            (ids[3], 2),
            (ids2[0], 5),
            (ids2[1], 8),
        ]);
    }

    #[test]
    fn test_add_edge() {
        let mut mesh = ComboMesh1::<usize, usize>::default();
        let ids = mesh.extend_vertices(vec![3, 6, 9, 2]);
        let prev = mesh.add_edge([ids[1], ids[3]], 54);
        assert_eq!(prev, None);
        assert_eq!(mesh.edge([ids[1], ids[3]]), Some(&54));
        assert_eq!(mesh.edge([ids[3], ids[1]]), None); // twin should not exist

        // Add twin
        let prev = mesh.add_edge([ids[3], ids[1]], 27);
        assert_eq!(prev, None);
        assert_eq!(mesh.edge([ids[1], ids[3]]), Some(&54));
        assert_eq!(mesh.edge([ids[3], ids[1]]), Some(&27));

        // Modify edge
        let prev = mesh.add_edge([ids[1], ids[3]], 1);
        assert_eq!(prev, Some(54));
        assert_eq!(mesh.edge([ids[1], ids[3]]), Some(&1));
        assert_eq!(mesh.edge([ids[3], ids[1]]), Some(&27));
    }

    #[test]
    fn test_extend_edges() {
        let mut mesh = ComboMesh1::<usize, usize>::default();
        let ids = mesh.extend_vertices(vec![3, 6, 9, 2]);
        let edges = vec![
            ([ids[0], ids[3]], 5),
            ([ids[1], ids[3]], 3),
            ([ids[3], ids[1]], 2),
            ([ids[2], ids[3]], 8),
            ([ids[1], ids[2]], 9),
        ];
        mesh.extend_edges(edges.clone());

        for (edge, value) in edges {
            assert_eq!(mesh.edge(edge), Some(&value))
        }
    }

    #[test]
    fn test_remove_vertex() {
        let mut mesh = ComboMesh1::<usize, usize>::default();
        let ids = mesh.extend_vertices(vec![3, 6, 9, 2, 5]);
        let edges = vec![
            ([ids[0], ids[3]], 5),
            ([ids[1], ids[3]], 3),
            ([ids[3], ids[1]], 2),
            ([ids[2], ids[3]], 8),
            ([ids[1], ids[2]], 9),
        ];
        mesh.extend_edges(edges.clone());

        mesh.remove_vertex(ids[4]); // edgeless vertex
        assert_vertices(&mesh, vec![(ids[0], 3), (ids[1], 6), (ids[2], 9), (ids[3], 2)]);
        assert_edges(&mesh, vec![
            ([ids[0], ids[3]], 5),
            ([ids[1], ids[3]], 3),
            ([ids[3], ids[1]], 2),
            ([ids[2], ids[3]], 8),
            ([ids[1], ids[2]], 9),
        ]);

        mesh.remove_vertex(ids[1]); // vertex with edge
        assert_vertices(&mesh, vec![(ids[0], 3), (ids[2], 9), (ids[3], 2)]);
        assert_edges(&mesh, vec![
            ([ids[0], ids[3]], 5),
            ([ids[2], ids[3]], 8),
        ]);
    }

    #[test]
    fn test_remove_edge() {
        let mut mesh = ComboMesh1::<usize, usize>::default();
        let ids = mesh.extend_vertices(vec![3, 6, 9, 2]);
        let edges = vec![
            ([ids[0], ids[3]], 5),
            ([ids[1], ids[3]], 3),
            ([ids[3], ids[1]], 2),
            ([ids[2], ids[3]], 8),
            ([ids[1], ids[2]], 9),
        ];
        mesh.extend_edges(edges.clone());

        mesh.remove_edge([ids[1], ids[3]]); // first outgoing edge from vertex
        assert_edges(&mesh, vec![
            ([ids[0], ids[3]], 5),
            ([ids[3], ids[1]], 2),
            ([ids[2], ids[3]], 8),
            ([ids[1], ids[2]], 9),
        ]);

        mesh.remove_edge([ids[1], ids[2]]); // last outgoing edge from vertex
        assert_edges(&mesh, vec![
            ([ids[0], ids[3]], 5),
            ([ids[3], ids[1]], 2),
            ([ids[2], ids[3]], 8),
        ]);

        mesh.remove_edge([ids[3], ids[0]]); // nonexistent edge
        assert_edges(&mesh, vec![
            ([ids[0], ids[3]], 5),
            ([ids[3], ids[1]], 2),
            ([ids[2], ids[3]], 8),
        ]);
    }
}