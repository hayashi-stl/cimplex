use idmap::OrderedIdMap;
use idmap::table::DenseEntryTable;
use typenum::{U2, U3};
use std::fmt::Debug;
use std::ops::{Index, IndexMut};
use std::iter::{Map, Filter};
use std::convert::{TryFrom, TryInto};
use std::collections::hash_map;
use fnv::FnvHashMap;
#[cfg(feature = "serde_")]
use serde::{Serialize, Deserialize};

use crate::{VecN, impl_integer_id};

/// An index to a vertex of an tri mesh.
/// Will not be invalidated unless the vertex gets removed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde_", derive(Serialize, Deserialize))]
pub struct VertexId(u64);
impl_integer_id!(VertexId);

/// An edge id is just the edge's vertices in order.
/// The vertices are not allowed to be the same.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde_", derive(Serialize, Deserialize))]
pub struct EdgeId([VertexId; 2]);

impl TryFrom<[VertexId; 2]> for EdgeId {
    type Error = &'static str;

    fn try_from(vertices: [VertexId; 2]) -> Result<Self, Self::Error> {
        if vertices[0] == vertices[1] {
            Err("Vertices are not allowed to be the same")
        } else {
            Ok(EdgeId(vertices))
        }
    }
}

impl EdgeId {
    /// Gets the vertices that this edge id is made of
    pub fn vertices(self) -> [VertexId; 2] {
        self.0
    }

    /// Reverses the vertices of this edge id to get the one for the twin edge
    fn twin(self) -> Self {
        Self([self.0[1], self.0[0]])
    }

    /// Sets the target. Assumes it doesn't equal the source.
    fn target(mut self, vertex: VertexId) -> Self {
        self.0[1] = vertex;
        debug_assert_ne!(self.0[0], self.0[1]);
        self
    }
}

/// An triangle id is just the triangle's vertices in winding order,
/// with the smallest index first.
/// No two vertices are allowed to be the same.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde_", derive(Serialize, Deserialize))]
pub struct TriId([VertexId; 3]);

impl TryFrom<[VertexId; 3]> for TriId {
    type Error = &'static str;

    fn try_from(mut v: [VertexId; 3]) -> Result<Self, Self::Error> {
        v = Self::canonicalize(v);
        
        if v[0] == v[1] || v[2] == v[0] || v[1] == v[2] {
            Err("Vertices are not allowed to be the same")
        } else {
            Ok(TriId(v))
        }
    }
}

impl TriId {
    fn canonicalize(mut v: [VertexId; 3]) -> [VertexId; 3] {
        let min_pos = v.iter().enumerate().min_by_key(|(_, value)| **value).unwrap().0;
        v.rotate_left(min_pos);
        v
    }

    /// Gets the vertices that this tri id is made of
    pub fn vertices(self) -> [VertexId; 3] {
        self.0
    }

    /// Gets the edges of the triangle, with sources in the order of the vertices
    pub fn edges(self) -> [EdgeId; 3] {
        [
            EdgeId([self.0[0], self.0[1]]),
            EdgeId([self.0[1], self.0[2]]),
            EdgeId([self.0[2], self.0[0]]),
        ]
    }

    /// Gets the edges of the triangle, with sources in the order of the vertices.
    /// Each edge includes its opposite vertex.
    pub fn edges_and_opposite(self) -> [(EdgeId, VertexId); 3] {
        [
            (EdgeId([self.0[0], self.0[1]]), self.0[2]),
            (EdgeId([self.0[1], self.0[2]]), self.0[0]),
            (EdgeId([self.0[2], self.0[0]]), self.0[1]),
        ]
    }

    /// Gets the index of a vertex, assuming it's part of the face
    fn index(self, vertex: VertexId) -> usize {
        self.0.iter().position(|v| *v == vertex).unwrap()
    }

    /// Reverses the tri so it winds the other way
    fn twin(self) -> Self {
        Self([self.0[0], self.0[2], self.0[1]])
    }

    fn target(mut self, vertex: VertexId) -> Self {
        self.0[2] = vertex;
        self.0 = Self::canonicalize(self.0);
        self
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

type TriFilterFn<'a, F> = for<'b> fn(&'b (&'a TriId, &'a Tri<F>)) -> bool;
type TriMapFn<'a, F> = fn((&'a TriId, &'a Tri<F>)) -> (&'a TriId, &'a F);
pub type Tris<'a, F> = Map<Filter<hash_map::Iter<'a, TriId, Tri<F>>, TriFilterFn<'a, F>>, TriMapFn<'a, F>>;
type TriFilterFnMut<'a, F> = for<'b> fn(&'b (&'a TriId, &'a mut Tri<F>)) -> bool;
type TriMapFnMut<'a, F> = fn((&'a TriId, &'a mut Tri<F>)) -> (&'a TriId, &'a mut F);
pub type TrisMut<'a, F> = Map<Filter<hash_map::IterMut<'a, TriId, Tri<F>>, TriFilterFnMut<'a, F>>, TriMapFnMut<'a, F>>;

//// A vertex of an tri mesh
#[derive(Clone, Debug)]
#[doc(hidden)]
#[cfg_attr(feature = "serde_", derive(Serialize, Deserialize))]
pub struct Vertex<V> {
    /// `target` is this vertex's id if there is no target
    target: VertexId,
    value: V,
}

/// An edge of an tri mesh
#[derive(Clone, Debug)]
#[doc(hidden)]
#[cfg_attr(feature = "serde_", derive(Serialize, Deserialize))]
pub struct Edge<E> {
    /// Previous outgoing target from the same vertex, whether the edge actually exists or not
    prev_target: VertexId,
    /// Next outgoing target from the same vertex, whether the edge actually exists or not
    next_target: VertexId,
    /// Some vertex opposite this edge in a triangle.
    /// This is the edge's first vertex if the edge is not part of a triangle.
    opp: VertexId,
    /// The edge does not actually exist if the value is None;
    /// it is just there for the structural purpose of
    /// ensuring that every edge has a twin.
    value: Option<E>,
}

/// A triangle of an tri mesh
#[derive(Clone, Debug)]
#[doc(hidden)]
#[cfg_attr(feature = "serde_", derive(Serialize, Deserialize))]
pub struct Tri<F> {
    /// Previous targets from the same edge for each of the edges,
    /// whether the face actually exists or not
    prev_targets: [VertexId; 3],
    /// Next targets from the same edge for each of the edges,
    /// whether the face actually exists or not
    next_targets: [VertexId; 3],
    /// The triangle does not actually exist if the value is None;
    /// it is just there for the structural purpose of
    /// ensuring that every triangle has a twin.
    value: Option<F>,
}

impl<F> Tri<F> {
    fn prev_target(&self, id: TriId, edge: EdgeId) -> VertexId {
        self.prev_targets[id.index(edge.0[0])]
    }

    fn prev_target_mut(&mut self, id: TriId, edge: EdgeId) -> &mut VertexId {
        &mut self.prev_targets[id.index(edge.0[0])]
    }

    fn next_target(&self, id: TriId, edge: EdgeId) -> VertexId {
        self.next_targets[id.index(edge.0[0])]
    }

    fn next_target_mut(&mut self, id: TriId, edge: EdgeId) -> &mut VertexId {
        &mut self.next_targets[id.index(edge.0[0])]
    }
}

/// A combinatorial simplicial 2-complex, containing only vertices, (oriented) edges, and (oriented) triangles.
/// Also known as an tri mesh.
/// Each vertex stores a value of type `V`.
/// Each edge stores its vertices and a value of type `E`.
/// Each triangle stores its vertices and a value of type `F`.
/// The edge manipulation methods can either be called with an array of 2 `VertexId`s
/// or an `EdgeId`.
/// The triangle manipulation methods can either be called with an array of 3 `VertexId`s
/// or an `TriId`.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde_", derive(Serialize, Deserialize))]
pub struct ComboMesh2<V, E, F> {
    vertices: OrderedIdMap<VertexId, Vertex<V>>,
    edges: FnvHashMap<EdgeId, Edge<E>>,
    tris: FnvHashMap<TriId, Tri<F>>,
    next_vertex_id: u64,
    /// Keep separate track because edge twins may or may not exist
    num_edges: usize,
    num_tris: usize,
}

impl<V, E, F> Default for ComboMesh2<V, E, F> {
    fn default() -> Self {
        ComboMesh2 {
            vertices: OrderedIdMap::default(),
            edges: FnvHashMap::default(),
            tris: FnvHashMap::default(),
            next_vertex_id: 0,
            num_edges: 0,
            num_tris: 0,
        }
    }
}

impl<V, E, F> ComboMesh2<V, E, F> {
    /// Creates an empty tri mesh.
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

    /// Gets the number of edges.
    pub fn num_edges(&self) -> usize {
        self.num_edges
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

    /// Gets the number of triangles.
    pub fn num_tris(&self) -> usize {
        self.num_tris
    }

    /// Iterates over the triangles of this mesh.
    /// Gives (id, value) pairs
    pub fn tris(&self) -> Tris<F> {
        self.tris.iter()
            .filter::<TriFilterFn<F>>(|(_, f)| f.value.is_some())
            .map::<_, TriMapFn<F>>(|(id, f)| (id, f.value.as_ref().unwrap()))
    }

    /// Iterates mutably over the triangles of this mesh.
    /// Gives (id, value) pairs
    pub fn tris_mut(&mut self) -> TrisMut<F> {
        self.tris.iter_mut()
            .filter::<TriFilterFnMut<F>>(|(_, f)| f.value.is_some())
            .map::<_, TriMapFnMut<F>>(|(id, f)| (id, f.value.as_mut().unwrap()))
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

    /// Iterates over the outgoing edges of a vertex.
    /// The vertex must exist.
    pub fn vertex_edges_out(&self, vertex: VertexId) -> VertexEdgesOut<V, E, F> {
        if let Some(walker) = self.edge_walker_from_vertex(vertex) {
            let start_target = walker.target();
            VertexEdgesOut {
                walker,
                start_target,
                finished: false,
            }
        } else {
            VertexEdgesOut {
                walker: EdgeWalker::dummy(self),
                start_target: VertexId(0),
                finished: true,
            }
        }
    }

    /// Iterates over the incoming edges of a vertex.
    /// The vertex must exist.
    pub fn vertex_edges_in(&self, vertex: VertexId) -> VertexEdgesIn<V, E, F> {
        if let Some(walker) = EdgeWalker::from_vertex_less_checked(self, vertex).and_then(|w| w.backward()) {
            let start_source = walker.vertex();
            VertexEdgesIn {
                walker,
                start_source,
                finished: false,
            }
        } else {
            VertexEdgesIn {
                walker: EdgeWalker::dummy(self),
                start_source: VertexId(0),
                finished: true,
            }
        }
    }

    /// Iterates over the triangles that an edge is part of.
    /// The edge must exist.
    pub fn edge_tris<EI: TryInto<EdgeId>>(&self, edge: EdgeId) -> EdgeTris<V, E, F> {
        if let Some(walker) = self.tri_walker_from_edge(edge) {
            let start_opp = walker.opp();
            EdgeTris {
                walker,
                start_opp,
                finished: false,
            }
        } else {
            EdgeTris {
                walker: TriWalker::dummy(self),
                start_opp: VertexId(0),
                finished: true,
            }
        }
    }

    /// Gets the value of the edge at a specific id.
    /// Returns None if not found.
    pub fn edge<EI: TryInto<EdgeId>>(&self, id: EI) -> Option<&E> {
        id.try_into().ok().and_then(|id| self.edges.get(&id)).and_then(|e| e.value.as_ref())
    }

    /// Gets the value of the edge at a specific id mutably.
    /// Returns None if not found.
    pub fn edge_mut<EI: TryInto<EdgeId>>(&mut self, id: EI) -> Option<&mut E> {
        id.try_into().ok().and_then(move |id| self.edges.get_mut(&id)).and_then(|e| e.value.as_mut())
    }
    
    /// Gets the value of the triangle at a specific id.
    /// Returns None if not found.
    pub fn tri<FI: TryInto<TriId>>(&self, id: FI) -> Option<&F> {
        id.try_into().ok().and_then(|id| self.tris.get(&id)).and_then(|f| f.value.as_ref())
    }

    /// Gets the value of the triangle at a specific id mutably.
    /// Returns None if not found.
    pub fn tri_mut<FI: TryInto<TriId>>(&mut self, id: FI) -> Option<&mut F> {
        id.try_into().ok().and_then(move |id| self.tris.get_mut(&id)).and_then(|f| f.value.as_mut())
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
    pub fn remove_vertex(&mut self, id: VertexId) -> (Option<V>, Vec<E>, Vec<F>) {
        let vertex = match self.vertices.get(id) {
            Some(vertex) => vertex,
            None => return (None, vec![], vec![]),
        };

        if vertex.target == id {
            (self.vertices.remove(id).map(|v| v.value), vec![], vec![])
        } else {
            // Get edges to remove
            let mut targets = vec![];
            let mut target = vertex.target;
            while targets.last() != Some(&vertex.target) {
                target = self.edges[&[id, target].try_into().ok().unwrap()].next_target;
                targets.push(target);
            }

            // Remove edges
            let (edge_values, face_values) = targets.into_iter().map(|target| {
                let fwd = self.remove_edge([id, target]);
                let inv = self.remove_edge([target, id]);
                let edge_values = vec![fwd.0, inv.0].into_iter().flatten().collect::<Vec<_>>();
                let tri_values = fwd.1.into_iter().chain(inv.1).collect::<Vec<_>>();
                (edge_values, tri_values)
            })
            .fold((vec![], vec![]), |(mut e_acc, mut f_acc), (e, f)| {
                e_acc.extend(e);
                f_acc.extend(f);
                (e_acc, f_acc)
            });
            (self.vertices.remove(id).map(|v| v.value), edge_values, face_values)
        }
    }

    /// Removes a list of vertices.
    pub fn remove_vertices<I: IntoIterator<Item = VertexId>>(&mut self, iter: I) {
        iter.into_iter().for_each(|id| { self.remove_vertex(id); })
    }

    /// Keeps only the vertices that satisfy a predicate
    pub fn retain_vertices<P: FnMut(VertexId, &V) -> bool>(&mut self, mut predicate: P) {
        let to_remove = self.vertices()
            .filter(|(id, v)| !predicate(**id, *v))
            .map(|(id, _)| *id)
            .collect::<Vec<_>>();
        self.remove_vertices(to_remove);
    }

    /// Removes all vertices from the mesh.
    /// Removes all edges as a side-effect.
    pub fn clear_vertices(&mut self) {
        self.tris.clear();
        self.num_tris = 0;
        self.edges.clear();
        self.num_edges = 0;
        self.vertices.clear();
    }

    /// Adds an edge to the mesh. Vertex order is important!
    /// If the edge was already there, this replaces the value.
    /// Returns the previous value of the edge, if there was one.
    ///
    /// # Panics
    /// Panics if either vertex doesn't exist or if the vertices are the same
    pub fn add_edge<EI: TryInto<EdgeId>>(&mut self, vertices: EI, value: E) -> Option<E> {
        let id = vertices.try_into().ok().unwrap();

        // Can't use entry().or_insert() because that would cause a
        // mutable borrow and an immutable borrow to exist at the same time
        if let Some(edge) = self.edges.get_mut(&id) {
            let old = edge.value.take();
            edge.value = Some(value);
            if old.is_none() {
                self.num_edges += 1;
            }
            old
        } else {
            self.num_edges += 1;

            let mut insert_edge = |id: EdgeId, value: Option<E>| {
                let target = self.vertices[id.0[0]].target;

                let (prev, next) = if target == id.0[0] { // First edge from vertex
                    self.vertices[id.0[0]].target = id.0[1];
                    (id.0[1], id.0[1])
                } else {
                    let prev = self.edges[&[id.0[0], target].try_into().ok().unwrap()].prev_target;
                    let next = target;
                    self.edges.get_mut(&id.target(prev)).unwrap().next_target = id.0[1];
                    self.edges.get_mut(&id.target(next)).unwrap().prev_target = id.0[1];
                    (prev, next)
                };

                self.edges.insert(id, Edge {
                    prev_target: prev,
                    next_target: next,
                    opp: id.0[0],
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
    pub fn extend_edges<EI: TryInto<EdgeId>, I: IntoIterator<Item = (EI, E)>>(&mut self, iter: I) {
        iter.into_iter().for_each(|(id, value)| { self.add_edge(id, value); })
    }

    /// Removes an edge from the mesh and returns the value that was there,
    /// or None if there was nothing there
    pub fn remove_edge<EI: TryInto<EdgeId>>(&mut self, id: EI) -> (Option<E>, Vec<F>) {
        // TODO: REMOVE APPROPRIATE FACES
        let id = match id.try_into() {
            Ok(id) => id,
            Err(_) => return (None, vec![]),
        };

        if !self.vertices.contains_key(id.0[0]) || !self.vertices.contains_key(id.0[1]) {
            return (None, vec![]);
        }

        let tri_values = if let Some(edge) = self.edges.get(&id) {
            if edge.opp != id.0[0] {
                // Get tris to remove
                let mut targets = vec![];
                let mut target = edge.opp;
                while targets.last() != Some(&edge.opp) {
                    let tri = [id.0[0], id.0[1], target].try_into().ok().unwrap();
                    target = self.tris[&tri].next_target(tri, id);
                    targets.push(target);
                }

                // Remove tris
                targets.into_iter().flat_map(|target| 
                    self.remove_tri([id.0[0], id.0[1], target])
                ).collect::<Vec<_>>()
            } else {
                vec![]
            }
        } else {
            vec![]
        };

        match self.edges.get(&id.twin()).map(|e| e.value.as_ref()) {
            // Twin actually exists; just set value to None
            Some(Some(_)) => {
                let old = self.edges.get_mut(&id).unwrap().value.take();
                if old.is_some() {
                    self.num_edges -= 1;
                }
                (old, tri_values)
            }

            // Twin is phantom, so remove both edge and twin from map
            Some(None) => {
                // Twin is phantom, so this edge actually exists.
                self.num_edges -= 1;

                let mut delete_edge = |id: EdgeId| {
                    let edge = &self.edges[&id];
                    let prev = edge.prev_target;
                    let next = edge.next_target;
                    let source = &mut self.vertices[id.0[0]];
                    self.edges.get_mut(&id.target(prev)).unwrap().next_target = next;
                    self.edges.get_mut(&id.target(next)).unwrap().prev_target = prev;

                    if id.0[1] == next { // this was the last edge from the vertex
                        source.target = id.0[0];
                    } else if id.0[1] == source.target {
                        source.target = next;
                    }

                    self.edges.remove(&id).and_then(|e| e.value)
                };
                
                delete_edge(id.twin());
                (delete_edge(id), tri_values)
            }

            // Twin isn't in map, and neither is the edge to remove
            None => (None, tri_values),
        }
    }

    /// Removes a list of edges.
    pub fn remove_edges<EI: TryInto<EdgeId>, I: IntoIterator<Item = EI>>(&mut self, iter: I) {
        iter.into_iter().for_each(|id| { self.remove_edge(id); })
    }

    /// Keeps only the edges that satisfy a predicate
    pub fn retain_edges<P: FnMut(EdgeId, &E) -> bool>(&mut self, mut predicate: P) {
        let to_remove = self.edges()
            .filter(|(id, e)| !predicate(**id, *e))
            .map(|(id, _)| *id)
            .collect::<Vec<_>>();
        self.remove_edges(to_remove);
    }

    /// Removes all edges from the mesh.
    pub fn clear_edges(&mut self) {
        self.tris.clear();
        self.num_tris = 0;
        self.edges.clear();
        self.num_edges = 0;
        // Fix vertex-target links
        for (id, vertex) in &mut self.vertices {
            vertex.target = *id;
        }
    }

    /// Adds a triangle to the mesh. Vertex order is important!
    /// If the triangle was already there, this replaces the value.
    /// Returns the previous value of the triangle, if there was one.
    ///
    /// # Panics
    /// Panics if any vertex doesn't exist or if any two vertices are the same,
    /// or if any of the required edges doesn't exist. Use `add_tri_and_edges`
    /// to automatically add the required edges.
    pub fn add_tri<FI: TryInto<TriId>>(&mut self, vertices: FI, value: F) -> Option<F> {
        let id = vertices.try_into().ok().unwrap();

        // Can't use entry().or_insert() because that would cause a
        // mutable borrow and an immutable borrow to exist at the same time
        if let Some(tri) = self.tris.get_mut(&id) {
            let old = tri.value.take();
            tri.value = Some(value);
            if old.is_none() {
                self.num_tris += 1;
            }
            old
        } else {
            self.num_tris += 1;

            let mut insert_tri = |id: TriId, value: Option<F>| {
                let mut prev_targets = [VertexId(0); 3];
                let mut next_targets = [VertexId(0); 3];

                for (i, (edge, opp)) in id.edges_and_opposite().iter().enumerate() {
                    let target = self.edges[edge].opp;
                    
                    let (prev, next) = if target == edge.0[0] { // First tri from edge
                        self.edges.get_mut(edge).unwrap().opp = *opp;
                        (*opp, *opp)
                    } else {
                        let side = [edge.0[0], edge.0[1], target].try_into().ok().unwrap();
                        let prev = self.tris[&side].prev_target(side, *edge);
                        let next = target;
                        let prev_tri = id.target(prev);
                        let next_tri = id.target(next);
                        *self.tris.get_mut(&prev_tri).unwrap().next_target_mut(prev_tri, *edge) = *opp;
                        *self.tris.get_mut(&next_tri).unwrap().prev_target_mut(next_tri, *edge) = *opp;
                        (prev, next)
                    };

                    prev_targets[i] = prev;
                    next_targets[i] = next;
                }

                self.tris.insert(id, Tri {
                    prev_targets,
                    next_targets,
                    value,
                });
            };

            insert_tri(id, Some(value));
            insert_tri(id.twin(), None);
            None
        }
    }

    /// Extends the triangle list with an iterator.
    ///
    /// # Panics
    /// Panics if any vertex doesn't exist or if any two vertices are the same
    /// in any of the triangles.
    pub fn extend_tris<FI: TryInto<TriId>, I: IntoIterator<Item = (FI, F)>>(&mut self, iter: I) {
        iter.into_iter().for_each(|(id, value)| { self.add_tri(id, value); })
    }

    /// Removes an triangle from the mesh and returns the value that was there,
    /// or None if there was nothing there
    pub fn remove_tri<FI: TryInto<TriId>>(&mut self, id: FI) -> Option<F> {
        let id = match id.try_into() {
            Ok(id) => id,
            Err(_) => return None,
        };

        if !self.vertices.contains_key(id.0[0]) || !self.vertices.contains_key(id.0[1]) || !self.vertices.contains_key(id.0[2]){
            return None;
        }

        match self.tris.get(&id.twin()).map(|f| f.value.as_ref()) {
            // Twin actually exists; just set value to None
            Some(Some(_)) => {
                let old = self.tris.get_mut(&id).unwrap().value.take();
                if old.is_some() {
                    self.num_tris -= 1;
                }
                old
            }

            // Twin is phantom, so remove both tri and twin from map
            Some(None) => {
                // Twin is phantom, so this tri actually exists.
                self.num_tris -= 1;

                let mut delete_tri = |id: TriId| {
                    for (i, (edge, opp)) in id.edges_and_opposite().iter().enumerate() {
                        let tri = &self.tris[&id];
                        let prev = tri.prev_targets[i];
                        let next = tri.next_targets[i];
                        let source = self.edges.get_mut(&edge).unwrap();
                        let prev_tri = id.target(prev);
                        let next_tri = id.target(next);
                        *self.tris.get_mut(&prev_tri).unwrap().next_target_mut(prev_tri, *edge) = next;
                        *self.tris.get_mut(&next_tri).unwrap().prev_target_mut(next_tri, *edge) = prev;

                        if *opp == next { // this was the last tri from the edge
                            source.opp = edge.0[0];
                        } else if *opp == source.opp {
                            source.opp = next;
                        }
                    }
                    
                    self.tris.remove(&id).and_then(|f| f.value)
                };
                
                delete_tri(id.twin());
                delete_tri(id)
            }

            // Twin isn't in map, and neither is the tri to remove
            None => None,
        }
    }

    /// Removes a list of triangles.
    pub fn remove_tris<FI: TryInto<TriId>, I: IntoIterator<Item = FI>>(&mut self, iter: I) {
        iter.into_iter().for_each(|id| { self.remove_tri(id); })
    }

    /// Keeps only the triangles that satisfy a predicate
    pub fn retain_tris<P: FnMut(TriId, &F) -> bool>(&mut self, mut predicate: P) {
        let to_remove = self.tris()
            .filter(|(id, f)| !predicate(**id, *f))
            .map(|(id, _)| *id)
            .collect::<Vec<_>>();
        self.remove_tris(to_remove);
    }

    /// Removes all triangles from the mesh.
    pub fn clear_tris(&mut self) {
        self.tris.clear();
        self.num_tris = 0;
        // Fix edge-target links
        for (id, edge) in &mut self.edges {
            edge.opp = id.0[0];
        }
    }

    /// Gets an edge walker that starts at the given vertex.
    /// Returns None if the vertex has no outgoing edge.
    pub fn edge_walker_from_vertex(&self, vertex: VertexId) -> Option<EdgeWalker<V, E, F>> {
        EdgeWalker::from_vertex(self, vertex)
    }

    /// Gets an edge walker that starts at the given edge.
    pub fn edge_walker_from_edge<EI: TryInto<EdgeId>>(&self, edge: EI) -> EdgeWalker<V, E, F> {
        EdgeWalker::new(self, edge)
    }

    /// Gets a triangle walker that starts at the given edge.
    /// Returns None if the edge has no triangle.
    pub fn tri_walker_from_edge<EI: TryInto<EdgeId>>(&self, edge: EI) -> Option<TriWalker<V, E, F>> {
        TriWalker::from_edge(self, edge)
    }

    /// Gets a triangle walker that starts at the given edge with the given opposite vertex.
    pub fn tri_walker_from_tri<EI: TryInto<EdgeId>>(&self, edge: EI, opp: VertexId) -> TriWalker<V, E, F> {
        TriWalker::new(self, edge, opp)
    }
}

/// A walker for navigating a simplicial 2-complex by edge
#[derive(Debug)]
pub struct EdgeWalker<'a, V, E, F> {
    mesh: &'a ComboMesh2<V, E, F>,
    edge: EdgeId,
}

impl<'a, V, E, F> Clone for EdgeWalker<'a, V, E, F> {
    fn clone(&self) -> Self {
        Self {
            mesh: self.mesh,
            edge: self.edge,
        }
    }
}

impl<'a, V, E, F> Copy for EdgeWalker<'a, V, E, F> {}

impl<'a, V, E, F> EdgeWalker<'a, V, E, F> {
    fn new<EI: TryInto<EdgeId>>(mesh: &'a ComboMesh2<V, E, F>, edge: EI) -> Self {
        Self {
            mesh,
            edge: edge.try_into().ok().unwrap(),
        }
    }

    /// Doesn't check that the starting edge actually exists
    fn from_vertex_less_checked(mesh: &'a ComboMesh2<V, E, F>, vertex: VertexId) -> Option<Self> {
        ([vertex, mesh.vertices[vertex].target].try_into().ok() as Option<EdgeId>)
            .map(|edge| Self::new(mesh, edge))
    }

    fn from_vertex(mesh: &'a ComboMesh2<V, E, F>, vertex: VertexId) -> Option<Self> {
        let start = match [vertex, mesh.vertices[vertex].target].try_into() {
            Ok(start) => start,
            Err(_) => return None,
        };
        let mut edge = start;
        while mesh.edges[&edge].value.is_none() {
            edge = edge.target(mesh.edges[&edge].next_target);
            if edge == start {
                return None;
            }
        }

        Some(Self::new(mesh, edge))
    }
    
    /// A walker that will not be used
    fn dummy(mesh: &'a ComboMesh2<V, E, F>) -> Self {
        Self {
            mesh,
            edge: EdgeId([VertexId(0), VertexId(1)])
        }
    }

    /// Get the mesh that the walker navigates
    pub fn mesh(&self) -> &ComboMesh2<V, E, F> {
        self.mesh
    }

    /// Get the current vertex id,
    /// which is the source of the current edge.
    pub fn vertex(&self) -> VertexId {
        self.edge.0[0]
    }

    /// Get the vertex id of the target of the current edge.
    pub fn target(&self) -> VertexId {
        self.edge.0[1]
    }

    /// Gets the current edge id
    pub fn edge(&self) -> EdgeId {
        self.edge
    }

    /// Reverse the walker's direction so its
    /// current edge is the opposite edge.
    /// Returns None if the resulting edge doesn't exist.
    pub fn twin(mut self) -> Option<Self> {
        self.edge = self.edge.twin();
        if self.mesh.edge(self.edge).is_some() {
            Some(self)
        } else {
            None
        }
    }

    /// Sets the current edge to the next one with the same source vertex.
    pub fn next(mut self) -> Self {
        while {
            self.edge = self.edge.target(self.mesh.edges[&self.edge].next_target);
            self.mesh.edge(self.edge).is_none()
        } {}
        self
    }

    /// Sets the current edge to the previous one with the same source vertex.
    pub fn prev(mut self) -> Self {
        while {
            self.edge = self.edge.target(self.mesh.edges[&self.edge].prev_target);
            self.mesh.edge(self.edge).is_none()
        } {}
        self
    }

    /// Sets the current edge to the next one with the same target vertex.
    pub fn next_in(mut self) -> Self {
        while {
            self.edge = self.edge.twin().target(self.mesh.edges[&self.edge.twin()].next_target).twin();
            self.mesh.edge(self.edge).is_none()
        } {}
        self
    }

    /// Sets the current edge to the previous one with the same target vertex.
    pub fn prev_in(mut self) -> Self {
        while {
            self.edge = self.edge.twin().target(self.mesh.edges[&self.edge.twin()].prev_target).twin();
            self.mesh.edge(self.edge).is_none()
        } {}
        self
    }

    /// Sets the current edge to an edge whose source vertex is the current edge's target vertex.
    /// Chooses a non-twin edge if possible.
    pub fn forward(mut self) -> Option<Self> {
        let twin = self.edge.twin();
        self.edge = twin;
        let mut found = true;
        while {
            self.edge = self.edge.target(self.mesh.edges[&self.edge].next_target);
            if self.mesh.edge(self.edge).is_some() {
                false
            } else if self.edge == twin {
                found = false;
                false
            } else {
                true
            }
        } {}
        if found { Some(self) } else { None }
    }

    /// Sets the current edge to an edge whose target vertex is the current edge's source vertex.
    /// Chooses a non-twin edge if possible.
    pub fn backward(mut self) -> Option<Self> {
        let twin = self.edge.twin();
        self.edge = twin;
        let mut found = true;
        while {
            self.edge = self.edge.twin().target(self.mesh.edges[&self.edge.twin()].next_target).twin();
            if self.mesh.edge(self.edge).is_some() {
                false
            } else if self.edge == twin {
                found = false;
                false
            } else {
                true
            }
        } {}
        if found { Some(self) } else { None }
    }

    pub fn tri_walker(self) -> Option<TriWalker<'a, V, E, F>> {
        TriWalker::from_edge(self.mesh, self.edge)
    }
}

/// A walker for navigating a simplicial 2-complex by triangle
#[derive(Debug)]
pub struct TriWalker<'a, V, E, F> {
    mesh: &'a ComboMesh2<V, E, F>,
    edge: EdgeId,
    opp: VertexId,
}

impl<'a, V, E, F> Clone for TriWalker<'a, V, E, F> {
    fn clone(&self) -> Self {
        Self {
            mesh: self.mesh,
            edge: self.edge,
            opp: self.opp,
        }
    }
}

impl<'a, V, E, F> Copy for TriWalker<'a, V, E, F> {}

impl<'a, V, E, F> TriWalker<'a, V, E, F> {
    fn new<EI: TryInto<EdgeId>>(mesh: &'a ComboMesh2<V, E, F>, edge: EI, opp: VertexId) -> Self {
        Self {
            mesh,
            edge: edge.try_into().ok().unwrap(),
            opp,
        }
    }

    fn from_edge<EI: TryInto<EdgeId>>(mesh: &'a ComboMesh2<V, E, F>, edge: EI) -> Option<Self> {
        let edge = edge.try_into().ok().unwrap();
        let start = match [edge.0[0], edge.0[1], mesh.edges[&edge].opp].try_into() {
            Ok(start) => start,
            Err(_) => return None,
        };
        let mut tri = start;
        while mesh.tris[&tri].value.is_none() {
            tri = tri.target(mesh.tris[&tri].next_target(tri, edge));
            if tri == start {
                return None;
            }
        }

        let index = tri.index(edge.0[0]);
        let (edge, opp) = tri.edges_and_opposite()[index];
        Some(Self::new(mesh, edge, opp))
    }
    
    /// A walker that will not be used
    fn dummy(mesh: &'a ComboMesh2<V, E, F>) -> Self {
        Self {
            mesh,
            edge: EdgeId([VertexId(0), VertexId(1)]),
            opp: VertexId(2),
        }
    }

    /// Get the mesh that the walker navigates
    pub fn mesh(&self) -> &ComboMesh2<V, E, F> {
        self.mesh
    }

    /// Get the current vertex id,
    /// which is the source of the current tri edge.
    pub fn vertex(&self) -> VertexId {
        self.edge.0[0]
    }

    /// Get the vertex id of the target of the current tri edge.
    pub fn second(&self) -> VertexId {
        self.edge.0[1]
    }

    /// Gets the current edge id
    pub fn edge(&self) -> EdgeId {
        self.edge
    }

    /// Gets the opposite vertex of the current triangle
    pub fn opp(&self) -> VertexId {
        self.opp
    }

    /// Gets the current triangle id
    pub fn tri(&self) -> TriId {
        TriId(TriId::canonicalize([self.vertex(), self.second(), self.opp()]))
    }

    /// Reverse the walker's direction so its
    /// current triangle is the opposite triangle without changing the opposite vertex
    /// Returns None if the resulting triangle doesn't exist.
    pub fn twin(mut self) -> Option<Self> {
        self.edge = self.edge.twin();
        if self.mesh.tri(self.tri()).is_some() {
            Some(self)
        } else {
            None
        }
    }

    /// Sets the current edge to the next one in the same triangle.
    pub fn next_edge(mut self) -> Self {
        let (edge, opp) = (EdgeId([self.second(), self.opp()]), self.vertex());
        self.edge = edge;
        self.opp = opp;
        self
    }

    /// Sets the current edge to the previous one in the same triangle.
    pub fn prev_edge(mut self) -> Self {
        let (edge, opp) = (EdgeId([self.opp(), self.vertex()]), self.second());
        self.edge = edge;
        self.opp = opp;
        self
    }

    /// Sets the current opposite vertex to the next one with the same edge.
    pub fn next_opp(mut self) -> Self {
        while {
            let tri = self.tri();
            self.opp = self.mesh.tris[&self.tri()].next_target(tri, self.edge);
            self.mesh.tris[&self.tri()].value.is_none()
        } {}
        self
    }

    /// Sets the current opposite vertex to the previoud one with the same edge.
    pub fn prev_opp(mut self) -> Self {
        while {
            let tri = self.tri();
            self.opp = self.mesh.tris[&self.tri()].prev_target(tri, self.edge);
            self.mesh.tris[&self.tri()].value.is_none()
        } {}
        self
    }

    /// Turns this into an edge walker that starts
    /// at the current edge.
    pub fn edge_walker(self) -> EdgeWalker<'a, V, E, F> {
        EdgeWalker::new(self.mesh, self.edge)
    }
}

/// An iterator over the outgoing edges of a vertex.
#[derive(Clone, Debug)]
pub struct VertexEdgesOut<'a, V, E, F> {
    walker: EdgeWalker<'a, V, E, F>,
    start_target: VertexId,
    finished: bool,
}

impl<'a, V, E, F> Iterator for VertexEdgesOut<'a, V, E, F> {
    type Item = EdgeId;
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        let edge = self.walker.edge();
        self.walker = self.walker.next();
        if self.walker.edge().0[1] == self.start_target {
            self.finished = true;
        }
        Some(edge)
    }
}

/// An iterator over the incoming edges of a vertex.
#[derive(Clone, Debug)]
pub struct VertexEdgesIn<'a, V, E, F> {
    walker: EdgeWalker<'a, V, E, F>,
    start_source: VertexId,
    finished: bool,
}

impl<'a, V, E, F> Iterator for VertexEdgesIn<'a, V, E, F> {
    type Item = EdgeId;
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        let edge = self.walker.edge();
        self.walker = self.walker.next_in();
        if self.walker.edge().0[0] == self.start_source {
            self.finished = true;
        }
        Some(edge)
    }
}

/// An iterator over the triangles of an edge.
#[derive(Clone, Debug)]
pub struct EdgeTris<'a, V, E, F> {
    walker: TriWalker<'a, V, E, F>,
    start_opp: VertexId,
    finished: bool,
}

impl<'a, V, E, F> Iterator for EdgeTris<'a, V, E, F> {
    type Item = TriId;
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        let tri = self.walker.tri();
        self.walker = self.walker.next_opp();
        if self.walker.opp() == self.start_opp {
            self.finished = true;
        }
        Some(tri)
    }
}


impl<V, E, F> Index<VertexId> for ComboMesh2<V, E, F> {
    type Output = V;

    fn index(&self, index: VertexId) -> &Self::Output {
        self.vertex(index).unwrap()
    }
}

impl<V, E, F> IndexMut<VertexId> for ComboMesh2<V, E, F> {
    fn index_mut(&mut self, index: VertexId) -> &mut Self::Output {
        self.vertex_mut(index).unwrap()
    }
}

impl<V, E, F> Index<[VertexId; 2]> for ComboMesh2<V, E, F> {
    type Output = E;

    fn index(&self, index: [VertexId; 2]) -> &Self::Output {
        self.edge(index).unwrap()
    }
}

impl<V, E, F> IndexMut<[VertexId; 2]> for ComboMesh2<V, E, F> {
    fn index_mut(&mut self, index: [VertexId; 2]) -> &mut Self::Output {
        self.edge_mut(index).unwrap()
    }
}

impl<V, E, F> Index<EdgeId> for ComboMesh2<V, E, F> {
    type Output = E;

    fn index(&self, index: EdgeId) -> &Self::Output {
        self.edge(index).unwrap()
    }
}

impl<V, E, F> IndexMut<EdgeId> for ComboMesh2<V, E, F> {
    fn index_mut(&mut self, index: EdgeId) -> &mut Self::Output {
        self.edge_mut(index).unwrap()
    }
}

impl<V, E, F> Index<[VertexId; 3]> for ComboMesh2<V, E, F> {
    type Output = F;

    fn index(&self, index: [VertexId; 3]) -> &Self::Output {
        self.tri(index).unwrap()
    }
}

impl<V, E, F> IndexMut<[VertexId; 3]> for ComboMesh2<V, E, F> {
    fn index_mut(&mut self, index: [VertexId; 3]) -> &mut Self::Output {
        self.tri_mut(index).unwrap()
    }
}

impl<V, E, F> Index<TriId> for ComboMesh2<V, E, F> {
    type Output = F;

    fn index(&self, index: TriId) -> &Self::Output {
        self.tri(index).unwrap()
    }
}

impl<V, E, F> IndexMut<TriId> for ComboMesh2<V, E, F> {
    fn index_mut(&mut self, index: TriId) -> &mut Self::Output {
        self.tri_mut(index).unwrap()
    }
}

/// A position-containing tri mesh
pub type Mesh2<V, E, F, D> = ComboMesh2<(VecN<D>, V), E, F>;

/// A 2D-position-containing tri mesh
pub type Mesh22<V, E, F> = Mesh2<V, E, F, U2>;

/// A 3D-position-containing tri mesh
pub type Mesh23<V, E, F> = Mesh2<V, E, F, U3>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt::Debug;
    use std::hash::Hash;
    use fnv::FnvHashSet;

    #[track_caller]
    fn assert_vertices<V: Clone + Debug + Eq + Hash, E, F, I: IntoIterator<Item = (VertexId, V)>>(mesh: &ComboMesh2<V, E, F>, vertices: I) {
        let result = mesh.vertices().map(|(id, v)| (*id, v.clone()))
            .collect::<FnvHashSet<_>>();
        let expect = vertices.into_iter().collect::<FnvHashSet<_>>();

        assert_eq!(result, expect);
    }

    #[track_caller]
    fn assert_edges<V, E: Clone + Debug + Eq + Hash, EI: TryInto<EdgeId>, F, I: IntoIterator<Item = (EI, E)>>(mesh: &ComboMesh2<V, E, F>, edges: I) {
        let result = mesh.edges().map(|(id, e)| (*id, e.clone()))
            .collect::<FnvHashSet<_>>();
        let expect = edges.into_iter()
            .map(|(vertices, e)| (vertices.try_into().ok().unwrap(), e))
            .collect::<FnvHashSet<_>>();

        assert_eq!(result, expect);
        assert_eq!(mesh.num_edges(), expect.len());
    }

    #[test]
    fn test_default() {
        let mesh = ComboMesh2::<(), (), ()>::default();
        assert!(mesh.vertices.is_empty());
        assert!(mesh.edges.is_empty());
        assert_eq!(mesh.num_edges(), 0);
    }

    #[test]
    fn test_add_vertex() {
        let mut mesh = ComboMesh2::<usize, usize, usize>::default();
        let id = mesh.add_vertex(3);
        assert_eq!(mesh.vertex(id), Some(&3));

        let id2 = mesh.add_vertex(9);
        assert_eq!(mesh.vertex(id), Some(&3));
        assert_eq!(mesh.vertex(id2), Some(&9));
    }

    #[test]
    fn test_extend_vertices() {
        let mut mesh = ComboMesh2::<usize, usize, usize>::default();
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
        let mut mesh = ComboMesh2::<usize, usize, usize>::default();
        let ids = mesh.extend_vertices(vec![3, 6, 9, 2]);
        let prev = mesh.add_edge([ids[1], ids[3]], 54);
        assert_eq!(prev, None);
        assert_eq!(mesh.edge([ids[1], ids[3]]), Some(&54));
        assert_eq!(mesh.edge([ids[3], ids[1]]), None); // twin should not exist
        assert_eq!(mesh.num_edges(), 1);

        // Add twin
        let prev = mesh.add_edge([ids[3], ids[1]], 27);
        assert_eq!(prev, None);
        assert_eq!(mesh.edge([ids[1], ids[3]]), Some(&54));
        assert_eq!(mesh.edge([ids[3], ids[1]]), Some(&27));
        assert_eq!(mesh.num_edges(), 2);

        // Modify edge
        let prev = mesh.add_edge([ids[1], ids[3]], 1);
        assert_eq!(prev, Some(54));
        assert_eq!(mesh.edge([ids[1], ids[3]]), Some(&1));
        assert_eq!(mesh.edge([ids[3], ids[1]]), Some(&27));
        assert_eq!(mesh.num_edges(), 2);
    }

    #[test]
    #[should_panic]
    fn test_add_edge_bad() {
        let mut mesh = ComboMesh2::<usize, usize, usize>::default();
        let ids = mesh.extend_vertices(vec![3, 6, 9, 2]);
        mesh.add_edge([ids[1], ids[1]], 4);
    }

    #[test]
    fn test_extend_edges() {
        let mut mesh = ComboMesh2::<usize, usize, usize>::default();
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
        assert_eq!(mesh.num_edges(), 5);
    }

    #[test]
    fn test_remove_vertex() {
        let mut mesh = ComboMesh2::<usize, usize, usize>::default();
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

        mesh.remove_vertex(ids[4]); // nonexistent vertex
        assert_vertices(&mesh, vec![(ids[0], 3), (ids[2], 9), (ids[3], 2)]);
        assert_edges(&mesh, vec![
            ([ids[0], ids[3]], 5),
            ([ids[2], ids[3]], 8),
        ]);
    }

    #[test]
    fn test_remove_add_vertex() {
        let mut mesh = ComboMesh2::<usize, usize, usize>::default();
        let ids = mesh.extend_vertices(vec![3, 6, 9, 2]);
        let edges = vec![
            ([ids[0], ids[3]], 5),
            ([ids[1], ids[3]], 3),
            ([ids[3], ids[1]], 2),
            ([ids[2], ids[3]], 8),
            ([ids[1], ids[2]], 9),
        ];
        mesh.extend_edges(edges.clone());

        mesh.remove_vertex(ids[1]);
        assert_vertices(&mesh, vec![(ids[0], 3), (ids[2], 9), (ids[3], 2)]);
        assert_edges(&mesh, vec![
            ([ids[0], ids[3]], 5),
            ([ids[2], ids[3]], 8),
        ]);

        let id2 = mesh.add_vertex(6);
        assert_vertices(&mesh, vec![(ids[0], 3), (ids[2], 9), (ids[3], 2), (id2, 6)]);
        assert_edges(&mesh, vec![
            ([ids[0], ids[3]], 5),
            ([ids[2], ids[3]], 8),
        ]);
    }

    #[test]
    fn test_remove_edge() {
        let mut mesh = ComboMesh2::<usize, usize, usize>::default();
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

    #[test]
    fn test_remove_add_edge() {
        let mut mesh = ComboMesh2::<usize, usize, usize>::default();
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

        mesh.add_edge([ids[1], ids[0]], 4);
        mesh.add_edge([ids[1], ids[3]], 6);
        assert_edges(&mesh, vec![
            ([ids[0], ids[3]], 5),
            ([ids[3], ids[1]], 2),
            ([ids[2], ids[3]], 8),
            ([ids[1], ids[2]], 9),
            ([ids[1], ids[0]], 4),
            ([ids[1], ids[3]], 6),
        ]);
    }

    #[test]
    fn test_clear_vertices() {
        let mut mesh = ComboMesh2::<usize, usize, usize>::default();
        let ids = mesh.extend_vertices(vec![3, 6, 9, 2]);
        let edges = vec![
            ([ids[0], ids[3]], 5),
            ([ids[1], ids[3]], 3),
            ([ids[3], ids[1]], 2),
            ([ids[2], ids[3]], 8),
            ([ids[1], ids[2]], 9),
        ];
        mesh.extend_edges(edges.clone());

        mesh.clear_vertices();
        assert_vertices(&mesh, vec![]);
        assert_edges(&mesh, vec![] as Vec<(EdgeId, _)>);
    }

    #[test]
    fn test_clear_edges() {
        let mut mesh = ComboMesh2::<usize, usize, usize>::default();
        let ids = mesh.extend_vertices(vec![3, 6, 9, 2]);
        let edges = vec![
            ([ids[0], ids[3]], 5),
            ([ids[1], ids[3]], 3),
            ([ids[3], ids[1]], 2),
            ([ids[2], ids[3]], 8),
            ([ids[1], ids[2]], 9),
        ];
        mesh.extend_edges(edges.clone());

        mesh.clear_edges();
        assert_vertices(&mesh, vec![(ids[0], 3), (ids[1], 6), (ids[2], 9), (ids[3], 2)]);
        assert_edges(&mesh, vec![] as Vec<(EdgeId, _)>);
    }

    #[test]
    fn test_walker() {
        let mut mesh = ComboMesh2::<usize, usize, usize>::default();
        let ids = mesh.extend_vertices(vec![3, 6, 9, 2, 5]);
        let edges = vec![
            ([ids[0], ids[3]], 5),
            ([ids[1], ids[3]], 3),
            ([ids[3], ids[1]], 2),
            ([ids[1], ids[2]], 9),
            ([ids[2], ids[3]], 8),
        ];
        mesh.extend_edges(edges.clone());

        assert!(mesh.edge_walker_from_vertex(ids[4]).is_none());

        let walker = mesh.edge_walker_from_edge([ids[3], ids[1]]);
        assert_eq!(walker.edge(), EdgeId([ids[3], ids[1]]));
        assert_eq!(walker.vertex(), ids[3]);
        assert_eq!(walker.target(), ids[1]);

        let walker = walker.next();
        assert_eq!(walker.edge(), EdgeId([ids[3], ids[1]]));

        let walker = walker.prev();
        assert_eq!(walker.edge(), EdgeId([ids[3], ids[1]]));

        let walker = walker.twin().unwrap();
        assert_eq!(walker.edge(), EdgeId([ids[1], ids[3]]));
        
        let walker = walker.twin().unwrap();
        assert_eq!(walker.edge(), EdgeId([ids[3], ids[1]]));

        let walker = walker.forward().unwrap();
        assert_eq!(walker.edge(), EdgeId([ids[1], ids[2]]));

        let walker = walker.next();
        assert_eq!(walker.edge(), EdgeId([ids[1], ids[3]]));

        let walker = walker.prev();
        assert_eq!(walker.edge(), EdgeId([ids[1], ids[2]]));

        let walker = walker.forward().unwrap();
        assert_eq!(walker.edge(), EdgeId([ids[2], ids[3]]));

        let walker = walker.backward().unwrap();
        assert_eq!(walker.edge(), EdgeId([ids[1], ids[2]]));

        assert!(walker.twin().is_none());

        let walker = mesh.edge_walker_from_edge([ids[0], ids[3]]);
        assert!(walker.backward().is_none());

        let walker = walker.next_in();
        assert_ne!(walker.vertex(), ids[0]);
        assert_eq!(walker.target(), ids[3]);

        let walker = walker.prev_in();
        assert_eq!(walker.edge(), EdgeId([ids[0], ids[3]]));
    }

    #[test]
    fn test_vertex_edges_out() {
        let mut mesh = ComboMesh2::<usize, usize, usize>::default();
        let ids = mesh.extend_vertices(vec![3, 6, 9, 2, 5]);
        let edges = vec![
            ([ids[0], ids[3]], 5),
            ([ids[1], ids[3]], 3),
            ([ids[3], ids[1]], 2),
            ([ids[1], ids[2]], 9),
            ([ids[2], ids[3]], 8),
        ];
        mesh.extend_edges(edges.clone());

        let set = mesh.vertex_edges_out(ids[4]).collect::<FnvHashSet<_>>();
        let expected = vec![].into_iter().collect::<FnvHashSet<_>>();
        assert_eq!(set, expected);

        let set = mesh.vertex_edges_out(ids[2]).collect::<FnvHashSet<_>>();
        let expected = vec![EdgeId([ids[2], ids[3]])].into_iter().collect::<FnvHashSet<_>>();
        assert_eq!(set, expected);

        let set = mesh.vertex_edges_out(ids[1]).collect::<FnvHashSet<_>>();
        let expected = vec![EdgeId([ids[1], ids[3]]), EdgeId([ids[1], ids[2]])].into_iter().collect::<FnvHashSet<_>>();
        assert_eq!(set, expected);
    }

    #[test]
    fn test_vertex_edges_in() {
        let mut mesh = ComboMesh2::<usize, usize, usize>::default();
        let ids = mesh.extend_vertices(vec![3, 6, 9, 2, 5]);
        let edges = vec![
            ([ids[0], ids[3]], 5),
            ([ids[1], ids[3]], 3),
            ([ids[3], ids[1]], 2),
            ([ids[1], ids[2]], 9),
        ];
        mesh.extend_edges(edges.clone());

        let set = mesh.vertex_edges_in(ids[4]).collect::<FnvHashSet<_>>();
        let expected = vec![].into_iter().collect::<FnvHashSet<_>>();
        assert_eq!(set, expected);

        let set = mesh.vertex_edges_in(ids[2]).collect::<FnvHashSet<_>>();
        let expected = vec![EdgeId([ids[1], ids[2]])].into_iter().collect::<FnvHashSet<_>>();
        assert_eq!(set, expected);

        let set = mesh.vertex_edges_in(ids[3]).collect::<FnvHashSet<_>>();
        let expected = vec![EdgeId([ids[0], ids[3]]), EdgeId([ids[1], ids[3]])].into_iter().collect::<FnvHashSet<_>>();
        assert_eq!(set, expected);
    }
}