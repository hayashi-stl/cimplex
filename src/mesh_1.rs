use idmap::OrderedIdMap;
use idmap::table::DenseEntryTable;
use typenum::{U2, U3};
use std::fmt::Debug;
use std::ops::{Index, IndexMut};
use std::iter::{Map, Filter};
use std::convert::{TryFrom, TryInto};
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

    /// Sets the source. Assumes it doesn't equal the target.
    fn source(mut self, vertex: VertexId) -> Self {
        self.0[1] = vertex;
        debug_assert_ne!(self.0[0], self.0[1]);
        self
    }

    /// Sets the target. Assumes it doesn't equal the source.
    fn target(mut self, vertex: VertexId) -> Self {
        self.0[1] = vertex;
        debug_assert_ne!(self.0[0], self.0[1]);
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
    /// Keep separate track because edge twins may or may not exist
    num_edges: usize,
}

impl<V, E> Default for ComboMesh1<V, E> {
    fn default() -> Self {
        ComboMesh1 {
            vertices: OrderedIdMap::default(),
            edges: FnvHashMap::default(),
            next_vertex_id: 0,
            num_edges: 0,
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
    pub fn vertex_edges_out(&self, vertex: VertexId) -> VertexEdgesOut<V, E> {
        if let Some(walker) = self.walker_from_vertex(vertex) {
            let start_target = walker.target();
            VertexEdgesOut {
                walker,
                start_target,
                finished: false,
            }
        } else {
            VertexEdgesOut {
                walker: Walker::dummy(self),
                start_target: VertexId(0),
                finished: true,
            }
        }
    }

    /// Iterates over the incoming edges of a vertex.
    pub fn vertex_edges_in(&self, vertex: VertexId) -> VertexEdgesIn<V, E> {
        if let Some(walker) = Walker::from_vertex_less_checked(self, vertex).and_then(|w| w.as_backward()) {
            let start_source = walker.vertex();
            VertexEdgesIn {
                walker,
                start_source,
                finished: false,
            }
        } else {
            VertexEdgesIn {
                walker: Walker::dummy(self),
                start_source: VertexId(0),
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
        let vertex = match self.vertices.get(id) {
            Some(vertex) => vertex,
            None => return (None, vec![]),
        };

        if vertex.target == id {
            (self.vertices.remove(id).map(|v| v.value), vec![])
        } else {
            // Get edges to remove
            let mut targets = vec![];
            let mut target = vertex.target;
            while targets.last() != Some(&vertex.target) {
                target = self.edges[&[id, target].try_into().ok().unwrap()].next_target;
                targets.push(target);
            }

            // Remove edges
            let edge_values = targets.into_iter().flat_map(|target|
                vec![self.remove_edge([id, target]), self.remove_edge([target, id])].into_iter().flatten())
                .collect();
            (self.vertices.remove(id).map(|v| v.value), edge_values)
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
    pub fn remove_edge<EI: TryInto<EdgeId>>(&mut self, id: EI) -> Option<E> {
        let id = match id.try_into() {
            Ok(id) => id,
            Err(_) => return None,
        };

        if !self.vertices.contains_key(id.0[0]) || !self.vertices.contains_key(id.0[1]) {
            return None;
        }

        match self.edges.get(&id.twin()).map(|e| e.value.as_ref()) {
            // Twin actually exists; just set value to None
            Some(Some(_)) => {
                let old = self.edges.get_mut(&id).unwrap().value.take();
                if old.is_some() {
                    self.num_edges -= 1;
                }
                old
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
                delete_edge(id)
            }

            // Twin isn't in map, and neither is the edge to remove
            None => None,
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
        self.edges.clear();
        self.num_edges = 0;
        // Fix vertex-target links
        for (id, vertex) in &mut self.vertices {
            vertex.target = *id;
        }
    }

    /// Gets a walker that starts at the given vertex.
    /// Returns None if the vertex has no outgoing edge.
    pub fn walker_from_vertex(&self, vertex: VertexId) -> Option<Walker<V, E>> {
        Walker::from_vertex(self, vertex)
    }

    /// Gets a walker that starts at the given edge.
    pub fn walker_from_edge<EI: TryInto<EdgeId>>(&self, edge: EI) -> Walker<V, E> {
        Walker::new(self, edge)
    }
}

/// A walker for navigating a simplicial 1-complex.
#[derive(Debug)]
pub struct Walker<'a, V, E> {
    mesh: &'a ComboMesh1<V, E>,
    edge: EdgeId,
}

impl<'a, V, E> Clone for Walker<'a, V, E> {
    fn clone(&self) -> Self {
        Self {
            mesh: self.mesh,
            edge: self.edge,
        }
    }
}

impl<'a, V, E> Copy for Walker<'a, V, E> {}

impl<'a, V, E> Walker<'a, V, E> {
    fn new<EI: TryInto<EdgeId>>(mesh: &'a ComboMesh1<V, E>, edge: EI) -> Self {
        Self {
            mesh,
            edge: edge.try_into().ok().unwrap(),
        }
    }

    /// Doesn't check that the starting edge actually exists
    fn from_vertex_less_checked(mesh: &'a ComboMesh1<V, E>, vertex: VertexId) -> Option<Self> {
        ([vertex, mesh.vertices[vertex].target].try_into().ok() as Option<EdgeId>)
            .map(|edge| Self::new(mesh, edge))
    }

    fn from_vertex(mesh: &'a ComboMesh1<V, E>, vertex: VertexId) -> Option<Self> {
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
    fn dummy(mesh: &'a ComboMesh1<V, E>) -> Self {
        Self {
            mesh,
            edge: EdgeId([VertexId(0), VertexId(1)])
        }
    }

    /// Get the mesh that the walker navigates
    pub fn mesh(&self) -> &ComboMesh1<V, E> {
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
    pub fn as_twin(mut self) -> Option<Self> {
        if let Some(twin) = self.twin() {
            self.edge = twin;
            Some(self)
        } else {
            None
        }
    }

    /// Returns the twin edge of the current edge, if it exists.
    pub fn twin(&self) -> Option<EdgeId> {
        let twin = self.edge.twin();
        self.mesh.edges[&twin].value.as_ref().map(|_| twin)
    }

    /// Sets the current edge to the next one with the same source vertex.
    pub fn as_next(mut self) -> Self {
        self.edge = self.next();
        self
    }

    /// Returns the next edge with the same source vertex.
    pub fn next(&self) -> EdgeId {
        let mut edge = self.edge;
        while {
            edge = edge.target(self.mesh.edges[&edge].next_target);
            self.mesh.edges[&edge].value.is_none()
        } {}
        edge
    }

    /// Sets the current edge to the previous one with the same source vertex.
    pub fn as_prev(mut self) -> Self {
        self.edge = self.prev();
        self
    }

    /// Returns the previous edge with the same source vertex.
    pub fn prev(&self) -> EdgeId {
        let mut edge = self.edge;
        while {
            edge = edge.target(self.mesh.edges[&edge].prev_target);
            self.mesh.edges[&edge].value.is_none()
        } {}
        edge
    }

    /// Sets the current edge to the next one with the same target vertex.
    pub fn as_next_in(mut self) -> Self {
        self.edge = self.next_in();
        self
    }

    /// Returns the next edge with the same target vertex.
    pub fn next_in(&self) -> EdgeId {
        let mut edge = self.edge;
        while {
            edge = edge.twin().target(self.mesh.edges[&edge.twin()].next_target).twin();
            self.mesh.edges[&edge].value.is_none()
        } {}
        edge
    }

    /// Sets the current edge to the previous one with the same target vertex.
    pub fn as_prev_in(mut self) -> Self {
        self.edge = self.prev_in();
        self
    }

    /// Returns the previous edge with the same target vertex.
    pub fn prev_in(&self) -> EdgeId {
        let mut edge = self.edge;
        while {
            edge = edge.twin().target(self.mesh.edges[&edge.twin()].prev_target).twin();
            self.mesh.edges[&edge].value.is_none()
        } {}
        edge
    }

    /// Sets the current edge to an edge whose source vertex is the current edge's target vertex.
    /// Chooses a non-twin edge if possible.
    pub fn as_forward(mut self) -> Option<Self> {
        if let Some(forward) = self.forward() {
            self.edge = forward;
            Some(self)
        } else {
            None
        }
    }

    /// Returns an edge whose source vertex is the current edge's target vertex.
    /// Chooses a non-twin edge if possible.
    pub fn forward(&self) -> Option<EdgeId> {
        let twin = self.edge.twin();
        let mut edge = twin;
        let mut found = true;
        while {
            edge = edge.target(self.mesh.edges[&edge].next_target);
            if self.mesh.edges[&edge].value.is_some() {
                false
            } else if edge == twin {
                found = false;
                false
            } else {
                true
            }
        } {}
        if found { Some(edge) } else { None }
    }

    /// Sets the current edge to an edge whose target vertex is the current edge's source vertex.
    /// Chooses a non-twin edge if possible.
    pub fn as_backward(mut self) -> Option<Self> {
        if let Some(backward) = self.backward() {
            self.edge = backward;
            Some(self)
        } else {
            None
        }
    }

    /// Returns an edge whose target vertex is the current edge's source vertex.
    /// Chooses a non-twin edge if possible.
    pub fn backward(&self) -> Option<EdgeId> {
        let twin = self.edge.twin();
        let mut edge = twin;
        let mut found = true;
        while {
            edge = edge.twin().target(self.mesh.edges[&edge.twin()].next_target).twin();
            if self.mesh.edges[&edge].value.is_some() {
                false
            } else if edge == twin {
                found = false;
                false
            } else {
                true
            }
        } {}
        if found { Some(edge) } else { None }
    }
}

/// An iterator over the outgoing edges of a vertex.
#[derive(Clone, Debug)]
pub struct VertexEdgesOut<'a, V, E> {
    walker: Walker<'a, V, E>,
    start_target: VertexId,
    finished: bool,
}

impl<'a, V, E> Iterator for VertexEdgesOut<'a, V, E> {
    type Item = EdgeId;
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        let edge = self.walker.edge();
        self.walker = self.walker.as_next();
        if self.walker.edge().0[1] == self.start_target {
            self.finished = true;
        }
        Some(edge)
    }
}

/// An iterator over the incoming edges of a vertex.
#[derive(Clone, Debug)]
pub struct VertexEdgesIn<'a, V, E> {
    walker: Walker<'a, V, E>,
    start_source: VertexId,
    finished: bool,
}

impl<'a, V, E> Iterator for VertexEdgesIn<'a, V, E> {
    type Item = EdgeId;
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        let edge = self.walker.edge();
        self.walker = self.walker.as_next_in();
        if self.walker.edge().0[0] == self.start_source {
            self.finished = true;
        }
        Some(edge)
    }
}

impl<V, E> Index<VertexId> for ComboMesh1<V, E> {
    type Output = V;

    fn index(&self, index: VertexId) -> &Self::Output {
        self.vertex(index).unwrap()
    }
}

impl<V, E> IndexMut<VertexId> for ComboMesh1<V, E> {
    fn index_mut(&mut self, index: VertexId) -> &mut Self::Output {
        self.vertex_mut(index).unwrap()
    }
}

impl<V, E> Index<[VertexId; 2]> for ComboMesh1<V, E> {
    type Output = E;

    fn index(&self, index: [VertexId; 2]) -> &Self::Output {
        self.edge(index).unwrap()
    }
}

impl<V, E> IndexMut<[VertexId; 2]> for ComboMesh1<V, E> {
    fn index_mut(&mut self, index: [VertexId; 2]) -> &mut Self::Output {
        self.edge_mut(index).unwrap()
    }
}

impl<V, E> Index<EdgeId> for ComboMesh1<V, E> {
    type Output = E;

    fn index(&self, index: EdgeId) -> &Self::Output {
        self.edge(index).unwrap()
    }
}

impl<V, E> IndexMut<EdgeId> for ComboMesh1<V, E> {
    fn index_mut(&mut self, index: EdgeId) -> &mut Self::Output {
        self.edge_mut(index).unwrap()
    }
}

/// A position-containing edge mesh
pub type Mesh1<V, E, D> = ComboMesh1<(VecN<D>, V), E>;

/// A 2D-position-containing edge mesh
pub type Mesh12<V, E> = Mesh1<V, E, U2>;

/// A 3D-position-containing edge mesh
pub type Mesh13<V, E> = Mesh1<V, E, U3>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt::Debug;
    use std::hash::Hash;

    #[track_caller]
    fn assert_vertices<V: Clone + Debug + Eq + Hash, E, I: IntoIterator<Item = (VertexId, V)>>(mesh: &ComboMesh1<V, E>, vertices: I) {
        let result = mesh.vertices().map(|(id, v)| (*id, v.clone()))
            .collect::<FnvHashSet<_>>();
        let expect = vertices.into_iter().collect::<FnvHashSet<_>>();

        assert_eq!(result, expect);
    }

    #[track_caller]
    fn assert_edges<V, E: Clone + Debug + Eq + Hash, EI: TryInto<EdgeId>, I: IntoIterator<Item = (EI, E)>>(mesh: &ComboMesh1<V, E>, edges: I) {
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
        let mesh = ComboMesh1::<(), ()>::default();
        assert!(mesh.vertices.is_empty());
        assert!(mesh.edges.is_empty());
        assert_eq!(mesh.num_edges(), 0);
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
        let mut mesh = ComboMesh1::<usize, usize>::default();
        let ids = mesh.extend_vertices(vec![3, 6, 9, 2]);
        mesh.add_edge([ids[1], ids[1]], 4);
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
        assert_eq!(mesh.num_edges(), 5);
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

        mesh.remove_vertex(ids[4]); // nonexistent vertex
        assert_vertices(&mesh, vec![(ids[0], 3), (ids[2], 9), (ids[3], 2)]);
        assert_edges(&mesh, vec![
            ([ids[0], ids[3]], 5),
            ([ids[2], ids[3]], 8),
        ]);
    }

    #[test]
    fn test_remove_add_vertex() {
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

    #[test]
    fn test_remove_add_edge() {
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

        mesh.clear_vertices();
        assert_vertices(&mesh, vec![]);
        assert_edges(&mesh, vec![] as Vec<(EdgeId, _)>);
    }

    #[test]
    fn test_clear_edges() {
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

        mesh.clear_edges();
        assert_vertices(&mesh, vec![(ids[0], 3), (ids[1], 6), (ids[2], 9), (ids[3], 2)]);
        assert_edges(&mesh, vec![] as Vec<(EdgeId, _)>);
    }

    #[test]
    fn test_walker() {
        let mut mesh = ComboMesh1::<usize, usize>::default();
        let ids = mesh.extend_vertices(vec![3, 6, 9, 2, 5]);
        let edges = vec![
            ([ids[0], ids[3]], 5),
            ([ids[1], ids[3]], 3),
            ([ids[3], ids[1]], 2),
            ([ids[1], ids[2]], 9),
            ([ids[2], ids[3]], 8),
        ];
        mesh.extend_edges(edges.clone());

        assert!(mesh.walker_from_vertex(ids[4]).is_none());

        let walker = mesh.walker_from_edge([ids[3], ids[1]]);
        assert_eq!(walker.edge(), EdgeId([ids[3], ids[1]]));
        assert_eq!(walker.vertex(), ids[3]);
        assert_eq!(walker.target(), ids[1]);

        let walker = walker.as_next();
        assert_eq!(walker.edge(), EdgeId([ids[3], ids[1]]));

        let walker = walker.as_prev();
        assert_eq!(walker.edge(), EdgeId([ids[3], ids[1]]));

        let walker = walker.as_twin().unwrap();
        assert_eq!(walker.edge(), EdgeId([ids[1], ids[3]]));
        
        let walker = walker.as_twin().unwrap();
        assert_eq!(walker.edge(), EdgeId([ids[3], ids[1]]));

        let walker = walker.as_forward().unwrap();
        assert_eq!(walker.edge(), EdgeId([ids[1], ids[2]]));

        let walker = walker.as_next();
        assert_eq!(walker.edge(), EdgeId([ids[1], ids[3]]));

        let walker = walker.as_prev();
        assert_eq!(walker.edge(), EdgeId([ids[1], ids[2]]));

        let walker = walker.as_forward().unwrap();
        assert_eq!(walker.edge(), EdgeId([ids[2], ids[3]]));

        let walker = walker.as_backward().unwrap();
        assert_eq!(walker.edge(), EdgeId([ids[1], ids[2]]));

        assert!(walker.as_twin().is_none());

        let walker = mesh.walker_from_edge([ids[0], ids[3]]);
        assert!(walker.as_backward().is_none());

        let walker = walker.as_next_in();
        assert_ne!(walker.vertex(), ids[0]);
        assert_eq!(walker.target(), ids[3]);

        let walker = walker.as_prev_in();
        assert_eq!(walker.edge(), EdgeId([ids[0], ids[3]]));
    }

    #[test]
    fn test_vertex_edges_out() {
        let mut mesh = ComboMesh1::<usize, usize>::default();
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
        let mut mesh = ComboMesh1::<usize, usize>::default();
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