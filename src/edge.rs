//! Traits and structs related to edges

use std::iter::{Map, Filter};
use std::convert::{TryFrom, TryInto};
use std::collections::hash_map;

use crate::vertex::{VertexId, internal::HigherVertex};

use internal::{Edge, Link, RemoveEdgeHigher, ClearEdgesHigher};

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

type EdgeFilterFn<'a, ET> = for<'b> fn(&'b (&'a EdgeId, &'a ET)) -> bool;
type EdgeMapFn<'a, ET> = fn((&'a EdgeId, &'a ET)) -> (&'a EdgeId, &'a <ET as Edge>::E);
pub type Edges<'a, ET> = Map<Filter<hash_map::Iter<'a, EdgeId, ET>, EdgeFilterFn<'a, ET>>, EdgeMapFn<'a, ET>>;
type EdgeFilterFnMut<'a, ET> = for<'b> fn(&'b (&'a EdgeId, &'a mut ET)) -> bool;
type EdgeMapFnMut<'a, ET> = fn((&'a EdgeId, &'a mut ET)) -> (&'a EdgeId, &'a mut <ET as Edge>::E);
pub type EdgesMut<'a, ET> = Map<Filter<hash_map::IterMut<'a, EdgeId, ET>, EdgeFilterFnMut<'a, ET>>, EdgeMapFnMut<'a, ET>>;

macro_rules! E {
    () => { <Self::Edge as Edge>::E };
}

/// For simplicial complexes that can have edges
pub trait HasEdges: internal::HasEdges + RemoveEdgeHigher + ClearEdgesHigher
where
    Self::Vertex: HigherVertex
{
    /// Gets the number of edges.
    fn num_edges(&self) -> usize {
        self.num_edges_raw()
    }

    /// Iterates over the edges of this mesh.
    /// Gives (id, value) pairs
    fn edges(&self) -> Edges<Self::Edge> {
        self.edges_raw().iter()
            .filter::<EdgeFilterFn<Self::Edge>>(|(_, e)| e.value().is_some())
            .map::<_, EdgeMapFn<Self::Edge>>(|(id, e)| (id, e.value().as_ref().unwrap()))
    }

    /// Iterates mutably over the edges of this mesh.
    /// Gives (id, value) pairs
    fn edges_mut(&mut self) -> EdgesMut<Self::Edge> {
        self.edges_raw_mut().iter_mut()
            .filter::<EdgeFilterFnMut<Self::Edge>>(|(_, e)| e.value().is_some())
            .map::<_, EdgeMapFnMut<Self::Edge>>(|(id, e)| (id, e.value_mut().as_mut().unwrap()))
    }

    /// Gets the value of the edge at a specific id.
    /// Returns None if not found.
    fn edge<EI: TryInto<EdgeId>>(&self, id: EI) -> Option<&E!()> {
        id.try_into().ok().and_then(|id| self.edges_raw().get(&id)).and_then(|e| e.value().as_ref())
    }

    /// Gets the value of the edge at a specific id mutably.
    /// Returns None if not found.
    fn edge_mut<EI: TryInto<EdgeId>>(&mut self, id: EI) -> Option<&mut E!()> {
        id.try_into().ok().and_then(move |id| self.edges_raw_mut().get_mut(&id)).and_then(|e| e.value_mut().as_mut())
    }
    
    /// Iterates over the outgoing edges of a vertex.
    /// The vertex must exist.
    fn vertex_edges_out(&self, vertex: VertexId) -> Box<dyn Iterator<Item = EdgeId>> {
        todo!()
    }

    /// Iterates over the incoming edges of a vertex.
    /// The vertex must exist.
    fn vertex_edges_in(&self, vertex: VertexId) -> Box<dyn Iterator<Item = EdgeId>> {
        todo!()
    }

    /// Adds an edge to the mesh. Vertex order is important!
    /// If the edge was already there, this replaces the value.
    /// Returns the previous value of the edge, if there was one.
    ///
    /// # Panics
    /// Panics if either vertex doesn't exist or if the vertices are the same
    fn add_edge<EI: TryInto<EdgeId>>(&mut self, vertices: EI, value: E!()) -> Option<E!()> {
        let id = vertices.try_into().ok().unwrap();

        // Can't use entry().or_insert() because that would cause a
        // mutable borrow and an immutable borrow to exist at the same time
        if let Some(edge) = self.edges_raw_mut().get_mut(&id) {
            let old = edge.value_mut().take();
            *edge.value_mut() = Some(value);
            if old.is_none() {
                *self.num_edges_raw_mut() += 1;
            }
            old
        } else {
            *self.num_edges_raw_mut() += 1;

            let mut insert_edge = |id: EdgeId, value: Option<E!()>| {
                let target = self.vertices_raw()[id.0[0]].target();

                let (prev, next) = if target == id.0[0] { // First edge from vertex
                    *self.vertices_raw_mut()[id.0[0]].target_mut() = id.0[1];
                    (id.0[1], id.0[1])
                } else {
                    let prev = self.edges_raw()[&[id.0[0], target].try_into().ok().unwrap()].targets().prev;
                    let next = target;
                    self.edges_raw_mut().get_mut(&id.target(prev)).unwrap().targets_mut().next = id.0[1];
                    self.edges_raw_mut().get_mut(&id.target(next)).unwrap().targets_mut().prev = id.0[1];
                    (prev, next)
                };

                self.edges_raw_mut().insert(id, Edge::new(id.0[0], Link::new(prev, next), value));
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
    fn extend_edges<EI: TryInto<EdgeId>, I: IntoIterator<Item = (EI, E!())>>(&mut self, iter: I) {
        iter.into_iter().for_each(|(id, value)| { self.add_edge(id, value); })
    }

    /// Removes an edge from the mesh and returns the value that was there,
    /// or None if there was nothing there
    fn remove_edge<EI: TryInto<EdgeId>>(&mut self, id: EI) -> Option<E!()> {
        let id = match id.try_into() {
            Ok(id) => id,
            Err(_) => return None,
        };

        if !self.vertices_raw().contains_key(id.0[0]) || !self.vertices_raw().contains_key(id.0[1]) {
            return None;
        }

        self.remove_edge_higher(id);

        match self.edges_raw().get(&id.twin()).map(|e| e.value().as_ref()) {
            // Twin actually exists; just set value to None
            Some(Some(_)) => {
                let old = self.edges_raw_mut().get_mut(&id).unwrap().value_mut().take();
                if old.is_some() {
                    *self.num_edges_raw_mut() -= 1;
                }
                old
            }

            // Twin is phantom, so remove both edge and twin from map
            Some(None) => {
                // Twin is phantom, so this edge actually exists.
                *self.num_edges_raw_mut() -= 1;

                let mut delete_edge = |id: EdgeId| {
                    let edge = &self.edges_raw()[&id];
                    let prev = edge.targets().prev;
                    let next = edge.targets().next;
                    self.edges_raw_mut().get_mut(&id.target(prev)).unwrap().targets_mut().next = next;
                    self.edges_raw_mut().get_mut(&id.target(next)).unwrap().targets_mut().prev = prev;

                    let source = &mut self.vertices_raw_mut()[id.0[0]];
                    if id.0[1] == next { // this was the last edge from the vertex
                        *source.target_mut() = id.0[0];
                    } else if id.0[1] == source.target() {
                        *source.target_mut() = next;
                    }

                    self.edges_raw_mut().remove(&id).and_then(|e| e.to_value())
                };
                
                delete_edge(id.twin());
                delete_edge(id)
            }

            // Twin isn't in map, and neither is the edge to remove
            None => None,
        }
    }

    /// Removes a list of edges.
    fn remove_edges<EI: TryInto<EdgeId>, I: IntoIterator<Item = EI>>(&mut self, iter: I) {
        iter.into_iter().for_each(|id| { self.remove_edge(id); })
    }

    /// Keeps only the edges that satisfy a predicate
    fn retain_edges<P: FnMut(EdgeId, &E!()) -> bool>(&mut self, mut predicate: P) {
        let to_remove = self.edges()
            .filter(|(id, e)| !predicate(**id, *e))
            .map(|(id, _)| *id)
            .collect::<Vec<_>>();
        self.remove_edges(to_remove);
    }

    /// Removes all edges from the mesh.
    fn clear_edges(&mut self) {
        self.clear_edges_higher();
        self.edges_raw_mut().clear();
        *self.num_edges_raw_mut() = 0;

        // Fix vertex-target links
        for (id, vertex) in self.vertices_raw_mut() {
            *vertex.target_mut() = *id;
        }
    }
}

mod internal {
    use super::EdgeId;
    use crate::vertex::VertexId;
    use crate::vertex::internal::{HasVertices as HasVerticesIntr, HigherVertex};
    use fnv::FnvHashMap;

    pub struct Link<T> {
        pub prev: T,
        pub next: T,
    }

    impl<T> Link<T> {
        pub fn new(prev: T, next: T) -> Self {
            Link { prev, next }
        }
    }

    pub trait Edge {
        type E;

        /// Takes the vertex id of the source, in case
        /// the edge needs to store a dummy value for the opposite vertex
        /// of the triangle.
        fn new(id: VertexId, link: Link<VertexId>, value: Option<Self::E>) -> Self;

        fn targets(&self) -> &Link<VertexId>;

        fn targets_mut(&mut self) -> &mut Link<VertexId>;

        fn to_value(self) -> Option<Self::E>;

        fn value(&self) -> &Option<Self::E>;

        fn value_mut(&mut self) -> &mut Option<Self::E>;
    }

    pub trait HasEdges: HasVerticesIntr
    where
        Self::Vertex: HigherVertex
    {
        type Edge: Edge;

        fn edges_raw(&self) -> &FnvHashMap<EdgeId, Self::Edge>;

        fn edges_raw_mut(&mut self) -> &mut FnvHashMap<EdgeId, Self::Edge>;

        fn num_edges_raw(&self) -> usize;

        fn num_edges_raw_mut(&mut self) -> &mut usize;
    }

    /// Removes higher-order simplexes that contain some edge
    pub trait RemoveEdgeHigher: HasEdges
    where
        Self::Vertex: HigherVertex
    {
        fn remove_edge_higher(&mut self, edge: EdgeId);
    }

    /// Clears higher-order simplexes
    pub trait ClearEdgesHigher: HasEdges
    where
        Self::Vertex: HigherVertex
    {
        fn clear_edges_higher(&mut self);
    }
}