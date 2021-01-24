//! Traits and structs related to edges

use nalgebra::allocator::Allocator;
use nalgebra::DefaultAllocator;
#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};
use std::collections::hash_map;
use std::convert::{TryFrom, TryInto};
use std::iter::Map;
use typenum::Bit;

use crate::{
    iter::{IteratorExt, MapWith},
    tri::{HasTris, TriWalker},
    vertex::HasPosition,
};
//use crate::tri::{HasTris, TriWalker};
use crate::vertex::internal::HasVertices as HasVerticesIntr;
use crate::vertex::{
    internal::{HigherVertex, Vertex},
    HasVertices, VertexId,
};
use crate::vertex::{HasPositionDim, HasPositionPoint, Position};

use internal::HasEdges as HasEdgesIntr;
use internal::{ClearEdgesHigher, Edge, HigherEdge, Link, RemoveEdgeHigher};

/// An edge id is just the edge's vertices in order.
/// The vertices are not allowed to be the same.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct EdgeId(pub(crate) [VertexId; 2]);

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

    pub(crate) fn dummy() -> Self {
        Self([VertexId::dummy(); 2])
    }

    /// Reverses the vertices of this edge id to get the one for the twin edge
    pub(crate) fn twin(self) -> Self {
        Self([self.0[1], self.0[0]])
    }

    /// Sets the source. Assumes it doesn't equal the target.
    fn with_source(mut self, vertex: VertexId) -> Self {
        self.0[0] = vertex;
        debug_assert_ne!(self.0[0], self.0[1]);
        self
    }

    /// Sets the target. Assumes it doesn't equal the source.
    fn with_target(mut self, vertex: VertexId) -> Self {
        self.0[1] = vertex;
        debug_assert_ne!(self.0[0], self.0[1]);
        self
    }
}

/// Iterator over the edges of a mesh.
pub type Edges<'a, ET> = Map<
    hash_map::Iter<'a, EdgeId, ET>,
    for<'b> fn((&'b EdgeId, &'b ET)) -> (&'b EdgeId, &'b <ET as Edge>::E),
>;
/// Iterator over the edges of a mesh mutably.
pub type EdgesMut<'a, ET> = Map<
    hash_map::IterMut<'a, EdgeId, ET>,
    for<'b> fn((&'b EdgeId, &'b mut ET)) -> (&'b EdgeId, &'b mut <ET as Edge>::E),
>;

/// Iterator over the edges pointing out from a vertex.
pub type VertexEdgesOut<'a, M> =
    MapWith<VertexId, EdgeId, VertexTargets<'a, M>, fn(VertexId, VertexId) -> EdgeId>;
/// Iterator over the edges pointing in to a vertex.
pub type VertexEdgesIn<'a, M> =
    MapWith<VertexId, EdgeId, VertexSources<'a, M>, fn(VertexId, VertexId) -> EdgeId>;

macro_rules! E {
    () => {
        <Self::Edge as Edge>::E
    };
}

/// For simplicial complexes that can have edges
pub trait HasEdges: internal::HasEdges + HasVertices + RemoveEdgeHigher + ClearEdgesHigher
where
    Self::Vertex: HigherVertex,
{
    /// Gets the number of edges.
    fn num_edges(&self) -> usize {
        self.edges_r().len()
    }

    /// Iterates over the edges of this mesh.
    /// Gives (id, value) pairs
    fn edges(&self) -> Edges<Self::Edge> {
        self.edges_r().iter().map(|(id, e)| (id, e.value()))
    }

    /// Iterates mutably over the edges of this mesh.
    /// Gives (id, value) pairs
    fn edges_mut(&mut self) -> EdgesMut<Self::Edge> {
        self.edges_r_mut()
            .iter_mut()
            .map(|(id, e)| (id, e.value_mut()))
    }

    /// Gets the value of the edge at a specific id.
    /// Returns None if not found.
    fn edge<EI: TryInto<EdgeId>>(&self, id: EI) -> Option<&E!()> {
        id.try_into()
            .ok()
            .and_then(|id| self.edges_r().get(&id))
            .map(|e| e.value())
    }

    /// Gets the value of the edge at a specific id mutably.
    /// Returns None if not found.
    fn edge_mut<EI: TryInto<EdgeId>>(&mut self, id: EI) -> Option<&mut E!()> {
        id.try_into()
            .ok()
            .and_then(move |id| self.edges_r_mut().get_mut(&id))
            .map(|e| e.value_mut())
    }

    /// Iterates over the targets of the outgoing edges of a vertex.
    /// The vertex must exist.
    fn vertex_targets(&self, vertex: VertexId) -> VertexTargets<Self> {
        if let Some(walker) = self.edge_walker_from_vertex(vertex) {
            let start_target = walker.second();
            VertexTargets {
                walker,
                start_target,
                finished: false,
            }
        } else {
            VertexTargets {
                walker: EdgeWalker::dummy(self),
                start_target: VertexId::dummy(),
                finished: true,
            }
        }
    }

    /// Gets the target of the ≤1 outgoing edge that the vertex is a source of.
    /// The vertex must exist.
    fn vertex_target(&self, vertex: VertexId) -> Option<VertexId>
    where
        Self::Edge: Edge<Manifold = typenum::B1>
    {
        let target = self.vertices_r()[vertex].target();
        if target != vertex {
            Some(target)
        } else {
            None
        }
    }

    /// Iterates over the outgoing edges of a vertex.
    /// The vertex must exist.
    fn vertex_edges_out(&self, vertex: VertexId) -> VertexEdgesOut<Self> {
        self.vertex_targets(vertex)
            .map_with(vertex, |s, t| EdgeId([s, t]))
    }
    
    /// Gets the ≤1 outgoing edge that the vertex is a source of.
    /// The vertex must exist.
    fn vertex_edge_out(&self, vertex: VertexId) -> Option<EdgeId>
    where
        Self::Edge: Edge<Manifold = typenum::B1>
    {
        Some(EdgeId([vertex, self.vertex_target(vertex)?]))
    }

    /// Iterates over the sources of the incoming edges of a vertex.
    /// The vertex must exist.
    fn vertex_sources(&self, vertex: VertexId) -> VertexSources<Self> {
        if let Some(walker) = EdgeWalker::from_target(self, vertex) {
            let start_source = walker.first();
            VertexSources {
                walker,
                start_source,
                finished: false,
            }
        } else {
            VertexSources {
                walker: EdgeWalker::dummy(self),
                start_source: VertexId::dummy(),
                finished: true,
            }
        }
    }

    /// Gets the source of the ≤1 incoming edge that the vertex is a target of.
    /// The vertex must exist.
    fn vertex_source(&self, vertex: VertexId) -> Option<VertexId>
    where
        Self::Edge: Edge<Manifold = typenum::B1>
    {
        let source = self.vertices_r()[vertex].source();
        if source != vertex {
            Some(source)
        } else {
            None
        }
    }

    /// Iterates over the incoming edges of a vertex.
    /// The vertex must exist.
    fn vertex_edges_in(&self, vertex: VertexId) -> VertexEdgesIn<Self> {
        self.vertex_sources(vertex)
            .map_with(vertex, |t, s| EdgeId([s, t]))
    }
    
    /// Gets the ≤1 incoming edge that the vertex is a source of.
    /// The vertex must exist.
    fn vertex_edge_in(&self, vertex: VertexId) -> Option<EdgeId>
    where
        Self::Edge: Edge<Manifold = typenum::B1>
    {
        Some(EdgeId([self.vertex_source(vertex)?, vertex]))
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
        if let Some(edge) = self.edges_r_mut().get_mut(&id) {
            Some(std::mem::replace(edge.value_mut(), value))
        } else {
            let target = self.vertices_r()[id.0[0]].target();

            let (prev_target, next_target) =
                if target == id.0[0] || <<Self::Edge as Edge>::Manifold as Bit>::BOOL {
                    if target != id.0[0] {
                        self.remove_edge(EdgeId([id.0[0], target]));
                    }
                    // First edge from vertex
                    *self.vertices_r_mut()[id.0[0]].target_mut() = id.0[1];
                    (id.0[1], id.0[1])
                } else {
                    let prev =
                        self.edges_r()[&[id.0[0], target].try_into().ok().unwrap()].links()[0].prev;
                    let next = target;
                    self.edges_r_mut()
                        .get_mut(&id.with_target(prev))
                        .unwrap()
                        .links_mut()[0]
                        .next = id.0[1];
                    self.edges_r_mut()
                        .get_mut(&id.with_target(next))
                        .unwrap()
                        .links_mut()[0]
                        .prev = id.0[1];
                    (prev, next)
                };

            let source = self.vertices_r()[id.0[1]].source();

            let (prev_source, next_source) =
                if source == id.0[1] || <<Self::Edge as Edge>::Manifold as Bit>::BOOL {
                    if source != id.0[1] {
                        self.remove_edge([source, id.0[1]]);
                    }
                    // First edge to vertex
                    *self.vertices_r_mut()[id.0[1]].source_mut() = id.0[0];
                    (id.0[0], id.0[0])
                } else {
                    let prev =
                        self.edges_r()[&[source, id.0[1]].try_into().ok().unwrap()].links()[1].prev;
                    let next = source;
                    self.edges_r_mut()
                        .get_mut(&id.with_source(prev))
                        .unwrap()
                        .links_mut()[1]
                        .next = id.0[0];
                    self.edges_r_mut()
                        .get_mut(&id.with_source(next))
                        .unwrap()
                        .links_mut()[1]
                        .prev = id.0[0];
                    (prev, next)
                };

            self.edges_r_mut().insert(
                id,
                Edge::new(
                    id.0[0],
                    [
                        Link::new(prev_target, next_target),
                        Link::new(prev_source, next_source),
                    ],
                    value,
                ),
            );

            None
        }
    }

    /// Extends the edge list with an iterator.
    ///
    /// # Panics
    /// Panics if either vertex doesn't exist or if the vertices are the same
    /// in any of the edges.
    fn extend_edges<EI: TryInto<EdgeId>, I: IntoIterator<Item = (EI, E!())>>(&mut self, iter: I) {
        iter.into_iter().for_each(|(id, value)| {
            self.add_edge(id, value);
        })
    }

    /// Removes an edge from the mesh and returns the value that was there,
    /// or None if there was nothing there
    fn remove_edge<EI: TryInto<EdgeId>>(&mut self, id: EI) -> Option<E!()> {
        let id = id.try_into().ok()?;

        if self.edge(id).is_some() {
            self.remove_edge_higher(id);

            let (next_source, next_target) = if <<Self::Edge as Edge>::Manifold as Bit>::BOOL {
                (id.0[0], id.0[1])
            } else {
                let edge = &self.edges_r()[&id];
                let prev = edge.links()[0].prev;
                let next = edge.links()[0].next;
                self.edges_r_mut()
                    .get_mut(&id.with_target(prev))
                    .unwrap()
                    .links_mut()[0]
                    .next = next;
                self.edges_r_mut()
                    .get_mut(&id.with_target(next))
                    .unwrap()
                    .links_mut()[0]
                    .prev = prev;
                let next_target = next;

                let edge = &self.edges_r()[&id];
                let prev = edge.links()[1].prev;
                let next = edge.links()[1].next;
                self.edges_r_mut()
                    .get_mut(&id.with_source(prev))
                    .unwrap()
                    .links_mut()[1]
                    .next = next;
                self.edges_r_mut()
                    .get_mut(&id.with_source(next))
                    .unwrap()
                    .links_mut()[1]
                    .prev = prev;

                (next, next_target)
            };

            let source = &mut self.vertices_r_mut()[id.0[0]];
            if id.0[1] == next_target {
                // this was the last edge from the vertex
                *source.target_mut() = id.0[0];
            } else if id.0[1] == source.target() {
                *source.target_mut() = next_target;
            }

            let target = &mut self.vertices_r_mut()[id.0[1]];
            if id.0[0] == next_source {
                // this was the last edge from the vertex
                *target.source_mut() = id.0[1];
            } else if id.0[0] == target.source() {
                *target.source_mut() = next_source;
            }

            self.edges_r_mut().remove(&id).map(|e| e.to_value())
        } else {
            None
        }
    }

    /// Removes a list of edges.
    fn remove_edges<EI: TryInto<EdgeId>, I: IntoIterator<Item = EI>>(&mut self, iter: I) {
        iter.into_iter().for_each(|id| {
            self.remove_edge(id);
        })
    }

    /// Keeps only the edges that satisfy a predicate
    fn retain_edges<P: FnMut(EdgeId, &E!()) -> bool>(&mut self, mut predicate: P) {
        let to_remove = self
            .edges()
            .filter(|(id, e)| !predicate(**id, *e))
            .map(|(id, _)| *id)
            .collect::<Vec<_>>();
        self.remove_edges(to_remove);
    }

    /// Removes all edges from the mesh.
    fn clear_edges(&mut self) {
        self.clear_edges_higher();
        self.edges_r_mut().clear();

        // Fix vertex-target links
        for (id, vertex) in self.vertices_r_mut() {
            *vertex.target_mut() = *id;
            *vertex.source_mut() = *id;
        }
    }

    /// Gets a walker that starts at the given vertex.
    /// Returns None if the vertex has no outgoing edge.
    fn edge_walker_from_vertex(&self, vertex: VertexId) -> Option<EdgeWalker<Self>> {
        EdgeWalker::from_vertex(self, vertex)
    }

    /// Gets a walker that starts at the given edge.
    /// The edge must actually exist.
    fn edge_walker_from_edge<EI: TryInto<EdgeId>>(&self, edge: EI) -> EdgeWalker<Self> {
        EdgeWalker::new(self, edge)
    }
}

/// A walker for navigating a simplicial complex by edge.
///
/// Anatomy of the edge walker:
/// ```notrust
///                  edge
///  first @ ─────────────────────> @ second
/// ```
///
/// Movement of the edge walker:
/// ```notrust
///                            twin
///         prev          <───────────────        next_in
///  @ <─────────────── @ ───────────────> @ <────────────── @
///    ───────────────> │r  · (edge)· · · /^ ──────────────>
///       backward      │ \ · · · · · ·  / │     forward
///                     │  \  · · · · · /  │        
///                next │   \tri_walker/   │ prev_in
///                     │    \  · · · /    │      
///                     │     \ · ·  /     │       
///                     │      \  · /      │
///                     v       \  /       │
///                     @        \L        @
///                              @
/// ```
/// The edges that `next` and `prev` reference is in no particular order.
/// The same is true for `next_in` and `prev_in`.
#[derive(Debug)]
pub struct EdgeWalker<'a, M: ?Sized>
where
    M: HasVertices,
    <M as HasVerticesIntr>::Vertex: HigherVertex,
    M: HasEdges,
{
    mesh: &'a M,
    edge: EdgeId,
}

impl<'a, M: ?Sized> Clone for EdgeWalker<'a, M>
where
    M: HasVertices,
    <M as HasVerticesIntr>::Vertex: HigherVertex,
    M: HasEdges,
{
    fn clone(&self) -> Self {
        Self {
            mesh: self.mesh,
            edge: self.edge,
        }
    }
}

impl<'a, M: ?Sized> Copy for EdgeWalker<'a, M>
where
    M: HasVertices,
    <M as HasVerticesIntr>::Vertex: HigherVertex,
    M: HasEdges,
{
}

impl<'a, M: ?Sized> EdgeWalker<'a, M>
where
    M: HasVertices,
    <M as HasVerticesIntr>::Vertex: HigherVertex,
    M: HasEdges,
{
    pub(crate) fn new<EI: TryInto<EdgeId>>(mesh: &'a M, edge: EI) -> Self {
        Self {
            mesh,
            edge: edge.try_into().ok().unwrap(),
        }
    }

    fn from_vertex(mesh: &'a M, vertex: VertexId) -> Option<Self> {
        match EdgeId::try_from([vertex, mesh.vertices_r()[vertex].target()]) {
            Ok(edge) => Some(Self::new(mesh, edge)),
            Err(_) => None,
        }
    }

    fn from_target(mesh: &'a M, vertex: VertexId) -> Option<Self> {
        match EdgeId::try_from([mesh.vertices_r()[vertex].source(), vertex]) {
            Ok(edge) => Some(Self::new(mesh, edge)),
            Err(_) => None,
        }
    }

    /// A walker that will not be used
    fn dummy(mesh: &'a M) -> Self {
        Self {
            mesh,
            edge: EdgeId::dummy(),
        }
    }

    /// Get the mesh that the walker navigates
    pub fn mesh(&self) -> &M {
        self.mesh
    }

    /// Get the current vertex id,
    /// which is the source of the current edge.
    pub fn first(&self) -> VertexId {
        self.edge.0[0]
    }

    /// Get the vertex id of the target of the current edge.
    pub fn second(&self) -> VertexId {
        self.edge.0[1]
    }

    /// Gets the current edge id
    pub fn edge(&self) -> EdgeId {
        self.edge
    }

    /// Gets the current list of vertices in order
    pub fn vertices(&self) -> [VertexId; 2] {
        [self.first(), self.second()]
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
        if !<<M::Edge as Edge>::Manifold as Bit>::BOOL {
            self.edge = self
                .edge
                .with_target(self.mesh.edges_r()[&self.edge].links()[0].next);
        }
        self
    }

    /// Sets the current edge to the previous one with the same source vertex.
    pub fn prev(mut self) -> Self {
        if !<<M::Edge as Edge>::Manifold as Bit>::BOOL {
            self.edge = self
                .edge
                .with_target(self.mesh.edges_r()[&self.edge].links()[0].prev);
        }
        self
    }

    /// Sets the current edge to the next one with the same target vertex.
    pub fn next_in(mut self) -> Self {
        if !<<M::Edge as Edge>::Manifold as Bit>::BOOL {
            self.edge = self
                .edge
                .with_source(self.mesh.edges_r()[&self.edge].links()[1].next);
        }
        self
    }

    /// Sets the current edge to the previous one with the same target vertex.
    pub fn prev_in(mut self) -> Self {
        if !<<M::Edge as Edge>::Manifold as Bit>::BOOL {
            self.edge = self
                .edge
                .with_source(self.mesh.edges_r()[&self.edge].links()[1].prev);
        }
        self
    }

    /// Sets the current edge to an edge whose source vertex is the current edge's target vertex.
    pub fn target_out(mut self) -> Option<Self> {
        let target = self.mesh.vertices_r()[self.edge.0[1]].target();
        if let Some(edge) = [self.edge.0[1], target].try_into().ok() {
            self.edge = edge;
            Some(self)
        } else {
            None
        }
    }

    /// Sets the current edge to an edge whose target vertex is the current edge's source vertex.
    pub fn source_in(mut self) -> Option<Self> {
        let source = self.mesh.vertices_r()[self.edge.0[0]].source();
        if let Some(edge) = [source, self.edge.0[0]].try_into().ok() {
            self.edge = edge;
            Some(self)
        } else {
            None
        }
    }

    pub fn tri_walker(self) -> Option<TriWalker<'a, M>>
    where
        <M as HasEdgesIntr>::Edge: HigherEdge,
        M: HasTris,
    {
        TriWalker::from_edge(self.mesh, self.edge)
    }
}

/// An iterator over the targets of the outgoing edges of a vertex.
#[derive(Clone, Debug)]
pub struct VertexTargets<'a, M: ?Sized>
where
    M: HasVertices,
    <M as HasVerticesIntr>::Vertex: HigherVertex,
    M: HasEdges,
{
    walker: EdgeWalker<'a, M>,
    start_target: VertexId,
    finished: bool,
}

impl<'a, M: ?Sized> Iterator for VertexTargets<'a, M>
where
    M: HasVertices,
    <M as HasVerticesIntr>::Vertex: HigherVertex,
    M: HasEdges,
{
    type Item = VertexId;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        let target = self.walker.second();
        self.walker = self.walker.next();
        if self.walker.edge().0[1] == self.start_target {
            self.finished = true;
        }
        Some(target)
    }
}

/// An iterator over the sources of the incoming edges of a vertex.
#[derive(Clone, Debug)]
pub struct VertexSources<'a, M: ?Sized>
where
    M: HasVertices,
    <M as HasVerticesIntr>::Vertex: HigherVertex,
    M: HasEdges,
{
    walker: EdgeWalker<'a, M>,
    start_source: VertexId,
    finished: bool,
}

impl<'a, M: ?Sized> Iterator for VertexSources<'a, M>
where
    M: HasVertices,
    <M as HasVerticesIntr>::Vertex: HigherVertex,
    M: HasEdges,
{
    type Item = VertexId;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        let source = self.walker.first();
        self.walker = self.walker.next_in();
        if self.walker.edge().0[0] == self.start_source {
            self.finished = true;
        }
        Some(source)
    }
}
#[macro_export]
#[doc(hidden)]
macro_rules! impl_index_edge {
    ($name:ident<$v:ident, $e:ident $(, $args:ident)*>) => {
        impl<$v, $e $(, $args)*> std::ops::Index<[crate::vertex::VertexId; 2]> for $name<$v, $e $(, $args)*> {
            type Output = $e;

            fn index(&self, index: [crate::vertex::VertexId; 2]) -> &Self::Output {
                self.edge(index).unwrap()
            }
        }

        impl<$v, $e $(, $args)*> std::ops::IndexMut<[crate::vertex::VertexId; 2]> for $name<$v, $e $(, $args)*> {
            fn index_mut(&mut self, index: [crate::vertex::VertexId; 2]) -> &mut Self::Output {
                self.edge_mut(index).unwrap()
            }
        }

        impl<$v, $e $(, $args)*> std::ops::Index<crate::edge::EdgeId> for $name<$v, $e $(, $args)*> {
            type Output = $e;

            fn index(&self, index: crate::edge::EdgeId) -> &Self::Output {
                self.edge(index).unwrap()
            }
        }

        impl<$v, $e $(, $args)*> std::ops::IndexMut<crate::edge::EdgeId> for $name<$v, $e $(, $args)*> {
            fn index_mut(&mut self, index: crate::edge::EdgeId) -> &mut Self::Output {
                self.edge_mut(index).unwrap()
            }
        }
    };
}

/// For concrete simplicial complexes with edges
pub trait HasPositionAndEdges: HasEdges + HasPosition
where
    <Self::Vertex as Vertex>::V: Position,
    DefaultAllocator: Allocator<f64, HasPositionDim<Self>>,
    Self::Vertex: HigherVertex,
{
    /// Gets the positions of the vertices of an edge
    fn edge_positions<EI: TryInto<EdgeId>>(&self, edge: EI) -> Option<[HasPositionPoint<Self>; 2]> {
        let edge = edge.try_into().ok()?;
        let v0 = self.position(edge.0[0])?;
        let v1 = self.position(edge.0[1])?;
        Some([v0, v1])
    }
}

pub(crate) mod internal {
    use super::EdgeId;
    use crate::vertex::internal::{HasVertices as HasVerticesIntr, HigherVertex};
    use crate::vertex::VertexId;
    use fnv::FnvHashMap;
    #[cfg(feature = "serialize")]
    use serde::{Deserialize, Serialize};
    use typenum::Bit;

    #[macro_export]
    #[doc(hidden)]
    macro_rules! impl_edge {
        ($name:ident<$e:ident>, new |$id:ident, $link:ident, $value:ident| $new:expr) => {
            impl<$e> crate::edge::internal::Edge for $name<$e> {
                type E = $e;
                type Manifold = typenum::B0;

                fn new(
                    $id: VertexId,
                    $link: [crate::edge::internal::Link<crate::vertex::VertexId>; 2],
                    $value: Self::E,
                ) -> Self {
                    $new
                }

                fn links(&self) -> [crate::edge::internal::Link<crate::vertex::VertexId>; 2] {
                    self.links
                }

                fn links_mut(
                    &mut self,
                ) -> &mut [crate::edge::internal::Link<crate::vertex::VertexId>; 2] {
                    &mut self.links
                }

                fn to_value(self) -> Self::E {
                    self.value
                }

                fn value(&self) -> &Self::E {
                    &self.value
                }

                fn value_mut(&mut self) -> &mut Self::E {
                    &mut self.value
                }
            }
        };
    }

    #[macro_export]
    #[doc(hidden)]
    macro_rules! impl_edge_manifold {
        ($name:ident<$e:ident>, new |$id:ident, $link:ident, $value:ident| $new:expr) => {
            impl<$e> crate::edge::internal::Edge for $name<$e> {
                type E = $e;
                type Manifold = typenum::B1;

                fn new(
                    $id: VertexId,
                    $link: [crate::edge::internal::Link<crate::vertex::VertexId>; 2],
                    $value: Self::E,
                ) -> Self {
                    $new
                }

                fn links(&self) -> [crate::edge::internal::Link<crate::vertex::VertexId>; 2] {
                    panic!("Cannot get links in \"manifold\" edge")
                }

                fn links_mut(
                    &mut self,
                ) -> &mut [crate::edge::internal::Link<crate::vertex::VertexId>; 2] {
                    panic!("Cannot get links in \"manifold\" edge")
                }

                fn to_value(self) -> Self::E {
                    self.value
                }

                fn value(&self) -> &Self::E {
                    &self.value
                }

                fn value_mut(&mut self) -> &mut Self::E {
                    &mut self.value
                }
            }
        };
    }

    #[macro_export]
    #[doc(hidden)]
    macro_rules! impl_higher_edge {
        ($name:ident<$e:ident>) => {
            impl<$e> crate::edge::internal::HigherEdge for $name<$e> {
                fn tri_opp(&self) -> crate::vertex::VertexId {
                    self.tri_opp
                }

                fn tri_opp_mut(&mut self) -> &mut crate::vertex::VertexId {
                    &mut self.tri_opp
                }
            }
        };
    }

    #[macro_export]
    #[doc(hidden)]
    macro_rules! impl_has_edges {
        ($name:ident<$v:ident, $e:ident $(, $args:ident)*>, $edge:ident) => {
            impl<$v, $e $(, $args)*> crate::edge::internal::HasEdges for $name<$v, $e $(, $args)*> {
                type Edge = $edge<$e>;

                fn edges_r(&self) -> &FnvHashMap<crate::edge::EdgeId, Self::Edge> {
                    &self.edges
                }

                fn edges_r_mut(&mut self) -> &mut FnvHashMap<crate::edge::EdgeId, Self::Edge> {
                    &mut self.edges
                }
            }
        };
    }

    #[derive(Clone, Copy, Debug)]
    #[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
    pub struct Link<T> {
        pub prev: T,
        pub next: T,
    }

    impl<T> Link<T> {
        pub fn new(prev: T, next: T) -> Self {
            Link { prev, next }
        }

        pub fn dummy(dummy_fn: impl Fn() -> T) -> Self {
            Link::new(dummy_fn(), dummy_fn())
        }
    }

    pub trait Edge {
        type E;
        type Manifold: Bit;

        /// Takes the vertex id of the source, in case
        /// the edge needs to store a dummy value for the opposite vertex
        /// of the triangle.
        fn new(id: VertexId, links: [Link<VertexId>; 2], value: Self::E) -> Self;

        /// Target link, then source link
        fn links(&self) -> [Link<VertexId>; 2];

        /// Panics for edges in "manifold" edge meshes
        fn links_mut(&mut self) -> &mut [Link<VertexId>; 2];

        fn to_value(self) -> Self::E;

        fn value(&self) -> &Self::E;

        fn value_mut(&mut self) -> &mut Self::E;
    }

    pub trait HigherEdge: Edge {
        fn tri_opp(&self) -> VertexId;

        fn tri_opp_mut(&mut self) -> &mut VertexId;
    }

    pub trait HasEdges: HasVerticesIntr
    where
        Self::Vertex: HigherVertex,
    {
        type Edge: Edge;

        fn edges_r(&self) -> &FnvHashMap<EdgeId, Self::Edge>;

        fn edges_r_mut(&mut self) -> &mut FnvHashMap<EdgeId, Self::Edge>;
    }

    /// Removes higher-order simplexes that contain some edge
    pub trait RemoveEdgeHigher: HasEdges
    where
        Self::Vertex: HigherVertex,
    {
        fn remove_edge_higher(&mut self, edge: EdgeId);
    }

    /// Clears higher-order simplexes
    pub trait ClearEdgesHigher: HasEdges
    where
        Self::Vertex: HigherVertex,
    {
        fn clear_edges_higher(&mut self);
    }
}
