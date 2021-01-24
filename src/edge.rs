//! Traits and structs related to edges

use nalgebra::allocator::Allocator;
use nalgebra::DefaultAllocator;
#[cfg(feature = "serde_")]
use serde::{Deserialize, Serialize};
use std::collections::hash_map;
use std::convert::{TryFrom, TryInto};
use std::iter::{Filter, Map};

use crate::tri::{HasTris, HasTrisWalker, TriWalk};
use crate::vertex::internal::HasVertices as HasVerticesIntr;
use crate::vertex::{
    internal::{HigherVertex, Vertex},
    HasVertices, VertexId,
};
use crate::vertex::{HasPositionDim, HasPositionPoint, Position};
use crate::{
    iter::{IteratorExt, MapWith},
    vertex::HasPosition,
};

use internal::HasEdges as HasEdgesIntr;
use internal::{ClearEdgesHigher, Edge, HigherEdge, Link, RemoveEdgeHigher};

/// An edge id is just the edge's vertices in order.
/// The vertices are not allowed to be the same.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde_", derive(Serialize, Deserialize))]
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

    /// Sets the target. Assumes it doesn't equal the source.
    fn with_target(mut self, vertex: VertexId) -> Self {
        self.0[1] = vertex;
        debug_assert_ne!(self.0[0], self.0[1]);
        self
    }
}

type EdgeFilterFn<'a, ET> = for<'b> fn(&'b (&'a EdgeId, &'a ET)) -> bool;
type EdgeMapFn<'a, ET> = fn((&'a EdgeId, &'a ET)) -> (&'a EdgeId, &'a <ET as Edge>::E);
/// Iterator over the edges of a mesh.
pub type Edges<'a, ET> =
    Map<Filter<hash_map::Iter<'a, EdgeId, ET>, EdgeFilterFn<'a, ET>>, EdgeMapFn<'a, ET>>;
type EdgeFilterFnMut<'a, ET> = for<'b> fn(&'b (&'a EdgeId, &'a mut ET)) -> bool;
type EdgeMapFnMut<'a, ET> = fn((&'a EdgeId, &'a mut ET)) -> (&'a EdgeId, &'a mut <ET as Edge>::E);
/// Iterator over the edges of a mesh mutably.
pub type EdgesMut<'a, ET> =
    Map<Filter<hash_map::IterMut<'a, EdgeId, ET>, EdgeFilterFnMut<'a, ET>>, EdgeMapFnMut<'a, ET>>;

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
        self.num_edges_r()
    }

    /// Iterates over the edges of this mesh.
    /// Gives (id, value) pairs
    fn edges(&self) -> Edges<Self::Edge> {
        self.edges_r()
            .iter()
            .filter::<EdgeFilterFn<Self::Edge>>(|(_, e)| e.value().is_some())
            .map::<_, EdgeMapFn<Self::Edge>>(|(id, e)| (id, e.value().as_ref().unwrap()))
    }

    /// Iterates mutably over the edges of this mesh.
    /// Gives (id, value) pairs
    fn edges_mut(&mut self) -> EdgesMut<Self::Edge> {
        self.edges_r_mut()
            .iter_mut()
            .filter::<EdgeFilterFnMut<Self::Edge>>(|(_, e)| e.value().is_some())
            .map::<_, EdgeMapFnMut<Self::Edge>>(|(id, e)| (id, e.value_mut().as_mut().unwrap()))
    }

    /// Gets the value of the edge at a specific id.
    /// Returns None if not found.
    fn edge<EI: TryInto<EdgeId>>(&self, id: EI) -> Option<&E!()> {
        id.try_into()
            .ok()
            .and_then(|id| self.edges_r().get(&id))
            .and_then(|e| e.value().as_ref())
    }

    /// Gets the value of the edge at a specific id mutably.
    /// Returns None if not found.
    fn edge_mut<EI: TryInto<EdgeId>>(&mut self, id: EI) -> Option<&mut E!()> {
        id.try_into()
            .ok()
            .and_then(move |id| self.edges_r_mut().get_mut(&id))
            .and_then(|e| e.value_mut().as_mut())
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

    /// Iterates over the outgoing edges of a vertex.
    /// The vertex must exist.
    fn vertex_edges_out(&self, vertex: VertexId) -> VertexEdgesOut<Self> {
        self.vertex_targets(vertex)
            .map_with(vertex, |s, t| EdgeId([s, t]))
    }

    /// Iterates over the sources of the incoming edges of a vertex.
    /// The vertex must exist.
    fn vertex_sources(&self, vertex: VertexId) -> VertexSources<Self> {
        if let Some(walker) =
            EdgeWalker::from_vertex_less_checked(self, vertex).and_then(|w| w.backward())
        {
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

    /// Iterates over the incoming edges of a vertex.
    /// The vertex must exist.
    fn vertex_edges_in(&self, vertex: VertexId) -> VertexEdgesIn<Self> {
        self.vertex_sources(vertex)
            .map_with(vertex, |t, s| EdgeId([s, t]))
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
            let old = edge.value_mut().take();
            *edge.value_mut() = Some(value);
            if old.is_none() {
                *self.num_edges_r_mut() += 1;
            }
            old
        } else {
            *self.num_edges_r_mut() += 1;

            let mut insert_edge = |id: EdgeId, value: Option<E!()>| {
                let target = self.vertices_r()[id.0[0]].target();

                let (prev, next) = if target == id.0[0] {
                    // First edge from vertex
                    *self.vertices_r_mut()[id.0[0]].target_mut() = id.0[1];
                    (id.0[1], id.0[1])
                } else {
                    let prev = self.edges_r()[&[id.0[0], target].try_into().ok().unwrap()]
                        .link()
                        .prev;
                    let next = target;
                    self.edges_r_mut()
                        .get_mut(&id.with_target(prev))
                        .unwrap()
                        .link_mut()
                        .next = id.0[1];
                    self.edges_r_mut()
                        .get_mut(&id.with_target(next))
                        .unwrap()
                        .link_mut()
                        .prev = id.0[1];
                    (prev, next)
                };

                self.edges_r_mut()
                    .insert(id, Edge::new(id.0[0], Link::new(prev, next), value));
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
        iter.into_iter().for_each(|(id, value)| {
            self.add_edge(id, value);
        })
    }

    /// Removes an edge from the mesh and returns the value that was there,
    /// or None if there was nothing there
    fn remove_edge<EI: TryInto<EdgeId>>(&mut self, id: EI) -> Option<E!()> {
        let id = match id.try_into() {
            Ok(id) => id,
            Err(_) => return None,
        };

        if self.edge(id).is_some() {
            self.remove_edge_higher(id);
        }

        match self.edges_r().get(&id.twin()).map(|e| e.value().as_ref()) {
            // Twin actually exists; just set value to None
            Some(Some(_)) => {
                let old = self.edges_r_mut().get_mut(&id).unwrap().value_mut().take();
                if old.is_some() {
                    *self.num_edges_r_mut() -= 1;
                }
                old
            }

            // Twin is phantom, so remove both edge and twin from map
            Some(None) => {
                // Twin is phantom, so this edge actually exists.
                *self.num_edges_r_mut() -= 1;

                let mut delete_edge = |id: EdgeId| {
                    let edge = &self.edges_r()[&id];
                    let prev = edge.link().prev;
                    let next = edge.link().next;
                    self.edges_r_mut()
                        .get_mut(&id.with_target(prev))
                        .unwrap()
                        .link_mut()
                        .next = next;
                    self.edges_r_mut()
                        .get_mut(&id.with_target(next))
                        .unwrap()
                        .link_mut()
                        .prev = prev;

                    let source = &mut self.vertices_r_mut()[id.0[0]];
                    if id.0[1] == next {
                        // this was the last edge from the vertex
                        *source.target_mut() = id.0[0];
                    } else if id.0[1] == source.target() {
                        *source.target_mut() = next;
                    }

                    self.edges_r_mut().remove(&id).and_then(|e| e.to_value())
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
        *self.num_edges_r_mut() = 0;

        // Fix vertex-target links
        for (id, vertex) in self.vertices_r_mut() {
            *vertex.target_mut() = *id;
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

    /// Doesn't check that the starting edge actually exists
    fn from_vertex_less_checked(mesh: &'a M, vertex: VertexId) -> Option<Self> {
        ([vertex, mesh.vertices_r()[vertex].target()].try_into().ok() as Option<EdgeId>)
            .map(|edge| Self::new(mesh, edge))
    }

    fn from_vertex(mesh: &'a M, vertex: VertexId) -> Option<Self> {
        let start = match [vertex, mesh.vertices_r()[vertex].target()].try_into() {
            Ok(start) => start,
            Err(_) => return None,
        };
        let mut edge = start;
        while mesh.edges_r()[&edge].value().is_none() {
            edge = edge.with_target(mesh.edges_r()[&edge].link().next);
            if edge == start {
                return None;
            }
        }

        Some(Self::new(mesh, edge))
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
        while {
            self.edge = self
                .edge
                .with_target(self.mesh.edges_r()[&self.edge].link().next);
            self.mesh.edge(self.edge).is_none()
        } {}
        self
    }

    /// Sets the current edge to the previous one with the same source vertex.
    pub fn prev(mut self) -> Self {
        while {
            self.edge = self
                .edge
                .with_target(self.mesh.edges_r()[&self.edge].link().prev);
            self.mesh.edge(self.edge).is_none()
        } {}
        self
    }

    /// Sets the current edge to the next one with the same target vertex.
    pub fn next_in(mut self) -> Self {
        while {
            self.edge = self
                .edge
                .twin()
                .with_target(self.mesh.edges_r()[&self.edge.twin()].link().next)
                .twin();
            self.mesh.edge(self.edge).is_none()
        } {}
        self
    }

    /// Sets the current edge to the previous one with the same target vertex.
    pub fn prev_in(mut self) -> Self {
        while {
            self.edge = self
                .edge
                .twin()
                .with_target(self.mesh.edges_r()[&self.edge.twin()].link().prev)
                .twin();
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
            self.edge = self
                .edge
                .with_target(self.mesh.edges_r()[&self.edge].link().next);
            if self.mesh.edge(self.edge).is_some() {
                false
            } else if self.edge == twin {
                found = false;
                false
            } else {
                true
            }
        } {}
        if found {
            Some(self)
        } else {
            None
        }
    }

    /// Sets the current edge to an edge whose target vertex is the current edge's source vertex.
    /// Chooses a non-twin edge if possible.
    pub fn backward(mut self) -> Option<Self> {
        let twin = self.edge.twin();
        self.edge = twin;
        let mut found = true;
        while {
            self.edge = self
                .edge
                .twin()
                .with_target(self.mesh.edges_r()[&self.edge.twin()].link().next)
                .twin();
            if self.mesh.edge(self.edge).is_some() {
                false
            } else if self.edge == twin {
                found = false;
                false
            } else {
                true
            }
        } {}
        if found {
            Some(self)
        } else {
            None
        }
    }

    pub fn tri_walker(self) -> Option<HasTrisWalker<'a, M>>
    where
        <M as HasEdgesIntr>::Edge: HigherEdge,
        M: HasTris,
        for<'b> HasTrisWalker<'b, M>: TriWalk<'b, Mesh = M>,
    {
        HasTrisWalker::from_edge(self.mesh, self.edge)
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
    ($name:ident<$v:ident, $e:ident $(, $args:ident)*> $(where $($wh:tt)*)?) => {
        impl<$v, $e $(, $args)*> std::ops::Index<[crate::vertex::VertexId; 2]> for $name<$v, $e $(, $args)*>
        $(where $($wh)*)?
        {
            type Output = $e;

            fn index(&self, index: [crate::vertex::VertexId; 2]) -> &Self::Output {
                self.edge(index).unwrap()
            }
        }

        impl<$v, $e $(, $args)*> std::ops::IndexMut<[crate::vertex::VertexId; 2]> for $name<$v, $e $(, $args)*>
        $(where $($wh)*)?
        {
            fn index_mut(&mut self, index: [crate::vertex::VertexId; 2]) -> &mut Self::Output {
                self.edge_mut(index).unwrap()
            }
        }

        impl<$v, $e $(, $args)*> std::ops::Index<crate::edge::EdgeId> for $name<$v, $e $(, $args)*>
        $(where $($wh)*)?
        {
            type Output = $e;

            fn index(&self, index: crate::edge::EdgeId) -> &Self::Output {
                self.edge(index).unwrap()
            }
        }

        impl<$v, $e $(, $args)*> std::ops::IndexMut<crate::edge::EdgeId> for $name<$v, $e $(, $args)*>
        $(where $($wh)*)?
        {
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
    #[cfg(feature = "serde_")]
    use serde::{Deserialize, Serialize};

    #[macro_export]
    #[doc(hidden)]
    macro_rules! impl_edge {
        ($name:ident<$e:ident>, new |$id:ident, $link:ident, $value:ident| $new:expr) => {
            impl<$e> crate::edge::internal::Edge for $name<$e> {
                type E = $e;

                fn new($id: VertexId, $link: Link<VertexId>, $value: Option<Self::E>) -> Self {
                    $new
                }

                fn link(&self) -> &crate::edge::internal::Link<crate::vertex::VertexId> {
                    &self.link
                }

                fn link_mut(
                    &mut self,
                ) -> &mut crate::edge::internal::Link<crate::vertex::VertexId> {
                    &mut self.link
                }

                fn to_value(self) -> Option<Self::E> {
                    self.value
                }

                fn value(&self) -> &Option<Self::E> {
                    &self.value
                }

                fn value_mut(&mut self) -> &mut Option<Self::E> {
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
        ($name:ident<$v:ident, $e:ident $(, $args:ident)*>, $edge:ident $(where $($wh:tt)*)?) => {
            impl<$v, $e $(, $args)*> crate::edge::internal::HasEdges for $name<$v, $e $(, $args)*>
            $(where $($wh)*)?
            {
                type Edge = $edge<$e>;

                fn edges_r(&self) -> &FnvHashMap<crate::edge::EdgeId, Self::Edge> {
                    &self.edges
                }

                fn edges_r_mut(&mut self) -> &mut FnvHashMap<crate::edge::EdgeId, Self::Edge> {
                    &mut self.edges
                }

                fn num_edges_r(&self) -> usize {
                    self.num_edges
                }

                fn num_edges_r_mut(&mut self) -> &mut usize {
                    &mut self.num_edges
                }
            }
        };
    }

    #[derive(Clone, Copy, Debug)]
    #[cfg_attr(feature = "serde_", derive(Serialize, Deserialize))]
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

        /// Takes the vertex id of the source, in case
        /// the edge needs to store a dummy value for the opposite vertex
        /// of the triangle.
        fn new(id: VertexId, link: Link<VertexId>, value: Option<Self::E>) -> Self;

        fn link(&self) -> &Link<VertexId>;

        fn link_mut(&mut self) -> &mut Link<VertexId>;

        fn to_value(self) -> Option<Self::E>;

        fn value(&self) -> &Option<Self::E>;

        fn value_mut(&mut self) -> &mut Option<Self::E>;
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

        fn num_edges_r(&self) -> usize;

        fn num_edges_r_mut(&mut self) -> &mut usize;
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
