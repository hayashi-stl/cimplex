//! Traits and structs related to triangles

use nalgebra::{DefaultAllocator, allocator::Allocator};
#[cfg(feature = "serde_")]
use serde::{Deserialize, Serialize};
use std::collections::hash_map;
use std::convert::{TryFrom, TryInto};
use std::iter::{Filter, Map};

use crate::{edge::{EdgeId, EdgeWalker, HasEdges, VertexEdgesOut}, vertex::{HasPosition, HasPositionDim, HasPositionPoint, Position, internal::Vertex}};
use crate::iter::{IteratorExt, MapWith};
use crate::tri::internal::HasTris as HasTrisIntr;
use crate::vertex::internal::{HasVertices as HasVerticesIntr, HigherVertex};
use crate::vertex::HasVertices;
use crate::{edge::internal::HigherEdge, vertex::VertexId};
use crate::{
    edge::internal::{Edge, HasEdges as HasEdgesIntr, Link},
    tet::{HasTets, TetWalker},
};

use internal::{ClearTrisHigher, RemoveTriHigher, Tri};

use self::internal::HigherTri;

/// An triangle id is just the triangle's vertices in winding order,
/// with the smallest index first.
/// No two vertices are allowed to be the same.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde_", derive(Serialize, Deserialize))]
pub struct TriId(pub(crate) [VertexId; 3]);

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
    pub(crate) fn canonicalize(mut v: [VertexId; 3]) -> [VertexId; 3] {
        let min_pos = v
            .iter()
            .enumerate()
            .min_by_key(|(_, value)| **value)
            .unwrap()
            .0;
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
    pub fn edges_and_opp(self) -> [(EdgeId, VertexId); 3] {
        [
            (EdgeId([self.0[0], self.0[1]]), self.0[2]),
            (EdgeId([self.0[1], self.0[2]]), self.0[0]),
            (EdgeId([self.0[2], self.0[0]]), self.0[1]),
        ]
    }

    /// Gets the index of a vertex, assuming it's part of the triangle
    fn index(self, vertex: VertexId) -> usize {
        self.0.iter().position(|v| *v == vertex).unwrap()
    }

    /// Gets the index of the opposite vertex of an edge, assuming it's part of the triangle
    pub(crate) fn opp_index(self, edge: EdgeId) -> usize {
        (self.index(edge.0[0]) + 2) % 3
    }

    /// Reverses the tri so it winds the other way
    fn twin(self) -> Self {
        Self([self.0[0], self.0[2], self.0[1]])
    }

    /// Sets the opposite vertex of some edge to some vertex.
    fn with_opp(mut self, edge: EdgeId, vertex: VertexId) -> Self {
        self.0[self.opp_index(edge)] = vertex;
        self.0 = Self::canonicalize(self.0);
        self
    }
}

type TriFilterFn<'a, FT> = for<'b> fn(&'b (&'a TriId, &'a FT)) -> bool;
type TriMapFn<'a, FT> = fn((&'a TriId, &'a FT)) -> (&'a TriId, &'a <FT as Tri>::F);
/// Iterator over the triangles of a mesh.
pub type Tris<'a, FT> =
    Map<Filter<hash_map::Iter<'a, TriId, FT>, TriFilterFn<'a, FT>>, TriMapFn<'a, FT>>;
type TriFilterFnMut<'a, FT> = for<'b> fn(&'b (&'a TriId, &'a mut FT)) -> bool;
type TriMapFnMut<'a, FT> = fn((&'a TriId, &'a mut FT)) -> (&'a TriId, &'a mut <FT as Tri>::F);
/// Iterator over the triangles of a mesh mutably.
pub type TrisMut<'a, FT> =
    Map<Filter<hash_map::IterMut<'a, TriId, FT>, TriFilterFnMut<'a, FT>>, TriMapFnMut<'a, FT>>;

/// Iterator over the triangles connected to an edge with the correct winding.
pub type EdgeTris<'a, M> =
    MapWith<EdgeId, TriId, EdgeVertexOpps<'a, M>, fn(EdgeId, VertexId) -> TriId>;
/// Iterator over the triangles connected to a vertex.
pub type VertexTris<'a, M> =
    MapWith<VertexId, TriId, VertexEdgeOpps<'a, M>, fn(VertexId, EdgeId) -> TriId>;

macro_rules! E {
    () => {
        <Self::Edge as Edge>::E
    };
}

macro_rules! F {
    () => {
        <Self::Tri as Tri>::F
    };
}

/// For simplicial complexes that can have triangles
pub trait HasTris: internal::HasTris + HasEdges + RemoveTriHigher + ClearTrisHigher
where
    Self::Vertex: HigherVertex,
    Self::Edge: HigherEdge,
{
    /// Gets the number of triangles.
    fn num_tris(&self) -> usize {
        self.num_tris_r()
    }

    /// Iterates over the triangles of this mesh.
    /// Gives (id, value) pairs
    fn tris(&self) -> Tris<Self::Tri> {
        self.tris_r()
            .iter()
            .filter::<TriFilterFn<Self::Tri>>(|(_, f)| f.value().is_some())
            .map::<_, TriMapFn<Self::Tri>>(|(id, f)| (id, f.value().as_ref().unwrap()))
    }

    /// Iterates mutably over the triangles of this mesh.
    /// Gives (id, value) pairs
    fn tris_mut(&mut self) -> TrisMut<Self::Tri> {
        self.tris_r_mut()
            .iter_mut()
            .filter::<TriFilterFnMut<Self::Tri>>(|(_, f)| f.value().is_some())
            .map::<_, TriMapFnMut<Self::Tri>>(|(id, f)| (id, f.value_mut().as_mut().unwrap()))
    }

    /// Gets the value of the triangle at a specific id.
    /// Returns None if not found.
    fn tri<FI: TryInto<TriId>>(&self, id: FI) -> Option<&F!()> {
        id.try_into()
            .ok()
            .and_then(|id| self.tris_r().get(&id))
            .and_then(|f| f.value().as_ref())
    }

    /// Gets the value of the triangle at a specific id mutably.
    /// Returns None if not found.
    fn tri_mut<FI: TryInto<TriId>>(&mut self, id: FI) -> Option<&mut F!()> {
        id.try_into()
            .ok()
            .and_then(move |id| self.tris_r_mut().get_mut(&id))
            .and_then(|f| f.value_mut().as_mut())
    }

    /// Iterates over the opposite edges of the triangles that a vertex is part of.
    /// The vertex must exist.
    fn vertex_edge_opps(&self, vertex: VertexId) -> VertexEdgeOpps<Self> {
        VertexEdgeOpps {
            mesh: self,
            edges: self.vertex_edges_out(vertex),
            opps: None,
        }
    }

    /// Iterates over the triangles that a vertex is part of.
    /// The vertex must exist.
    fn vertex_tris(&self, vertex: VertexId) -> VertexTris<Self> {
        self.vertex_edge_opps(vertex)
            .map_with(vertex, |vertex, opp| {
                TriId(TriId::canonicalize([vertex, opp.0[0], opp.0[1]]))
            })
    }

    /// Iterates over the opposite vertices of the triangles that an edge is part of.
    /// The edge must exist.
    fn edge_vertex_opps<EI: TryInto<EdgeId>>(&self, edge: EI) -> EdgeVertexOpps<Self> {
        if let Some(walker) = self.tri_walker_from_edge(edge) {
            let start_opp = walker.third();
            EdgeVertexOpps {
                walker,
                start_opp,
                finished: false,
            }
        } else {
            EdgeVertexOpps {
                walker: TriWalker::dummy(self),
                start_opp: VertexId::dummy(),
                finished: true,
            }
        }
    }

    /// Iterates over the triangles that an edge is part of.
    /// The edge must exist.
    fn edge_tris<EI: TryInto<EdgeId>>(&self, edge: EI) -> EdgeTris<Self> {
        let edge = edge.try_into().ok().unwrap();
        self.edge_vertex_opps(edge).map_with(edge, |edge, opp| {
            TriId(TriId::canonicalize([edge.0[0], edge.0[1], opp]))
        })
    }

    /// Adds a triangle to the mesh. Vertex order is important!
    /// If the triangle was already there, this replaces the value.
    /// Adds in the required edges if they aren't there already.
    /// Returns the previous value of the triangle, if there was one.
    ///
    /// # Panics
    /// Panics if any vertex doesn't exist or if any two vertices are the same.
    fn add_tri<FI: TryInto<TriId>>(
        &mut self,
        vertices: FI,
        value: F!(),
        edge_value: impl Fn() -> E!(),
    ) -> Option<F!()> {
        let id = vertices.try_into().ok().unwrap();

        for edge in &id.edges() {
            if self.edge(*edge).is_none() {
                self.add_edge(*edge, edge_value());
            }
        }

        // Can't use entry().or_insert() because that would cause a
        // mutable borrow and an immutable borrow to exist at the same time
        if let Some(tri) = self.tris_r_mut().get_mut(&id) {
            let old = tri.value_mut().take();
            *tri.value_mut() = Some(value);
            if old.is_none() {
                *self.num_tris_r_mut() += 1;
            }
            old
        } else {
            *self.num_tris_r_mut() += 1;

            let mut insert_tri = |id: TriId, value: Option<F!()>| {
                let mut opps = [Link::dummy(VertexId::dummy); 3];

                for (i, (edge, opp)) in id.edges_and_opp().iter().enumerate() {
                    let target = self.edges_r()[edge].tri_opp();

                    let (prev, next) = if target == edge.0[0] {
                        // First tri from edge
                        *self.edges_r_mut().get_mut(edge).unwrap().tri_opp_mut() = *opp;
                        (*opp, *opp)
                    } else {
                        let side = [edge.0[0], edge.0[1], target].try_into().ok().unwrap();
                        let prev = self.tris_r()[&side].link(side, *edge).prev;
                        let next = target;
                        let prev_tri = id.with_opp(*edge, prev);
                        let next_tri = id.with_opp(*edge, next);
                        self.tris_r_mut()
                            .get_mut(&prev_tri)
                            .unwrap()
                            .link_mut(prev_tri, *edge)
                            .next = *opp;
                        self.tris_r_mut()
                            .get_mut(&next_tri)
                            .unwrap()
                            .link_mut(next_tri, *edge)
                            .prev = *opp;
                        (prev, next)
                    };

                    opps[i] = Link::new(prev, next);
                }

                self.tris_r_mut().insert(id, Tri::new(id.0[0], opps, value));
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
    fn extend_tris<FI: TryInto<TriId>, I: IntoIterator<Item = (FI, F!())>>(
        &mut self,
        iter: I,
        edge_value: impl Fn() -> E!() + Clone,
    ) {
        iter.into_iter().for_each(|(id, value)| {
            self.add_tri(id, value, edge_value.clone());
        })
    }

    /// Removes an triangle from the mesh and returns the value that was there,
    /// or None if there was nothing there.
    /// Removes the edges that are part of the triangle if they are part of no other triangles
    /// and the triangle to be removed exists.
    fn remove_tri<FI: TryInto<TriId>>(&mut self, id: FI) -> Option<F!()> {
        let id = match id.try_into() {
            Ok(id) => id,
            Err(_) => return None,
        };

        if let Some(value) = self.remove_tri_keep_edges(id) {
            for edge in &id.edges() {
                if self.edge_tris(*edge).next().is_none() {
                    self.remove_edge(*edge);
                }
            }

            Some(value)
        } else {
            None
        }
    }

    /// Removes an triangle from the mesh and returns the value that was there,
    /// or None if there was nothing there.
    /// Keeps the edges that are part of the triangle.
    fn remove_tri_keep_edges<FI: TryInto<TriId>>(&mut self, id: FI) -> Option<F!()> {
        let id = match id.try_into() {
            Ok(id) => id,
            Err(_) => return None,
        };

        if self.tri(id).is_some() {
            self.remove_tri_higher(id);
        }

        match self.tris_r().get(&id.twin()).map(|f| f.value().as_ref()) {
            // Twin actually exists; just set value to None
            Some(Some(_)) => {
                let old = self.tris_r_mut().get_mut(&id).unwrap().value_mut().take();
                if old.is_some() {
                    *self.num_tris_r_mut() -= 1;
                }
                old
            }

            // Twin is phantom, so remove both tri and twin from map
            Some(None) => {
                // Twin is phantom, so this tri actually exists.
                *self.num_tris_r_mut() -= 1;

                let mut delete_tri = |id: TriId| {
                    for (i, (edge, opp)) in id.edges_and_opp().iter().enumerate() {
                        let tri = &self.tris_r()[&id];
                        let prev = tri.links()[i].prev;
                        let next = tri.links()[i].next;
                        let prev_tri = id.with_opp(*edge, prev);
                        let next_tri = id.with_opp(*edge, next);
                        self.tris_r_mut()
                            .get_mut(&prev_tri)
                            .unwrap()
                            .link_mut(prev_tri, *edge)
                            .next = next;
                        self.tris_r_mut()
                            .get_mut(&next_tri)
                            .unwrap()
                            .link_mut(next_tri, *edge)
                            .prev = prev;

                        let source = self.edges_r_mut().get_mut(&edge).unwrap();
                        if *opp == next {
                            // this was the last tri from the edge
                            *source.tri_opp_mut() = edge.0[0];
                        } else if *opp == source.tri_opp() {
                            *source.tri_opp_mut() = next;
                        }
                    }

                    self.tris_r_mut().remove(&id).and_then(|f| f.to_value())
                };

                delete_tri(id.twin());
                delete_tri(id)
            }

            // Twin isn't in map, and neither is the tri to remove
            None => None,
        }
    }

    /// Removes a list of triangles.
    fn remove_tris<FI: TryInto<TriId>, I: IntoIterator<Item = FI>>(&mut self, iter: I) {
        iter.into_iter().for_each(|id| {
            self.remove_tri(id);
        })
    }

    /// Removes a list of triangles.
    fn remove_tris_keep_edges<FI: TryInto<TriId>, I: IntoIterator<Item = FI>>(&mut self, iter: I) {
        iter.into_iter().for_each(|id| {
            self.remove_tri_keep_edges(id);
        })
    }

    /// Keeps only the triangles that satisfy a predicate
    fn retain_tris<P: FnMut(TriId, &F!()) -> bool>(&mut self, mut predicate: P) {
        let to_remove = self
            .tris()
            .filter(|(id, f)| !predicate(**id, *f))
            .map(|(id, _)| *id)
            .collect::<Vec<_>>();
        self.remove_tris(to_remove);
    }

    /// Keeps only the triangles that satisfy a predicate
    fn retain_tris_keep_edges<P: FnMut(TriId, &F!()) -> bool>(&mut self, mut predicate: P) {
        let to_remove = self
            .tris()
            .filter(|(id, f)| !predicate(**id, *f))
            .map(|(id, _)| *id)
            .collect::<Vec<_>>();
        self.remove_tris_keep_edges(to_remove);
    }

    /// Removes all triangles from the mesh.
    fn clear_tris(&mut self) {
        self.clear_tris_higher();
        self.tris_r_mut().clear();
        *self.num_tris_r_mut() = 0;

        // Fix edge-target links
        for (id, edge) in self.edges_r_mut() {
            *edge.tri_opp_mut() = id.0[0];
        }
    }

    /// Gets a triangle walker that starts at the given edge.
    /// Returns None if the edge has no triangle.
    fn tri_walker_from_edge<EI: TryInto<EdgeId>>(&self, edge: EI) -> Option<TriWalker<Self>> {
        TriWalker::from_edge(self, edge)
    }

    /// Gets a triangle walker that starts at the given edge with the given opposite vertex.
    /// They must actually exist.
    fn tri_walker_from_edge_vertex<EI: TryInto<EdgeId>>(
        &self,
        edge: EI,
        opp: VertexId,
    ) -> TriWalker<Self> {
        TriWalker::new(self, edge, opp)
    }

    /// Gets a triangle walker that starts at the given triangle.
    /// It must actually exist.
    /// Be warned that this does not preserve the order of the vertices
    /// because the triangle id is canonicalized.
    fn tri_walker_from_tri<FI: TryInto<TriId>>(&self, tri: FI) -> TriWalker<Self> {
        let tri = tri.try_into().ok().unwrap();
        TriWalker::new(self, tri.edges()[0], tri.0[2])
    }
}

/// A walker for navigating a simplicial complex by triangle
#[derive(Debug)]
pub struct TriWalker<'a, M: ?Sized>
where
    M: HasVertices,
    <M as HasVerticesIntr>::Vertex: HigherVertex,
    M: HasEdges,
    <M as HasEdgesIntr>::Edge: HigherEdge,
    M: HasTris,
{
    mesh: &'a M,
    edge: EdgeId,
    opp: VertexId,
}

impl<'a, M: ?Sized> Clone for TriWalker<'a, M>
where
    M: HasVertices,
    <M as HasVerticesIntr>::Vertex: HigherVertex,
    M: HasEdges,
    <M as HasEdgesIntr>::Edge: HigherEdge,
    M: HasTris,
{
    fn clone(&self) -> Self {
        Self {
            mesh: self.mesh,
            edge: self.edge,
            opp: self.opp,
        }
    }
}

impl<'a, M: ?Sized> Copy for TriWalker<'a, M>
where
    M: HasVertices,
    <M as HasVerticesIntr>::Vertex: HigherVertex,
    M: HasEdges,
    <M as HasEdgesIntr>::Edge: HigherEdge,
    M: HasTris,
{
}

impl<'a, M: ?Sized> TriWalker<'a, M>
where
    M: HasVertices,
    <M as HasVerticesIntr>::Vertex: HigherVertex,
    M: HasEdges,
    <M as HasEdgesIntr>::Edge: HigherEdge,
    M: HasTris,
{
    pub(crate) fn new<EI: TryInto<EdgeId>>(mesh: &'a M, edge: EI, opp: VertexId) -> Self {
        Self {
            mesh,
            edge: edge.try_into().ok().unwrap(),
            opp,
        }
    }

    pub(crate) fn from_edge<EI: TryInto<EdgeId>>(mesh: &'a M, edge: EI) -> Option<Self> {
        let edge = edge.try_into().ok().unwrap();
        let start = match [edge.0[0], edge.0[1], mesh.edges_r()[&edge].tri_opp()].try_into() {
            Ok(start) => start,
            Err(_) => return None,
        };
        let mut tri = start;
        while mesh.tris_r()[&tri].value().is_none() {
            tri = tri.with_opp(edge, mesh.tris_r()[&tri].link(tri, edge).next);
            if tri == start {
                return None;
            }
        }

        let index = tri.index(edge.0[0]);
        let (edge, opp) = tri.edges_and_opp()[index];
        Some(Self::new(mesh, edge, opp))
    }

    /// A walker that will not be used
    fn dummy(mesh: &'a M) -> Self {
        Self {
            mesh,
            edge: EdgeId::dummy(),
            opp: VertexId::dummy(),
        }
    }

    /// Get the mesh that the walker navigates
    pub fn mesh(&self) -> &M {
        self.mesh
    }

    /// Get the current vertex id,
    /// which is the source of the current tri edge.
    pub fn first(&self) -> VertexId {
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
    pub fn third(&self) -> VertexId {
        self.opp
    }

    /// Gets the current triangle id
    pub fn tri(&self) -> TriId {
        TriId(TriId::canonicalize([
            self.first(),
            self.second(),
            self.third(),
        ]))
    }

    /// Gets the current list of vertices in order
    pub fn vertices(&self) -> [VertexId; 3] {
        [self.first(), self.second(), self.third()]
    }

    /// Reverse the walker's direction so its
    /// current triangle is the opposite triangle without changing the opposite vertex.
    /// Returns None if the resulting triangle doesn't exist.
    pub fn twin(mut self) -> Option<Self> {
        self.edge = self.edge.twin();
        if self.mesh.tri(self.tri()).is_some() {
            Some(self)
        } else {
            None
        }
    }

    /// Set the current triangle to one that contains the twin
    /// of the current edge. Useful for getting the
    /// other triangle of the undirected edge in a manifold.
    pub fn on_twin_edge(self) -> Option<Self> {
        self.edge_walker().twin().and_then(|w| w.tri_walker())
    }

    /// Sets the current edge to the next one in the same triangle.
    pub fn next_edge(mut self) -> Self {
        let (edge, opp) = (EdgeId([self.second(), self.third()]), self.first());
        self.edge = edge;
        self.opp = opp;
        self
    }

    /// Sets the current edge to the previous one in the same triangle.
    pub fn prev_edge(mut self) -> Self {
        let (edge, opp) = (EdgeId([self.third(), self.first()]), self.second());
        self.edge = edge;
        self.opp = opp;
        self
    }

    /// Sets the current opposite vertex to the next one with the same edge.
    pub fn next_opp(mut self) -> Self {
        while {
            let tri = self.tri();
            self.opp = self.mesh.tris_r()[&self.tri()].link(tri, self.edge).next;
            self.mesh.tris_r()[&self.tri()].value().is_none()
        } {}
        self
    }

    /// Sets the current opposite vertex to the previous one with the same edge.
    pub fn prev_opp(mut self) -> Self {
        while {
            let tri = self.tri();
            self.opp = self.mesh.tris_r()[&self.tri()].link(tri, self.edge).prev;
            self.mesh.tris_r()[&self.tri()].value().is_none()
        } {}
        self
    }

    /// Turns this into an edge walker that starts
    /// at the current edge.
    pub fn edge_walker(self) -> EdgeWalker<'a, M> {
        EdgeWalker::new(self.mesh, self.edge)
    }

    pub fn tet_walker(self) -> Option<TetWalker<'a, M>>
    where
        <M as HasTrisIntr>::Tri: HigherTri,
        M: HasTets,
    {
        TetWalker::from_edge_vertex(self.mesh, self.edge, self.opp)
    }
}

/// An iterator over the opposite edges of the triangles of a vertex.
#[derive(Clone)]
pub struct VertexEdgeOpps<'a, M: ?Sized>
where
    M: HasVertices,
    <M as HasVerticesIntr>::Vertex: HigherVertex,
    M: HasEdges,
    <M as HasEdgesIntr>::Edge: HigherEdge,
    M: HasTris,
{
    mesh: &'a M,
    edges: VertexEdgesOut<'a, M>,
    opps: Option<EdgeVertexOpps<'a, M>>,
}

impl<'a, M: ?Sized> Iterator for VertexEdgeOpps<'a, M>
where
    M: HasVertices,
    <M as HasVerticesIntr>::Vertex: HigherVertex,
    M: HasEdges,
    <M as HasEdgesIntr>::Edge: HigherEdge,
    M: HasTris,
{
    type Item = EdgeId;

    fn next(&mut self) -> Option<Self::Item> {
        let mut next;
        while let Some(edge) = {
            next = self
                .opps
                .as_mut()
                .and_then(|iter| iter.next().map(|opp| EdgeId([iter.walker.second(), opp])));
            if next.is_none() {
                self.edges.next()
            } else {
                None
            }
        } {
            self.opps = Some(self.mesh.edge_vertex_opps(edge));
        }

        next
    }
}

/// An iterator over the opposite vertices of the triangles of an edge.
#[derive(Clone, Debug)]
pub struct EdgeVertexOpps<'a, M: ?Sized>
where
    M: HasVertices,
    <M as HasVerticesIntr>::Vertex: HigherVertex,
    M: HasEdges,
    <M as HasEdgesIntr>::Edge: HigherEdge,
    M: HasTris,
{
    pub(crate) walker: TriWalker<'a, M>,
    start_opp: VertexId,
    finished: bool,
}

impl<'a, M: ?Sized> Iterator for EdgeVertexOpps<'a, M>
where
    M: HasVertices,
    <M as HasVerticesIntr>::Vertex: HigherVertex,
    M: HasEdges,
    <M as HasEdgesIntr>::Edge: HigherEdge,
    M: HasTris,
{
    type Item = VertexId;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        let opp = self.walker.third();
        self.walker = self.walker.next_opp();
        if self.walker.third() == self.start_opp {
            self.finished = true;
        }
        Some(opp)
    }
}

#[macro_export]
#[doc(hidden)]
macro_rules! impl_index_tri {
    ($name:ident<$v:ident, $e:ident, $f: ident $(, $args:ident)*>) => {
        impl<$v, $e, $f $(, $args)*> std::ops::Index<[crate::vertex::VertexId; 3]> for $name<$v, $e, $f $(, $args)*> {
            type Output = $f;

            fn index(&self, index: [crate::vertex::VertexId; 3]) -> &Self::Output {
                self.tri(index).unwrap()
            }
        }

        impl<$v, $e, $f $(, $args)*> std::ops::IndexMut<[crate::vertex::VertexId; 3]> for $name<$v, $e, $f $(, $args)*> {
            fn index_mut(&mut self, index: [crate::vertex::VertexId; 3]) -> &mut Self::Output {
                self.tri_mut(index).unwrap()
            }
        }

        impl<$v, $e, $f $(, $args)*> std::ops::Index<crate::tri::TriId> for $name<$v, $e, $f $(, $args)*> {
            type Output = $f;

            fn index(&self, index: crate::tri::TriId) -> &Self::Output {
                self.tri(index).unwrap()
            }
        }

        impl<$v, $e, $f $(, $args)*> std::ops::IndexMut<crate::tri::TriId> for $name<$v, $e, $f $(, $args)*> {
            fn index_mut(&mut self, index: crate::tri::TriId) -> &mut Self::Output {
                self.tri_mut(index).unwrap()
            }
        }
    };
}

/// For concrete simplicial complexes with triangles
pub trait HasPositionAndTris: HasTris + HasPosition
where
    <Self::Vertex as Vertex>::V: Position,
    DefaultAllocator: Allocator<f64, HasPositionDim<Self>>,
    Self::Vertex: HigherVertex,
    Self::Edge: HigherEdge,
{
    /// Gets the positions of the vertices of an triangle
    fn tri_positions<FI: TryInto<TriId>>(&self, tri: FI) -> Option<[HasPositionPoint<Self>; 3]> {
        let tri = tri.try_into().ok()?;
        let v0 = self.position(tri.0[0])?;
        let v1 = self.position(tri.0[1])?;
        let v2 = self.position(tri.0[2])?;
        Some([v0, v1, v2])
    }
}

pub(crate) mod internal {
    use fnv::FnvHashMap;

    use super::TriId;
    use crate::edge::internal::{HasEdges as HasEdgesIntr, HigherEdge, Link};
    use crate::edge::EdgeId;
    use crate::vertex::internal::HigherVertex;
    use crate::vertex::VertexId;

    #[macro_export]
    #[doc(hidden)]
    macro_rules! impl_tri {
        ($name:ident<$f:ident>, new |$id:ident, $links:ident, $value:ident| $new:expr) => {
            impl<$f> crate::tri::internal::Tri for $name<$f> {
                type F = $f;

                fn new(
                    $id: crate::vertex::VertexId,
                    $links: [crate::edge::internal::Link<crate::vertex::VertexId>; 3],
                    $value: Option<Self::F>,
                ) -> Self {
                    $new
                }

                fn links(&self) -> &[crate::edge::internal::Link<crate::vertex::VertexId>; 3] {
                    &self.links
                }

                fn links_mut(
                    &mut self,
                ) -> &mut [crate::edge::internal::Link<crate::vertex::VertexId>; 3] {
                    &mut self.links
                }

                fn to_value(self) -> Option<Self::F> {
                    self.value
                }

                fn value(&self) -> &Option<Self::F> {
                    &self.value
                }

                fn value_mut(&mut self) -> &mut Option<Self::F> {
                    &mut self.value
                }
            }
        };
    }

    #[macro_export]
    #[doc(hidden)]
    macro_rules! impl_higher_tri {
        ($name:ident<$f:ident>) => {
            impl<$f> crate::tri::internal::HigherTri for $name<$f> {
                fn tet_opp(&self) -> crate::vertex::VertexId {
                    self.tet_opp
                }

                fn tet_opp_mut(&mut self) -> &mut crate::vertex::VertexId {
                    &mut self.tet_opp
                }
            }
        };
    }

    #[macro_export]
    #[doc(hidden)]
    macro_rules! impl_has_tris {
        ($name:ident<$v:ident, $e:ident, $f:ident $(, $args:ident)*>, $tri:ident) => {
            impl<$v, $e, $f $(, $args)*> crate::tri::internal::HasTris for $name<$v, $e, $f $(, $args)*> {
                type Tri = $tri<$f>;

                fn tris_r(&self) -> &FnvHashMap<crate::tri::TriId, Self::Tri> {
                    &self.tris
                }

                fn tris_r_mut(&mut self) -> &mut FnvHashMap<crate::tri::TriId, Self::Tri> {
                    &mut self.tris
                }

                fn num_tris_r(&self) -> usize {
                    self.num_tris
                }

                fn num_tris_r_mut(&mut self) -> &mut usize {
                    &mut self.num_tris
                }
            }
        };
    }

    /// Triangle storage
    pub trait Tri {
        type F;

        /// Takes the vertex id of the source, in case
        /// the triangle needs to store a dummy value for the opposite vertex
        /// of the tetrahedron.
        fn new(id: VertexId, links: [Link<VertexId>; 3], value: Option<Self::F>) -> Self;

        fn links(&self) -> &[Link<VertexId>; 3];

        fn links_mut(&mut self) -> &mut [Link<VertexId>; 3];

        fn to_value(self) -> Option<Self::F>;

        fn value(&self) -> &Option<Self::F>;

        fn value_mut(&mut self) -> &mut Option<Self::F>;

        fn link(&self, id: TriId, edge: EdgeId) -> &Link<VertexId> {
            &self.links()[id.index(edge.0[0])]
        }

        fn link_mut(&mut self, id: TriId, edge: EdgeId) -> &mut Link<VertexId> {
            &mut self.links_mut()[id.index(edge.0[0])]
        }
    }

    pub trait HigherTri: Tri {
        fn tet_opp(&self) -> VertexId;

        fn tet_opp_mut(&mut self) -> &mut VertexId;
    }

    pub trait HasTris: HasEdgesIntr
    where
        Self::Vertex: HigherVertex,
        Self::Edge: HigherEdge,
    {
        type Tri: Tri;

        fn tris_r(&self) -> &FnvHashMap<TriId, Self::Tri>;

        fn tris_r_mut(&mut self) -> &mut FnvHashMap<TriId, Self::Tri>;

        fn num_tris_r(&self) -> usize;

        fn num_tris_r_mut(&mut self) -> &mut usize;
    }

    /// Removes higher-order simplexes that contain some triangle
    pub trait RemoveTriHigher: HasTris
    where
        Self::Vertex: HigherVertex,
        Self::Edge: HigherEdge,
    {
        fn remove_tri_higher(&mut self, tri: TriId);
    }

    /// Clears higher-order simplexes
    pub trait ClearTrisHigher: HasTris
    where
        Self::Vertex: HigherVertex,
        Self::Edge: HigherEdge,
    {
        fn clear_tris_higher(&mut self);
    }
}
