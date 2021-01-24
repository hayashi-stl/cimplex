//! Traits and structs related to triangles

use nalgebra::{allocator::Allocator, DefaultAllocator};
#[cfg(feature = "serde_")]
use serde::{Deserialize, Serialize};
use std::collections::hash_map;
use std::convert::{TryFrom, TryInto};
use std::iter::{Filter, Map};
use typenum::{Bit, B0, B1};
use std::fmt::Debug;

use crate::iter::{IteratorExt, MapWith};
use crate::tri::internal::HasTris as HasTrisIntr;
use crate::vertex::internal::{HasVertices as HasVerticesIntr, HigherVertex};
use crate::vertex::HasVertices;
use crate::{edge::internal::HigherEdge, vertex::VertexId};
use crate::{
    edge::internal::{Edge, HasEdges as HasEdgesIntr, Link},
    tet::{HasTets, HasTetsWalker, TetWalker, TetWalk},
};
use crate::{
    edge::{EdgeId, EdgeWalker, HasEdges, VertexEdgesOut},
    vertex::{internal::Vertex, HasPosition, HasPositionDim, HasPositionPoint, Position},
};

use internal::{ClearTrisHigher, RemoveTriHigher, Tri};

use self::internal::{HigherTri, NonManifoldTri, ManifoldTri};

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

pub type HasTrisWalker<'a, M> = TriWalker<'a, M, <<M as HasTrisIntr>::Tri as Tri>::Manifold>;

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
            .map::<_, TriMapFn<Self::Tri>>(|(id, f)| (id, f.value().unwrap()))
    }

    /// Iterates mutably over the triangles of this mesh.
    /// Gives (id, value) pairs
    fn tris_mut(&mut self) -> TrisMut<Self::Tri> {
        self.tris_r_mut()
            .iter_mut()
            .filter::<TriFilterFnMut<Self::Tri>>(|(_, f)| f.value().is_some())
            .map::<_, TriMapFnMut<Self::Tri>>(|(id, f)| (id, f.value_mut().unwrap()))
    }

    /// Gets the value of the triangle at a specific id.
    /// Returns None if not found.
    fn tri<FI: TryInto<TriId>>(&self, id: FI) -> Option<&F!()> {
        id.try_into()
            .ok()
            .and_then(|id| self.tris_r().get(&id))
            .and_then(|f| f.value())
    }

    /// Gets the value of the triangle at a specific id mutably.
    /// Returns None if not found.
    fn tri_mut<FI: TryInto<TriId>>(&mut self, id: FI) -> Option<&mut F!()> {
        id.try_into()
            .ok()
            .and_then(move |id| self.tris_r_mut().get_mut(&id))
            .and_then(|f| f.value_mut())
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
    fn vertex_tris(&self, vertex: VertexId) -> VertexTris<Self>
    where
        for<'b> HasTrisWalker<'b, Self>: TriWalk<'b, Mesh = Self>,
    {
        self.vertex_edge_opps(vertex)
            .map_with(vertex, |vertex, opp| {
                TriId(TriId::canonicalize([vertex, opp.0[0], opp.0[1]]))
            })
    }

    /// Iterates over the opposite vertices of the triangles that an edge is part of.
    /// The edge must exist.
    fn edge_vertex_opps<EI: TryInto<EdgeId>>(&self, edge: EI) -> EdgeVertexOpps<Self>
    where
        for<'b> HasTrisWalker<'b, Self>: TriWalk<'b, Mesh = Self>,
    {
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
    fn edge_tris<EI: TryInto<EdgeId>>(&self, edge: EI) -> EdgeTris<Self>
    where
        for<'b> HasTrisWalker<'b, Self>: TriWalk<'b, Mesh = Self>,
    {
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
    ) -> Option<F!()>;

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
    fn remove_tri<FI: TryInto<TriId>>(&mut self, id: FI) -> Option<F!()>
    where
        for<'b> HasTrisWalker<'b, Self>: TriWalk<'b, Mesh = Self>,
    {
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
    fn remove_tri_keep_edges<FI: TryInto<TriId>>(&mut self, id: FI) -> Option<F!()>;

    /// Removes a list of triangles.
    fn remove_tris<FI: TryInto<TriId>, I: IntoIterator<Item = FI>>(&mut self, iter: I)
    where
        for<'b> HasTrisWalker<'b, Self>: TriWalk<'b, Mesh = Self>,
    {
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
    fn retain_tris<P: FnMut(TriId, &F!()) -> bool>(&mut self, mut predicate: P)
    where
        for<'b> HasTrisWalker<'b, Self>: TriWalk<'b, Mesh = Self>,
    {
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
    fn tri_walker_from_edge<EI: TryInto<EdgeId>>(&self, edge: EI) -> Option<HasTrisWalker<Self>>
    where
        for<'b> HasTrisWalker<'b, Self>: TriWalk<'b, Mesh = Self>,
    {
        TriWalker::from_edge(self, edge)
    }

    /// Gets a triangle walker that starts at the given edge with the given opposite vertex.
    /// They must actually exist.
    fn tri_walker_from_edge_vertex<EI: TryInto<EdgeId>>(
        &self,
        edge: EI,
        opp: VertexId,
    ) -> HasTrisWalker<Self>
    where
        for<'b> HasTrisWalker<'b, Self>: TriWalk<'b, Mesh = Self>,
    {
        HasTrisWalker::<Self>::new(self, edge, opp)
    }

    /// Gets a triangle walker that starts at the given triangle.
    /// It must actually exist.
    /// Be warned that this does not preserve the order of the vertices
    /// because the triangle id is canonicalized.
    fn tri_walker_from_tri<FI: TryInto<TriId>>(&self, tri: FI) -> HasTrisWalker<Self>
    where
        for<'b> HasTrisWalker<'b, Self>: TriWalk<'b, Mesh = Self>,
    {
        let tri = tri.try_into().ok().unwrap();
        HasTrisWalker::<Self>::new(self, tri.edges()[0], tri.0[2])
    }
}

/// Triangle walker generic over whether the mesh has the "manifold" restriction
pub trait TriWalk<'m>
where
    Self::Mesh: HasVertices,
    <Self::Mesh as HasVerticesIntr>::Vertex: HigherVertex,
    Self::Mesh: HasEdges,
    <Self::Mesh as HasEdgesIntr>::Edge: HigherEdge,
    Self: Sized,
{
    type Mesh: HasTris + ?Sized + 'm;

    #[doc(hidden)]
    fn new<EI: TryInto<EdgeId>>(mesh: &'m Self::Mesh, edge: EI, opp: VertexId) -> Self;

    #[doc(hidden)]
    fn from_edge<EI: TryInto<EdgeId>>(mesh: &'m Self::Mesh, edge: EI) -> Option<Self>;

    /// A walker that will not be used
    #[doc(hidden)]
    fn dummy(mesh: &'m Self::Mesh) -> Self {
        Self::new(mesh, EdgeId::dummy(), VertexId::dummy())
    }

    /// Get the mesh that the walker navigates
    fn mesh(&self) -> &'m Self::Mesh;

    /// Get the current vertex id,
    /// which is the source of the current tri edge.
    fn first(&self) -> VertexId {
        self.edge().0[0]
    }

    /// Get the vertex id of the target of the current tri edge.
    fn second(&self) -> VertexId {
        self.edge().0[1]
    }

    /// Gets the current edge id
    fn edge(&self) -> EdgeId;

    #[doc(hidden)]
    fn edge_mut(&mut self) -> &mut EdgeId;

    /// Gets the opposite vertex of the current triangle
    fn third(&self) -> VertexId;

    #[doc(hidden)]
    fn third_mut(&mut self) -> &mut VertexId;

    /// Gets the current triangle id
    fn tri(&self) -> TriId {
        TriId(TriId::canonicalize([
            self.first(),
            self.second(),
            self.third(),
        ]))
    }

    /// Gets the current list of vertices in order
    fn vertices(&self) -> [VertexId; 3] {
        [self.first(), self.second(), self.third()]
    }

    /// Reverse the walker's direction so its
    /// current triangle is the opposite triangle without changing the opposite vertex.
    /// Returns None if the resulting triangle doesn't exist.
    fn twin(mut self) -> Option<Self> {
        *self.edge_mut() = self.edge().twin();
        if self.mesh().tri(self.tri()).is_some() {
            Some(self)
        } else {
            None
        }
    }

    /// Set the current triangle to one that contains the twin
    /// of the current edge. Useful for getting the
    /// other triangle of the undirected edge in a manifold.
    fn on_twin_edge(self) -> Option<Self>;

    /// Sets the current edge to the next one in the same triangle.
    fn next_edge(mut self) -> Self {
        let (edge, opp) = (EdgeId([self.second(), self.third()]), self.first());
        *self.edge_mut() = edge;
        *self.third_mut() = opp;
        self
    }

    /// Sets the current edge to the previous one in the same triangle.
    fn prev_edge(mut self) -> Self {
        let (edge, opp) = (EdgeId([self.third(), self.first()]), self.second());
        *self.edge_mut() = edge;
        *self.third_mut() = opp;
        self
    }

    /// Sets the current opposite vertex to the next one with the same edge.
    fn next_opp(self) -> Self;

    /// Sets the current opposite vertex to the previous one with the same edge.
    fn prev_opp(self) -> Self;

    /// Turns this into an edge walker that starts
    /// at the current edge.
    fn edge_walker(self) -> EdgeWalker<'m, Self::Mesh> {
        EdgeWalker::new(self.mesh(), self.edge())
    }

    fn tet_walker(self) -> Option<HasTetsWalker<'m, Self::Mesh>>
    where
        <Self::Mesh as HasTrisIntr>::Tri: HigherTri,
        Self::Mesh: HasTets,
        HasTetsWalker<'m, Self::Mesh>: TetWalk<'m, Mesh = Self::Mesh>,
    {
        HasTetsWalker::from_edge_vertex(self.mesh(), self.edge(), self.third())
    }
}
/// A walker for navigating a simplicial complex by triangle
#[derive(Debug)]
pub struct TriWalker<'a, M: ?Sized, MF: Bit + Debug>
where
    M: HasVertices,
    <M as HasVerticesIntr>::Vertex: HigherVertex,
    M: HasEdges,
    <M as HasEdgesIntr>::Edge: HigherEdge,
    M: HasTris,
    <M as HasTrisIntr>::Tri: Tri<Manifold = MF>,
{
    mesh: &'a M,
    edge: EdgeId,
    opp: VertexId,
}

impl<'a, M: ?Sized, MF: Bit + Debug> Clone for TriWalker<'a, M, MF>
where
    M: HasVertices,
    <M as HasVerticesIntr>::Vertex: HigherVertex,
    M: HasEdges,
    <M as HasEdgesIntr>::Edge: HigherEdge,
    M: HasTris,
    <M as HasTrisIntr>::Tri: Tri<Manifold = MF>,
{
    fn clone(&self) -> Self {
        Self {
            mesh: self.mesh,
            edge: self.edge,
            opp: self.opp,
        }
    }
}

impl<'a, M: ?Sized, MF: Bit + Debug> Copy for TriWalker<'a, M, MF>
where
    M: HasVertices,
    <M as HasVerticesIntr>::Vertex: HigherVertex,
    M: HasEdges,
    <M as HasEdgesIntr>::Edge: HigherEdge,
    M: HasTris,
    <M as HasTrisIntr>::Tri: Tri<Manifold = MF>,
{
}

impl<'a, M: ?Sized> TriWalk<'a> for TriWalker<'a, M, B1>
where
    M: HasVertices,
    <M as HasVerticesIntr>::Vertex: HigherVertex,
    M: HasEdges,
    <M as HasEdgesIntr>::Edge: HigherEdge,
    M: HasTris,
    <M as HasTrisIntr>::Tri: ManifoldTri + Tri<Manifold = B1>,
{
    type Mesh = M;

    fn new<EI: TryInto<EdgeId>>(mesh: &'a M, edge: EI, opp: VertexId) -> Self {
        Self {
            mesh,
            edge: edge.try_into().ok().unwrap(),
            opp,
        }
    }

    fn from_edge<EI: TryInto<EdgeId>>(mesh: &'a M, edge: EI) -> Option<Self> {
        let edge = edge.try_into().ok().unwrap();
        let opp = mesh.edges_r()[&edge].tri_opp();

        // Ensure that the triangle exists
        let _: TriId = [edge.0[0], edge.0[1], opp].try_into().ok()?;
        Some(Self::new(mesh, edge, opp))
    }

    /// Get the mesh that the walker navigates
    fn mesh(&self) -> &'a M {
        self.mesh
    }

    /// Gets the current edge id
    fn edge(&self) -> EdgeId {
        self.edge
    }

    fn edge_mut(&mut self) -> &mut EdgeId {
        &mut self.edge
    }

    /// Gets the opposite vertex of the current triangle
    fn third(&self) -> VertexId {
        self.opp
    }

    fn third_mut(&mut self) -> &mut VertexId {
        &mut self.opp
    }

    /// Set the current triangle to one that contains the twin
    /// of the current edge. Useful for getting the
    /// other triangle of the undirected edge in a manifold.
    fn on_twin_edge(self) -> Option<Self> {
        self.edge_walker().twin().and_then(|w| w.tri_walker())
    }

    /// Sets the current opposite vertex to the next one with the same edge.
    fn next_opp(mut self) -> Self {
        self
    }

    /// Sets the current opposite vertex to the previous one with the same edge.
    fn prev_opp(mut self) -> Self {
        self
    }
}

impl<'a, M: ?Sized> TriWalk<'a> for TriWalker<'a, M, B0>
where
    M: HasVertices,
    <M as HasVerticesIntr>::Vertex: HigherVertex,
    M: HasEdges,
    <M as HasEdgesIntr>::Edge: HigherEdge,
    M: HasTris,
    <M as HasTrisIntr>::Tri: NonManifoldTri + Tri<Manifold = B0>,
{
    type Mesh = M;

    fn new<EI: TryInto<EdgeId>>(mesh: &'a M, edge: EI, opp: VertexId) -> Self {
        Self {
            mesh,
            edge: edge.try_into().ok().unwrap(),
            opp,
        }
    }

    fn from_edge<EI: TryInto<EdgeId>>(mesh: &'a M, edge: EI) -> Option<Self> {
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

    /// Get the mesh that the walker navigates
    fn mesh(&self) -> &'a M {
        self.mesh
    }

    /// Gets the current edge id
    fn edge(&self) -> EdgeId {
        self.edge
    }

    fn edge_mut(&mut self) -> &mut EdgeId {
        &mut self.edge
    }

    /// Gets the opposite vertex of the current triangle
    fn third(&self) -> VertexId {
        self.opp
    }

    fn third_mut(&mut self) -> &mut VertexId {
        &mut self.opp
    }

    /// Set the current triangle to one that contains the twin
    /// of the current edge. Useful for getting the
    /// other triangle of the undirected edge in a manifold.
    fn on_twin_edge(self) -> Option<Self> {
        self.edge_walker().twin().and_then(|w| w.tri_walker())
    }

    /// Sets the current opposite vertex to the next one with the same edge.
    fn next_opp(mut self) -> Self {
        while {
            let tri = self.tri();
            self.opp = self.mesh.tris_r()[&self.tri()].link(tri, self.edge).next;
            self.mesh.tris_r()[&self.tri()].value().is_none()
        } {}
        self
    }

    /// Sets the current opposite vertex to the previous one with the same edge.
    fn prev_opp(mut self) -> Self {
        while {
            let tri = self.tri();
            self.opp = self.mesh.tris_r()[&self.tri()].link(tri, self.edge).prev;
            self.mesh.tris_r()[&self.tri()].value().is_none()
        } {}
        self
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
    for<'b> HasTrisWalker<'b, M>: TriWalk<'b, Mesh = M>,
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
    pub(crate) walker: HasTrisWalker<'a, M>,
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
    for<'b> HasTrisWalker<'b, M>: TriWalk<'b, Mesh = M>,
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
    ($name:ident<$v:ident, $e:ident, $f: ident $(, $args:ident)*> $(where $($wh:tt)*)?) => {
        impl<$v, $e, $f $(, $args)*> std::ops::Index<[crate::vertex::VertexId; 3]> for $name<$v, $e, $f $(, $args)*>
        $(where $($wh)*)?
        {
            type Output = $f;

            fn index(&self, index: [crate::vertex::VertexId; 3]) -> &Self::Output {
                self.tri(index).unwrap()
            }
        }

        impl<$v, $e, $f $(, $args)*> std::ops::IndexMut<[crate::vertex::VertexId; 3]> for $name<$v, $e, $f $(, $args)*>
        $(where $($wh)*)?
        {
            fn index_mut(&mut self, index: [crate::vertex::VertexId; 3]) -> &mut Self::Output {
                self.tri_mut(index).unwrap()
            }
        }

        impl<$v, $e, $f $(, $args)*> std::ops::Index<crate::tri::TriId> for $name<$v, $e, $f $(, $args)*>
        $(where $($wh)*)?
        {
            type Output = $f;

            fn index(&self, index: crate::tri::TriId) -> &Self::Output {
                self.tri(index).unwrap()
            }
        }

        impl<$v, $e, $f $(, $args)*> std::ops::IndexMut<crate::tri::TriId> for $name<$v, $e, $f $(, $args)*>
        $(where $($wh)*)?
        {
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
    use std::convert::TryInto;

    use fnv::FnvHashMap;
    use typenum::{Bit, B0, B1};

    use super::{TriId, HasTrisWalker, TriWalk};
    use crate::edge::internal::{Edge, HasEdges as HasEdgesIntr, HigherEdge, Link};
    use crate::edge::EdgeId;
    use crate::vertex::internal::HigherVertex;
    use crate::vertex::VertexId;

    #[macro_export]
    #[doc(hidden)]
    macro_rules! impl_manifold_tri {
        ($name:ident<$f:ident>, new |$value:ident| $new:expr) => {
            impl<$f> crate::tri::internal::Tri for $name<$f> {
                type F = $f;
                type Manifold = typenum::B1;

                fn to_value(self) -> Option<Self::F> {
                    Some(self.value)
                }

                fn value(&self) -> Option<&Self::F> {
                    Some(&self.value)
                }

                fn value_mut(&mut self) -> Option<&mut Self::F> {
                    Some(&mut self.value)
                }
            }

            impl<$f> crate::tri::internal::ManifoldTri for $name<$f> {
                fn new($value: F) -> Self {
                    $new
                }
            }
        };
    }

    #[macro_export]
    #[doc(hidden)]
    macro_rules! impl_non_manifold_tri {
        ($name:ident<$f:ident>, with_links |$id:ident, $links:ident, $value:ident| $new:expr) => {
            impl<$f> crate::tri::internal::Tri for $name<$f> {
                type F = $f;
                type Manifold = typenum::B0;

                fn to_value(self) -> Option<Self::F> {
                    self.value
                }

                fn value(&self) -> Option<&Self::F> {
                    self.value.as_ref()
                }

                fn value_mut(&mut self) -> Option<&mut Self::F> {
                    self.value.as_mut()
                }
            }

            impl<$f> crate::tri::internal::NonManifoldTri for $name<$f> {
                fn with_links(
                    $id: crate::vertex::VertexId,
                    $links: [crate::edge::internal::Link<crate::vertex::VertexId>; 3],
                    $value: Option<Self::F>,
                ) -> Self {
                    $new
                }

                fn option_value_mut(&mut self) -> &mut Option<Self::F> {
                    &mut self.value
                }

                fn links(&self) -> &[crate::edge::internal::Link<crate::vertex::VertexId>; 3] {
                    &self.links
                }

                fn links_mut(
                    &mut self,
                ) -> &mut [crate::edge::internal::Link<crate::vertex::VertexId>; 3] {
                    &mut self.links
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
    macro_rules! impl_has_tris_manifold {
        ($name:ident<$v:ident, $e:ident, $f:ident $(, $args:ident)*>, $tri:ident $(where $($wh:tt)*)?) => {
            impl<$v, $e, $f $(, $args)*> crate::tri::internal::HasTris for $name<$v, $e, $f $(, $args)*>
            $(where $($wh)*)?
            {
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

            impl<$v, $e, $f $(, $args)*> crate::tri::HasTris for $name<$v, $e, $f $(, $args)*>
            $(where $($wh)*)?
            {
                fn add_tri<FI: std::convert::TryInto<TriId>>(
                    &mut self,
                    vertices: FI,
                    value: <Self::Tri as crate::tri::internal::Tri>::F,
                    edge_value: impl Fn() -> <Self::Edge as crate::edge::internal::Edge>::E,
                ) -> Option<<Self::Tri as crate::tri::internal::Tri>::F> {
                    crate::tri::internal::add_tri_manifold(self, vertices, value, edge_value)
                }

                fn remove_tri_keep_edges<FI: std::convert::TryInto<TriId>>(
                    &mut self,
                    id: FI,
                ) -> Option<<Self::Tri as crate::tri::internal::Tri>::F> {
                    crate::tri::internal::remove_tri_manifold(self, id)
                }
            }
        };
    }

    #[macro_export]
    #[doc(hidden)]
    macro_rules! impl_has_tris_non_manifold {
        ($name:ident<$v:ident, $e:ident, $f:ident $(, $args:ident)*>, $tri:ident $(where $($wh:tt)*)?) => {
            impl<$v, $e, $f $(, $args)*> crate::tri::internal::HasTris for $name<$v, $e, $f $(, $args)*>
            $(where $($wh)*)?
            {
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

            impl<$v, $e, $f $(, $args)*> crate::tri::HasTris for $name<$v, $e, $f $(, $args)*>
            $(where $($wh)*)?
            {
                fn add_tri<FI: std::convert::TryInto<TriId>>(
                    &mut self,
                    vertices: FI,
                    value: <Self::Tri as crate::tri::internal::Tri>::F,
                    edge_value: impl Fn() -> <Self::Edge as crate::edge::internal::Edge>::E,
                ) -> Option<<Self::Tri as crate::tri::internal::Tri>::F> {
                    crate::tri::internal::add_tri_non_manifold(self, vertices, value, edge_value)
                }

                fn remove_tri_keep_edges<FI: std::convert::TryInto<TriId>>(
                    &mut self,
                    id: FI,
                ) -> Option<<Self::Tri as crate::tri::internal::Tri>::F> {
                    crate::tri::internal::remove_tri_non_manifold(self, id)
                }
            }
        };
    }

    /// Triangle storage
    pub trait Tri {
        type F;
        type Manifold: Bit + std::fmt::Debug;

        fn to_value(self) -> Option<Self::F>;

        fn value(&self) -> Option<&Self::F>;

        fn value_mut(&mut self) -> Option<&mut Self::F>;
    }

    /// Triangle storage
    pub trait ManifoldTri: Tri {
        fn new(value: Self::F) -> Self;
    }

    /// Triangle storage
    pub trait NonManifoldTri: Tri {
        fn with_links(id: VertexId, links: [Link<VertexId>; 3], value: Option<Self::F>) -> Self;

        fn option_value_mut(&mut self) -> &mut Option<Self::F>;

        fn links(&self) -> &[Link<VertexId>; 3];

        fn links_mut(&mut self) -> &mut [Link<VertexId>; 3];

        fn link(&self, id: TriId, edge: EdgeId) -> &Link<VertexId> {
            &self.links()[id.index(edge.0[0])]
        }

        fn link_mut(&mut self, id: TriId, edge: EdgeId) -> &mut Link<VertexId> {
            &mut self.links_mut()[id.index(edge.0[0])]
        }
    }

    pub trait HigherTri: NonManifoldTri {
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

    pub(crate) fn add_tri_manifold<M: super::HasTris, FI: TryInto<TriId>>(
        mesh: &mut M,
        vertices: FI,
        value: <M::Tri as Tri>::F,
        edge_value: impl Fn() -> <M::Edge as Edge>::E,
    ) -> Option<<M::Tri as Tri>::F>
    where
        M::Vertex: HigherVertex,
        M::Edge: HigherEdge,
        M::Tri: ManifoldTri,
        for<'b> HasTrisWalker<'b, M>: TriWalk<'b, Mesh = M>,
    {
        let id = vertices.try_into().ok().unwrap();

        for edge in &id.edges() {
            if mesh.edge(*edge).is_none() {
                mesh.add_edge(*edge, edge_value());
            }
        }

        // Can't use entry().or_insert() because that would cause a
        // mutable borrow and an immutable borrow to exist at the same time
        if let Some(tri) = mesh.tris_r_mut().get_mut(&id) {
            Some(std::mem::replace(tri.value_mut().unwrap(), value))
        } else {
            *mesh.num_tris_r_mut() += 1;

            for (i, (edge, opp)) in id.edges_and_opp().iter().enumerate() {
                let target = mesh.edges_r()[edge].tri_opp();

                if target != edge.0[0] {
                    mesh.remove_tri([edge.0[0], edge.0[1], target]);
                }
                *mesh.edges_r_mut().get_mut(edge).unwrap().tri_opp_mut() = *opp;
            }

            mesh.tris_r_mut().insert(id, ManifoldTri::new(value));
            None
        }
    }

    pub(crate) fn add_tri_non_manifold<M: super::HasTris, FI: TryInto<TriId>>(
        mesh: &mut M,
        vertices: FI,
        value: <M::Tri as Tri>::F,
        edge_value: impl Fn() -> <M::Edge as Edge>::E,
    ) -> Option<<M::Tri as Tri>::F>
    where
        M::Vertex: HigherVertex,
        M::Edge: HigherEdge,
        M::Tri: NonManifoldTri,
    {
        let id = vertices.try_into().ok().unwrap();

        for edge in &id.edges() {
            if mesh.edge(*edge).is_none() {
                mesh.add_edge(*edge, edge_value());
            }
        }

        // Can't use entry().or_insert() because that would cause a
        // mutable borrow and an immutable borrow to exist at the same time
        if let Some(tri) = mesh.tris_r_mut().get_mut(&id) {
            let old = tri.option_value_mut().take();
            *tri.option_value_mut() = Some(value);
            if old.is_none() {
                *mesh.num_tris_r_mut() += 1;
            }
            old
        } else {
            *mesh.num_tris_r_mut() += 1;

            let mut insert_tri = |id: TriId, value: Option<<M::Tri as Tri>::F>| {
                let mut opps = [Link::dummy(VertexId::dummy); 3];

                for (i, (edge, opp)) in id.edges_and_opp().iter().enumerate() {
                    let target = mesh.edges_r()[edge].tri_opp();

                    let (prev, next) = if target == edge.0[0] {
                        // First tri from edge
                        *mesh.edges_r_mut().get_mut(edge).unwrap().tri_opp_mut() = *opp;
                        (*opp, *opp)
                    } else {
                        let side = [edge.0[0], edge.0[1], target].try_into().ok().unwrap();
                        let prev = mesh.tris_r()[&side].link(side, *edge).prev;
                        let next = target;
                        let prev_tri = id.with_opp(*edge, prev);
                        let next_tri = id.with_opp(*edge, next);
                        mesh.tris_r_mut()
                            .get_mut(&prev_tri)
                            .unwrap()
                            .link_mut(prev_tri, *edge)
                            .next = *opp;
                        mesh.tris_r_mut()
                            .get_mut(&next_tri)
                            .unwrap()
                            .link_mut(next_tri, *edge)
                            .prev = *opp;
                        (prev, next)
                    };

                    opps[i] = Link::new(prev, next);
                }

                mesh.tris_r_mut()
                    .insert(id, NonManifoldTri::with_links(id.0[0], opps, value));
            };

            insert_tri(id, Some(value));
            insert_tri(id.twin(), None);
            None
        }
    }

    pub(crate) fn remove_tri_manifold<M: super::HasTris, FI: TryInto<TriId>>(
        mesh: &mut M,
        id: FI,
    ) -> Option<<M::Tri as Tri>::F>
    where
        M::Vertex: HigherVertex,
        M::Edge: HigherEdge,
        M::Tri: ManifoldTri,
    {
        let id = match id.try_into() {
            Ok(id) => id,
            Err(_) => return None,
        };

        if mesh.tri(id).is_some() {
            mesh.remove_tri_higher(id);
        }

        match mesh.tri(id) {
            Some(_) => {
                *mesh.num_tris_r_mut() -= 1;

                for (i, (edge, opp)) in id.edges_and_opp().iter().enumerate() {
                    // Because of the "manifold" condition, this has to be the last triangle from the edge
                    *mesh.edges_r_mut().get_mut(&edge).unwrap().tri_opp_mut() = edge.0[0];
                }

                mesh.tris_r_mut().remove(&id).and_then(|f| f.to_value())
            }

            // Twin isn't in map, and neither is the tri to remove
            None => None,
        }
    }

    pub(crate) fn remove_tri_non_manifold<M: super::HasTris, FI: TryInto<TriId>>(
        mesh: &mut M,
        id: FI,
    ) -> Option<<M::Tri as Tri>::F>
    where
        M::Vertex: HigherVertex,
        M::Edge: HigherEdge,
        M::Tri: NonManifoldTri,
    {
        let id = match id.try_into() {
            Ok(id) => id,
            Err(_) => return None,
        };

        if mesh.tri(id).is_some() {
            mesh.remove_tri_higher(id);
        }

        match mesh.tris_r().get(&id.twin()).map(|f| f.value()) {
            // Twin actually exists; just set value to None
            Some(Some(_)) => {
                let old = mesh
                    .tris_r_mut()
                    .get_mut(&id)
                    .unwrap()
                    .option_value_mut()
                    .take();
                if old.is_some() {
                    *mesh.num_tris_r_mut() -= 1;
                }
                old
            }

            // Twin is phantom, so remove both tri and twin from map
            Some(None) => {
                // Twin is phantom, so this tri actually exists.
                *mesh.num_tris_r_mut() -= 1;

                let mut delete_tri = |id: TriId| {
                    for (i, (edge, opp)) in id.edges_and_opp().iter().enumerate() {
                        let tri = &mesh.tris_r()[&id];
                        let prev = tri.links()[i].prev;
                        let next = tri.links()[i].next;
                        let prev_tri = id.with_opp(*edge, prev);
                        let next_tri = id.with_opp(*edge, next);
                        mesh.tris_r_mut()
                            .get_mut(&prev_tri)
                            .unwrap()
                            .link_mut(prev_tri, *edge)
                            .next = next;
                        mesh.tris_r_mut()
                            .get_mut(&next_tri)
                            .unwrap()
                            .link_mut(next_tri, *edge)
                            .prev = prev;

                        let source = mesh.edges_r_mut().get_mut(&edge).unwrap();
                        if *opp == next {
                            // this was the last tri from the edge
                            *source.tri_opp_mut() = edge.0[0];
                        } else if *opp == source.tri_opp() {
                            *source.tri_opp_mut() = next;
                        }
                    }

                    mesh.tris_r_mut().remove(&id).and_then(|f| f.to_value())
                };

                delete_tri(id.twin());
                delete_tri(id)
            }

            // Twin isn't in map, and neither is the tri to remove
            None => None,
        }
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
