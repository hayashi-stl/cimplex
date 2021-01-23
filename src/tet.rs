//! Traits and structs related to tetrahedrons

use nalgebra::{allocator::Allocator, DefaultAllocator};
#[cfg(feature = "serde_")]
use serde::{Deserialize, Serialize};
use std::convert::{TryFrom, TryInto};
use std::iter::{Filter, Map};
use std::{collections::hash_map, fmt::Debug};
use typenum::{Bit, B0, B1};

use crate::edge::{EdgeId, HasEdges, VertexEdgesOut};
use crate::iter::{IteratorExt, MapWith};
use crate::tri::{
    internal::{HasTris as HasTrisIntr, HigherTri, Tri},
    EdgeVertexOpps,
};
use crate::tri::{HasTris, TriId, HasTrisWalker, TriWalk};
use crate::vertex::internal::{HasVertices as HasVerticesIntr, HigherVertex};
use crate::vertex::HasVertices;
use crate::{edge::internal::HigherEdge, vertex::VertexId};
use crate::{
    edge::internal::{Edge, HasEdges as HasEdgesIntr},
    vertex::{internal::Vertex, HasPosition, HasPositionDim, HasPositionPoint, Position},
};

use internal::{
    ClearTetsHigher, HasTets as HasTetsIntr, ManifoldTet, NonManifoldTet, RemoveTetHigher, Tet,
};

/// An tetrahedron id is just the tetrahedrons's vertices in winding order,
/// with the smallest two indexes first.
/// No two vertices are allowed to be the same.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde_", derive(Serialize, Deserialize))]
pub struct TetId(pub(crate) [VertexId; 4]);

impl TryFrom<[VertexId; 4]> for TetId {
    type Error = &'static str;

    fn try_from(mut v: [VertexId; 4]) -> Result<Self, Self::Error> {
        v = Self::canonicalize(v);

        if v[0] == v[1] || v[1] == v[2] || v[3] == v[1] || v[2] == v[3] {
            Err("Vertices are not allowed to be the same")
        } else {
            Ok(TetId(v))
        }
    }
}

impl TetId {
    fn canonicalize(mut v: [VertexId; 4]) -> [VertexId; 4] {
        let min_pos = v
            .iter()
            .enumerate()
            .min_by_key(|(_, value)| **value)
            .unwrap()
            .0;
        v.swap(0, min_pos);

        // Swap is odd permutation; fix it
        if min_pos != 0 {
            v.swap(2, 3);
        }

        let min_pos = v[1..]
            .iter()
            .enumerate()
            .min_by_key(|(_, value)| **value)
            .unwrap()
            .0;
        v[1..].rotate_left(min_pos);
        v
    }

    /// Gets the vertices that this tet id is made of
    pub fn vertices(self) -> [VertexId; 4] {
        self.0
    }

    /// Gets the edges of the tetrahedron in no particular order.
    /// Beware that both directions of each edge are included,
    /// so there are 12 edges, not 6.
    pub fn edges(self) -> [EdgeId; 12] {
        let v = self.0;
        [
            EdgeId([v[0], v[1]]),
            EdgeId([v[1], v[2]]),
            EdgeId([v[2], v[0]]),
            EdgeId([v[3], v[2]]),
            EdgeId([v[2], v[1]]),
            EdgeId([v[1], v[3]]),
            EdgeId([v[2], v[3]]),
            EdgeId([v[3], v[0]]),
            EdgeId([v[0], v[2]]),
            EdgeId([v[1], v[0]]),
            EdgeId([v[0], v[3]]),
            EdgeId([v[3], v[1]]),
        ]
    }

    /// Gets the edges of the tetrahedron in no particular order.
    /// Each edge includes its opposite edge.
    /// Beware that both directions of each edge are included,
    /// so there are 12 edges, not 6.
    pub fn edges_and_opp(self) -> [(EdgeId, EdgeId); 12] {
        let v = self.0;
        [
            (EdgeId([v[0], v[1]]), EdgeId([v[2], v[3]])),
            (EdgeId([v[1], v[2]]), EdgeId([v[0], v[3]])),
            (EdgeId([v[2], v[0]]), EdgeId([v[1], v[3]])),
            (EdgeId([v[3], v[2]]), EdgeId([v[1], v[0]])),
            (EdgeId([v[2], v[1]]), EdgeId([v[3], v[0]])),
            (EdgeId([v[1], v[3]]), EdgeId([v[2], v[0]])),
            (EdgeId([v[2], v[3]]), EdgeId([v[0], v[1]])),
            (EdgeId([v[3], v[0]]), EdgeId([v[2], v[1]])),
            (EdgeId([v[0], v[2]]), EdgeId([v[3], v[1]])),
            (EdgeId([v[1], v[0]]), EdgeId([v[3], v[2]])),
            (EdgeId([v[0], v[3]]), EdgeId([v[1], v[2]])),
            (EdgeId([v[3], v[1]]), EdgeId([v[0], v[2]])),
        ]
    }

    /// Gets the triangles of the tetrahedron in the order of the vertices.
    pub fn tris(self) -> [TriId; 4] {
        let v = self.0;
        [
            TriId(TriId::canonicalize([v[0], v[1], v[2]])),
            TriId(TriId::canonicalize([v[3], v[2], v[1]])),
            TriId(TriId::canonicalize([v[2], v[3], v[0]])),
            TriId(TriId::canonicalize([v[1], v[0], v[3]])),
        ]
    }

    /// Gets the triangles of the tetrahedron in the order of the vertices.
    /// Each triangle includes its opposite vertex.
    pub fn tris_and_opp(self) -> [(TriId, VertexId); 4] {
        let v = self.0;
        [
            (TriId(TriId::canonicalize([v[0], v[1], v[2]])), v[3]),
            (TriId(TriId::canonicalize([v[3], v[2], v[1]])), v[0]),
            (TriId(TriId::canonicalize([v[2], v[3], v[0]])), v[1]),
            (TriId(TriId::canonicalize([v[1], v[0], v[3]])), v[2]),
        ]
    }

    /// Gets the index of a vertex, assuming it's part of the tetrahedron
    fn index(self, vertex: VertexId) -> usize {
        self.0.iter().position(|v| *v == vertex).unwrap()
    }

    /// Gets the index of the opposite vertex of a triangle, assuming it's part of the tetrahedron
    fn opp_index(self, tri: TriId) -> usize {
        6 - self.index(tri.0[0]) - self.index(tri.0[1]) - self.index(tri.0[2])
    }

    /// Reverses the tetrahedron so it winds the other way
    fn twin(self) -> Self {
        Self([self.0[0], self.0[1], self.0[3], self.0[2]])
    }

    /// Sets the opposite vertex of some triangle to some vertex.
    fn with_opp(mut self, tri: TriId, vertex: VertexId) -> Self {
        self.0[self.opp_index(tri)] = vertex;
        self.0 = Self::canonicalize(self.0);
        self
    }
}

type TetFilterFn<'a, TT> = for<'b> fn(&'b (&'a TetId, &'a TT)) -> bool;
type TetMapFn<'a, TT> = fn((&'a TetId, &'a TT)) -> (&'a TetId, &'a <TT as Tet>::T);
/// Iterator over the tetrahedrons of a mesh.
pub type Tets<'a, TT> =
    Map<Filter<hash_map::Iter<'a, TetId, TT>, TetFilterFn<'a, TT>>, TetMapFn<'a, TT>>;
type TetFilterFnMut<'a, TT> = for<'b> fn(&'b (&'a TetId, &'a mut TT)) -> bool;
type TetMapFnMut<'a, TT> = fn((&'a TetId, &'a mut TT)) -> (&'a TetId, &'a mut <TT as Tet>::T);
/// Iterator over the tetrahedrons of a mesh mutably.
pub type TetsMut<'a, TT> =
    Map<Filter<hash_map::IterMut<'a, TetId, TT>, TetFilterFnMut<'a, TT>>, TetMapFnMut<'a, TT>>;

/// Iterator over the tetrahedrons connected to a triangle with the correct winding.
pub type TriTets<'a, M> = MapWith<TriId, TetId, TriVertexOpps<'a, M>, fn(TriId, VertexId) -> TetId>;
/// Iterator over the tetrahedrons connected to an edge.
pub type EdgeTets<'a, M> = MapWith<EdgeId, TetId, EdgeEdgeOpps<'a, M>, fn(EdgeId, EdgeId) -> TetId>;
/// Iterator over the tetrahedrons connected to a vertex.
pub type VertexTets<'a, M> =
    MapWith<VertexId, TetId, VertexTriOpps<'a, M>, fn(VertexId, TriId) -> TetId>;

pub type HasTetsWalker<'a, M> = TetWalker<'a, M, <<M as HasTetsIntr>::Tet as Tet>::Manifold>;

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

macro_rules! T {
    () => {
        <Self::Tet as Tet>::T
    };
}

/// For simplicial complexes that can have tetrahedrons.
pub trait HasTets: internal::HasTets + HasTris + RemoveTetHigher + ClearTetsHigher
where
    Self::Vertex: HigherVertex,
    Self::Edge: HigherEdge,
    Self::Tri: HigherTri,
{
    /// Gets the number of tetrahedrons.
    fn num_tets(&self) -> usize {
        self.num_tets_r()
    }

    /// Iterates over the tetrahedrons of this mesh.
    /// Gives (id, value) pairs
    fn tets(&self) -> Tets<Self::Tet> {
        self.tets_r()
            .iter()
            .filter::<TetFilterFn<Self::Tet>>(|(_, t)| t.value().is_some())
            .map::<_, TetMapFn<Self::Tet>>(|(id, t)| (id, t.value().unwrap()))
    }

    /// Iterates mutably over the tetrahedrons of this mesh.
    /// Gives (id, value) pairs
    fn tets_mut(&mut self) -> TetsMut<Self::Tet> {
        self.tets_r_mut()
            .iter_mut()
            .filter::<TetFilterFnMut<Self::Tet>>(|(_, t)| t.value().is_some())
            .map::<_, TetMapFnMut<Self::Tet>>(|(id, t)| (id, t.value_mut().unwrap()))
    }

    /// Gets the value of the tetrahedron at a specific id.
    /// Returns None if not found.
    fn tet<TI: TryInto<TetId>>(&self, id: TI) -> Option<&T!()> {
        id.try_into()
            .ok()
            .and_then(|id| self.tets_r().get(&id))
            .and_then(|t| t.value())
    }

    /// Gets the value of the tetrahedron at a specific id mutably.
    /// Returns None if not found.
    fn tet_mut<TI: TryInto<TetId>>(&mut self, id: TI) -> Option<&mut T!()> {
        id.try_into()
            .ok()
            .and_then(move |id| self.tets_r_mut().get_mut(&id))
            .and_then(|t| t.value_mut())
    }

    /// Iterates over the opposite triangles of the tetrahedrons that a vertex is part of.
    /// The vertex must exist.
    fn vertex_tri_opps(&self, vertex: VertexId) -> VertexTriOpps<Self> {
        VertexTriOpps {
            mesh: self,
            edges: self.vertex_edges_out(vertex),
            opps: None,
        }
    }

    /// Iterates over the tetrahedrons that a vertex is part of.
    /// The vertex must exist.
    fn vertex_tets(&self, vertex: VertexId) -> VertexTets<Self>
    where
        for<'b> HasTetsWalker<'b, Self>: TetWalk<'b, Mesh = Self>,
        for<'b> HasTrisWalker<'b, Self>: TriWalk<'b, Mesh = Self>,
    {
        self.vertex_tri_opps(vertex)
            .map_with(vertex, |vertex, opp| {
                TetId(TetId::canonicalize([vertex, opp.0[0], opp.0[2], opp.0[1]]))
            })
    }

    /// Iterates over the opposite edges of the tetrahedrons that an edge is part of.
    /// The edge must exist.
    fn edge_edge_opps<EI: TryInto<EdgeId>>(&self, edge: EI) -> EdgeEdgeOpps<Self>
    where
        for<'b> HasTrisWalker<'b, Self>: TriWalk<'b, Mesh = Self>,
    {
        EdgeEdgeOpps {
            mesh: self,
            tris: self.edge_vertex_opps(edge),
            opps: None,
        }
    }

    /// Iterates over the tetrahedrons that an edge is part of.
    /// The edge must exist.
    fn edge_tets<EI: TryInto<EdgeId>>(&self, edge: EI) -> EdgeTets<Self>
    where
        for<'b> HasTetsWalker<'b, Self>: TetWalk<'b, Mesh = Self>,
        for<'b> HasTrisWalker<'b, Self>: TriWalk<'b, Mesh = Self>,
    {
        let edge = edge.try_into().ok().unwrap();
        self.edge_edge_opps(edge).map_with(edge, |edge, opp| {
            TetId(TetId::canonicalize([
                edge.0[0], edge.0[1], opp.0[0], opp.0[1],
            ]))
        })
    }

    /// Iterates over the opposite vertices of the tetrahedrons that a triangle is part of.
    /// The triangle must exist.
    fn tri_vertex_opps<TI: TryInto<TriId>>(&self, tri: TI) -> TriVertexOpps<Self>
    where
        for<'b> HasTetsWalker<'b, Self>: TetWalk<'b, Mesh = Self>,
    {
        if let Some(walker) = self.tet_walker_from_tri(tri) {
            let start_opp = walker.fourth();
            TriVertexOpps {
                walker,
                start_opp,
                finished: false,
            }
        } else {
            TriVertexOpps {
                walker: TetWalker::dummy(self),
                start_opp: VertexId::dummy(),
                finished: true,
            }
        }
    }

    /// Gets the opposite vertex of the ≤1 tetrahedron that a triangle is part of because of the "manifold" condition.
    /// The triangle must exist
    fn tri_vertex_opp<FI: TryInto<TriId>>(&self, tri: FI) -> Option<VertexId>
    where
        Self::Tet: ManifoldTet,
        for<'b> HasTetsWalker<'b, Self>: TetWalk<'b, Mesh = Self>,
    {
        Some(self.tet_walker_from_tri(tri)?.fourth())
    }

    /// Iterates over the tetrahedrons that an triangle is part of.
    /// The triangle must exist.
    fn tri_tets<FI: TryInto<TriId>>(&self, tri: FI) -> TriTets<Self>
    where
        for<'b> HasTetsWalker<'b, Self>: TetWalk<'b, Mesh = Self>,
    {
        let tri = tri.try_into().ok().unwrap();
        self.tri_vertex_opps(tri).map_with(tri, |tri, opp| {
            TetId(TetId::canonicalize([tri.0[0], tri.0[1], tri.0[2], opp]))
        })
    }

    /// Iterates over the ≤1 tetrahedron that an triangle is part of because of the "manifold" condition.
    /// The triangle must exist.
    fn tri_tet<FI: TryInto<TriId>>(&self, tri: FI) -> Option<TetId>
    where
        Self::Tet: ManifoldTet,
        for<'b> HasTetsWalker<'b, Self>: TetWalk<'b, Mesh = Self>,
    {
        let tri = tri.try_into().ok().unwrap();
        let opp = self.tri_vertex_opp(tri)?;
        Some(TetId(TetId::canonicalize([
            tri.0[0], tri.0[1], tri.0[2], opp,
        ])))
    }

    /// Adds a tetrahedron to the mesh. Vertex order is important!
    /// If the tetrahedron was already there, this replaces the value.
    /// Adds in the required edges and triangles if they aren't there already.
    /// Returns the previous value of the tetrahedron, if there was one.
    ///
    /// # Panics
    /// Panics if any vertex doesn't exist or if any two vertices are the same.
    fn add_tet<TI: TryInto<TetId>>(
        &mut self,
        vertices: TI,
        value: T!(),
        tri_value: impl Fn() -> F!(),
        edge_value: impl Fn() -> E!() + Clone,
    ) -> Option<T!()>;

    /// Extends the tetrahedron list with an iterator.
    ///
    /// # Panics
    /// Panics if any vertex doesn't exist or if any two vertices are the same
    /// in any of the tetrahedrons.
    fn extend_tets<TI: TryInto<TetId>, I: IntoIterator<Item = (TI, T!())>>(
        &mut self,
        iter: I,
        tri_value: impl Fn() -> F!() + Clone,
        edge_value: impl Fn() -> E!() + Clone,
    ) {
        iter.into_iter().for_each(|(id, value)| {
            self.add_tet(id, value, tri_value.clone(), edge_value.clone());
        })
    }

    /// Removes an tetrahedron from the mesh and returns the value that was there,
    /// or None if there was nothing there.
    /// Removes the edges and tris that are part of the tetrahedron if they are part of no other tetrahedrons
    /// and the tetrahedron to be removed exists.
    fn remove_tet<TI: TryInto<TetId>>(&mut self, id: TI) -> Option<T!()>
    where
        for<'b> HasTetsWalker<'b, Self>: TetWalk<'b, Mesh = Self>,
        for<'b> HasTrisWalker<'b, Self>: TriWalk<'b, Mesh = Self>,
    {
        let id = match id.try_into() {
            Ok(id) => id,
            Err(_) => return None,
        };

        if let Some(value) = self.remove_tet_keep_tris(id) {
            for tri in &id.tris() {
                if self.tri_tets(*tri).next().is_none() {
                    self.remove_tri(*tri);
                }
            }

            Some(value)
        } else {
            None
        }
    }

    /// Removes an tetrahedron from the mesh and returns the value that was there,
    /// or None if there was nothing there.
    /// Keeps the triangles that are part of the tetrahedron.
    fn remove_tet_keep_tris<TI: TryInto<TetId>>(&mut self, id: TI) -> Option<T!()>;

    /// Removes a list of tetrahedrons.
    fn remove_tets<TI: TryInto<TetId>, I: IntoIterator<Item = TI>>(&mut self, iter: I)
    where
        for<'b> HasTetsWalker<'b, Self>: TetWalk<'b, Mesh = Self>,
        for<'b> HasTrisWalker<'b, Self>: TriWalk<'b, Mesh = Self>,
    {
        for tet in iter {
            self.remove_tet(tet);
        }
    }

    /// Removes a list of tetrahedrons.
    fn remove_tets_keep_tris<TI: TryInto<TetId>, I: IntoIterator<Item = TI>>(&mut self, iter: I) {
        iter.into_iter().for_each(|id| {
            self.remove_tet_keep_tris(id);
        })
    }

    /// Keeps only the tetrahedrons that satisfy a predicate
    fn retain_tets<P: FnMut(TetId, &T!()) -> bool>(&mut self, mut predicate: P)
    where
        for<'b> HasTetsWalker<'b, Self>: TetWalk<'b, Mesh = Self>,
        for<'b> HasTrisWalker<'b, Self>: TriWalk<'b, Mesh = Self>,
    {
        let to_remove = self
            .tets()
            .filter(|(id, f)| !predicate(**id, *f))
            .map(|(id, _)| *id)
            .collect::<Vec<_>>();
        self.remove_tets(to_remove);
    }

    /// Keeps only the tetrahedrons that satisfy a predicate
    fn retain_tets_keep_tris<P: FnMut(TetId, &T!()) -> bool>(&mut self, mut predicate: P) {
        let to_remove = self
            .tets()
            .filter(|(id, f)| !predicate(**id, *f))
            .map(|(id, _)| *id)
            .collect::<Vec<_>>();
        self.remove_tets_keep_tris(to_remove);
    }

    /// Removes all tetrahedrons from the mesh.
    fn clear_tets(&mut self) {
        self.clear_tets_higher();
        self.tets_r_mut().clear();
        *self.num_tets_r_mut() = 0;

        // Fix tri-target links
        for (id, tri) in self.tris_r_mut() {
            *tri.tet_opp_mut() = id.0[0];
        }
    }

    /// Gets a tetrahedron walker that starts at the given edge with a given vertex
    /// to form the starting triangle.
    /// Returns None if the triangle formed has no tetrahedron.
    fn tet_walker_from_edge_vertex<EI: TryInto<EdgeId>>(
        &self,
        edge: EI,
        vertex: VertexId,
    ) -> Option<HasTetsWalker<Self>>
    where
        for<'b> HasTetsWalker<'b, Self>: TetWalk<'b, Mesh = Self>,
    {
        HasTetsWalker::<Self>::from_edge_vertex(self, edge, vertex)
    }

    /// Gets a tetrahedron walker that starts at the given triangle.
    /// Returns None if the triangle has no tetrahedron.
    /// Be warned that this does not preserve the order of the vertices
    /// because the triangle id is canonicalized.
    fn tet_walker_from_tri<FI: TryInto<TriId>>(&self, tri: FI) -> Option<HasTetsWalker<Self>>
    where
        for<'b> HasTetsWalker<'b, Self>: TetWalk<'b, Mesh = Self>,
    {
        let tri = tri.try_into().ok().unwrap();
        self.tet_walker_from_edge_vertex(tri.edges()[0], tri.0[2])
    }

    /// Gets a tetrahedron walker that starts at the given edge with the given opposite edge.
    /// They must actually exist.
    fn tet_walker_from_edge_edge<EI: TryInto<EdgeId>, EJ: TryInto<EdgeId>>(
        &self,
        edge: EI,
        opp: EJ,
    ) -> HasTetsWalker<Self>
    where
        for<'b> HasTetsWalker<'b, Self>: TetWalk<'b, Mesh = Self>,
    {
        HasTetsWalker::<Self>::new(self, edge, opp)
    }

    /// Gets a tetrahedron walker that starts at the given tetrahedron.
    /// It must actually exist.
    /// Be warned that this does not preserve the order of the vertices
    /// because the tetrahedron id is canonicalized.
    fn tet_walker_from_tet<TI: TryInto<TetId>>(&self, tet: TI) -> HasTetsWalker<Self>
    where
        for<'b> HasTetsWalker<'b, Self>: TetWalk<'b, Mesh = Self>,
    {
        let tet = tet.try_into().ok().unwrap();
        HasTetsWalker::<Self>::new(
            self,
            EdgeId([tet.0[0], tet.0[1]]),
            EdgeId([tet.0[2], tet.0[3]]),
        )
    }
}

/// Tetrahedron walker generic over whether the mesh has the "manifold" restriction
pub trait TetWalk<'m>
where
    Self::Mesh: HasVertices,
    <Self::Mesh as HasVerticesIntr>::Vertex: HigherVertex,
    Self::Mesh: HasEdges,
    <Self::Mesh as HasEdgesIntr>::Edge: HigherEdge,
    Self::Mesh: HasTris,
    <Self::Mesh as HasTrisIntr>::Tri: HigherTri,
    Self: Sized,
{
    type Mesh: HasTets + ?Sized + 'm;

    #[doc(hidden)]
    fn new<EI: TryInto<EdgeId>, EJ: TryInto<EdgeId>>(
        mesh: &'m Self::Mesh,
        edge: EI,
        opp: EJ,
    ) -> Self;

    #[doc(hidden)]
    fn from_edge_vertex<EI: TryInto<EdgeId>>(
        mesh: &'m Self::Mesh,
        edge: EI,
        vertex: VertexId,
    ) -> Option<Self>;

    /// A walker that will not be used
    #[doc(hidden)]
    fn dummy(mesh: &'m Self::Mesh) -> Self {
        Self::new(mesh, EdgeId::dummy(), EdgeId::dummy())
    }

    /// Gets the mesh the walker references
    fn mesh(&self) -> &'m Self::Mesh;

    /// Get the current vertex id,
    /// which is the source of the current edge.
    fn first(&self) -> VertexId {
        self.edge().0[0]
    }

    /// Get the vertex id of the target of the current edge.
    fn second(&self) -> VertexId {
        self.edge().0[1]
    }

    /// Gets the current edge id
    fn edge(&self) -> EdgeId;

    #[doc(hidden)]
    fn edge_mut(&mut self) -> &mut EdgeId;

    /// Get the vertex id of the source of the current opposite edge.
    fn third(&self) -> VertexId {
        self.opp_edge().0[0]
    }

    /// Gets the opposite vertex of the current triangle
    fn fourth(&self) -> VertexId {
        self.opp_edge().0[1]
    }

    /// Gets the opposite edge of the current edge
    fn opp_edge(&self) -> EdgeId;

    #[doc(hidden)]
    fn opp_edge_mut(&mut self) -> &mut EdgeId;

    /// Gets the current triangle id
    fn tri(&self) -> TriId {
        TriId(TriId::canonicalize([
            self.first(),
            self.second(),
            self.third(),
        ]))
    }

    /// Gets the current list of vertices in order
    fn vertices(&self) -> [VertexId; 4] {
        [self.first(), self.second(), self.third(), self.fourth()]
    }

    /// Gets the current tetrahedron id
    fn tet(&self) -> TetId {
        TetId(TetId::canonicalize([
            self.first(),
            self.second(),
            self.third(),
            self.fourth(),
        ]))
    }

    /// Reverse the walker's direction so its
    /// current tetrahedron is the opposite tetrahedron without changing the opposite edge.
    /// Returns None if the resulting tetrahedron doesn't exist.
    fn twin(mut self) -> Option<Self> {
        *self.edge_mut() = self.edge().twin();
        if self.mesh().tet(self.tet()).is_some() {
            Some(self)
        } else {
            None
        }
    }

    /// Set the current tetrahedron to one that contains the twin
    /// of the current triangle. Useful for getting the
    /// other tetrahedron of the unoriented triangle in a manifold.
    fn on_twin_tri(self) -> Option<Self>
    where
        HasTrisWalker<'m, Self::Mesh>: TriWalk<'m, Mesh = Self::Mesh>;

    /// Sets the current edge to the next one in the same current triangle.
    fn next_edge(mut self) -> Self {
        let (edge, opp) = (EdgeId([self.second(), self.third()]), self.first());
        *self.edge_mut() = edge;
        *self.opp_edge_mut() = EdgeId([opp, self.fourth()]);
        self
    }

    /// Sets the current edge to the previous one in the same current triangle.
    fn prev_edge(mut self) -> Self {
        let (edge, opp) = (EdgeId([self.third(), self.first()]), self.second());
        *self.edge_mut() = edge;
        *self.opp_edge_mut() = EdgeId([opp, self.fourth()]);
        self
    }

    /// Sets the current triangle to the next one in the same tetrahedron.
    fn next_tri(mut self) -> Self {
        let (v0, v1, v2, v3) = if self
            .vertices()
            .iter()
            .enumerate()
            .min_by_key(|(_, id)| **id)
            .unwrap()
            .0
            % 2
            == 0
        {
            (self.fourth(), self.third(), self.second(), self.first())
        } else {
            (self.second(), self.first(), self.fourth(), self.third())
        };

        *self.edge_mut() = EdgeId([v0, v1]);
        *self.opp_edge_mut() = EdgeId([v2, v3]);
        self
    }

    /// Switches the current edge and opposite edge
    fn flip_tri(mut self) -> Self {
        let (v0, v1, v2, v3) = (self.third(), self.fourth(), self.first(), self.second());
        *self.edge_mut() = EdgeId([v0, v1]);
        *self.opp_edge_mut() = EdgeId([v2, v3]);
        self
    }

    /// Sets the current triangle to the previous one in the same tetrahedron.
    fn prev_tri(mut self) -> Self {
        let (v0, v1, v2, v3) = if self
            .vertices()
            .iter()
            .enumerate()
            .min_by_key(|(_, id)| **id)
            .unwrap()
            .0
            % 2
            != 0
        {
            (self.fourth(), self.third(), self.second(), self.first())
        } else {
            (self.second(), self.first(), self.fourth(), self.third())
        };

        *self.edge_mut() = EdgeId([v0, v1]);
        *self.opp_edge_mut() = EdgeId([v2, v3]);
        self
    }

    /// Sets the current opposite vertex to the next one with the same triangle.
    fn next_opp(self) -> Self;

    /// Sets the current opposite vertex to the previous one with the same triangle.
    fn prev_opp(self) -> Self;

    /// Turns this into an triangle walker that starts
    /// at the current edge and opposite vertex of the current triangle.
    fn tri_walker(self) -> HasTrisWalker<'m, Self::Mesh>
    where
        HasTrisWalker<'m, Self::Mesh>: TriWalk<'m, Mesh = Self::Mesh>,
    {
        HasTrisWalker::new(self.mesh(), self.edge(), self.third())
    }
}

/// A walker for navigating a simplicial complex by tetrahedron.
#[derive(Debug)]
pub struct TetWalker<'a, M: ?Sized, MF: Bit + Debug>
where
    M: HasVertices,
    <M as HasVerticesIntr>::Vertex: HigherVertex,
    M: HasEdges,
    <M as HasEdgesIntr>::Edge: HigherEdge,
    M: HasTris,
    <M as HasTrisIntr>::Tri: HigherTri,
    M: HasTets,
    <M as HasTetsIntr>::Tet: Tet<Manifold = MF>,
{
    mesh: &'a M,
    edge: EdgeId,
    opp: EdgeId,
}

impl<'a, M: ?Sized, MF: Bit + Debug> Clone for TetWalker<'a, M, MF>
where
    M: HasVertices,
    <M as HasVerticesIntr>::Vertex: HigherVertex,
    M: HasEdges,
    <M as HasEdgesIntr>::Edge: HigherEdge,
    M: HasTris,
    <M as HasTrisIntr>::Tri: HigherTri,
    M: HasTets,
    <M as HasTetsIntr>::Tet: Tet<Manifold = MF>,
{
    fn clone(&self) -> Self {
        Self {
            mesh: self.mesh,
            edge: self.edge,
            opp: self.opp,
        }
    }
}

impl<'a, M: ?Sized, MF: Bit + Debug> Copy for TetWalker<'a, M, MF>
where
    M: HasVertices,
    <M as HasVerticesIntr>::Vertex: HigherVertex,
    M: HasEdges,
    <M as HasEdgesIntr>::Edge: HigherEdge,
    M: HasTris,
    <M as HasTrisIntr>::Tri: HigherTri,
    M: HasTets,
    <M as HasTetsIntr>::Tet: Tet<Manifold = MF>,
{
}

impl<'m, M: 'm + ?Sized> TetWalk<'m> for TetWalker<'m, M, B1>
where
    M: HasVertices,
    <M as HasVerticesIntr>::Vertex: HigherVertex,
    M: HasEdges,
    <M as HasEdgesIntr>::Edge: HigherEdge,
    M: HasTris,
    <M as HasTrisIntr>::Tri: HigherTri,
    M: HasTets,
    <M as HasTetsIntr>::Tet: ManifoldTet,
{
    type Mesh = M;

    fn new<EI: TryInto<EdgeId>, EJ: TryInto<EdgeId>>(
        mesh: &'m Self::Mesh,
        edge: EI,
        opp: EJ,
    ) -> Self {
        Self {
            mesh,
            edge: edge.try_into().ok().unwrap(),
            opp: opp.try_into().ok().unwrap(),
        }
    }

    fn from_edge_vertex<EI: TryInto<EdgeId>>(
        mesh: &'m Self::Mesh,
        edge: EI,
        vertex: VertexId,
    ) -> Option<Self> {
        let edge = edge.try_into().ok().unwrap();
        let tri = [edge.0[0], edge.0[1], vertex].try_into().ok().unwrap();
        let opp = mesh.tris_r()[&tri].tet_opp();

        // Ensure that the tet exists
        let _: TetId = [edge.0[0], edge.0[1], vertex, opp].try_into().ok()?;
        Some(Self::new(mesh, edge, EdgeId([vertex, opp])))
    }

    fn mesh(&self) -> &'m Self::Mesh {
        self.mesh
    }

    fn edge(&self) -> EdgeId {
        self.edge
    }

    fn edge_mut(&mut self) -> &mut EdgeId {
        &mut self.edge
    }

    fn opp_edge(&self) -> EdgeId {
        self.opp
    }

    fn opp_edge_mut(&mut self) -> &mut EdgeId {
        &mut self.opp
    }

    fn on_twin_tri(self) -> Option<Self>
    where
        HasTrisWalker<'m, M>: TriWalk<'m, Mesh = M>,
    {
        self.tri_walker().twin().and_then(|w| w.tet_walker())
    }

    fn next_opp(self) -> Self {
        self
    }

    fn prev_opp(self) -> Self {
        self
    }
}

impl<'m, M: 'm + ?Sized> TetWalk<'m> for TetWalker<'m, M, B0>
where
    M: HasVertices,
    <M as HasVerticesIntr>::Vertex: HigherVertex,
    M: HasEdges,
    <M as HasEdgesIntr>::Edge: HigherEdge,
    M: HasTris,
    <M as HasTrisIntr>::Tri: HigherTri,
    M: HasTets,
    <M as HasTetsIntr>::Tet: NonManifoldTet,
{
    type Mesh = M;

    fn new<EI: TryInto<EdgeId>, EJ: TryInto<EdgeId>>(
        mesh: &'m Self::Mesh,
        edge: EI,
        opp: EJ,
    ) -> Self {
        Self {
            mesh,
            edge: edge.try_into().ok().unwrap(),
            opp: opp.try_into().ok().unwrap(),
        }
    }

    fn from_edge_vertex<EI: TryInto<EdgeId>>(
        mesh: &'m Self::Mesh,
        edge: EI,
        vertex: VertexId,
    ) -> Option<Self> {
        let edge = edge.try_into().ok().unwrap();
        let tri = [edge.0[0], edge.0[1], vertex].try_into().ok().unwrap();
        let start = match [edge.0[0], edge.0[1], vertex, mesh.tris_r()[&tri].tet_opp()].try_into() {
            Ok(start) => start,
            Err(_) => return None,
        };

        let mut tet = start;
        while mesh.tets_r()[&tet].value().is_none() {
            tet = tet.with_opp(tri, mesh.tets_r()[&tet].link(tet, tri).next);
            if tet == start {
                return None;
            }
        }

        let opp = tet.0[tet.opp_index(tri)];
        Some(Self::new(mesh, edge, EdgeId([vertex, opp])))
    }

    fn mesh(&self) -> &'m Self::Mesh {
        self.mesh
    }

    fn edge(&self) -> EdgeId {
        self.edge
    }

    fn edge_mut(&mut self) -> &mut EdgeId {
        &mut self.edge
    }

    fn opp_edge(&self) -> EdgeId {
        self.opp
    }

    fn opp_edge_mut(&mut self) -> &mut EdgeId {
        &mut self.opp
    }

    fn on_twin_tri(self) -> Option<Self>
    where
        HasTrisWalker<'m, M>: TriWalk<'m, Mesh = M>,
    {
        self.tri_walker().twin().and_then(|w| w.tet_walker())
    }

    fn next_opp(mut self) -> Self {
        while {
            let tet = self.tet();
            self.opp.0[1] = self.mesh.tets_r()[&self.tet()].link(tet, self.tri()).next;
            self.mesh.tets_r()[&self.tet()].value().is_none()
        } {}
        self
    }

    fn prev_opp(mut self) -> Self {
        while {
            let tet = self.tet();
            self.opp.0[1] = self.mesh.tets_r()[&self.tet()].link(tet, self.tri()).prev;
            self.mesh.tets_r()[&self.tet()].value().is_none()
        } {}
        self
    }
}

/// An iterator over the opposite triangles of the tetrahedrons of a vertex.
#[derive(Clone)]
pub struct VertexTriOpps<'a, M: ?Sized>
where
    M: HasVertices,
    <M as HasVerticesIntr>::Vertex: HigherVertex,
    M: HasEdges,
    <M as HasEdgesIntr>::Edge: HigherEdge,
    M: HasTris,
    <M as HasTrisIntr>::Tri: HigherTri,
    M: HasTets,
{
    mesh: &'a M,
    edges: VertexEdgesOut<'a, M>,
    opps: Option<EdgeEdgeOpps<'a, M>>,
}

impl<'a, M: ?Sized> Iterator for VertexTriOpps<'a, M>
where
    M: HasVertices,
    <M as HasVerticesIntr>::Vertex: HigherVertex,
    M: HasEdges,
    <M as HasEdgesIntr>::Edge: HigherEdge,
    M: HasTris,
    <M as HasTrisIntr>::Tri: HigherTri,
    M: HasTets,
    for<'b> HasTetsWalker<'b, M>: TetWalk<'b, Mesh = M>,
    for<'b> HasTrisWalker<'b, M>: TriWalk<'b, Mesh = M>,
{
    type Item = TriId;

    fn next(&mut self) -> Option<Self::Item> {
        let mut next;

        while let Some(edge) = {
            while {
                let mut non_canonical = false;

                next = self.opps.as_mut().and_then(|iter| {
                    iter.next().map(|opp| {
                        let tri = TriId(TriId::canonicalize([
                            iter.tris.walker.second(),
                            opp.0[1],
                            opp.0[0],
                        ]));

                        // Triangles would have 3 copies, one for each cyclic permutation,
                        // if we don't deduplicate.
                        if tri.0[0] != iter.tris.walker.second() {
                            non_canonical = true;
                        }
                        tri
                    })
                });

                non_canonical
            } {}

            if next.is_none() {
                self.edges.next()
            } else {
                None
            }
        } {
            self.opps = Some(self.mesh.edge_edge_opps(edge));
        }

        next
    }
}

/// An iterator over the opposite edges of the tetrahedrons of a edge.
#[derive(Clone)]
pub struct EdgeEdgeOpps<'a, M: ?Sized>
where
    M: HasVertices,
    <M as HasVerticesIntr>::Vertex: HigherVertex,
    M: HasEdges,
    <M as HasEdgesIntr>::Edge: HigherEdge,
    M: HasTris,
    <M as HasTrisIntr>::Tri: HigherTri,
    M: HasTets,
{
    mesh: &'a M,
    tris: EdgeVertexOpps<'a, M>,
    opps: Option<TriVertexOpps<'a, M>>,
}

impl<'a, M: ?Sized> Iterator for EdgeEdgeOpps<'a, M>
where
    M: HasVertices,
    <M as HasVerticesIntr>::Vertex: HigherVertex,
    M: HasEdges,
    <M as HasEdgesIntr>::Edge: HigherEdge,
    M: HasTris,
    <M as HasTrisIntr>::Tri: HigherTri,
    M: HasTets,
    for<'b> HasTetsWalker<'b, M>: TetWalk<'b, Mesh = M>,
    for<'b> HasTrisWalker<'b, M>: TriWalk<'b, Mesh = M>,
{
    type Item = EdgeId;

    fn next(&mut self) -> Option<Self::Item> {
        let mut next;
        let edge = self.tris.walker.edge();

        while let Some(third) = {
            next = self.opps.as_mut().and_then(|iter| {
                iter.next().map(|opp| {
                    let tri = iter.walker.tri();
                    let third = tri.0[tri.opp_index(edge)];
                    EdgeId([third, opp])
                })
            });
            if next.is_none() {
                self.tris.next()
            } else {
                None
            }
        } {
            self.opps = Some(
                self.mesh
                    .tri_vertex_opps(TriId(TriId::canonicalize([edge.0[0], edge.0[1], third]))),
            );
        }

        next
    }
}

/// An iterator over the opposite vertices of the tetrahedrons of an triangle.
#[derive(Clone, Debug)]
pub struct TriVertexOpps<'a, M: ?Sized>
where
    M: HasVertices,
    <M as HasVerticesIntr>::Vertex: HigherVertex,
    M: HasEdges,
    <M as HasEdgesIntr>::Edge: HigherEdge,
    M: HasTris,
    <M as HasTrisIntr>::Tri: HigherTri,
    M: HasTets,
{
    walker: HasTetsWalker<'a, M>,
    start_opp: VertexId,
    finished: bool,
}

impl<'a, M: ?Sized> Iterator for TriVertexOpps<'a, M>
where
    M: HasVertices,
    <M as HasVerticesIntr>::Vertex: HigherVertex,
    M: HasEdges,
    <M as HasEdgesIntr>::Edge: HigherEdge,
    M: HasTris,
    <M as HasTrisIntr>::Tri: HigherTri,
    M: HasTets,
    for<'b> HasTetsWalker<'b, M>: TetWalk<'b, Mesh = M>,
{
    type Item = VertexId;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        let opp = self.walker.fourth();
        self.walker = self.walker.next_opp();
        if self.walker.fourth() == self.start_opp {
            self.finished = true;
        }
        Some(opp)
    }
}

#[macro_export]
#[doc(hidden)]
macro_rules! impl_index_tet {
    ($name:ident<$v:ident, $e:ident, $f: ident, $t: ident $(, $args:ident)*> $(where $($wh:tt)*)?) => {
        impl<$v, $e, $f, $t $(, $args)*> std::ops::Index<[crate::vertex::VertexId; 4]> for $name<$v, $e, $f, $t $(, $args)*>
        $(where $($wh)*)?
        {
            type Output = $t;

            fn index(&self, index: [crate::vertex::VertexId; 4]) -> &Self::Output {
                self.tet(index).unwrap()
            }
        }

        impl<$v, $e, $f, $t $(, $args)*> std::ops::IndexMut<[crate::vertex::VertexId; 4]> for $name<$v, $e, $f, $t $(, $args)*>
        $(where $($wh)*)?
        {
            fn index_mut(&mut self, index: [crate::vertex::VertexId; 4]) -> &mut Self::Output {
                self.tet_mut(index).unwrap()
            }
        }

        impl<$v, $e, $f, $t $(, $args)*> std::ops::Index<crate::tet::TetId> for $name<$v, $e, $f, $t $(, $args)*>
        $(where $($wh)*)?
        {
            type Output = $t;

            fn index(&self, index: crate::tet::TetId) -> &Self::Output {
                self.tet(index).unwrap()
            }
        }

        impl<$v, $e, $f, $t $(, $args)*> std::ops::IndexMut<crate::tet::TetId> for $name<$v, $e, $f, $t $(, $args)*>
        $(where $($wh)*)?
        {
            fn index_mut(&mut self, index: crate::tet::TetId) -> &mut Self::Output {
                self.tet_mut(index).unwrap()
            }
        }
    };
}

/// For concrete simplicial complexes with tetrahedrons
pub trait HasPositionAndTets: HasTets + HasPosition
where
    <Self::Vertex as Vertex>::V: Position,
    DefaultAllocator: Allocator<f64, HasPositionDim<Self>>,
    Self::Vertex: HigherVertex,
    Self::Edge: HigherEdge,
    Self::Tri: HigherTri,
{
    /// Gets the positions of the vertices of an tetrahedron
    fn tet_positions<TI: TryInto<TetId>>(&self, tet: TI) -> Option<[HasPositionPoint<Self>; 4]> {
        let tet = tet.try_into().ok()?;
        let v0 = self.position(tet.0[0])?;
        let v1 = self.position(tet.0[1])?;
        let v2 = self.position(tet.0[2])?;
        let v3 = self.position(tet.0[3])?;
        Some([v0, v1, v2, v3])
    }
}

pub(crate) mod internal {
    use fnv::FnvHashMap;
    use std::convert::TryInto;
    use typenum::{Bit, B0, B1};

    use super::{HasTetsWalker, TetId, TetWalk};
    use crate::edge::internal::{Edge, HigherEdge, Link};
    use crate::tri::internal::{HasTris as HasTrisIntr, HigherTri, Tri};
    use crate::tri::{TriId, TriWalk, HasTrisWalker};
    use crate::vertex::internal::HigherVertex;
    use crate::vertex::VertexId;

    #[macro_export]
    #[doc(hidden)]
    macro_rules! impl_manifold_tet {
        ($name:ident<$t:ident>, new |$value:ident| $new:expr) => {
            impl<$t> crate::tet::internal::Tet for $name<$t> {
                type T = $t;
                type Manifold = typenum::B1;

                fn to_value(self) -> Option<Self::T> {
                    Some(self.value)
                }

                fn value(&self) -> Option<&Self::T> {
                    Some(&self.value)
                }

                fn value_mut(&mut self) -> Option<&mut Self::T> {
                    Some(&mut self.value)
                }
            }

            impl<$t> crate::tet::internal::ManifoldTet for $name<$t> {
                fn new($value: Self::T) -> Self {
                    $new
                }
            }
        };
    }

    #[macro_export]
    #[doc(hidden)]
    macro_rules! impl_non_manifold_tet {
        ($name:ident<$t:ident>, with_links |$id:ident, $links:ident, $value:ident| $new:expr) => {
            impl<$t> crate::tet::internal::Tet for $name<$t> {
                type T = $t;
                type Manifold = typenum::B0;

                fn to_value(self) -> Option<Self::T> {
                    self.value
                }

                fn value(&self) -> Option<&Self::T> {
                    self.value.as_ref()
                }

                fn value_mut(&mut self) -> Option<&mut Self::T> {
                    self.value.as_mut()
                }
            }

            impl<$t> crate::tet::internal::NonManifoldTet for $name<$t> {
                fn option_value_mut(&mut self) -> &mut Option<Self::T> {
                    &mut self.value
                }

                fn with_links(
                    $id: crate::vertex::VertexId,
                    $links: [crate::edge::internal::Link<crate::vertex::VertexId>; 4],
                    $value: Option<Self::T>,
                ) -> Self {
                    $new
                }

                fn links(&self) -> &[crate::edge::internal::Link<crate::vertex::VertexId>; 4] {
                    &self.links
                }

                fn links_mut(
                    &mut self,
                ) -> &mut [crate::edge::internal::Link<crate::vertex::VertexId>; 4] {
                    &mut self.links
                }
            }
        };
    }

    #[macro_export]
    #[doc(hidden)]
    macro_rules! impl_has_tets_manifold {
        ($name:ident<$v:ident, $e:ident, $f:ident, $t:ident $(, $args:ident)*>, $tet:ident $(where $($wh:tt)*)?) => {
            impl<$v, $e, $f, $t $(, $args)*> crate::tet::internal::HasTets for $name<$v, $e, $f, $t $(, $args)*>
            $(where $($wh)*)?
            {
                type Tet = $tet<$t>;

                fn tets_r(&self) -> &FnvHashMap<crate::tet::TetId, Self::Tet> {
                    &self.tets
                }

                fn tets_r_mut(&mut self) -> &mut FnvHashMap<crate::tet::TetId, Self::Tet> {
                    &mut self.tets
                }

                fn num_tets_r(&self) -> usize {
                    self.num_tets
                }

                fn num_tets_r_mut(&mut self) -> &mut usize {
                    &mut self.num_tets
                }
            }

            impl<$v, $e, $f, $t $(, $args)*> crate::tet::HasTets for $name<$v, $e, $f, $t $(, $args)*>
            $(where $($wh)*)?
            {
                fn add_tet<TI: std::convert::TryInto<TetId>>(
                    &mut self,
                    vertices: TI,
                    value: <Self::Tet as crate::tet::internal::Tet>::T,
                    tri_value: impl Fn() -> <Self::Tri as crate::tri::internal::Tri>::F,
                    edge_value: impl Fn() -> <Self::Edge as crate::edge::internal::Edge>::E + Clone,
                ) -> Option<<Self::Tet as crate::tet::internal::Tet>::T> {
                    crate::tet::internal::add_tet_manifold(self, vertices, value, tri_value, edge_value)
                }

                fn remove_tet_keep_tris<TI: std::convert::TryInto<TetId>>(
                    &mut self,
                    id: TI,
                ) -> Option<<Self::Tet as crate::tet::internal::Tet>::T> {
                    crate::tet::internal::remove_tet_manifold(self, id)
                }
            }
        };
    }

    #[macro_export]
    #[doc(hidden)]
    macro_rules! impl_has_tets_non_manifold {
        ($name:ident<$v:ident, $e:ident, $f:ident, $t:ident $(, $args:ident)*>, $tet:ident $(where $($wh:tt)*)?) => {
            impl<$v, $e, $f, $t $(, $args)*> crate::tet::internal::HasTets for $name<$v, $e, $f, $t $(, $args)*>
            $(where $($wh)*)?
            {
                type Tet = $tet<$t>;

                fn tets_r(&self) -> &FnvHashMap<crate::tet::TetId, Self::Tet> {
                    &self.tets
                }

                fn tets_r_mut(&mut self) -> &mut FnvHashMap<crate::tet::TetId, Self::Tet> {
                    &mut self.tets
                }

                fn num_tets_r(&self) -> usize {
                    self.num_tets
                }

                fn num_tets_r_mut(&mut self) -> &mut usize {
                    &mut self.num_tets
                }
            }

            impl<$v, $e, $f, $t $(, $args)*> crate::tet::HasTets for $name<$v, $e, $f, $t $(, $args)*>
            $(where $($wh)*)?
            {
                fn add_tet<TI: std::convert::TryInto<TetId>>(
                    &mut self,
                    vertices: TI,
                    value: <Self::Tet as crate::tet::internal::Tet>::T,
                    tri_value: impl Fn() -> <Self::Tri as crate::tri::internal::Tri>::F,
                    edge_value: impl Fn() -> <Self::Edge as crate::edge::internal::Edge>::E + Clone,
                ) -> Option<<Self::Tet as crate::tet::internal::Tet>::T> {
                    crate::tet::internal::add_tet_non_manifold(self, vertices, value, tri_value, edge_value)
                }

                fn remove_tet_keep_tris<TI: std::convert::TryInto<TetId>>(
                    &mut self,
                    id: TI,
                ) -> Option<<Self::Tet as crate::tet::internal::Tet>::T> {
                    crate::tet::internal::remove_tet_non_manifold(self, id)
                }
            }
        };
    }

    /// Tetrahedron storage
    pub trait Tet {
        type T;
        type Manifold: Bit + std::fmt::Debug;

        fn to_value(self) -> Option<Self::T>;

        fn value(&self) -> Option<&Self::T>;

        fn value_mut(&mut self) -> Option<&mut Self::T>;
    }

    pub trait ManifoldTet: Tet<Manifold = B1> {
        fn new(value: Self::T) -> Self;
    }

    /// Tetrahedron storage
    pub trait NonManifoldTet: Tet<Manifold = B0> {
        fn option_value_mut(&mut self) -> &mut Option<Self::T>;

        fn with_links(id: VertexId, links: [Link<VertexId>; 4], value: Option<Self::T>) -> Self;

        fn links(&self) -> &[Link<VertexId>; 4];

        fn links_mut(&mut self) -> &mut [Link<VertexId>; 4];

        fn link(&self, id: TetId, tri: TriId) -> &Link<VertexId> {
            &self.links()[(id.opp_index(tri) + 1) % 4]
        }

        fn link_mut(&mut self, id: TetId, tri: TriId) -> &mut Link<VertexId> {
            &mut self.links_mut()[(id.opp_index(tri) + 1) % 4]
        }
    }

    pub trait HasTets: HasTrisIntr
    where
        Self::Vertex: HigherVertex,
        Self::Edge: HigherEdge,
        Self::Tri: HigherTri,
    {
        type Tet: Tet;

        fn tets_r(&self) -> &FnvHashMap<TetId, Self::Tet>;

        fn tets_r_mut(&mut self) -> &mut FnvHashMap<TetId, Self::Tet>;

        fn num_tets_r(&self) -> usize;

        fn num_tets_r_mut(&mut self) -> &mut usize;
    }

    pub(crate) fn add_tet_manifold<M: super::HasTets, TI: TryInto<TetId>>(
        mesh: &mut M,
        vertices: TI,
        value: <M::Tet as Tet>::T,
        tri_value: impl Fn() -> <M::Tri as Tri>::F,
        edge_value: impl Fn() -> <M::Edge as Edge>::E + Clone,
    ) -> Option<<M::Tet as Tet>::T>
    where
        M::Vertex: HigherVertex,
        M::Edge: HigherEdge,
        M::Tri: HigherTri,
        M::Tet: ManifoldTet,
        for<'b> HasTetsWalker<'b, M>: TetWalk<'b, Mesh = M>,
        for<'b> HasTrisWalker<'b, M>: TriWalk<'b, Mesh = M>,
    {
        let id = vertices.try_into().ok().unwrap();

        for tri in &id.tris() {
            if mesh.tri(*tri).is_none() {
                mesh.add_tri(*tri, tri_value(), edge_value.clone());
            }
        }

        // Can't use entry().or_insert() because that would cause a
        // mutable borrow and an immutable borrow to exist at the same time
        if let Some(tet) = mesh.tets_r_mut().get_mut(&id) {
            Some(std::mem::replace(tet.value_mut().unwrap(), value))
        } else {
            *mesh.num_tets_r_mut() += 1;

            for (i, (tri, opp)) in id.tris_and_opp().iter().enumerate() {
                let target = mesh.tris_r()[tri].tet_opp();

                // Don't violate the restriction
                if target != tri.0[0] {
                    mesh.remove_tet([tri.0[0], tri.0[1], tri.0[2], target]);
                }
                *mesh.tris_r_mut().get_mut(tri).unwrap().tet_opp_mut() = *opp;
            }

            mesh.tets_r_mut().insert(id, ManifoldTet::new(value));
            None
        }
    }

    pub(crate) fn add_tet_non_manifold<M: super::HasTets, TI: TryInto<TetId>>(
        mesh: &mut M,
        vertices: TI,
        value: <M::Tet as Tet>::T,
        tri_value: impl Fn() -> <M::Tri as Tri>::F,
        edge_value: impl Fn() -> <M::Edge as Edge>::E + Clone,
    ) -> Option<<M::Tet as Tet>::T>
    where
        M::Vertex: HigherVertex,
        M::Edge: HigherEdge,
        M::Tri: HigherTri,
        M::Tet: NonManifoldTet,
    {
        let id = vertices.try_into().ok().unwrap();

        for tri in &id.tris() {
            if mesh.tri(*tri).is_none() {
                mesh.add_tri(*tri, tri_value(), edge_value.clone());
            }
        }

        // Can't use entry().or_insert() because that would cause a
        // mutable borrow and an immutable borrow to exist at the same time
        if let Some(tet) = mesh.tets_r_mut().get_mut(&id) {
            let old = tet.option_value_mut().take();
            *tet.option_value_mut() = Some(value);
            if old.is_none() {
                *mesh.num_tets_r_mut() += 1;
            }
            old
        } else {
            *mesh.num_tets_r_mut() += 1;

            let mut insert_tet = |id: TetId, value: Option<<M::Tet as Tet>::T>| {
                let mut opps = [Link::dummy(VertexId::dummy); 4];

                for (i, (tri, opp)) in id.tris_and_opp().iter().enumerate() {
                    let target = mesh.tris_r()[tri].tet_opp();

                    let (prev, next) = if target == tri.0[0] {
                        // First tet from tri
                        *mesh.tris_r_mut().get_mut(tri).unwrap().tet_opp_mut() = *opp;
                        (*opp, *opp)
                    } else {
                        let side = [tri.0[0], tri.0[1], tri.0[2], target]
                            .try_into()
                            .ok()
                            .unwrap();
                        let prev = mesh.tets_r()[&side].link(side, *tri).prev;
                        let next = target;
                        let prev_tet = id.with_opp(*tri, prev);
                        let next_tet = id.with_opp(*tri, next);
                        mesh.tets_r_mut()
                            .get_mut(&prev_tet)
                            .unwrap()
                            .link_mut(prev_tet, *tri)
                            .next = *opp;
                        mesh.tets_r_mut()
                            .get_mut(&next_tet)
                            .unwrap()
                            .link_mut(next_tet, *tri)
                            .prev = *opp;
                        (prev, next)
                    };

                    opps[i] = Link::new(prev, next);
                }

                mesh.tets_r_mut()
                    .insert(id, NonManifoldTet::with_links(id.0[0], opps, value));
            };

            insert_tet(id, Some(value));
            insert_tet(id.twin(), None);
            None
        }
    }

    pub(crate) fn remove_tet_manifold<M: super::HasTets, TI: TryInto<TetId>>(
        mesh: &mut M,
        id: TI,
    ) -> Option<<M::Tet as Tet>::T>
    where
        M::Vertex: HigherVertex,
        M::Edge: HigherEdge,
        M::Tri: HigherTri,
        M::Tet: ManifoldTet,
    {
        let id = match id.try_into() {
            Ok(id) => id,
            Err(_) => return None,
        };

        if mesh.tet(id).is_some() {
            mesh.remove_tet_higher(id);
        }

        match mesh.tet(id) {
            Some(_) => {
                *mesh.num_tets_r_mut() -= 1;

                for (i, (tri, opp)) in id.tris_and_opp().iter().enumerate() {
                    // Because of the "manifold" condition, this has to be the last tet from the triangle
                    *mesh.tris_r_mut().get_mut(&tri).unwrap().tet_opp_mut() = tri.0[0];
                }

                mesh.tets_r_mut().remove(&id).and_then(|f| f.to_value())
            }

            None => None,
        }
    }
    pub(crate) fn remove_tet_non_manifold<M: super::HasTets, TI: TryInto<TetId>>(
        mesh: &mut M,
        id: TI,
    ) -> Option<<M::Tet as Tet>::T>
    where
        M::Vertex: HigherVertex,
        M::Edge: HigherEdge,
        M::Tri: HigherTri,
        M::Tet: NonManifoldTet,
    {
        let id = match id.try_into() {
            Ok(id) => id,
            Err(_) => return None,
        };

        if mesh.tet(id).is_some() {
            mesh.remove_tet_higher(id);
        }

        match mesh.tets_r().get(&id.twin()).map(|t| t.value()) {
            // Twin actually exists; just set value to None
            Some(Some(_)) => {
                let old = mesh
                    .tets_r_mut()
                    .get_mut(&id)
                    .unwrap()
                    .option_value_mut()
                    .take();
                if old.is_some() {
                    *mesh.num_tets_r_mut() -= 1;
                }
                old
            }

            // Twin is phantom, so remove both tet and twin from map
            Some(None) => {
                // Twin is phantom, so this tet actually exists.
                *mesh.num_tets_r_mut() -= 1;

                let mut delete_tet = |id: TetId| {
                    for (i, (tri, opp)) in id.tris_and_opp().iter().enumerate() {
                        let tet = &mesh.tets_r()[&id];
                        let prev = tet.links()[i].prev;
                        let next = tet.links()[i].next;
                        let prev_tet = id.with_opp(*tri, prev);
                        let next_tet = id.with_opp(*tri, next);
                        mesh.tets_r_mut()
                            .get_mut(&prev_tet)
                            .unwrap()
                            .link_mut(prev_tet, *tri)
                            .next = next;
                        mesh.tets_r_mut()
                            .get_mut(&next_tet)
                            .unwrap()
                            .link_mut(next_tet, *tri)
                            .prev = prev;

                        let source = mesh.tris_r_mut().get_mut(&tri).unwrap();
                        if *opp == next {
                            // this was the last tet from the triangle
                            *source.tet_opp_mut() = tri.0[0];
                        } else if *opp == source.tet_opp() {
                            *source.tet_opp_mut() = next;
                        }
                    }

                    mesh.tets_r_mut().remove(&id).and_then(|f| f.to_value())
                };

                delete_tet(id.twin());
                delete_tet(id)
            }

            // Twin isn't in map, and neither is the tet to remove
            None => None,
        }
    }

    /// Removes higher-order simplexes that contain some tetangle
    pub trait RemoveTetHigher: HasTets
    where
        Self::Vertex: HigherVertex,
        Self::Edge: HigherEdge,
        Self::Tri: HigherTri,
    {
        fn remove_tet_higher(&mut self, tet: TetId);
    }
    /// Clears higher-order simplexes
    pub trait ClearTetsHigher: HasTets
    where
        Self::Vertex: HigherVertex,
        Self::Edge: HigherEdge,
        Self::Tri: HigherTri,
    {
        fn clear_tets_higher(&mut self);
    }
}
