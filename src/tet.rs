//! Traits and structs related to tetrahedrons

use simplicity as sim;
use fnv::FnvHashMap;
use idmap::OrderedIdMap;
use nalgebra::{allocator::Allocator, DefaultAllocator};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::collections::hash_map;
use std::convert::{TryFrom, TryInto};
use std::iter::Map;
use std::vec;
use typenum::{Bit, B0, B1};
use nalgebra::dimension::U3;

use crate::{iter::{FlatMapWith, IteratorExt, MapWith}, vertex::HasPosition3D};
use crate::private::{Key, Lock};
use crate::tri::{EdgeVertexOpps, Tri};
use crate::tri::{HasTris, TriId, TriWalker};
use crate::vertex::VertexId;
use crate::{
    edge::Link,
    vertex::{HasPosition, HasPositionDim, HasPositionPoint, Position},
};
use crate::{
    edge::{EdgeId, HasEdges, IntoEdges, VertexEdgesOut},
    tri::IntoTris,
    vertex::{HasVertices, IntoVertices},
};
use crate::tetrahedralize::index_fn;

/// An tetrahedron id is just the tetrahedrons's vertices in winding order,
/// with the smallest two indexes first.
/// No two vertices are allowed to be the same.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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

    /// Skips the vertex inequality check but does canonicalization
    pub(crate) fn from_valid(v: [VertexId; 4]) -> Self {
        Self(Self::canonicalize(v))
    }

    /// Canonicalizes this tet id into an undirected version.
    pub fn undirected(mut self) -> TetId {
        if self.0[2] > self.0[3] {
            self.0.swap(2, 3);
        }
        self
    }

    /// Gets the vertices that this tet id is made of
    pub fn vertices(self) -> [VertexId; 4] {
        self.0
    }

    /// Whether this contains some vertex
    pub fn contains_vertex(self, vertex: VertexId) -> bool {
        self.0.contains(&vertex)
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

    /// Gets the opposite triangle of a vertex.
    pub fn opp_tri(self, vertex: VertexId) -> TriId {
        let index = self.index(vertex);
        let v = self.0;
        match index {
            0 => TriId::from_valid([v[3], v[2], v[1]]),
            1 => TriId::from_valid([v[2], v[3], v[0]]),
            2 => TriId::from_valid([v[1], v[0], v[3]]),
            3 => TriId::from_valid([v[0], v[1], v[2]]),
            _ => unreachable!(),
        }
    }

    /// Gets the opposite edge of an edge, oriented such
    /// that the triangle `[edge.source(), edge.target(), self.opp_edge().source()]`
    /// is a triangle of this tetrahedron.
    pub fn opp_edge(self, edge: EdgeId) -> EdgeId {
        let i0 = self.index(edge.0[0]);
        let i1 = self.index(edge.0[1]);
        let v = self.0;
        match (i0, i1) {
            (0, 1) => EdgeId([v[2], v[3]]),
            (1, 2) => EdgeId([v[0], v[3]]),
            (2, 0) => EdgeId([v[1], v[3]]),
            (3, 2) => EdgeId([v[1], v[0]]),
            (2, 1) => EdgeId([v[3], v[0]]),
            (1, 3) => EdgeId([v[2], v[0]]),
            (2, 3) => EdgeId([v[0], v[1]]),
            (3, 0) => EdgeId([v[2], v[1]]),
            (0, 2) => EdgeId([v[3], v[1]]),
            (1, 0) => EdgeId([v[3], v[2]]),
            (0, 3) => EdgeId([v[1], v[2]]),
            (3, 1) => EdgeId([v[0], v[2]]),
            _ => unreachable!(),
        }
    }

    /// Gets the opposite vertex of a triangle.
    pub fn opp_vertex(self, tri: TriId) -> VertexId {
        self.0[self.opp_index(tri)]
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
    #[allow(dead_code)]
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

/// Iterator over the tetrahedron ids of a mesh.
pub type TetIds<'a, TT> = hash_map::Keys<'a, TetId, TT>;

/// Iterator over the tetrahedrons of a mesh.
pub type IntoTets<TT> =
    Map<hash_map::IntoIter<TetId, TT>, fn((TetId, TT)) -> (TetId, <TT as Tet>::T)>;

/// Iterator over the tetrahedrons of a mesh.
pub type Tets<'a, TT> = Map<
    hash_map::Iter<'a, TetId, TT>,
    for<'b> fn((&'b TetId, &'b TT)) -> (&'b TetId, &'b <TT as Tet>::T),
>;

/// Iterator over the tetrahedrons of a mesh mutably.
pub type TetsMut<'a, TT> = Map<
    hash_map::IterMut<'a, TetId, TT>,
    for<'b> fn((&'b TetId, &'b mut TT)) -> (&'b TetId, &'b mut <TT as Tet>::T),
>;

/// Iterator over the tetrahedrons connected to a triangle with the correct winding.
pub type TriTets<'a, M> = MapWith<TriId, TriVertexOpps<'a, M>, fn(TriId, VertexId) -> TetId>;
/// Iterator over the tetrahedrons connected to an edge.
pub type EdgeTets<'a, M> = MapWith<EdgeId, EdgeEdgeOpps<'a, M>, fn(EdgeId, EdgeId) -> TetId>;
/// Iterator over the tetrahedrons connected to a vertex.
pub type VertexTets<'a, M> = MapWith<VertexId, VertexTriOpps<'a, M>, fn(VertexId, TriId) -> TetId>;

/// Iterator over the adjacent tetrahedrons of a tetrahedron.
pub type AdjacentTets<'a, M> =
    FlatMapWith<&'a M, vec::IntoIter<TriId>, TriTets<'a, M>, fn(&'a M, TriId) -> TriTets<'a, M>>;

/// Tetrahedron attributes
pub trait Tet {
    type T;
    type Mwb: Bit;

    #[doc(hidden)]
    fn new<L: Lock>(id: VertexId, links: [Link<VertexId>; 4], value: Self::T) -> Self;

    #[doc(hidden)]
    fn links<L: Lock>(&self) -> [Link<VertexId>; 4];

    #[doc(hidden)]
    fn links_mut<L: Lock>(&mut self) -> &mut [Link<VertexId>; 4];

    #[doc(hidden)]
    fn to_value<L: Lock>(self) -> Self::T;

    #[doc(hidden)]
    fn value<L: Lock>(&self) -> &Self::T;

    #[doc(hidden)]
    fn value_mut<L: Lock>(&mut self) -> &mut Self::T;

    #[doc(hidden)]
    fn link<L: Lock>(&self, id: TetId, tri: TriId) -> Link<VertexId> {
        self.links::<Key>()[(id.opp_index(tri) + 1) % 4]
    }

    #[doc(hidden)]
    fn link_mut<L: Lock>(&mut self, id: TetId, tri: TriId) -> &mut Link<VertexId> {
        &mut self.links_mut::<Key>()[(id.opp_index(tri) + 1) % 4]
    }
}

/// Allows upgrading to a simplicial 3-complex.
pub trait WithTets<V, E, F, T> {
    type WithTets: HasVertices<V = V> + HasEdges<E = E> + HasTris<F = F> + HasTets<T = T>;
}

/// For simplicial complexes that can have tetrahedrons.
pub trait HasTets: HasTris<HigherF = B1> {
    type Tet: Tet<T = Self::T, Mwb = Self::MwbT>;
    type T;
    type MwbT: Bit;
    type WithoutTets: HasVertices<V = Self::V> + HasEdges<E = Self::E> + HasTris<F = Self::F>;
    type WithMwbT: HasVertices<V = Self::V>
        + HasEdges<E = Self::E>
        + HasTris<F = Self::F>
        + HasTets<T = Self::T, MwbT = B1>;
    type WithoutMwbT: HasVertices<V = Self::V>
        + HasEdges<E = Self::E>
        + HasTris<F = Self::F>
        + HasTets<T = Self::T, MwbT = B0>;

    #[doc(hidden)]
    fn from_veft_r<
        VI: IntoIterator<Item = (VertexId, Self::V)>,
        EI: IntoIterator<Item = (EdgeId, Self::E)>,
        FI: IntoIterator<Item = (TriId, Self::F)>,
        TI: IntoIterator<Item = (TetId, Self::T)>,
        L: Lock,
    >(
        vertices: VI,
        edges: EI,
        tris: FI,
        tets: TI,
        default_v: fn() -> Self::V,
        default_e: fn() -> Self::E,
        default_f: fn() -> Self::F,
        default_t: fn() -> Self::T,
    ) -> Self;

    #[doc(hidden)]
    fn into_veft_r<L: Lock>(
        self,
    ) -> (
        IntoVertices<Self::Vertex>,
        IntoEdges<Self::Edge>,
        IntoTris<Self::Tri>,
        IntoTets<Self::Tet>,
    );

    #[doc(hidden)]
    fn tets_r<L: Lock>(&self) -> &FnvHashMap<TetId, Self::Tet>;

    #[doc(hidden)]
    fn tets_r_mut<L: Lock>(&mut self) -> &mut FnvHashMap<TetId, Self::Tet>;

    #[doc(hidden)]
    fn remove_tet_higher<L: Lock>(&mut self, tet: TetId);

    #[doc(hidden)]
    fn clear_tets_higher<L: Lock>(&mut self);

    #[doc(hidden)]
    fn default_t_r<L: Lock>(&self) -> fn() -> Self::T;

    #[doc(hidden)]
    #[cfg(feature = "obj")]
    fn obj_with_tets<L: Lock>(&self, _data: &mut obj::ObjData, _v_inv: &FnvHashMap<VertexId, usize>) {
        panic!("Can't export tet mesh as obj");
    }

    /// Gets the default value of a tetrahedron.
    fn default_tet(&self) -> Self::T {
        self.default_t_r::<Key>()()
    }

    /// Gets the number of tetrahedrons.
    fn num_tets(&self) -> usize {
        self.tets_r::<Key>().len()
    }

    /// Iterates over the tetrahedron ids of this mesh.
    fn tet_ids(&self) -> TetIds<Self::Tet> {
        self.tets_r::<Key>().keys()
    }

    /// Iterates over the tetrahedrons of this mesh.
    /// Gives (id, value) pairs
    fn tets(&self) -> Tets<Self::Tet> {
        self.tets_r::<Key>()
            .iter()
            .map(|(id, t)| (id, t.value::<Key>()))
    }

    /// Iterates mutably over the tetrahedrons of this mesh.
    /// Gives (id, value) pairs
    fn tets_mut(&mut self) -> TetsMut<Self::Tet> {
        self.tets_r_mut::<Key>()
            .iter_mut()
            .map(|(id, t)| (id, t.value_mut::<Key>()))
    }

    /// Gets whether the mesh contains some tetrahedron.
    fn contains_tet<TI: TryInto<TetId>>(&self, id: TI) -> bool {
        id.try_into()
            .ok()
            .and_then(|id| self.tets_r::<Key>().get(&id))
            .is_some()
    }

    /// Takes a tet id and returns it back if the tetrahedron exists,
    /// or None if it doesn't.
    /// Useful for composing with functions that assume the tetrahedron exists.
    fn tet_id<TI: TryInto<TetId>>(&self, id: TI) -> Option<TetId> {
        id.try_into().ok().and_then(|id| if self.contains_tet(id) { Some(id) } else { None })
    }

    /// Gets the value of the tetrahedron at a specific id.
    /// Returns None if not found.
    fn tet<TI: TryInto<TetId>>(&self, id: TI) -> Option<&Self::T> {
        id.try_into()
            .ok()
            .and_then(|id| self.tets_r::<Key>().get(&id))
            .map(|t| t.value::<Key>())
    }

    /// Gets the value of the tetrahedron at a specific id mutably.
    /// Returns None if not found.
    fn tet_mut<TI: TryInto<TetId>>(&mut self, id: TI) -> Option<&mut Self::T> {
        id.try_into()
            .ok()
            .and_then(move |id| self.tets_r_mut::<Key>().get_mut(&id))
            .map(|t| t.value_mut::<Key>())
    }

    /// Iterates over the opposite triangles of the tetrahedrons that a vertex is part of.
    fn vertex_tri_opps(&self, vertex: VertexId) -> VertexTriOpps<Self> {
        VertexTriOpps {
            mesh: self,
            edges: self.vertex_edges_out(vertex),
            opps: None,
        }
    }

    /// Iterates over the tetrahedrons that a vertex is part of.
    fn vertex_tets(&self, vertex: VertexId) -> VertexTets<Self> {
        self.vertex_tri_opps(vertex)
            .map_with(vertex, |vertex, opp| {
                TetId::from_valid([vertex, opp.0[0], opp.0[2], opp.0[1]])
            })
    }

    /// Iterates over the opposite edges of the tetrahedrons that an edge is part of.
    fn edge_edge_opps<EI: TryInto<EdgeId>>(&self, edge: EI) -> EdgeEdgeOpps<Self> {
        EdgeEdgeOpps {
            mesh: self,
            tris: self.edge_vertex_opps(edge),
            opps: None,
        }
    }

    /// Iterates over the tetrahedrons that an edge is part of.
    fn edge_tets<EI: TryInto<EdgeId>>(&self, edge: EI) -> EdgeTets<Self> {
        let edge = edge.try_into().unwrap_or(EdgeId::invalid());

        // edge_vertex_opps is an empty iterator if the edge does not exist.
        self.edge_edge_opps(edge).map_with(edge, |edge, opp| {
            TetId::from_valid([edge.0[0], edge.0[1], opp.0[0], opp.0[1]])
        })
    }

    /// Iterates over the opposite vertices of the tetrahedrons that a triangle is part of.
    fn tri_vertex_opps<TI: TryInto<TriId>>(&self, tri: TI) -> TriVertexOpps<Self> {
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

    /// Gets the opposite vertex of the ≤1 outgoing tetrahedron that the triangle is part of.
    fn tri_vertex_opp<FI: TryInto<TriId>>(&self, tri: FI) -> Option<VertexId>
    where
        Self: HasTets<MwbT = typenum::B1>,
    {
        let tri = tri.try_into().ok()?;
        let opp = self.tris_r::<Key>().get(&tri)?.tet_opp::<Key>();
        if opp != tri.0[0] {
            Some(opp)
        } else {
            None
        }
    }

    /// Iterates over the tetrahedrons that an triangle is part of.
    fn tri_tets<FI: TryInto<TriId>>(&self, tri: FI) -> TriTets<Self> {
        let tri = tri.try_into().unwrap_or(TriId::invalid());
        self.tri_vertex_opps(tri).map_with(tri, |tri, opp| {
            TetId::from_valid([tri.0[0], tri.0[1], tri.0[2], opp])
        })
    }

    /// Gets the ≤1 tetrahedron that the triangle is part of.
    fn tri_tet<FI: TryInto<TriId>>(&self, tri: FI) -> Option<TetId>
    where
        Self: HasTets<MwbT = typenum::B1>,
    {
        let tri = tri.try_into().ok()?;
        Some(TetId::from_valid([
            tri.0[0],
            tri.0[1],
            tri.0[2],
            self.tri_vertex_opp(tri)?,
        ]))
    }

    /// Gets the tetrahedrons that are adjacent to this tetrahedron.
    /// Beware that for now, the twin tetrahedron is returned 4 times if it exists.
    fn adjacent_tets(&self, tet: TetId) -> AdjacentTets<Self> {
        let tris = tet.tris().to_vec();
        tris.into_iter()
            .flat_map_with(self, |mesh, tri| mesh.tri_tets(tri.twin()))
    }

    /// Attempts to flip away the triangle of the first 3 vertices, creating an edge connecting the last 2 vertices.
    /// [v1, v2, v3, vp] must have a positive orientation.
    /// This sets custom values to their default.
    /// This method is pretty much unchecked.
    fn flip23(&mut self, v1: VertexId, v2: VertexId, v3: VertexId, vp: VertexId, vn: VertexId)
    where
        Self: HasTets<MwbT = B1>
    {
        self.remove_tet(TetId::from_valid([v1, v2, v3, vp]));
        self.remove_tet(TetId::from_valid([v3, v2, v1, vn]));
        self.add_tet(TetId::from_valid([v1, v2, vn, vp]), self.default_tet());
        self.add_tet(TetId::from_valid([v2, v3, vn, vp]), self.default_tet());
        self.add_tet(TetId::from_valid([v3, v1, vn, vp]), self.default_tet());
    }

    /// Attempts to flip away the edge connecting the last 2 vertices, creating a triangle of the first 3 vertices.
    /// [v1, v2, v3, vp] must have a positive orientation.
    /// This sets custom values to their default.
    /// This method is pretty much unchecked.
    fn flip32(&mut self, v1: VertexId, v2: VertexId, v3: VertexId, vp: VertexId, vn: VertexId)
    where
        Self: HasTets<MwbT = B1>
    {
        self.remove_tet(TetId::from_valid([v1, v2, vn, vp]));
        self.remove_tet(TetId::from_valid([v2, v3, vn, vp]));
        self.remove_tet(TetId::from_valid([v3, v1, vn, vp]));
        self.add_tet(TetId::from_valid([v1, v2, v3, vp]), self.default_tet());
        self.add_tet(TetId::from_valid([v3, v2, v1, vn]), self.default_tet());
    }

    /// Attempts to remove a triangle with flips and returns whether this succeeded.
    fn remove_tri_via_flips<FI: TryInto<TriId>>(&mut self, tri: FI, depth: usize,
        edge_removable: impl Fn(&Self, EdgeId) -> bool + Clone,
        tri_removable: impl Fn(&Self, TriId) -> bool + Clone,
        edge_addable: impl Fn(&Self, VertexId, VertexId) -> bool + Clone,
        tri_addable: impl Fn(&Self, VertexId, VertexId, VertexId) -> bool + Clone,
        bad_edges: &mut Vec<EdgeId>,
    ) -> bool
    where
        Self: HasTets<MwbT = B1> + HasPosition3D + Sized,
        Self::V: Position<Dim = U3>,
    {
        if depth == 0 {
            return false;
        }

        let tri = match tri.try_into() {
            Ok(tri) => tri,
            Err(_) => return false,
        };
        if !tri_removable(self, tri) {
            return false;
        }

        let (vp, vn) = match (self.tri_vertex_opp(tri), self.tri_vertex_opp(tri.twin())) {
            (Some(vp), Some(vn)) => (vp, vn),
            _ => return false,
        };

        // From here on out, the edge may be removed in a roundabout way.
        // So check if the edge was removed before lying and saying it wasn't.

        // Check for concavity, then addability
        if sim::orient_3d(self, index_fn, tri.0[0], tri.0[1], vp, vn) {
            self.remove_edge_via_flips(EdgeId([tri.0[0], tri.0[1]]), depth - 1,
                edge_removable, tri_removable, edge_addable, tri_addable, bad_edges) || !self.contains_tri(tri)
        } else if sim::orient_3d(self, index_fn, tri.0[1], tri.0[2], vp, vn) {
            self.remove_edge_via_flips(EdgeId([tri.0[1], tri.0[2]]), depth - 1,
                edge_removable, tri_removable, edge_addable, tri_addable, bad_edges) || !self.contains_tri(tri)
        } else if sim::orient_3d(self, index_fn, tri.0[2], tri.0[0], vp, vn) {
            self.remove_edge_via_flips(EdgeId([tri.0[2], tri.0[0]]), depth - 1,
                edge_removable, tri_removable, edge_addable, tri_addable, bad_edges) || !self.contains_tri(tri)
        } else if edge_addable(self, vp, vn) &&
            tri_addable(self, tri.0[0], vp, vn) && bad_edges.iter().all(|edge| {let tri = TriId::from_valid([tri.0[0], vp, vn]); !tri.edges().contains(edge) && !tri.edges().contains(&edge.twin())}) &&
            tri_addable(self, tri.0[1], vp, vn) && bad_edges.iter().all(|edge| {let tri = TriId::from_valid([tri.0[1], vp, vn]); !tri.edges().contains(edge) && !tri.edges().contains(&edge.twin())}) &&
            tri_addable(self, tri.0[2], vp, vn) && bad_edges.iter().all(|edge| {let tri = TriId::from_valid([tri.0[2], vp, vn]); !tri.edges().contains(edge) && !tri.edges().contains(&edge.twin())})
        {
            self.flip23(tri.0[0], tri.0[1], tri.0[2], vp, vn);
            true
        } else {
            !self.contains_tri(tri)
        }
    }

    /// Attempts to remove an edge with flips and returns whether this succeeded.
    fn remove_edge_via_flips<EI: TryInto<EdgeId>>(&mut self, edge: EI, depth: usize,
        edge_removable: impl Fn(&Self, EdgeId) -> bool + Clone,
        tri_removable: impl Fn(&Self, TriId) -> bool + Clone,
        edge_addable: impl Fn(&Self, VertexId, VertexId) -> bool + Clone,
        tri_addable: impl Fn(&Self, VertexId, VertexId, VertexId) -> bool + Clone,
        bad_edges: &mut Vec<EdgeId>,
    ) -> bool 
    where
        Self: HasTets<MwbT = B1> + HasPosition3D + Sized,
        Self::V: Position<Dim = U3>,
    {
        if depth == 0 {
            return false;
        }

        let edge = match edge.try_into() {
            Ok(edge) => edge,
            Err(_) => return false,
        };

        if !edge_removable(self, edge) {
            return false;
        }

        let mut vertex_opps = self.edge_vertex_opps(edge).collect::<Vec<_>>();

        // Make sure this isn't a boundary edge
        if vertex_opps.iter().any(|opp| !self.contains_tri(TriId::from_valid([edge.0[1], edge.0[0], *opp]))) {
            return false;
        }

        // From here on out, the edge may be removed in a roundabout way.
        // So check if the edge was removed before lying and saying it wasn't.

        // Attempt to remove triangles around the edge until there are just 3 left
        while vertex_opps.len() > 3 {
            let opp = vertex_opps.pop().unwrap();

            // Avoid flipping a triangle to the edge I'm trying to remove triangles from!
            bad_edges.push(edge);
            let removed = self.remove_tri_via_flips(TriId::from_valid([edge.0[0], edge.0[1], opp]), depth,
                edge_removable.clone(),
                tri_removable.clone(),
                edge_addable.clone(),
                tri_addable.clone(),
                bad_edges
            );
            bad_edges.pop();

            if !removed {
                return !self.contains_edge(edge);
            }
        }

        let [v0, v1, v2] = [vertex_opps[0], vertex_opps[1], vertex_opps[2]];
        let v012 = TriId::from_valid([v0, v1, v2]);

        // Check for concavities
        let orient = sim::orient_3d(self, index_fn, v0, v1, v2, edge.0[0]);
        if  orient == sim::orient_3d(self, index_fn, v2, v1, v0, edge.0[1]) &&
            tri_removable(self, TriId::from_valid([v0, edge.0[0], edge.0[1]])) &&
            tri_removable(self, TriId::from_valid([v1, edge.0[0], edge.0[1]])) &&
            tri_removable(self, TriId::from_valid([v2, edge.0[0], edge.0[1]])) &&
            tri_addable(self, v0, v1, v2) && bad_edges.iter().all(|edge| !v012.edges().contains(edge) && !v012.edges().contains(&edge.twin()))
        {
            let index = if orient { 0 } else { 1 };
            self.flip32(v0, v1, v2, edge.0[index], edge.0[1 - index]);
            true
        } else {
            !self.contains_edge(edge)
        }
    }

    /// Adds a tetrahedron to the mesh. Vertex order is important!
    /// If the tetrahedron was already there, this replaces the value.
    /// Adds in the required edges and triangles if they aren't there already.
    /// Returns the previous value of the tetrahedron, if there was one.
    ///
    /// In case of a mwb tet mesh, any tetrahedrons that were already
    /// attached to an oriented triangle of the new tetrahedron get removed, along with their triangles and edges.
    ///
    /// # Panics
    /// Panics if any vertex doesn't exist or if any two vertices are the same.
    fn add_tet<TI: TryInto<TetId>>(&mut self, vertices: TI, value: Self::T) -> Option<Self::T> {
        let id = vertices.try_into().ok().unwrap();

        for tri in &id.tris() {
            if self.tri(*tri).is_none() {
                self.add_tri(*tri, self.default_tri());
            }
        }

        // Can't use entry().or_insert() because that would cause a
        // mutable borrow and an immutable borrow to exist at the same time
        if let Some(tet) = self.tets_r_mut::<Key>().get_mut(&id) {
            Some(std::mem::replace(tet.value_mut::<Key>(), value))
        } else {
            let mut opps = [Link::dummy(VertexId::dummy); 4];

            for (i, (tri, opp)) in id.tris_and_opp().iter().enumerate() {
                let target = self.tris_r::<Key>()[tri].tet_opp::<Key>();

                let (prev, next) = if target == tri.0[0] || <<Self::Tet as Tet>::Mwb as Bit>::BOOL {
                    if target != tri.0[0] {
                        // "Mwb" condition requires ≤1 tetrahedron per oriented triangle!
                        self.remove_tet_keep_tris(TetId::from_valid([
                            tri.0[0], tri.0[1], tri.0[2], target,
                        ]));
                        // Triangles were attached to that tetrahedron and should be removed
                        self.remove_tri(TriId::from_valid([target, tri.0[2], tri.0[1]]));
                        self.remove_tri(TriId::from_valid([tri.0[2], target, tri.0[0]]));
                        self.remove_tri(TriId::from_valid([tri.0[1], tri.0[0], target]));
                    }
                    // First tet from tri
                    *self
                        .tris_r_mut::<Key>()
                        .get_mut(tri)
                        .unwrap()
                        .tet_opp_mut::<Key>() = *opp;
                    (*opp, *opp)
                } else {
                    let side = [tri.0[0], tri.0[1], tri.0[2], target]
                        .try_into()
                        .ok()
                        .unwrap();
                    let prev = self.tets_r::<Key>()[&side].link::<Key>(side, *tri).prev;
                    let next = target;
                    let prev_tet = id.with_opp(*tri, prev);
                    let next_tet = id.with_opp(*tri, next);
                    self.tets_r_mut::<Key>()
                        .get_mut(&prev_tet)
                        .unwrap()
                        .link_mut::<Key>(prev_tet, *tri)
                        .next = *opp;
                    self.tets_r_mut::<Key>()
                        .get_mut(&next_tet)
                        .unwrap()
                        .link_mut::<Key>(next_tet, *tri)
                        .prev = *opp;
                    (prev, next)
                };

                opps[i] = Link::new(prev, next);
            }

            self.tets_r_mut::<Key>()
                .insert(id, Tet::new::<Key>(id.0[0], opps, value));
            None
        }
    }

    /// Extends the tetrahedron list with an iterator.
    ///
    /// # Panics
    /// Panics if any vertex doesn't exist or if any two vertices are the same
    /// in any of the tetrahedrons.
    fn extend_tets<TI: TryInto<TetId>, I: IntoIterator<Item = (TI, Self::T)>>(&mut self, iter: I) {
        iter.into_iter().for_each(|(id, value)| {
            self.add_tet(id, value);
        })
    }

    /// Removes an tetrahedron from the mesh and returns the value that was there,
    /// or None if there was nothing there.
    /// Removes the edges and tris that are part of the tetrahedron if they are part of no other tetrahedrons
    /// and the tetrahedron to be removed exists.
    fn remove_tet<TI: TryInto<TetId>>(&mut self, id: TI) -> Option<Self::T> {
        let id = id.try_into().ok()?;

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
    fn remove_tet_keep_tris<TI: TryInto<TetId>>(&mut self, id: TI) -> Option<Self::T> {
        let id = id.try_into().ok()?;

        if self.tet(id).is_some() {
            self.remove_tet_higher::<Key>(id);

            for (i, (tri, opp)) in id.tris_and_opp().iter().enumerate() {
                let next = if <<Self::Tet as Tet>::Mwb as Bit>::BOOL {
                    *opp
                } else {
                    let tet = &self.tets_r::<Key>()[&id];
                    let prev = tet.links::<Key>()[i].prev;
                    let next = tet.links::<Key>()[i].next;
                    let prev_tet = id.with_opp(*tri, prev);
                    let next_tet = id.with_opp(*tri, next);
                    self.tets_r_mut::<Key>()
                        .get_mut(&prev_tet)
                        .unwrap()
                        .link_mut::<Key>(prev_tet, *tri)
                        .next = next;
                    self.tets_r_mut::<Key>()
                        .get_mut(&next_tet)
                        .unwrap()
                        .link_mut::<Key>(next_tet, *tri)
                        .prev = prev;
                    next
                };

                let source = self.tris_r_mut::<Key>().get_mut(&tri).unwrap();
                if *opp == next {
                    // this was the last tet from the triangle
                    *source.tet_opp_mut::<Key>() = tri.0[0];
                } else if *opp == source.tet_opp::<Key>() {
                    *source.tet_opp_mut::<Key>() = next;
                }
            }

            self.tets_r_mut::<Key>()
                .remove(&id)
                .map(|f| f.to_value::<Key>())
        } else {
            None
        }
    }

    /// Removes a list of tetrahedrons.
    fn remove_tets<TI: TryInto<TetId>, I: IntoIterator<Item = TI>>(&mut self, iter: I) {
        iter.into_iter().for_each(|id| {
            self.remove_tet(id);
        })
    }

    /// Removes a list of tetrahedrons.
    fn remove_tets_keep_tris<TI: TryInto<TetId>, I: IntoIterator<Item = TI>>(&mut self, iter: I) {
        iter.into_iter().for_each(|id| {
            self.remove_tet_keep_tris(id);
        })
    }

    /// Keeps only the tetrahedrons that satisfy a predicate
    fn retain_tets<P: FnMut(TetId, &Self::T) -> bool>(&mut self, mut predicate: P) {
        let to_remove = self
            .tets()
            .filter(|(id, f)| !predicate(**id, *f))
            .map(|(id, _)| *id)
            .collect::<Vec<_>>();
        self.remove_tets(to_remove);
    }

    /// Keeps only the tetrahedrons that satisfy a predicate
    fn retain_tets_keep_tris<P: FnMut(TetId, &Self::T) -> bool>(&mut self, mut predicate: P) {
        let to_remove = self
            .tets()
            .filter(|(id, f)| !predicate(**id, *f))
            .map(|(id, _)| *id)
            .collect::<Vec<_>>();
        self.remove_tets_keep_tris(to_remove);
    }

    /// Removes all tetrahedrons from the mesh.
    fn clear_tets(&mut self) {
        self.clear_tets_higher::<Key>();
        self.tets_r_mut::<Key>().clear();

        // Fix tri-target links
        for (id, tri) in self.tris_r_mut::<Key>() {
            *tri.tet_opp_mut::<Key>() = id.0[0];
        }
    }

    /// Gets a tetrahedron walker that starts at the given edge with a given vertex
    /// to form the starting triangle.
    /// Returns None if the triangle formed has no tetrahedron.
    fn tet_walker_from_edge_vertex<EI: TryInto<EdgeId>>(
        &self,
        edge: EI,
        vertex: VertexId,
    ) -> Option<TetWalker<Self>> {
        let edge = self.edge_id(edge)?;
        let _ = self.tri_id([edge.0[0], edge.0[1], vertex])?;
        TetWalker::from_edge_vertex(self, edge, vertex)
    }

    /// Gets a tetrahedron walker that starts at the given triangle.
    /// Returns None if the triangle has no tetrahedron.
    /// Be warned that this does not preserve the order of the vertices
    /// because the triangle id is canonicalized.
    fn tet_walker_from_tri<FI: TryInto<TriId>>(&self, tri: FI) -> Option<TetWalker<Self>> {
        let tri = tri.try_into().ok()?;
        // This method checks if the triangle exists already, so no need to do it above
        TetWalker::from_edge_vertex(self, tri.edges()[0], tri.0[2])
    }

    /// Gets a tetrahedron walker that starts at the given edge with the given opposite edge.
    /// They must actually exist.
    fn tet_walker_from_edge_edge<EI: TryInto<EdgeId>, EJ: TryInto<EdgeId>>(
        &self,
        edge: EI,
        opp: EJ,
    ) -> TetWalker<Self> {
        TetWalker::new(self, edge, opp)
    }

    /// Gets a tetrahedron walker that starts at the given tetrahedron.
    /// It must actually exist.
    /// Be warned that this does not preserve the order of the vertices
    /// because the tetrahedron id is canonicalized.
    fn tet_walker_from_tet<TI: TryInto<TetId>>(&self, tet: TI) -> TetWalker<Self> {
        let tet = tet.try_into().ok().unwrap();
        TetWalker::new(
            self,
            EdgeId([tet.0[0], tet.0[1]]),
            EdgeId([tet.0[2], tet.0[3]]),
        )
    }

    /// Converts this into a triangle mesh where each tetrahedron
    /// turns into 4 separate vertices and 4 triangles with those vertices.
    fn to_separate_tets(&self) -> Self::WithoutTets
    where
        Self::V: Clone,
        Self::E: Clone,
        Self::F: Clone,
        Self::WithoutTets: HasTris<HigherF = B0>
    {
        let mut mesh = <Self::WithoutTets as HasTris>::from_vef_r::<_, _, _, Key>(vec![], vec![], vec![],
            self.default_v_r::<Key>(), self.default_e_r::<Key>(), self.default_f_r::<Key>());

        for tet in self.tet_ids() {
            let v_map = tet.vertices().iter().map(|v_id| {
                (*v_id, mesh.add_vertex(self.vertex(*v_id).unwrap().clone()))
            }).collect::<OrderedIdMap<_, _>>();

            for edge in &tet.edges() {
                mesh.add_edge(EdgeId([v_map[edge.0[0]], v_map[edge.0[1]]]), self.edge(*edge).unwrap().clone());
            }

            for tri in &tet.tris() {
                mesh.add_tri(TriId::from_valid([v_map[tri.0[0]], v_map[tri.0[1]], v_map[tri.0[2]]]),
                    self.tri(*tri).unwrap().clone());
            }
        }

        mesh
    }
}

/// A walker for navigating a simplicial complex by tetrahedron.
#[derive(Debug)]
pub struct TetWalker<'a, M: ?Sized>
where
    M: HasTets,
{
    mesh: &'a M,
    edge: EdgeId,
    opp: EdgeId,
}

impl<'a, M: ?Sized> Clone for TetWalker<'a, M>
where
    M: HasTets,
{
    fn clone(&self) -> Self {
        Self {
            mesh: self.mesh,
            edge: self.edge,
            opp: self.opp,
        }
    }
}

impl<'a, M: ?Sized> Copy for TetWalker<'a, M> where M: HasTets {}

impl<'a, M: ?Sized> TetWalker<'a, M>
where
    M: HasTets,
{
    fn new<EI: TryInto<EdgeId>, EJ: TryInto<EdgeId>>(mesh: &'a M, edge: EI, opp: EJ) -> Self {
        Self {
            mesh,
            edge: edge.try_into().ok().unwrap(),
            opp: opp.try_into().ok().unwrap(),
        }
    }

    pub(crate) fn from_edge_vertex<EI: TryInto<EdgeId>>(
        mesh: &'a M,
        edge: EI,
        vertex: VertexId,
    ) -> Option<Self> {
        let edge = edge.try_into().ok()?;
        let tri = [edge.0[0], edge.0[1], vertex].try_into().ok()?;
        let opp = mesh.tris_r::<Key>().get(&tri)?.tet_opp::<Key>();
        let _: TetId = [edge.0[0], edge.0[1], vertex, opp].try_into().ok()?;
        Some(Self::new(mesh, edge, EdgeId([vertex, opp])))
    }

    /// A walker that will not be used
    fn dummy(mesh: &'a M) -> Self {
        Self {
            mesh,
            edge: EdgeId::dummy(),
            opp: EdgeId::dummy(),
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

    /// Get the vertex id of the source of the current opposite edge.
    pub fn third(&self) -> VertexId {
        self.opp.0[0]
    }

    /// Gets the opposite vertex of the current triangle
    pub fn fourth(&self) -> VertexId {
        self.opp.0[1]
    }

    /// Gets the opposite edge of the current edge
    pub fn opp_edge(&self) -> EdgeId {
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
    pub fn vertices(&self) -> [VertexId; 4] {
        [self.first(), self.second(), self.third(), self.fourth()]
    }

    /// Gets the current tetrahedron id
    pub fn tet(&self) -> TetId {
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
    pub fn twin(mut self) -> Option<Self> {
        self.edge = self.edge.twin();
        if self.mesh.tet(self.tet()).is_some() {
            Some(self)
        } else {
            None
        }
    }

    /// Set the current tetrahedron to one that contains the twin
    /// of the current triangle. Useful for getting the
    /// other tetrahedron of the unoriented triangle in a mwb.
    pub fn on_twin_tri(self) -> Option<Self> {
        self.tri_walker().twin().and_then(|w| w.tet_walker())
    }

    /// Sets the current edge to the next one in the same current triangle.
    pub fn next_edge(mut self) -> Self {
        let (edge, opp) = (EdgeId([self.second(), self.third()]), self.first());
        self.edge = edge;
        self.opp = EdgeId([opp, self.fourth()]);
        self
    }

    /// Sets the current edge to the previous one in the same current triangle.
    pub fn prev_edge(mut self) -> Self {
        let (edge, opp) = (EdgeId([self.third(), self.first()]), self.second());
        self.edge = edge;
        self.opp = EdgeId([opp, self.fourth()]);
        self
    }

    /// Sets the current triangle to the next one in the same tetrahedron.
    pub fn next_tri(mut self) -> Self {
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

        self.edge = EdgeId([v0, v1]);
        self.opp = EdgeId([v2, v3]);
        self
    }

    /// Switches the current edge and opposite edge
    pub fn flip_tri(mut self) -> Self {
        let (v0, v1, v2, v3) = (self.third(), self.fourth(), self.first(), self.second());
        self.edge = EdgeId([v0, v1]);
        self.opp = EdgeId([v2, v3]);
        self
    }

    /// Sets the current triangle to the previous one in the same tetrahedron.
    pub fn prev_tri(mut self) -> Self {
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

        self.edge = EdgeId([v0, v1]);
        self.opp = EdgeId([v2, v3]);
        self
    }

    /// Sets the current opposite vertex to the next one with the same triangle.
    pub fn next_opp(mut self) -> Self {
        if !<<M::Tet as Tet>::Mwb as Bit>::BOOL {
            let tet = self.tet();
            self.opp.0[1] = self.mesh.tets_r::<Key>()[&tet]
                .link::<Key>(tet, self.tri())
                .next;
        }
        self
    }

    /// Sets the current opposite vertex to the previous one with the same triangle.
    pub fn prev_opp(mut self) -> Self {
        if !<<M::Tet as Tet>::Mwb as Bit>::BOOL {
            let tet = self.tet();
            self.opp.0[1] = self.mesh.tets_r::<Key>()[&tet]
                .link::<Key>(tet, self.tri())
                .prev;
        }
        self
    }

    /// Turns this into an triangle walker that starts
    /// at the current edge and opposite vertex of the current triangle.
    pub fn tri_walker(self) -> TriWalker<'a, M> {
        TriWalker::new(self.mesh, self.edge, self.third())
    }
}

/// An iterator over the opposite triangles of the tetrahedrons of a vertex.
#[derive(Clone)]
pub struct VertexTriOpps<'a, M: ?Sized>
where
    M: HasTets,
{
    mesh: &'a M,
    edges: VertexEdgesOut<'a, M>,
    opps: Option<EdgeEdgeOpps<'a, M>>,
}

impl<'a, M: ?Sized> Iterator for VertexTriOpps<'a, M>
where
    M: HasTets,
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
    M: HasTets,
{
    mesh: &'a M,
    tris: EdgeVertexOpps<'a, M>,
    opps: Option<TriVertexOpps<'a, M>>,
}

impl<'a, M: ?Sized> Iterator for EdgeEdgeOpps<'a, M>
where
    M: HasTets,
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
    M: HasTets,
{
    walker: TetWalker<'a, M>,
    start_opp: VertexId,
    finished: bool,
}

impl<'a, M: ?Sized> Iterator for TriVertexOpps<'a, M>
where
    M: HasTets,
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
    ($name:ident<$v:ident, $e:ident, $f: ident, $t: ident $(, $args:ident)*>) => {
        impl<$v, $e, $f, $t $(, $args)*> std::ops::Index<[crate::vertex::VertexId; 4]> for $name<$v, $e, $f, $t $(, $args)*> {
            type Output = $t;

            fn index(&self, index: [crate::vertex::VertexId; 4]) -> &Self::Output {
                self.tet(index).unwrap()
            }
        }

        impl<$v, $e, $f, $t $(, $args)*> std::ops::IndexMut<[crate::vertex::VertexId; 4]> for $name<$v, $e, $f, $t $(, $args)*> {
            fn index_mut(&mut self, index: [crate::vertex::VertexId; 4]) -> &mut Self::Output {
                self.tet_mut(index).unwrap()
            }
        }

        impl<$v, $e, $f, $t $(, $args)*> std::ops::Index<crate::tet::TetId> for $name<$v, $e, $f, $t $(, $args)*> {
            type Output = $t;

            fn index(&self, index: crate::tet::TetId) -> &Self::Output {
                self.tet(index).unwrap()
            }
        }

        impl<$v, $e, $f, $t $(, $args)*> std::ops::IndexMut<crate::tet::TetId> for $name<$v, $e, $f, $t $(, $args)*> {
            fn index_mut(&mut self, index: crate::tet::TetId) -> &mut Self::Output {
                self.tet_mut(index).unwrap()
            }
        }
    };
}

/// For concrete simplicial complexes with tetrahedrons
pub trait HasPositionAndTets: HasTets + HasPosition
where
    Self::V: Position,
    DefaultAllocator: Allocator<f64, HasPositionDim<Self>>,
{
    /// Gets the positions of the vertices of an tetrahedron
    fn tet_positions<TI: TryInto<TetId>>(&self, tet: TI) -> [HasPositionPoint<Self>; 4] {
        let tet = tet.try_into().ok().unwrap();
        let v0 = self.position(tet.0[0]);
        let v1 = self.position(tet.0[1]);
        let v2 = self.position(tet.0[2]);
        let v3 = self.position(tet.0[3]);
        [v0, v1, v2, v3]
    }
}

#[macro_export]
#[doc(hidden)]
macro_rules! impl_tet {
    ($name:ident<$t:ident>, new |$id:ident, $links:ident, $value:ident| $new:expr) => {
        impl<$t> crate::tet::Tet for $name<$t> {
            type T = $t;
            type Mwb = typenum::B0;

            fn new<L: crate::private::Lock>(
                $id: crate::vertex::VertexId,
                $links: [crate::edge::Link<crate::vertex::VertexId>; 4],
                $value: Self::T,
            ) -> Self {
                $new
            }

            fn links<L: crate::private::Lock>(
                &self,
            ) -> [crate::edge::Link<crate::vertex::VertexId>; 4] {
                self.links
            }

            fn links_mut<L: crate::private::Lock>(
                &mut self,
            ) -> &mut [crate::edge::Link<crate::vertex::VertexId>; 4] {
                &mut self.links
            }

            fn to_value<L: crate::private::Lock>(self) -> Self::T {
                self.value
            }

            fn value<L: crate::private::Lock>(&self) -> &Self::T {
                &self.value
            }

            fn value_mut<L: crate::private::Lock>(&mut self) -> &mut Self::T {
                &mut self.value
            }
        }
    };
}

#[macro_export]
#[doc(hidden)]
macro_rules! impl_tet_mwb {
    ($name:ident<$t:ident>, new |$id:ident, $links:ident, $value:ident| $new:expr) => {
        impl<$t> crate::tet::Tet for $name<$t> {
            type T = $t;
            type Mwb = typenum::B1;

            fn new<L: crate::private::Lock>(
                $id: crate::vertex::VertexId,
                $links: [crate::edge::Link<crate::vertex::VertexId>; 4],
                $value: Self::T,
            ) -> Self {
                $new
            }

            fn links<L: crate::private::Lock>(
                &self,
            ) -> [crate::edge::Link<crate::vertex::VertexId>; 4] {
                panic!("Cannot get links in \"mwb\" tet")
            }

            fn links_mut<L: crate::private::Lock>(
                &mut self,
            ) -> &mut [crate::edge::Link<crate::vertex::VertexId>; 4] {
                panic!("Cannot get links in \"mwb\" tet")
            }

            fn to_value<L: crate::private::Lock>(self) -> Self::T {
                self.value
            }

            fn value<L: crate::private::Lock>(&self) -> &Self::T {
                &self.value
            }

            fn value_mut<L: crate::private::Lock>(&mut self) -> &mut Self::T {
                &mut self.value
            }
        }
    };
}

#[macro_export]
#[doc(hidden)]
macro_rules! impl_has_tets {
    ($tet:ident<$t:ident>, Mwb = $mwb:ty) => {
        type Tet = $tet<$t>;
        type T = $t;
        type MwbT = $mwb;

        fn from_veft_r<
            VI: IntoIterator<
                Item = (
                    crate::vertex::VertexId,
                    <Self::Vertex as crate::vertex::Vertex>::V,
                ),
            >,
            EI: IntoIterator<Item = (crate::edge::EdgeId, <Self::Edge as crate::edge::Edge>::E)>,
            FI: IntoIterator<Item = (crate::tri::TriId, <Self::Tri as crate::tri::Tri>::F)>,
            TI: IntoIterator<Item = (crate::tet::TetId, <Self::Tet as crate::tet::Tet>::T)>,
            L: crate::private::Lock,
        >(
            vertices: VI,
            edges: EI,
            tris: FI,
            tets: TI,
            default_v: fn() -> Self::V,
            default_e: fn() -> Self::E,
            default_f: fn() -> Self::F,
            default_t: fn() -> Self::T,
        ) -> Self {
            let mut mesh = Self::with_defaults(default_v, default_e, default_f, default_t);
            mesh.extend_vertices_with_ids(vertices);
            mesh.extend_edges(edges);
            mesh.extend_tris(tris);
            mesh.extend_tets(tets);
            mesh
        }

        fn into_veft_r<L: crate::private::Lock>(
            self,
        ) -> (
            crate::vertex::IntoVertices<Self::Vertex>,
            crate::edge::IntoEdges<Self::Edge>,
            crate::tri::IntoTris<Self::Tri>,
            crate::tet::IntoTets<Self::Tet>,
        ) {
            use crate::edge::Edge;
            use crate::tet::Tet;
            use crate::tri::Tri;
            use crate::vertex::Vertex;
            (
                self.vertices
                    .into_iter()
                    .map(|(id, v)| (id, v.to_value::<crate::private::Key>())),
                self.edges
                    .into_iter()
                    .map(|(id, e)| (id, e.to_value::<crate::private::Key>())),
                self.tris
                    .into_iter()
                    .map(|(id, f)| (id, f.to_value::<crate::private::Key>())),
                self.tets
                    .into_iter()
                    .map(|(id, t)| (id, t.to_value::<crate::private::Key>())),
            )
        }

        fn tets_r<L: crate::private::Lock>(&self) -> &FnvHashMap<crate::tet::TetId, Self::Tet> {
            &self.tets
        }

        fn tets_r_mut<L: crate::private::Lock>(
            &mut self,
        ) -> &mut FnvHashMap<crate::tet::TetId, Self::Tet> {
            &mut self.tets
        }

        fn default_t_r<L: crate::private::Lock>(&self) -> fn() -> Self::T {
            self.default_t
        }
    };
}
