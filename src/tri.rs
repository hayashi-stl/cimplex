//! Traits and structs related to triangles

use fnv::{FnvHashMap, FnvHashSet};
use nalgebra::{allocator::Allocator, DefaultAllocator};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::collections::hash_map;
use std::convert::{TryFrom, TryInto};
use std::iter::Map;
use typenum::{Bit, B0, B1};

use crate::iter::{IteratorExt, MapWith};
use crate::private::{Key, Lock};
use crate::vertex::VertexId;
use crate::{
    edge::{Edge, IntoEdges, Link},
    vertex::{HasVertices, IntoVertices},
};
use crate::{
    edge::{EdgeId, EdgeWalker, HasEdges, VertexEdgesOut},
    tet::{HasTets, TetWalker},
    vertex::{HasPosition, HasPositionDim, HasPositionPoint, Position},
};

/// An triangle id is just the triangle's vertices in winding order,
/// with the smallest index first.
/// No two vertices are allowed to be the same.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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

    pub(crate) fn invalid() -> Self {
        Self([VertexId(0); 3])
    }

    /// Conversion without checking for inequality of the vertices
    pub(crate) fn from_valid(v: [VertexId; 3]) -> Self {
        Self(Self::canonicalize(v))
    }

    /// Canonicalizes this tri id into an undirected version.
    pub fn undirected(mut self) -> TriId {
        if self.0[1] > self.0[2] {
            self.0.swap(1, 2);
        }
        self
    }

    /// Gets the vertices that this tri id is made of
    pub fn vertices(self) -> [VertexId; 3] {
        self.0
    }

    /// Whether this contains some vertex
    pub fn contains_vertex(self, vertex: VertexId) -> bool {
        self.0.contains(&vertex)
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

    /// Gets the opposite edge of a vertex.
    pub fn opp_edge(self, vertex: VertexId) -> EdgeId {
        let index = self.index(vertex);
        EdgeId([self.0[(index + 1) % 3], self.0[(index + 2) % 3]])
    }

    /// Gets the opposite vertex of an edge.
    pub fn opp_vertex(self, edge: EdgeId) -> VertexId {
        self.0[self.opp_index(edge)]
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
    #[allow(dead_code)]
    pub(crate) fn twin(self) -> Self {
        Self([self.0[0], self.0[2], self.0[1]])
    }

    /// Sets the opposite vertex of some edge to some vertex.
    fn with_opp(mut self, edge: EdgeId, vertex: VertexId) -> Self {
        self.0[self.opp_index(edge)] = vertex;
        self.0 = Self::canonicalize(self.0);
        self
    }
}

/// Iterator over the triangle ids of a mesh.
pub type TriIds<'a, FT> = hash_map::Keys<'a, TriId, FT>;

/// Iterator over the triangles of a mesh.
pub type IntoTris<FT> =
    Map<hash_map::IntoIter<TriId, FT>, fn((TriId, FT)) -> (TriId, <FT as Tri>::F)>;

/// Iterator over the triangles of a mesh.
pub type Tris<'a, FT> = Map<
    hash_map::Iter<'a, TriId, FT>,
    for<'b> fn((&'b TriId, &'b FT)) -> (&'b TriId, &'b <FT as Tri>::F),
>;

/// Iterator over the triangles of a mesh mutably.
pub type TrisMut<'a, FT> = Map<
    hash_map::IterMut<'a, TriId, FT>,
    for<'b> fn((&'b TriId, &'b mut FT)) -> (&'b TriId, &'b mut <FT as Tri>::F),
>;

/// Iterator over the triangles connected to an edge with the correct winding.
pub type EdgeTris<'a, M> = MapWith<EdgeId, EdgeVertexOpps<'a, M>, fn(EdgeId, VertexId) -> TriId>;
/// Iterator over the triangles connected to a vertex.
pub type VertexTris<'a, M> =
    MapWith<VertexId, VertexEdgeOpps<'a, M>, fn(VertexId, EdgeId) -> TriId>;

/// Triangle attributes
pub trait Tri {
    type F;
    type Mwb: Bit;
    type Higher: Bit;

    /// Takes the vertex id of the source, in case
    /// the triangle needs to store a dummy value for the opposite vertex
    /// of the tetrahedron.
    #[doc(hidden)]
    fn new<L: Lock>(id: VertexId, links: [Link<VertexId>; 3], value: Self::F) -> Self;

    #[doc(hidden)]
    fn links<L: Lock>(&self) -> [Link<VertexId>; 3];

    #[doc(hidden)]
    fn links_mut<L: Lock>(&mut self) -> &mut [Link<VertexId>; 3];

    #[doc(hidden)]
    fn to_value<L: Lock>(self) -> Self::F;

    #[doc(hidden)]
    fn value<L: Lock>(&self) -> &Self::F;

    #[doc(hidden)]
    fn value_mut<L: Lock>(&mut self) -> &mut Self::F;

    #[doc(hidden)]
    fn link<L: Lock>(&self, id: TriId, edge: EdgeId) -> Link<VertexId> {
        self.links::<Key>()[id.index(edge.0[0])]
    }

    #[doc(hidden)]
    fn link_mut<L: Lock>(&mut self, id: TriId, edge: EdgeId) -> &mut Link<VertexId> {
        &mut self.links_mut::<Key>()[id.index(edge.0[0])]
    }

    #[doc(hidden)]
    fn tet_opp<L: Lock>(&self) -> VertexId
    where
        Self: Tri<Higher = B1>;

    #[doc(hidden)]
    fn tet_opp_mut<L: Lock>(&mut self) -> &mut VertexId
    where
        Self: Tri<Higher = B1>;
}

/// Allows upgrading to a simplicial 2-complex.
pub trait WithTris<V, E, F> {
    type WithTris: HasVertices<V = V> + HasEdges<E = E> + HasTris<F = F>;
}

/// For simplicial complexes that can have triangles
pub trait HasTris: HasEdges<HigherE = B1> {
    type Tri: Tri<F = Self::F, Mwb = Self::MwbF, Higher = Self::HigherF>;
    type F;
    type MwbF: Bit;
    type HigherF: Bit;
    type WithoutTris: HasVertices<V = Self::V> + HasEdges<E = Self::E, HigherE = B0>;
    // Can't upgrade without GATs
    //type WithTets: HasVertices<V = Self::V> + HasEdges<E = Self::E> + HasTris<F = Self::F> + HasTets;
    type WithMwbF: HasVertices<V = Self::V>
        + HasEdges<E = Self::E>
        + HasTris<F = Self::F, MwbF = B1>;
    type WithoutMwbF: HasVertices<V = Self::V>
        + HasEdges<E = Self::E>
        + HasTris<F = Self::F, MwbF = B0>;

    #[doc(hidden)]
    fn from_vef_r<
        VI: IntoIterator<Item = (VertexId, Self::V)>,
        EI: IntoIterator<Item = (EdgeId, Self::E)>,
        FI: IntoIterator<Item = (TriId, Self::F)>,
        L: Lock,
    >(
        vertices: VI,
        edges: EI,
        tris: FI,
        default_v: fn() -> Self::V,
        default_e: fn() -> Self::E,
        default_f: fn() -> Self::F,
    ) -> Self
    where
        Self: HasTris<HigherF = B0>;

    #[doc(hidden)]
    fn into_vef_r<L: Lock>(
        self,
    ) -> (
        IntoVertices<Self::Vertex>,
        IntoEdges<Self::Edge>,
        IntoTris<Self::Tri>,
    );

    #[doc(hidden)]
    fn tris_r<L: Lock>(&self) -> &FnvHashMap<TriId, Self::Tri>;

    #[doc(hidden)]
    fn tris_r_mut<L: Lock>(&mut self) -> &mut FnvHashMap<TriId, Self::Tri>;

    #[doc(hidden)]
    fn remove_tri_higher<L: Lock>(&mut self, tri: TriId);

    #[doc(hidden)]
    fn clear_tris_higher<L: Lock>(&mut self);

    #[doc(hidden)]
    fn default_f_r<L: Lock>(&self) -> fn() -> Self::F;

    #[doc(hidden)]
    #[cfg(feature = "obj")]
    fn obj_with_tris<L: Lock>(&self, data: &mut obj::ObjData, v_inv: &FnvHashMap<VertexId, usize>) {
        // Triangles
        data.objects[0].groups[0].polys.extend(
            self.tri_ids().map(|tri| {
                obj::SimplePolygon(vec![
                    obj::IndexTuple(v_inv[&tri.0[0]], None, None),
                    obj::IndexTuple(v_inv[&tri.0[1]], None, None),
                    obj::IndexTuple(v_inv[&tri.0[2]], None, None),
                ])
            })
        );

        // Isolated edges
        data.objects[0].groups[0].polys.extend(
            self.edge_ids().flat_map(|e| self.edge_vertex_opps(*e).next().map(|_| e.undirected()))
                .collect::<FnvHashSet<_>>().into_iter().map(|edge| {
                    obj::SimplePolygon(vec![
                        obj::IndexTuple(v_inv[&edge.0[0]], None, None),
                        obj::IndexTuple(v_inv[&edge.0[1]], None, None),
                    ])
                })
        );

        self.obj_with_tris_higher::<Key>(data, v_inv);
    }

    #[doc(hidden)]
    #[cfg(feature = "obj")]
    fn obj_with_tris_higher<L: Lock>(&self, data: &mut obj::ObjData, v_inv: &FnvHashMap<VertexId, usize>);

    /// Gets the default value of a triangle.
    fn default_tri(&self) -> Self::F {
        self.default_f_r::<Key>()()
    }

    /// Gets the number of triangles.
    fn num_tris(&self) -> usize {
        self.tris_r::<Key>().len()
    }

    /// Iterates over the triangle ids of this mesh.
    fn tri_ids(&self) -> TriIds<Self::Tri> {
        self.tris_r::<Key>().keys()
    }

    /// Iterates over the triangles of this mesh.
    /// Gives (id, value) pairs
    fn tris(&self) -> Tris<Self::Tri> {
        self.tris_r::<Key>()
            .iter()
            .map(|(id, f)| (id, f.value::<Key>()))
    }

    /// Iterates mutably over the triangles of this mesh.
    /// Gives (id, value) pairs
    fn tris_mut(&mut self) -> TrisMut<Self::Tri> {
        self.tris_r_mut::<Key>()
            .iter_mut()
            .map(|(id, f)| (id, f.value_mut::<Key>()))
    }

    /// Gets whether the mesh contains some triangle.
    fn contains_tri<FI: TryInto<TriId>>(&self, id: FI) -> bool {
        id.try_into()
            .ok()
            .and_then(|id| self.tris_r::<Key>().get(&id))
            .is_some()
    }

    /// Takes a tri id and returns it back if the triangle exists,
    /// or None if it doesn't.
    /// Useful for composing with functions that assume the triangle exists.
    fn tri_id<FI: TryInto<TriId>>(&self, id: FI) -> Option<TriId> {
        id.try_into().ok().and_then(|id| if self.contains_tri(id) { Some(id) } else { None })
    }

    /// Gets the value of the triangle at a specific id.
    /// Returns None if not found.
    fn tri<FI: TryInto<TriId>>(&self, id: FI) -> Option<&Self::F> {
        id.try_into()
            .ok()
            .and_then(|id| self.tris_r::<Key>().get(&id))
            .map(|f| f.value::<Key>())
    }

    /// Gets the value of the triangle at a specific id mutably.
    /// Returns None if not found.
    fn tri_mut<FI: TryInto<TriId>>(&mut self, id: FI) -> Option<&mut Self::F> {
        id.try_into()
            .ok()
            .and_then(move |id| self.tris_r_mut::<Key>().get_mut(&id))
            .map(|f| f.value_mut::<Key>())
    }

    /// Iterates over the opposite edges of the triangles that a vertex is part of.
    fn vertex_edge_opps(&self, vertex: VertexId) -> VertexEdgeOpps<Self> {
        VertexEdgeOpps {
            mesh: self,
            edges: self.vertex_edges_out(vertex),
            opps: None,
        }
    }

    /// Iterates over the triangles that a vertex is part of.
    fn vertex_tris(&self, vertex: VertexId) -> VertexTris<Self> {
        self.vertex_edge_opps(vertex)
            .map_with(vertex, |vertex, opp| {
                TriId::from_valid([vertex, opp.0[0], opp.0[1]])
            })
    }

    /// Iterates over the opposite vertices of the triangles that an edge is part of.
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

    /// Gets the opposite vertex of the ≤1 outgoing triangle that the edge is part of.
    fn edge_vertex_opp<EI: TryInto<EdgeId>>(&self, edge: EI) -> Option<VertexId>
    where
        Self: HasTris<MwbF = typenum::B1>,
    {
        let edge = edge.try_into().ok()?;
        let opp = self.edges_r::<Key>().get(&edge)?.tri_opp::<Key>();
        if opp != edge.0[0] {
            Some(opp)
        } else {
            None
        }
    }

    /// Iterates over the triangles that an edge is part of.
    fn edge_tris<EI: TryInto<EdgeId>>(&self, edge: EI) -> EdgeTris<Self> {
        let edge = edge.try_into().unwrap_or(EdgeId::invalid());
        self.edge_vertex_opps(edge).map_with(edge, |edge, opp| {
            TriId::from_valid([edge.0[0], edge.0[1], opp])
        })
    }

    /// Gets the ≤1 triangle that the edge is part of.
    fn edge_tri<EI: TryInto<EdgeId>>(&self, edge: EI) -> Option<TriId>
    where
        Self: HasTris<MwbF = typenum::B1>,
    {
        let edge = edge.try_into().ok()?;
        Some(TriId::from_valid([
            edge.0[0],
            edge.0[1],
            self.edge_vertex_opp(edge)?,
        ]))
    }

    /// Adds a triangle to the mesh. Vertex order is important!
    /// If the triangle was already there, this replaces the value.
    /// Adds in the required edges if they aren't there already.
    /// Returns the previous value of the triangle, if there was one.
    ///
    /// In case of a mwb tri mesh, any triangles that were already
    /// attached to an oriented edge of the new triangle get removed, along with their edges.
    ///
    /// # Panics
    /// Panics if any vertex doesn't exist or if any two vertices are the same.
    fn add_tri<FI: TryInto<TriId>>(&mut self, vertices: FI, value: Self::F) -> Option<Self::F> {
        let id = vertices.try_into().ok().unwrap();

        for edge in &id.edges() {
            if self.edge(*edge).is_none() {
                self.add_edge(*edge, self.default_edge());
            }
        }

        // Can't use entry().or_insert() because that would cause a
        // mutable borrow and an immutable borrow to exist at the same time
        if let Some(tri) = self.tris_r_mut::<Key>().get_mut(&id) {
            Some(std::mem::replace(tri.value_mut::<Key>(), value))
        } else {
            let mut opps = [Link::dummy(VertexId::dummy); 3];

            for (i, (edge, opp)) in id.edges_and_opp().iter().enumerate() {
                let target = self.edges_r::<Key>()[edge].tri_opp::<Key>();

                let (prev, next) = if target == edge.0[0] || <<Self::Tri as Tri>::Mwb as Bit>::BOOL
                {
                    if target != edge.0[0] {
                        // "Mwb" condition requires ≤1 triangle per oriented edge!
                        self.remove_tri_keep_edges(TriId::from_valid([
                            edge.0[0], edge.0[1], target,
                        ]));
                        // Edges were attached to that triangle and should be removed
                        self.remove_edge(EdgeId([edge.0[1], target]));
                        self.remove_edge(EdgeId([target, edge.0[0]]));
                    }
                    // First tri from edge
                    *self
                        .edges_r_mut::<Key>()
                        .get_mut(edge)
                        .unwrap()
                        .tri_opp_mut::<Key>() = *opp;
                    (*opp, *opp)
                } else {
                    let side = [edge.0[0], edge.0[1], target].try_into().ok().unwrap();
                    let prev = self.tris_r::<Key>()[&side].link::<Key>(side, *edge).prev;
                    let next = target;
                    let prev_tri = id.with_opp(*edge, prev);
                    let next_tri = id.with_opp(*edge, next);
                    self.tris_r_mut::<Key>()
                        .get_mut(&prev_tri)
                        .unwrap()
                        .link_mut::<Key>(prev_tri, *edge)
                        .next = *opp;
                    self.tris_r_mut::<Key>()
                        .get_mut(&next_tri)
                        .unwrap()
                        .link_mut::<Key>(next_tri, *edge)
                        .prev = *opp;
                    (prev, next)
                };

                opps[i] = Link::new(prev, next);
            }

            self.tris_r_mut::<Key>()
                .insert(id, Tri::new::<Key>(id.0[0], opps, value));
            None
        }
    }

    /// Extends the triangle list with an iterator.
    ///
    /// # Panics
    /// Panics if any vertex doesn't exist or if any two vertices are the same
    /// in any of the triangles.
    fn extend_tris<FI: TryInto<TriId>, I: IntoIterator<Item = (FI, Self::F)>>(&mut self, iter: I) {
        iter.into_iter().for_each(|(id, value)| {
            self.add_tri(id, value);
        })
    }

    /// Removes an triangle from the mesh and returns the value that was there,
    /// or None if there was nothing there.
    /// Removes the edges that are part of the triangle if they are part of no other triangles
    /// and the triangle to be removed exists.
    fn remove_tri<FI: TryInto<TriId>>(&mut self, id: FI) -> Option<Self::F> {
        let id = id.try_into().ok()?;

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
    fn remove_tri_keep_edges<FI: TryInto<TriId>>(&mut self, id: FI) -> Option<Self::F> {
        let id = id.try_into().ok()?;

        if self.tri(id).is_some() {
            self.remove_tri_higher::<Key>(id);

            for (i, (edge, opp)) in id.edges_and_opp().iter().enumerate() {
                let next = if <<Self::Tri as Tri>::Mwb as Bit>::BOOL {
                    *opp
                } else {
                    let tri = &self.tris_r::<Key>()[&id];
                    let prev = tri.links::<Key>()[i].prev;
                    let next = tri.links::<Key>()[i].next;
                    let prev_tri = id.with_opp(*edge, prev);
                    let next_tri = id.with_opp(*edge, next);
                    self.tris_r_mut::<Key>()
                        .get_mut(&prev_tri)
                        .unwrap()
                        .link_mut::<Key>(prev_tri, *edge)
                        .next = next;
                    self.tris_r_mut::<Key>()
                        .get_mut(&next_tri)
                        .unwrap()
                        .link_mut::<Key>(next_tri, *edge)
                        .prev = prev;

                    next
                };

                let source = self.edges_r_mut::<Key>().get_mut(&edge).unwrap();
                if *opp == next {
                    // this was the last tri from the edge
                    *source.tri_opp_mut::<Key>() = edge.0[0];
                } else if *opp == source.tri_opp::<Key>() {
                    *source.tri_opp_mut::<Key>() = next;
                }
            }

            self.tris_r_mut::<Key>()
                .remove(&id)
                .map(|f| f.to_value::<Key>())
        } else {
            None
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
    fn retain_tris<P: FnMut(TriId, &Self::F) -> bool>(&mut self, mut predicate: P) {
        let to_remove = self
            .tris()
            .filter(|(id, f)| !predicate(**id, *f))
            .map(|(id, _)| *id)
            .collect::<Vec<_>>();
        self.remove_tris(to_remove);
    }

    /// Keeps only the triangles that satisfy a predicate
    fn retain_tris_keep_edges<P: FnMut(TriId, &Self::F) -> bool>(&mut self, mut predicate: P) {
        let to_remove = self
            .tris()
            .filter(|(id, f)| !predicate(**id, *f))
            .map(|(id, _)| *id)
            .collect::<Vec<_>>();
        self.remove_tris_keep_edges(to_remove);
    }

    /// Removes all triangles from the mesh.
    fn clear_tris(&mut self) {
        self.clear_tris_higher::<Key>();
        self.tris_r_mut::<Key>().clear();

        // Fix edge-target links
        for (id, edge) in self.edges_r_mut::<Key>() {
            *edge.tri_opp_mut::<Key>() = id.0[0];
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
    M: HasTris,
{
    mesh: &'a M,
    edge: EdgeId,
    opp: VertexId,
}

impl<'a, M: ?Sized> Clone for TriWalker<'a, M>
where
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

impl<'a, M: ?Sized> Copy for TriWalker<'a, M> where M: HasTris {}

impl<'a, M: ?Sized> TriWalker<'a, M>
where
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
        let edge = edge.try_into().ok()?;
        let opp = mesh.edges_r::<Key>().get(&edge)?.tri_opp::<Key>();
        let _: TriId = [
            edge.0[0],
            edge.0[1],
            mesh.edges_r::<Key>()[&edge].tri_opp::<Key>(),
        ]
        .try_into()
        .ok()?;
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
    /// other triangle of the undirected edge in a mwb.
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
        if !<<M::Tri as Tri>::Mwb as Bit>::BOOL {
            let tri = self.tri();
            self.opp = self.mesh.tris_r::<Key>()[&tri]
                .link::<Key>(tri, self.edge)
                .next;
        }
        self
    }

    /// Sets the current opposite vertex to the previous one with the same edge.
    pub fn prev_opp(mut self) -> Self {
        if !<<M::Tri as Tri>::Mwb as Bit>::BOOL {
            let tri = self.tri();
            self.opp = self.mesh.tris_r::<Key>()[&tri]
                .link::<Key>(tri, self.edge)
                .prev;
        }
        self
    }

    /// Turns this into an edge walker that starts
    /// at the current edge.
    pub fn edge_walker(self) -> EdgeWalker<'a, M> {
        EdgeWalker::new(self.mesh, self.edge)
    }

    pub fn tet_walker(self) -> Option<TetWalker<'a, M>>
    where
        M: HasTets,
    {
        TetWalker::from_edge_vertex(self.mesh, self.edge, self.opp)
    }
}

/// An iterator over the opposite edges of the triangles of a vertex.
#[derive(Clone)]
pub struct VertexEdgeOpps<'a, M: ?Sized>
where
    M: HasTris,
{
    mesh: &'a M,
    edges: VertexEdgesOut<'a, M>,
    opps: Option<EdgeVertexOpps<'a, M>>,
}

impl<'a, M: ?Sized> Iterator for VertexEdgeOpps<'a, M>
where
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
    M: HasTris,
{
    pub(crate) walker: TriWalker<'a, M>,
    start_opp: VertexId,
    finished: bool,
}

impl<'a, M: ?Sized> Iterator for EdgeVertexOpps<'a, M>
where
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
    Self::V: Position,
    DefaultAllocator: Allocator<f64, HasPositionDim<Self>>,
{
    /// Gets the positions of the vertices of an triangle.
    /// Assumes the triangle exists.
    fn tri_positions<FI: TryInto<TriId>>(&self, tri: FI) -> [HasPositionPoint<Self>; 3] {
        let tri = tri.try_into().ok().unwrap();
        let v0 = self.position(tri.0[0]);
        let v1 = self.position(tri.0[1]);
        let v2 = self.position(tri.0[2]);
        [v0, v1, v2]
    }
}

#[macro_export]
#[doc(hidden)]
macro_rules! impl_tri {
    ($name:ident<$f:ident>, new |$id:ident, $links:ident, $value:ident| $new:expr) => {
        impl<$f> crate::tri::Tri for $name<$f> {
            type F = $f;
            type Mwb = typenum::B0;
            type Higher = typenum::B0;

            fn new<L: crate::private::Lock>(
                $id: crate::vertex::VertexId,
                $links: [crate::edge::Link<crate::vertex::VertexId>; 3],
                $value: Self::F,
            ) -> Self {
                $new
            }

            fn links<L: crate::private::Lock>(
                &self,
            ) -> [crate::edge::Link<crate::vertex::VertexId>; 3] {
                self.links
            }

            fn links_mut<L: crate::private::Lock>(
                &mut self,
            ) -> &mut [crate::edge::Link<crate::vertex::VertexId>; 3] {
                &mut self.links
            }

            fn to_value<L: crate::private::Lock>(self) -> Self::F {
                self.value
            }

            fn value<L: crate::private::Lock>(&self) -> &Self::F {
                &self.value
            }

            fn value_mut<L: crate::private::Lock>(&mut self) -> &mut Self::F {
                &mut self.value
            }

            fn tet_opp<L: crate::private::Lock>(&self) -> VertexId
            where
                Self: crate::tri::Tri<Higher = typenum::B1>,
            {
                unreachable!()
            }

            fn tet_opp_mut<L: crate::private::Lock>(&mut self) -> &mut VertexId
            where
                Self: crate::tri::Tri<Higher = typenum::B1>,
            {
                unreachable!()
            }
        }
    };
}

#[macro_export]
#[doc(hidden)]
macro_rules! impl_tri_mwb {
    ($name:ident<$f:ident>, new |$id:ident, $links:ident, $value:ident| $new:expr) => {
        impl<$f> crate::tri::Tri for $name<$f> {
            type F = $f;
            type Mwb = typenum::B1;
            type Higher = typenum::B0;

            fn new<L: crate::private::Lock>(
                $id: crate::vertex::VertexId,
                $links: [crate::edge::Link<crate::vertex::VertexId>; 3],
                $value: Self::F,
            ) -> Self {
                $new
            }

            fn links<L: crate::private::Lock>(
                &self,
            ) -> [crate::edge::Link<crate::vertex::VertexId>; 3] {
                panic!("Cannot get links in \"mwb\" tri")
            }

            fn links_mut<L: crate::private::Lock>(
                &mut self,
            ) -> &mut [crate::edge::Link<crate::vertex::VertexId>; 3] {
                panic!("Cannot get links in \"mwb\" tri")
            }

            fn to_value<L: crate::private::Lock>(self) -> Self::F {
                self.value
            }

            fn value<L: crate::private::Lock>(&self) -> &Self::F {
                &self.value
            }

            fn value_mut<L: crate::private::Lock>(&mut self) -> &mut Self::F {
                &mut self.value
            }

            fn tet_opp<L: crate::private::Lock>(&self) -> VertexId
            where
                Self: crate::tri::Tri<Higher = typenum::B1>,
            {
                unreachable!()
            }

            fn tet_opp_mut<L: crate::private::Lock>(&mut self) -> &mut VertexId
            where
                Self: crate::tri::Tri<Higher = typenum::B1>,
            {
                unreachable!()
            }
        }
    };
}

#[macro_export]
#[doc(hidden)]
macro_rules! impl_tri_higher {
    ($name:ident<$f:ident>, new |$id:ident, $links:ident, $value:ident| $new:expr) => {
        impl<$f> crate::tri::Tri for $name<$f> {
            type F = $f;
            type Mwb = typenum::B0;
            type Higher = typenum::B1;

            fn new<L: crate::private::Lock>(
                $id: crate::vertex::VertexId,
                $links: [crate::edge::Link<crate::vertex::VertexId>; 3],
                $value: Self::F,
            ) -> Self {
                $new
            }

            fn links<L: crate::private::Lock>(
                &self,
            ) -> [crate::edge::Link<crate::vertex::VertexId>; 3] {
                self.links
            }

            fn links_mut<L: crate::private::Lock>(
                &mut self,
            ) -> &mut [crate::edge::Link<crate::vertex::VertexId>; 3] {
                &mut self.links
            }

            fn to_value<L: crate::private::Lock>(self) -> Self::F {
                self.value
            }

            fn value<L: crate::private::Lock>(&self) -> &Self::F {
                &self.value
            }

            fn value_mut<L: crate::private::Lock>(&mut self) -> &mut Self::F {
                &mut self.value
            }

            fn tet_opp<L: crate::private::Lock>(&self) -> crate::vertex::VertexId {
                self.tet_opp
            }

            fn tet_opp_mut<L: crate::private::Lock>(&mut self) -> &mut crate::vertex::VertexId {
                &mut self.tet_opp
            }
        }
    };
}

#[macro_export]
#[doc(hidden)]
macro_rules! impl_has_tris {
    ($tri:ident<$f:ident> $($z:ident)*, Mwb = $mwb:ty, Higher = $higher:ident) => {
        type Tri = $tri<$f>;
        type F = $f;
        type MwbF = $mwb;
        type HigherF = $higher;

        fn from_vef_r<
            VI: IntoIterator<
                Item = (
                    crate::vertex::VertexId,
                    <Self::Vertex as crate::vertex::Vertex>::V,
                ),
            >,
            EI: IntoIterator<Item = (crate::edge::EdgeId, <Self::Edge as crate::edge::Edge>::E)>,
            FI: IntoIterator<Item = (crate::tri::TriId, <Self::Tri as crate::tri::Tri>::F)>,
            L: crate::private::Lock,
        >(
            vertices: VI,
            edges: EI,
            tris: FI,
            default_v: fn() -> Self::V,
            default_e: fn() -> Self::E,
            default_f: fn() -> Self::F,
        ) -> Self {
            use typenum::Bit;
            if <$higher>::BOOL {
                unreachable!()
            }
            // The code below will not be executed if the value is invalid.
            #[allow(invalid_value)]
            let mut mesh = Self::with_defaults(default_v, default_e, default_f $(, unsafe { std::mem::$z() } )*);
            mesh.extend_vertices_with_ids(vertices);
            mesh.extend_edges(edges);
            mesh.extend_tris(tris);
            mesh
        }

        fn into_vef_r<L: crate::private::Lock>(
            self,
        ) -> (
            crate::vertex::IntoVertices<Self::Vertex>,
            crate::edge::IntoEdges<Self::Edge>,
            crate::tri::IntoTris<Self::Tri>,
        ) {
            use crate::edge::Edge;
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
            )
        }

        fn tris_r<L: crate::private::Lock>(&self) -> &FnvHashMap<crate::tri::TriId, Self::Tri> {
            &self.tris
        }

        fn tris_r_mut<L: crate::private::Lock>(
            &mut self,
        ) -> &mut FnvHashMap<crate::tri::TriId, Self::Tri> {
            &mut self.tris
        }

        fn default_f_r<L: crate::private::Lock>(&self) -> fn() -> Self::F {
            self.default_f
        }

        crate::if_b0! { $higher =>
            #[cfg(feature = "obj")]
            fn obj_with_tris_higher<L: crate::private::Lock>(&self, _: &mut obj::ObjData, _: &fnv::FnvHashMap<crate::vertex::VertexId, usize>) {}
        }

        crate::if_b1! { $higher =>
            #[cfg(feature = "obj")]
            fn obj_with_tris_higher<L: crate::private::Lock>(&self, data: &mut obj::ObjData, v_inv: &fnv::FnvHashMap<crate::vertex::VertexId, usize>) {
                self.obj_with_tets::<crate::private::Key>(data, v_inv);
            }
        }
    };
}
