//! Traits and structs related to edges

use fnv::{FnvHashMap, FnvHashSet};
use nalgebra::allocator::Allocator;
use nalgebra::DefaultAllocator;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::collections::hash_map;
use std::convert::{TryFrom, TryInto};
use std::iter::Map;
use typenum::{Bit, B0, B1};

use crate::{
    iter::{IteratorExt, MapWith},
    tri::{HasTris, TriWalker},
    vertex::{HasPosition, IntoVertices},
};
//use crate::tri::{HasTris, TriWalker};
use crate::private::{Key, Lock};
use crate::vertex::{HasPositionDim, HasPositionPoint, Position};
use crate::vertex::{HasVertices, Vertex, VertexId};

/// An edge id is just the edge's vertices in order.
/// The vertices are not allowed to be the same.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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

    /// Canonicalizes this edge id into an undirected version.
    pub fn undirected(mut self) -> EdgeId {
        if self.0[0] > self.0[1] {
            self.0.swap(0, 1);
        }
        self
    }

    /// Gets the source vertex.
    pub fn source(self) -> VertexId {
        self.0[0]
    }

    /// Gets the target vertex.
    pub fn target(self) -> VertexId {
        self.0[1]
    }

    /// Whether this contains some vertex
    pub fn contains_vertex(self, vertex: VertexId) -> bool {
        self.0.contains(&vertex)
    }

    /// Gets the opposite vertex of a vertex.
    pub fn opp_vertex(self, vertex: VertexId) -> VertexId {
        self.0[1 - self.index(vertex)]
    }

    /// Gets the index of a vertex, assuming it's part of the edge
    fn index(self, vertex: VertexId) -> usize {
        self.0.iter().position(|v| *v == vertex).unwrap()
    }

    pub(crate) fn invalid() -> Self {
        // Same ID
        Self([VertexId(0); 2])
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

/// Iterator over the edges ids of a mesh.
pub type EdgeIds<'a, ET> = hash_map::Keys<'a, EdgeId, ET>;

/// Iterator over the edges of a mesh.
pub type IntoEdges<ET> =
    Map<hash_map::IntoIter<EdgeId, ET>, fn((EdgeId, ET)) -> (EdgeId, <ET as Edge>::E)>;

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
    MapWith<VertexId, VertexTargets<'a, M>, fn(VertexId, VertexId) -> EdgeId>;
/// Iterator over the edges pointing in to a vertex.
pub type VertexEdgesIn<'a, M> =
    MapWith<VertexId, VertexSources<'a, M>, fn(VertexId, VertexId) -> EdgeId>;

/// A link. Too lazy to refactor this into an internal module
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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
    type Mwb: Bit;
    type Higher: Bit;

    /// Takes the vertex id of the source, in case
    /// the edge needs to store a dummy value for the opposite vertex
    /// of the triangle.
    #[doc(hidden)]
    fn new<L: Lock>(id: VertexId, links: [Link<VertexId>; 2], value: Self::E) -> Self;

    /// Target link, then source link
    #[doc(hidden)]
    fn links<L: Lock>(&self) -> [Link<VertexId>; 2];

    /// Panics for edges in mwb edge meshes
    #[doc(hidden)]
    fn links_mut<L: Lock>(&mut self) -> &mut [Link<VertexId>; 2];

    #[doc(hidden)]
    fn to_value<L: Lock>(self) -> Self::E;

    #[doc(hidden)]
    fn value<L: Lock>(&self) -> &Self::E;

    #[doc(hidden)]
    fn value_mut<L: Lock>(&mut self) -> &mut Self::E;

    #[doc(hidden)]
    fn tri_opp<L: Lock>(&self) -> VertexId
    where
        Self: Edge<Higher = B1>;

    #[doc(hidden)]
    fn tri_opp_mut<L: Lock>(&mut self) -> &mut VertexId
    where
        Self: Edge<Higher = B1>;
}

/// Allows upgrading to a simplicial 1-complex.
pub trait WithEdges<V, E> {
    type WithEdges: HasVertices<V = V> + HasEdges<E = E>;
}

/// For simplicial complexes that can have edges
pub trait HasEdges: HasVertices<HigherV = B1> {
    type Edge: Edge<E = Self::E, Mwb = Self::MwbE, Higher = Self::HigherE>;
    type E;
    type MwbE: Bit;
    type HigherE: Bit;
    type WithoutEdges: HasVertices<V = Self::V, HigherV = B0>;
    type WithMwbE: HasVertices<V = Self::V> + HasEdges<E = Self::E, MwbE = B1>;
    type WithoutMwbE: HasVertices<V = Self::V> + HasEdges<E = Self::E, MwbE = B0>;

    #[doc(hidden)]
    fn from_ve_r<
        VI: IntoIterator<Item = (VertexId, Self::V)>,
        EI: IntoIterator<Item = (EdgeId, Self::E)>,
        L: Lock,
    >(
        vertices: VI,
        edges: EI,
        default_v: fn() -> Self::V,
        default_e: fn() -> Self::E,
    ) -> Self
    where
        Self: HasEdges<HigherE = B0>;

    #[doc(hidden)]
    fn into_ve_r<L: Lock>(self) -> (IntoVertices<Self::Vertex>, IntoEdges<Self::Edge>);

    #[doc(hidden)]
    fn edges_r<L: Lock>(&self) -> &FnvHashMap<EdgeId, Self::Edge>;

    #[doc(hidden)]
    fn edges_r_mut<L: Lock>(&mut self) -> &mut FnvHashMap<EdgeId, Self::Edge>;

    #[doc(hidden)]
    fn remove_edge_higher<L: Lock>(&mut self, edge: EdgeId);

    #[doc(hidden)]
    fn clear_edges_higher<L: Lock>(&mut self);

    #[doc(hidden)]
    fn default_e_r<L: Lock>(&self) -> fn() -> Self::E;

    #[doc(hidden)]
    #[cfg(feature = "obj")]
    fn obj_with_edges<L: Lock>(
        &self,
        data: &mut obj::ObjData,
        v_inv: &FnvHashMap<VertexId, usize>,
    ) {
        // If there are triangles, only isolated edges should be added separately.
        if !<Self::HigherE as Bit>::BOOL {
            data.objects[0].groups[0].polys.extend(
                self.edge_ids()
                    .map(|edge| edge.undirected())
                    .collect::<FnvHashSet<_>>()
                    .into_iter()
                    .map(|edge| {
                        obj::SimplePolygon(vec![
                            obj::IndexTuple(v_inv[&edge.0[0]], None, None),
                            obj::IndexTuple(v_inv[&edge.0[1]], None, None),
                        ])
                    }),
            )
        }

        self.obj_with_edges_higher::<Key>(data, v_inv);
    }

    #[doc(hidden)]
    #[cfg(feature = "obj")]
    fn obj_with_edges_higher<L: Lock>(
        &self,
        data: &mut obj::ObjData,
        v_inv: &FnvHashMap<VertexId, usize>,
    );

    /// Flips an edge into 2 edges with `vertex` between them.
    #[doc(hidden)]
    fn flip12_edge_higher<EI: TryInto<EdgeId>, L: Lock, C: FnMut(&mut Self)>(&mut self, edge: EI, vertex: VertexId, callback: C);

    /// Gets the default value of an edge.
    fn default_edge(&self) -> Self::E {
        self.default_e_r::<Key>()()
    }

    /// Gets the number of edges.
    fn num_edges(&self) -> usize {
        self.edges_r::<Key>().len()
    }

    /// Iterates over the edge ids of this mesh.
    fn edge_ids(&self) -> EdgeIds<Self::Edge> {
        self.edges_r::<Key>().keys()
    }

    /// Iterates over the edges of this mesh.
    /// Gives (id, value) pairs
    fn edges(&self) -> Edges<Self::Edge> {
        self.edges_r::<Key>()
            .iter()
            .map(|(id, e)| (id, e.value::<Key>()))
    }

    /// Iterates mutably over the edges of this mesh.
    /// Gives (id, value) pairs
    fn edges_mut(&mut self) -> EdgesMut<Self::Edge> {
        self.edges_r_mut::<Key>()
            .iter_mut()
            .map(|(id, e)| (id, e.value_mut::<Key>()))
    }

    /// Gets whether the mesh contains some edge.
    fn contains_edge<EI: TryInto<EdgeId>>(&self, id: EI) -> bool {
        id.try_into()
            .ok()
            .and_then(|id| self.edges_r::<Key>().get(&id))
            .is_some()
    }

    /// Takes a edge id and returns it back if the edge exists,
    /// or None if it doesn't.
    /// Useful for composing with functions that assume the edge exists.
    fn edge_id<EI: TryInto<EdgeId>>(&self, id: EI) -> Option<EdgeId> {
        id.try_into().ok().and_then(|id| {
            if self.contains_edge(id) {
                Some(id)
            } else {
                None
            }
        })
    }

    /// Gets the value of the edge at a specific id.
    /// Returns None if not found.
    fn edge<EI: TryInto<EdgeId>>(&self, id: EI) -> Option<&Self::E> {
        id.try_into()
            .ok()
            .and_then(|id| self.edges_r::<Key>().get(&id))
            .map(|e| e.value::<Key>())
    }

    /// Gets the value of the edge at a specific id mutably.
    /// Returns None if not found.
    fn edge_mut<EI: TryInto<EdgeId>>(&mut self, id: EI) -> Option<&mut Self::E> {
        id.try_into()
            .ok()
            .and_then(move |id| self.edges_r_mut::<Key>().get_mut(&id))
            .map(|e| e.value_mut::<Key>())
    }

    /// Iterates over the targets of the outgoing edges of a vertex.
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
    fn vertex_target(&self, vertex: VertexId) -> Option<VertexId>
    where
        Self: HasEdges<MwbE = B1>,
    {
        let target = self.vertices_r::<Key>().get(vertex)?.target::<Key>();
        if target != vertex {
            Some(target)
        } else {
            None
        }
    }

    /// Iterates over the outgoing edges of a vertex.
    fn vertex_edges_out(&self, vertex: VertexId) -> VertexEdgesOut<Self> {
        self.vertex_targets(vertex)
            .map_with(vertex, |s, t| EdgeId([s, t]))
    }

    /// Gets the ≤1 outgoing edge that the vertex is a source of.
    fn vertex_edge_out(&self, vertex: VertexId) -> Option<EdgeId>
    where
        Self: HasEdges<MwbE = B1>,
    {
        Some(EdgeId([vertex, self.vertex_target(vertex)?]))
    }

    /// Iterates over the sources of the incoming edges of a vertex.
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
    fn vertex_source(&self, vertex: VertexId) -> Option<VertexId>
    where
        Self: HasEdges<MwbE = B1>,
    {
        let source = self.vertices_r::<Key>().get(vertex)?.source::<Key>();
        if source != vertex {
            Some(source)
        } else {
            None
        }
    }

    /// Iterates over the incoming edges of a vertex.
    fn vertex_edges_in(&self, vertex: VertexId) -> VertexEdgesIn<Self> {
        self.vertex_sources(vertex)
            .map_with(vertex, |t, s| EdgeId([s, t]))
    }

    /// Gets the ≤1 incoming edge that the vertex is a source of.
    fn vertex_edge_in(&self, vertex: VertexId) -> Option<EdgeId>
    where
        Self: HasEdges<MwbE = B1>,
    {
        Some(EdgeId([self.vertex_source(vertex)?, vertex]))
    }

    /// Flips an edge into 2 edges with `vertex` between them.
    fn flip12<EI: TryInto<EdgeId>>(&mut self, edge: EI, vertex: VertexId)
    where
        Self::E: Clone
    {
        let edge = edge.try_into().ok().unwrap();

        assert!(self.contains_edge(edge) || self.contains_edge(edge.twin()));
        if self.contains_edge(edge) {
            self.flip12_edge_higher::<_, Key, _>(edge, vertex, |mesh| {
                let value = mesh.remove_edge(edge).unwrap();
                mesh.add_edge(EdgeId([edge.0[0], vertex]), value.clone());
                mesh.add_edge(EdgeId([vertex, edge.0[1]]), value);
            });
        }

        let edge = edge.twin();
        if self.contains_edge(edge) {
            self.flip12_edge_higher::<_, Key, _>(edge, vertex, |mesh| {
                let value = mesh.remove_edge(edge).unwrap();
                mesh.add_edge(EdgeId([edge.0[0], vertex]), value.clone());
                mesh.add_edge(EdgeId([vertex, edge.0[1]]), value);
            });
        }
    }

    /// Adds an edge to the mesh. Vertex order is important!
    /// If the edge was already there, this replaces the value.
    /// Returns the previous value of the edge, if there was one.
    ///
    /// # Panics
    /// Panics if either vertex doesn't exist or if the vertices are the same
    fn add_edge<EI: TryInto<EdgeId>>(&mut self, vertices: EI, value: Self::E) -> Option<Self::E> {
        let id = vertices.try_into().ok().unwrap();

        // Can't use entry().or_insert() because that would cause a
        // mutable borrow and an immutable borrow to exist at the same time
        if let Some(edge) = self.edges_r_mut::<Key>().get_mut(&id) {
            Some(std::mem::replace(edge.value_mut::<Key>(), value))
        } else {
            let target = self.vertices_r::<Key>()[id.0[0]].target::<Key>();

            let (prev_target, next_target) =
                if target == id.0[0] || <<Self::Edge as Edge>::Mwb as Bit>::BOOL {
                    if target != id.0[0] {
                        self.remove_edge(EdgeId([id.0[0], target]));
                    }
                    // First edge from vertex
                    *self.vertices_r_mut::<Key>()[id.0[0]].target_mut::<Key>() = id.0[1];
                    (id.0[1], id.0[1])
                } else {
                    let prev = self.edges_r::<Key>()[&[id.0[0], target].try_into().ok().unwrap()]
                        .links::<Key>()[0]
                        .prev;
                    let next = target;
                    self.edges_r_mut::<Key>()
                        .get_mut(&id.with_target(prev))
                        .unwrap()
                        .links_mut::<Key>()[0]
                        .next = id.0[1];
                    self.edges_r_mut::<Key>()
                        .get_mut(&id.with_target(next))
                        .unwrap()
                        .links_mut::<Key>()[0]
                        .prev = id.0[1];
                    (prev, next)
                };

            let source = self.vertices_r::<Key>()[id.0[1]].source::<Key>();

            let (prev_source, next_source) =
                if source == id.0[1] || <<Self::Edge as Edge>::Mwb as Bit>::BOOL {
                    if source != id.0[1] {
                        self.remove_edge([source, id.0[1]]);
                    }
                    // First edge to vertex
                    *self.vertices_r_mut::<Key>()[id.0[1]].source_mut::<Key>() = id.0[0];
                    (id.0[0], id.0[0])
                } else {
                    let prev = self.edges_r::<Key>()[&[source, id.0[1]].try_into().ok().unwrap()]
                        .links::<Key>()[1]
                        .prev;
                    let next = source;
                    self.edges_r_mut::<Key>()
                        .get_mut(&id.with_source(prev))
                        .unwrap()
                        .links_mut::<Key>()[1]
                        .next = id.0[0];
                    self.edges_r_mut::<Key>()
                        .get_mut(&id.with_source(next))
                        .unwrap()
                        .links_mut::<Key>()[1]
                        .prev = id.0[0];
                    (prev, next)
                };

            self.edges_r_mut::<Key>().insert(
                id,
                Edge::new::<Key>(
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
    fn extend_edges<EI: TryInto<EdgeId>, I: IntoIterator<Item = (EI, Self::E)>>(
        &mut self,
        iter: I,
    ) {
        iter.into_iter().for_each(|(id, value)| {
            self.add_edge(id, value);
        })
    }

    /// Removes an edge from the mesh and returns the value that was there,
    /// or None if there was nothing there
    fn remove_edge<EI: TryInto<EdgeId>>(&mut self, id: EI) -> Option<Self::E> {
        let id = id.try_into().ok()?;

        if self.edge(id).is_some() {
            self.remove_edge_higher::<Key>(id);

            let (next_source, next_target) = if <<Self::Edge as Edge>::Mwb as Bit>::BOOL {
                (id.0[0], id.0[1])
            } else {
                let edge = &self.edges_r::<Key>()[&id];
                let prev = edge.links::<Key>()[0].prev;
                let next = edge.links::<Key>()[0].next;
                self.edges_r_mut::<Key>()
                    .get_mut(&id.with_target(prev))
                    .unwrap()
                    .links_mut::<Key>()[0]
                    .next = next;
                self.edges_r_mut::<Key>()
                    .get_mut(&id.with_target(next))
                    .unwrap()
                    .links_mut::<Key>()[0]
                    .prev = prev;
                let next_target = next;

                let edge = &self.edges_r::<Key>()[&id];
                let prev = edge.links::<Key>()[1].prev;
                let next = edge.links::<Key>()[1].next;
                self.edges_r_mut::<Key>()
                    .get_mut(&id.with_source(prev))
                    .unwrap()
                    .links_mut::<Key>()[1]
                    .next = next;
                self.edges_r_mut::<Key>()
                    .get_mut(&id.with_source(next))
                    .unwrap()
                    .links_mut::<Key>()[1]
                    .prev = prev;

                (next, next_target)
            };

            let source = &mut self.vertices_r_mut::<Key>()[id.0[0]];
            if id.0[1] == next_target {
                // this was the last edge from the vertex
                *source.target_mut::<Key>() = id.0[0];
            } else if id.0[1] == source.target::<Key>() {
                *source.target_mut::<Key>() = next_target;
            }

            let target = &mut self.vertices_r_mut::<Key>()[id.0[1]];
            if id.0[0] == next_source {
                // this was the last edge from the vertex
                *target.source_mut::<Key>() = id.0[1];
            } else if id.0[0] == target.source::<Key>() {
                *target.source_mut::<Key>() = next_source;
            }

            self.edges_r_mut::<Key>()
                .remove(&id)
                .map(|e| e.to_value::<Key>())
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
    fn retain_edges<P: FnMut(EdgeId, &Self::E) -> bool>(&mut self, mut predicate: P) {
        let to_remove = self
            .edges()
            .filter(|(id, e)| !predicate(**id, *e))
            .map(|(id, _)| *id)
            .collect::<Vec<_>>();
        self.remove_edges(to_remove);
    }

    /// Removes all edges from the mesh.
    fn clear_edges(&mut self) {
        self.clear_edges_higher::<Key>();
        self.edges_r_mut::<Key>().clear();

        // Fix vertex-target links
        for (id, vertex) in self.vertices_r_mut::<Key>() {
            *vertex.target_mut::<Key>() = *id;
            *vertex.source_mut::<Key>() = *id;
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
///       source_in     │ \ · · · · · ·  / │     target_out
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
/// The edges that `source_in` and `target_out` reference could
/// also be the current edge's twin.
#[derive(Debug)]
pub struct EdgeWalker<'a, M: ?Sized>
where
    M: HasEdges,
{
    mesh: &'a M,
    edge: EdgeId,
}

impl<'a, M: ?Sized> Clone for EdgeWalker<'a, M>
where
    M: HasEdges,
{
    fn clone(&self) -> Self {
        Self {
            mesh: self.mesh,
            edge: self.edge,
        }
    }
}

impl<'a, M: ?Sized> Copy for EdgeWalker<'a, M> where M: HasEdges {}

impl<'a, M: ?Sized> EdgeWalker<'a, M>
where
    M: HasEdges,
{
    pub(crate) fn new<EI: TryInto<EdgeId>>(mesh: &'a M, edge: EI) -> Self {
        Self {
            mesh,
            edge: edge.try_into().ok().unwrap(),
        }
    }

    fn from_vertex(mesh: &'a M, vertex: VertexId) -> Option<Self> {
        match EdgeId::try_from([
            vertex,
            mesh.vertices_r::<Key>().get(vertex)?.target::<Key>(),
        ]) {
            Ok(edge) => Some(Self::new(mesh, edge)),
            Err(_) => None,
        }
    }

    fn from_target(mesh: &'a M, vertex: VertexId) -> Option<Self> {
        match EdgeId::try_from([
            mesh.vertices_r::<Key>().get(vertex)?.source::<Key>(),
            vertex,
        ]) {
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
        if !<<M::Edge as Edge>::Mwb as Bit>::BOOL {
            self.edge = self
                .edge
                .with_target(self.mesh.edges_r::<Key>()[&self.edge].links::<Key>()[0].next);
        }
        self
    }

    /// Sets the current edge to the previous one with the same source vertex.
    pub fn prev(mut self) -> Self {
        if !<<M::Edge as Edge>::Mwb as Bit>::BOOL {
            self.edge = self
                .edge
                .with_target(self.mesh.edges_r::<Key>()[&self.edge].links::<Key>()[0].prev);
        }
        self
    }

    /// Sets the current edge to the next one with the same target vertex.
    pub fn next_in(mut self) -> Self {
        if !<<M::Edge as Edge>::Mwb as Bit>::BOOL {
            self.edge = self
                .edge
                .with_source(self.mesh.edges_r::<Key>()[&self.edge].links::<Key>()[1].next);
        }
        self
    }

    /// Sets the current edge to the previous one with the same target vertex.
    pub fn prev_in(mut self) -> Self {
        if !<<M::Edge as Edge>::Mwb as Bit>::BOOL {
            self.edge = self
                .edge
                .with_source(self.mesh.edges_r::<Key>()[&self.edge].links::<Key>()[1].prev);
        }
        self
    }

    /// Sets the current edge to an edge whose source vertex is the current edge's target vertex.
    pub fn target_out(mut self) -> Option<Self> {
        let target = self.mesh.vertices_r::<Key>()[self.edge.0[1]].target::<Key>();
        if let Some(edge) = [self.edge.0[1], target].try_into().ok() {
            self.edge = edge;
            Some(self)
        } else {
            None
        }
    }

    /// Sets the current edge to an edge whose target vertex is the current edge's source vertex.
    pub fn source_in(mut self) -> Option<Self> {
        let source = self.mesh.vertices_r::<Key>()[self.edge.0[0]].source::<Key>();
        if let Some(edge) = [source, self.edge.0[0]].try_into().ok() {
            self.edge = edge;
            Some(self)
        } else {
            None
        }
    }

    pub fn tri_walker(self) -> Option<TriWalker<'a, M>>
    where
        M: HasTris,
    {
        TriWalker::from_edge(self.mesh, self.edge)
    }
}

/// An iterator over the targets of the outgoing edges of a vertex.
#[derive(Clone, Debug)]
pub struct VertexTargets<'a, M: ?Sized>
where
    M: HasEdges,
{
    walker: EdgeWalker<'a, M>,
    start_target: VertexId,
    finished: bool,
}

impl<'a, M: ?Sized> Iterator for VertexTargets<'a, M>
where
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
    M: HasEdges,
{
    walker: EdgeWalker<'a, M>,
    start_source: VertexId,
    finished: bool,
}

impl<'a, M: ?Sized> Iterator for VertexSources<'a, M>
where
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
    Self::V: Position,
    DefaultAllocator: Allocator<f64, HasPositionDim<Self>>,
{
    /// Gets the positions of the vertices of an edge.
    /// Assumes the edge exists.
    fn edge_positions<EI: TryInto<EdgeId>>(&self, edge: EI) -> [HasPositionPoint<Self>; 2] {
        let edge = edge.try_into().ok().unwrap();
        let v0 = self.position(edge.0[0]);
        let v1 = self.position(edge.0[1]);
        [v0, v1]
    }
}

impl<M: HasEdges + HasPosition> HasPositionAndEdges for M
where
    M::V: Position,
    DefaultAllocator: Allocator<f64, HasPositionDim<M>>,
{}

#[macro_export]
#[doc(hidden)]
macro_rules! impl_edge {
    ($name:ident<$e:ident>, new |$id:ident, $link:ident, $value:ident| $new:expr) => {
        impl<$e> crate::edge::Edge for $name<$e> {
            type E = $e;
            type Mwb = typenum::B0;
            type Higher = typenum::B0;

            fn new<L: crate::private::Lock>(
                $id: VertexId,
                $link: [crate::edge::Link<crate::vertex::VertexId>; 2],
                $value: Self::E,
            ) -> Self {
                $new
            }

            fn links<L: crate::private::Lock>(
                &self,
            ) -> [crate::edge::Link<crate::vertex::VertexId>; 2] {
                self.links
            }

            fn links_mut<L: crate::private::Lock>(
                &mut self,
            ) -> &mut [crate::edge::Link<crate::vertex::VertexId>; 2] {
                &mut self.links
            }

            fn to_value<L: crate::private::Lock>(self) -> Self::E {
                self.value
            }

            fn value<L: crate::private::Lock>(&self) -> &Self::E {
                &self.value
            }

            fn value_mut<L: crate::private::Lock>(&mut self) -> &mut Self::E {
                &mut self.value
            }

            fn tri_opp<L: crate::private::Lock>(&self) -> crate::vertex::VertexId
            where
                Self: crate::edge::Edge<Higher = typenum::B1>,
            {
                unreachable!()
            }

            fn tri_opp_mut<L: crate::private::Lock>(&mut self) -> &mut crate::vertex::VertexId
            where
                Self: crate::edge::Edge<Higher = typenum::B1>,
            {
                unreachable!()
            }
        }
    };
}

#[macro_export]
#[doc(hidden)]
macro_rules! impl_edge_mwb {
    ($name:ident<$e:ident>, new |$id:ident, $link:ident, $value:ident| $new:expr) => {
        impl<$e> crate::edge::Edge for $name<$e> {
            type E = $e;
            type Mwb = typenum::B1;
            type Higher = typenum::B0;

            fn new<L: crate::private::Lock>(
                $id: VertexId,
                $link: [crate::edge::Link<crate::vertex::VertexId>; 2],
                $value: Self::E,
            ) -> Self {
                $new
            }

            fn links<L: crate::private::Lock>(
                &self,
            ) -> [crate::edge::Link<crate::vertex::VertexId>; 2] {
                panic!("Cannot get links in \"mwb\" edge")
            }

            fn links_mut<L: crate::private::Lock>(
                &mut self,
            ) -> &mut [crate::edge::Link<crate::vertex::VertexId>; 2] {
                panic!("Cannot get links in \"mwb\" edge")
            }

            fn to_value<L: crate::private::Lock>(self) -> Self::E {
                self.value
            }

            fn value<L: crate::private::Lock>(&self) -> &Self::E {
                &self.value
            }

            fn value_mut<L: crate::private::Lock>(&mut self) -> &mut Self::E {
                &mut self.value
            }

            fn tri_opp<L: crate::private::Lock>(&self) -> crate::vertex::VertexId
            where
                Self: crate::edge::Edge<Higher = typenum::B1>,
            {
                unreachable!()
            }

            fn tri_opp_mut<L: crate::private::Lock>(&mut self) -> &mut crate::vertex::VertexId
            where
                Self: crate::edge::Edge<Higher = typenum::B1>,
            {
                unreachable!()
            }
        }
    };
}

#[macro_export]
#[doc(hidden)]
macro_rules! impl_edge_higher {
    ($name:ident<$e:ident>, new |$id:ident, $link:ident, $value:ident| $new:expr) => {
        impl<$e> crate::edge::Edge for $name<$e> {
            type E = $e;
            type Mwb = typenum::B0;
            type Higher = typenum::B1;

            fn new<L: crate::private::Lock>(
                $id: VertexId,
                $link: [crate::edge::Link<crate::vertex::VertexId>; 2],
                $value: Self::E,
            ) -> Self {
                $new
            }

            fn links<L: crate::private::Lock>(
                &self,
            ) -> [crate::edge::Link<crate::vertex::VertexId>; 2] {
                self.links
            }

            fn links_mut<L: crate::private::Lock>(
                &mut self,
            ) -> &mut [crate::edge::Link<crate::vertex::VertexId>; 2] {
                &mut self.links
            }

            fn to_value<L: crate::private::Lock>(self) -> Self::E {
                self.value
            }

            fn value<L: crate::private::Lock>(&self) -> &Self::E {
                &self.value
            }

            fn value_mut<L: crate::private::Lock>(&mut self) -> &mut Self::E {
                &mut self.value
            }

            fn tri_opp<L: crate::private::Lock>(&self) -> crate::vertex::VertexId
            where
                Self: crate::edge::Edge<Higher = typenum::B1>,
            {
                self.tri_opp
            }

            fn tri_opp_mut<L: crate::private::Lock>(&mut self) -> &mut crate::vertex::VertexId
            where
                Self: crate::edge::Edge<Higher = typenum::B1>,
            {
                &mut self.tri_opp
            }
        }
    };
}

#[macro_export]
#[doc(hidden)]
macro_rules! impl_has_edges {
    ($edge:ident<$e:ident> $($z:ident)*, Mwb = $mwb:ty, Higher = $higher:ident) => {
        type Edge = $edge<$e>;
        type E = $e;
        type MwbE = $mwb;
        type HigherE = $higher;

        fn from_ve_r<
            VI: IntoIterator<
                Item = (
                    crate::vertex::VertexId,
                    <Self::Vertex as crate::vertex::Vertex>::V,
                ),
            >,
            EI: IntoIterator<Item = (crate::edge::EdgeId, <Self::Edge as crate::edge::Edge>::E)>,
            L: crate::private::Lock,
        >(
            vertices: VI,
            edges: EI,
            default_v: fn() -> Self::V,
            default_e: fn() -> Self::E,
        ) -> Self {
            use typenum::Bit;
            if <$higher>::BOOL {
                unreachable!()
            }
            // The code below will not be executed if the value is invalid.
            #[allow(invalid_value)]
            let mut mesh = Self::with_defaults(default_v, default_e $(, unsafe { std::mem::$z() })*);
            mesh.extend_vertices_with_ids(vertices);
            mesh.extend_edges(edges);
            mesh
        }

        fn into_ve_r<L: crate::private::Lock>(
            self,
        ) -> (
            crate::vertex::IntoVertices<Self::Vertex>,
            crate::edge::IntoEdges<Self::Edge>,
        ) {
            use crate::edge::Edge;
            use crate::vertex::Vertex;
            (
                self.vertices
                    .into_iter()
                    .map(|(id, v)| (id, v.to_value::<crate::private::Key>())),
                self.edges
                    .into_iter()
                    .map(|(id, e)| (id, e.to_value::<crate::private::Key>())),
            )
        }

        fn edges_r<L: crate::private::Lock>(&self) -> &FnvHashMap<crate::edge::EdgeId, Self::Edge> {
            &self.edges
        }

        fn edges_r_mut<L: crate::private::Lock>(
            &mut self,
        ) -> &mut FnvHashMap<crate::edge::EdgeId, Self::Edge> {
            &mut self.edges
        }

        fn default_e_r<L: crate::private::Lock>(&self) -> fn() -> Self::E {
            self.default_e
        }

        crate::if_b0! { $higher =>
            #[cfg(feature = "obj")]
            fn obj_with_edges_higher<L: crate::private::Lock>(&self, _: &mut obj::ObjData, _: &fnv::FnvHashMap<crate::vertex::VertexId, usize>) {}

            fn flip12_edge_higher<
                EI: std::convert::TryInto<crate::edge::EdgeId>, L: crate::private::Lock, C: FnMut(&mut Self)
            > (&mut self, _edge: EI, _vertex: crate::vertex::VertexId, mut callback: C) {
                callback(self);
            }
        }

        crate::if_b1! { $higher =>
            #[cfg(feature = "obj")]
            fn obj_with_edges_higher<L: crate::private::Lock>(&self, data: &mut obj::ObjData, v_inv: &fnv::FnvHashMap<crate::vertex::VertexId, usize>) {
                self.obj_with_tris::<crate::private::Key>(data, v_inv);
            }

            fn flip12_edge_higher<
                EI: std::convert::TryInto<crate::edge::EdgeId>, L: crate::private::Lock, C: FnMut(&mut Self)
            > (&mut self, edge: EI, vertex: crate::vertex::VertexId, callback: C) {
                self.flip12_tri::<_, crate::private::Key, _>(edge, vertex, callback);
            }
        }
    };
}
