use fnv::FnvHashMap;
use idmap::OrderedIdMap;
#[cfg(feature = "serde_")]
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use typenum::{U2, U3};

use crate::edge::{EdgeId, HasEdges};
use crate::mesh_1::internal::HigherVertex;
use crate::mesh_2::internal::HigherEdge;
use crate::tri::{HasTris, TriId};
use crate::tet::{HasTets, TetId};
use crate::vertex::{HasVertices, VertexId};
use crate::VecN;

use internal::{HigherTri, Tet};

/// A combinatorial simplicial 3-complex, containing only vertices, (oriented) edges, (oriented) triangles, and (oriented) tetrahedrons.
/// Also known as an tet mesh.
/// Each vertex stores a value of type `V`.
/// Each edge stores its vertices and a value of type `E`.
/// Each triangle stores its vertices and a value of type `F`.
/// Each tetrahedron stores its vertices and a value of type `T`.
/// The edge manipulation methods can either be called with an array of 2 `VertexId`s
/// or an `EdgeId`.
/// The triangle manipulation methods can either be called with an array of 3 `VertexId`s
/// or an `TriId`.
/// The tetrahedron manipulation methods can either be called with an array of 4 `VertexId`s
/// or an `TetId`.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde_", derive(Serialize, Deserialize))]
pub struct ComboMesh3<V, E, F, T> {
    vertices: OrderedIdMap<VertexId, HigherVertex<V>>,
    edges: FnvHashMap<EdgeId, HigherEdge<E>>,
    tris: FnvHashMap<TriId, HigherTri<F>>,
    tets: FnvHashMap<TetId, Tet<T>>,
    next_vertex_id: u64,
    /// Keep separate track because edge twins may or may not exist
    num_edges: usize,
    num_tris: usize,
    num_tets: usize,
}
crate::impl_has_vertices!(ComboMesh3<V, E, F, T>, HigherVertex);
crate::impl_has_edges!(ComboMesh3<V, E, F, T>, HigherEdge);
crate::impl_has_tris!(ComboMesh3<V, E, F, T>, HigherTri);
crate::impl_has_tets!(ComboMesh3<V, E, F, T>, Tet);
crate::impl_index_vertex!(ComboMesh3<V, E, F, T>);
crate::impl_index_edge!(ComboMesh3<V, E, F, T>);
crate::impl_index_tri!(ComboMesh3<V, E, F, T>);
crate::impl_index_tet!(ComboMesh3<V, E, F, T>);

impl<V, E, F, T> HasVertices for ComboMesh3<V, E, F, T> {}
impl<V, E, F, T> HasEdges for ComboMesh3<V, E, F, T> {}
impl<V, E, F, T> HasTris for ComboMesh3<V, E, F, T> {}
impl<V, E, F, T> HasTets for ComboMesh3<V, E, F, T> {}

impl<V, E, F, T> Default for ComboMesh3<V, E, F, T> {
    fn default() -> Self {
        ComboMesh3 {
            vertices: OrderedIdMap::default(),
            edges: FnvHashMap::default(),
            tris: FnvHashMap::default(),
            tets: FnvHashMap::default(),
            next_vertex_id: 0,
            num_edges: 0,
            num_tris: 0,
            num_tets: 0,
        }
    }
}

impl<V, E, F, T> ComboMesh3<V, E, F, T> {
    /// Creates an empty tri mesh.
    pub fn new() -> Self {
        Self::default()
    }
}

/// A position-containing tri mesh
pub type Mesh3<V, E, F, T, D> = ComboMesh3<(VecN<D>, V), E, F, T>;

/// A 2D-position-containing tri mesh
pub type Mesh32<V, E, F, T> = Mesh3<V, E, F, T, U2>;

/// A 3D-position-containing tri mesh
pub type Mesh33<V, E, F, T> = Mesh3<V, E, F, T, U3>;


mod internal {
    use super::ComboMesh3;
    use crate::edge::internal::{ClearEdgesHigher, Link, RemoveEdgeHigher};
    use crate::edge::{EdgeId, HasEdges};
    use crate::tri::internal::{ClearTrisHigher, RemoveTriHigher};
    use crate::tri::{HasTris, TriId};
    use crate::tet::internal::{ClearTetsHigher, RemoveTetHigher};
    use crate::tet::{HasTets, TetId};
    use crate::vertex::internal::{ClearVerticesHigher, RemoveVertexHigher};
    use crate::vertex::VertexId;
    #[cfg(feature = "serde_")]
    use serde::{Deserialize, Serialize};

    #[derive(Clone, Debug)]
    #[doc(hidden)]
    #[cfg_attr(feature = "serde_", derive(Serialize, Deserialize))]
    pub struct HigherTri<F> {
        /// Targets from the same edge for each of the edges,
        /// whether the triangle actually exists or not
        links: [Link<VertexId>; 3],
        tet_opp: VertexId,
        /// The triangle does not actually exist if the value is None;
        /// it is just there for the structural purpose of
        /// ensuring that every triangle has a twin.
        value: Option<F>,
    }
    #[rustfmt::skip]
    crate::impl_tri!(
        HigherTri<F>,
        new |id, links, value| {
            HigherTri {
                tet_opp: id,
                links,
                value,
            }
        }
    );
    crate::impl_higher_tri!(HigherTri<F>);

    /// A tetrahedron of an tet mesh
    #[derive(Clone, Debug)]
    #[doc(hidden)]
    #[cfg_attr(feature = "serde_", derive(Serialize, Deserialize))]
    pub struct Tet<F> {
        /// Targets from the same triangle for each of the triangle,
        /// whether the tetrahedron actually exists or not
        links: [Link<VertexId>; 4],
        /// The tetrahedron does not actually exist if the value is None;
        /// it is just there for the structural purpose of
        /// ensuring that every tetrahedron has a twin.
        value: Option<F>,
    }
    #[rustfmt::skip]
    crate::impl_tet!(Tet<F>, new |_id, links, value| Tet { links, value });

    impl<V, E, F, T> RemoveVertexHigher for ComboMesh3<V, E, F, T> {
        fn remove_vertex_higher(&mut self, vertex: VertexId) {
            self.remove_edges(
                self.vertex_edges_out(vertex)
                    .chain(self.vertex_edges_in(vertex))
                    .collect::<Vec<_>>(),
            );
        }
    }

    impl<V, E, F, T> ClearVerticesHigher for ComboMesh3<V, E, F, T> {
        fn clear_vertices_higher(&mut self) {
            self.tris.clear();
            self.num_tris = 0;
            self.edges.clear();
            self.num_edges = 0;
        }
    }

    impl<V, E, F, T> RemoveEdgeHigher for ComboMesh3<V, E, F, T> {
        fn remove_edge_higher(&mut self, edge: EdgeId) {
            self.remove_tris(self.edge_tris(edge).collect::<Vec<_>>());
        }
    }

    impl<V, E, F, T> ClearEdgesHigher for ComboMesh3<V, E, F, T> {
        fn clear_edges_higher(&mut self) {
            self.tris.clear();
            self.num_tris = 0;
        }
    }

    impl<V, E, F, T> RemoveTriHigher for ComboMesh3<V, E, F, T> {
        fn remove_tri_higher(&mut self, tri: TriId) {
            self.remove_tets(self.tri_tets(tri).collect::<Vec<_>>());
        }
    }

    impl<V, E, F, T> ClearTrisHigher for ComboMesh3<V, E, F, T> {
        fn clear_tris_higher(&mut self) {
            self.tets.clear();
            self.num_tets = 0;
        }
    }

    impl<V, E, F, T> RemoveTetHigher for ComboMesh3<V, E, F, T> {
        fn remove_tet_higher(&mut self, _: TetId) {}
    }

    impl<V, E, F, T> ClearTetsHigher for ComboMesh3<V, E, F, T> {
        fn clear_tets_higher(&mut self) {}
    }
}