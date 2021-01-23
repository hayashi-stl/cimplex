use fnv::FnvHashMap;
use idmap::OrderedIdMap;
#[cfg(feature = "serde_")]
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use typenum::{U2, U3};

use crate::{edge::{EdgeId, HasEdges}, tet::TetWalker};
use crate::mesh_1::internal::HigherVertex;
use crate::mesh_2::internal::HigherEdge;
use crate::tet::{HasTets, TetId, TetWalk};
use crate::tri::{HasTris, TriId};
use crate::vertex::{HasVertices, VertexId};
use crate::VecN;

use internal::{HigherTri, Tet, ManifoldTet};

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
impl<V, E, F, T> HasTets for ComboMesh3<V, E, F, T> {
    fn add_tet<TI: std::convert::TryInto<crate::tet::TetId>>(
        &mut self,
        vertices: TI,
        value: <Self::Tet as crate::tet::internal::Tet>::T,
        tri_value: impl Fn() -> <Self::Tri as crate::tri::internal::Tri>::F,
        edge_value: impl Fn() -> <Self::Edge as crate::edge::internal::Edge>::E + Clone,
    ) -> Option<<Self::Tet as crate::tet::internal::Tet>::T> {
        crate::tet::internal::add_tet_non_manifold(self, vertices, value, tri_value, edge_value)
    }

    fn remove_tet_keep_tris<TI: std::convert::TryInto<crate::tet::TetId>>(
        &mut self,
        id: TI,
    ) -> Option<<Self::Tet as crate::tet::internal::Tet>::T> {
        crate::tet::internal::remove_tet_non_manifold(self, id)
    }
}

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

/// A simplicial 3-complex optimized for manifolds with boundary.
/// Each oriented face can be part of at most 1 tetrahedron.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde_", derive(Serialize, Deserialize))]
pub struct ManifoldComboMesh3<V, E, F, T> {
    vertices: OrderedIdMap<VertexId, HigherVertex<V>>,
    edges: FnvHashMap<EdgeId, HigherEdge<E>>,
    tris: FnvHashMap<TriId, HigherTri<F>>,
    tets: FnvHashMap<TetId, ManifoldTet<T>>,
    next_vertex_id: u64,
    /// Keep separate track because edge twins may or may not exist
    num_edges: usize,
    num_tris: usize,
    num_tets: usize,
}
crate::impl_has_vertices!(ManifoldComboMesh3<V, E, F, T>, HigherVertex);
crate::impl_has_edges!(ManifoldComboMesh3<V, E, F, T>, HigherEdge);
crate::impl_has_tris!(ManifoldComboMesh3<V, E, F, T>, HigherTri);
crate::impl_has_tets!(ManifoldComboMesh3<V, E, F, T>, ManifoldTet);
crate::impl_index_vertex!(ManifoldComboMesh3<V, E, F, T>);
crate::impl_index_edge!(ManifoldComboMesh3<V, E, F, T>);
crate::impl_index_tri!(ManifoldComboMesh3<V, E, F, T>);
crate::impl_index_tet!(ManifoldComboMesh3<V, E, F, T>);

impl<V, E, F, T> HasVertices for ManifoldComboMesh3<V, E, F, T> {}
impl<V, E, F, T> HasEdges for ManifoldComboMesh3<V, E, F, T> {}
impl<V, E, F, T> HasTris for ManifoldComboMesh3<V, E, F, T> {}
impl<V, E, F, T> HasTets for ManifoldComboMesh3<V, E, F, T> {
    fn add_tet<TI: std::convert::TryInto<crate::tet::TetId>>(
        &mut self,
        vertices: TI,
        value: <Self::Tet as crate::tet::internal::Tet>::T,
        tri_value: impl Fn() -> <Self::Tri as crate::tri::internal::Tri>::F,
        edge_value: impl Fn() -> <Self::Edge as crate::edge::internal::Edge>::E + Clone,
    ) -> Option<<Self::Tet as crate::tet::internal::Tet>::T> {
        todo!()
        //crate::tet::internal::add_tet_manifold(self, vertices, value, tri_value, edge_value)
    }

    fn remove_tet_keep_tris<TI: std::convert::TryInto<crate::tet::TetId>>(
        &mut self,
        id: TI,
    ) -> Option<<Self::Tet as crate::tet::internal::Tet>::T> {
        crate::tet::internal::remove_tet_manifold(self, id)
    }
}

impl<V, E, F, T> Default for ManifoldComboMesh3<V, E, F, T> {
    fn default() -> Self {
        ManifoldComboMesh3 {
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

impl<V, E, F, T> ManifoldComboMesh3<V, E, F, T> {
    /// Creates an empty tri mesh.
    pub fn new() -> Self {
        Self::default()
    }
}

pub(crate) mod internal {
    use super::{ComboMesh3, ManifoldComboMesh3};
    use crate::edge::internal::{ClearEdgesHigher, Link, RemoveEdgeHigher};
    use crate::edge::{EdgeId, HasEdges};
    use crate::tet::internal::{ClearTetsHigher, RemoveTetHigher};
    use crate::tet::{HasTets, TetId};
    use crate::tri::internal::{ClearTrisHigher, RemoveTriHigher};
    use crate::tri::{HasTris, TriId};
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
    pub struct Tet<T> {
        /// Targets from the same triangle for each of the triangle,
        /// whether the tetrahedron actually exists or not
        links: [Link<VertexId>; 4],
        /// The tetrahedron does not actually exist if the value is None;
        /// it is just there for the structural purpose of
        /// ensuring that every tetrahedron has a twin.
        value: Option<T>,
    }
    #[rustfmt::skip]
    crate::impl_non_manifold_tet!(Tet<T>, with_links |_id, links, value| Tet { links, value });

    /// A tetrahedron of a manifold tet mesh, possibly with boundary
    #[derive(Clone, Debug)]
    #[doc(hidden)]
    #[cfg_attr(feature = "serde_", derive(Serialize, Deserialize))]
    pub struct ManifoldTet<T> {
        value: T,
    }
    #[rustfmt::skip]
    crate::impl_manifold_tet!(ManifoldTet<T>, new |value| ManifoldTet { value });

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

    impl<V, E, F, T> RemoveVertexHigher for ManifoldComboMesh3<V, E, F, T> {
        fn remove_vertex_higher(&mut self, vertex: VertexId) {
            self.remove_edges(
                self.vertex_edges_out(vertex)
                    .chain(self.vertex_edges_in(vertex))
                    .collect::<Vec<_>>(),
            );
        }
    }

    impl<V, E, F, T> ClearVerticesHigher for ManifoldComboMesh3<V, E, F, T> {
        fn clear_vertices_higher(&mut self) {
            self.tris.clear();
            self.num_tris = 0;
            self.edges.clear();
            self.num_edges = 0;
        }
    }

    impl<V, E, F, T> RemoveEdgeHigher for ManifoldComboMesh3<V, E, F, T> {
        fn remove_edge_higher(&mut self, edge: EdgeId) {
            self.remove_tris(self.edge_tris(edge).collect::<Vec<_>>());
        }
    }

    impl<V, E, F, T> ClearEdgesHigher for ManifoldComboMesh3<V, E, F, T> {
        fn clear_edges_higher(&mut self) {
            self.tris.clear();
            self.num_tris = 0;
        }
    }

    impl<V, E, F, T> RemoveTriHigher for ManifoldComboMesh3<V, E, F, T> {
        fn remove_tri_higher(&mut self, tri: TriId) {
            todo!()
            //self.remove_tets(self.tri_tets(tri).collect::<Vec<_>>());
        }
    }

    impl<V, E, F, T> ClearTrisHigher for ManifoldComboMesh3<V, E, F, T> {
        fn clear_tris_higher(&mut self) {
            self.tets.clear();
            self.num_tets = 0;
        }
    }

    impl<V, E, F, T> RemoveTetHigher for ManifoldComboMesh3<V, E, F, T> {
        fn remove_tet_higher(&mut self, _: TetId) {}
    }

    impl<V, E, F, T> ClearTetsHigher for ManifoldComboMesh3<V, E, F, T> {
        fn clear_tets_higher(&mut self) {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fnv::FnvHashSet;
    use std::convert::TryInto;
    use std::fmt::Debug;
    use std::hash::Hash;

    #[track_caller]
    fn assert_vertices<
        V: Clone + Debug + Eq + Hash,
        E,
        F,
        T,
        I: IntoIterator<Item = (VertexId, V)>,
    >(
        mesh: &ComboMesh3<V, E, F, T>,
        vertices: I,
    ) {
        let result = mesh
            .vertices()
            .map(|(id, v)| (*id, v.clone()))
            .collect::<FnvHashSet<_>>();
        let expect = vertices.into_iter().collect::<FnvHashSet<_>>();

        assert_eq!(result, expect);
    }

    #[track_caller]
    fn assert_edges<
        V,
        E: Clone + Debug + Eq + Hash,
        EI: TryInto<EdgeId>,
        F,
        T,
        I: IntoIterator<Item = (EI, E)>,
    >(
        mesh: &ComboMesh3<V, E, F, T>,
        edges: I,
    ) {
        let result = mesh
            .edges()
            .map(|(id, e)| (*id, e.clone()))
            .collect::<FnvHashSet<_>>();
        let expect = edges
            .into_iter()
            .map(|(vertices, e)| (vertices.try_into().ok().unwrap(), e))
            .collect::<FnvHashSet<_>>();

        assert_eq!(result, expect);
        assert_eq!(mesh.num_edges(), expect.len());
    }

    #[track_caller]
    fn assert_tris<
        V,
        E,
        F: Clone + Debug + Eq + Hash,
        FI: TryInto<TriId>,
        T,
        I: IntoIterator<Item = (FI, F)>,
    >(
        mesh: &ComboMesh3<V, E, F, T>,
        tris: I,
    ) {
        let result = mesh
            .tris()
            .map(|(id, f)| (*id, f.clone()))
            .collect::<FnvHashSet<_>>();
        let expect = tris
            .into_iter()
            .map(|(vertices, f)| (vertices.try_into().ok().unwrap(), f))
            .collect::<FnvHashSet<_>>();

        assert_eq!(result, expect);
        assert_eq!(mesh.num_tris(), expect.len());
    }

    #[track_caller]
    fn assert_tets<
        V,
        E,
        F,
        T: Clone + Debug + Eq + Hash,
        TI: TryInto<TetId>,
        I: IntoIterator<Item = (TI, T)>,
    >(
        mesh: &ComboMesh3<V, E, F, T>,
        tets: I,
    ) {
        let result = mesh
            .tets()
            .map(|(id, f)| (*id, f.clone()))
            .collect::<FnvHashSet<_>>();
        let expect = tets
            .into_iter()
            .map(|(vertices, f)| (vertices.try_into().ok().unwrap(), f))
            .collect::<FnvHashSet<_>>();

        assert_eq!(result, expect);
        assert_eq!(mesh.num_tets(), expect.len());
    }

    #[test]
    fn test_default() {
        let mesh = ComboMesh3::<(), (), (), ()>::default();
        assert!(mesh.vertices.is_empty());
        assert!(mesh.edges.is_empty());
        assert!(mesh.tris.is_empty());
        assert!(mesh.tets.is_empty());
        assert_eq!(mesh.num_edges(), 0);
        assert_eq!(mesh.num_tris(), 0);
        assert_eq!(mesh.num_tets(), 0);
    }

    #[test]
    fn test_add_vertex() {
        let mut mesh = ComboMesh3::<usize, usize, usize, usize>::default();
        let id = mesh.add_vertex(3);
        assert_eq!(mesh.vertex(id), Some(&3));

        let id2 = mesh.add_vertex(9);
        assert_eq!(mesh.vertex(id), Some(&3));
        assert_eq!(mesh.vertex(id2), Some(&9));
    }

    #[test]
    fn test_extend_vertices() {
        let mut mesh = ComboMesh3::<usize, usize, usize, usize>::default();
        let ids = mesh.extend_vertices(vec![3, 6, 9, 2]);
        assert_eq!(mesh.vertex(ids[0]), Some(&3));
        assert_eq!(mesh.vertex(ids[1]), Some(&6));
        assert_eq!(mesh.vertex(ids[2]), Some(&9));
        assert_eq!(mesh.vertex(ids[3]), Some(&2));

        let ids2 = mesh.extend_vertices(vec![5, 8]);
        assert_vertices(
            &mesh,
            vec![
                (ids[0], 3),
                (ids[1], 6),
                (ids[2], 9),
                (ids[3], 2),
                (ids2[0], 5),
                (ids2[1], 8),
            ],
        );
    }

    #[test]
    fn test_add_edge() {
        let mut mesh = ComboMesh3::<usize, usize, usize, usize>::default();
        let ids = mesh.extend_vertices(vec![3, 6, 9, 2]);
        let prev = mesh.add_edge([ids[1], ids[3]], 54);
        assert_eq!(prev, None);
        assert_eq!(mesh.edge([ids[1], ids[3]]), Some(&54));
        assert_eq!(mesh.edge([ids[3], ids[1]]), None); // twin should not exist
        assert_eq!(mesh.num_edges(), 1);

        // Add twin
        let prev = mesh.add_edge([ids[3], ids[1]], 27);
        assert_eq!(prev, None);
        assert_eq!(mesh.edge([ids[1], ids[3]]), Some(&54));
        assert_eq!(mesh.edge([ids[3], ids[1]]), Some(&27));
        assert_eq!(mesh.num_edges(), 2);

        // Modify edge
        let prev = mesh.add_edge([ids[1], ids[3]], 1);
        assert_eq!(prev, Some(54));
        assert_eq!(mesh.edge([ids[1], ids[3]]), Some(&1));
        assert_eq!(mesh.edge([ids[3], ids[1]]), Some(&27));
        assert_eq!(mesh.num_edges(), 2);
    }

    #[test]
    #[should_panic]
    fn test_add_edge_bad() {
        let mut mesh = ComboMesh3::<usize, usize, usize, usize>::default();
        let ids = mesh.extend_vertices(vec![3, 6, 9, 2]);
        mesh.add_edge([ids[1], ids[1]], 4);
    }

    #[test]
    fn test_extend_edges() {
        let mut mesh = ComboMesh3::<usize, usize, usize, usize>::default();
        let ids = mesh.extend_vertices(vec![3, 6, 9, 2]);
        let edges = vec![
            ([ids[0], ids[3]], 5),
            ([ids[1], ids[3]], 3),
            ([ids[3], ids[1]], 2),
            ([ids[2], ids[3]], 8),
            ([ids[1], ids[2]], 9),
        ];
        mesh.extend_edges(edges.clone());

        for (edge, value) in edges {
            assert_eq!(mesh.edge(edge), Some(&value))
        }
        assert_eq!(mesh.num_edges(), 5);
    }

    #[test]
    fn test_add_tri() {
        let mut mesh = ComboMesh3::<usize, usize, usize, usize>::default();
        let ids = mesh.extend_vertices(vec![3, 6, 9, 2]);
        assert_eq!(mesh.add_tri([ids[1], ids[0], ids[2]], 5, || 0), None);

        assert_edges(
            &mesh,
            vec![
                ([ids[1], ids[0]], 0),
                ([ids[0], ids[2]], 0),
                ([ids[2], ids[1]], 0),
            ],
        );
        assert_tris(&mesh, vec![([ids[0], ids[2], ids[1]], 5)]);

        // Prematurely add edge
        mesh.add_edge([ids[1], ids[2]], 1);

        // Add twin
        assert_eq!(mesh.add_tri([ids[1], ids[2], ids[0]], 6, || 0), None);
        assert_edges(
            &mesh,
            vec![
                ([ids[1], ids[0]], 0),
                ([ids[0], ids[2]], 0),
                ([ids[2], ids[1]], 0),
                ([ids[1], ids[2]], 1),
                ([ids[0], ids[1]], 0),
                ([ids[2], ids[0]], 0),
            ],
        );
        assert_tris(
            &mesh,
            vec![([ids[0], ids[2], ids[1]], 5), ([ids[0], ids[1], ids[2]], 6)],
        );

        // Modify tri
        assert_eq!(mesh.add_tri([ids[1], ids[2], ids[0]], 7, || 0), Some(6));
        assert_edges(
            &mesh,
            vec![
                ([ids[1], ids[0]], 0),
                ([ids[0], ids[2]], 0),
                ([ids[2], ids[1]], 0),
                ([ids[1], ids[2]], 1),
                ([ids[0], ids[1]], 0),
                ([ids[2], ids[0]], 0),
            ],
        );
        assert_tris(
            &mesh,
            vec![([ids[0], ids[2], ids[1]], 5), ([ids[0], ids[1], ids[2]], 7)],
        );
    }

    #[test]
    fn test_extend_tris() {
        let mut mesh = ComboMesh3::<usize, usize, usize, usize>::default();
        let ids = mesh.extend_vertices(vec![3, 6, 9, 2, 5, 8, 1, 4]);

        let tris = vec![
            ([ids[0], ids[1], ids[2]], 1),
            ([ids[3], ids[1], ids[2]], 2),
            ([ids[4], ids[2], ids[1]], 3),
            ([ids[4], ids[1], ids[5]], 4),
            ([ids[5], ids[6], ids[4]], 5),
            ([ids[4], ids[6], ids[5]], 6),
        ];
        mesh.extend_tris(tris.clone(), || 0);

        assert_edges(
            &mesh,
            vec![
                ([ids[0], ids[1]], 0),
                ([ids[1], ids[2]], 0),
                ([ids[2], ids[0]], 0),
                ([ids[3], ids[1]], 0),
                ([ids[2], ids[3]], 0),
                ([ids[4], ids[2]], 0),
                ([ids[2], ids[1]], 0),
                ([ids[1], ids[4]], 0),
                ([ids[4], ids[1]], 0),
                ([ids[1], ids[5]], 0),
                ([ids[5], ids[4]], 0),
                ([ids[5], ids[6]], 0),
                ([ids[6], ids[4]], 0),
                ([ids[4], ids[5]], 0),
                ([ids[4], ids[6]], 0),
                ([ids[6], ids[5]], 0),
            ],
        );
        assert_tris(&mesh, tris);
    }

    #[test]
    fn test_add_tet() {
        let mut mesh = ComboMesh3::<usize, usize, usize, usize>::default();
        let ids = mesh.extend_vertices(vec![3, 6, 9, 2]);
        assert_eq!(
            mesh.add_tet([ids[1], ids[0], ids[2], ids[3]], 1, || 0, || 0),
            None
        );
        assert_tris(
            &mesh,
            vec![
                ([ids[2], ids[1], ids[0]], 0),
                ([ids[1], ids[2], ids[3]], 0),
                ([ids[0], ids[3], ids[2]], 0),
                ([ids[3], ids[0], ids[1]], 0),
            ],
        );
        assert_tets(&mesh, vec![([ids[0], ids[1], ids[3], ids[2]], 1)]);

        // Prematurely add triangle
        mesh.add_tri([ids[1], ids[2], ids[0]], 1, || 0);

        // Add twin
        assert_eq!(
            mesh.add_tet([ids[1], ids[0], ids[3], ids[2]], 2, || 0, || 0),
            None
        );
        assert_tris(
            &mesh,
            vec![
                ([ids[2], ids[1], ids[0]], 0),
                ([ids[1], ids[2], ids[3]], 0),
                ([ids[0], ids[3], ids[2]], 0),
                ([ids[3], ids[0], ids[1]], 0),
                ([ids[0], ids[1], ids[2]], 1),
                ([ids[3], ids[2], ids[1]], 0),
                ([ids[2], ids[3], ids[0]], 0),
                ([ids[1], ids[0], ids[3]], 0),
            ],
        );
        assert_tets(
            &mesh,
            vec![
                ([ids[0], ids[1], ids[3], ids[2]], 1),
                ([ids[0], ids[1], ids[2], ids[3]], 2),
            ],
        );

        // Modify tet
        assert_eq!(
            mesh.add_tet([ids[3], ids[2], ids[1], ids[0]], 3, || 0, || 0),
            Some(2)
        );
        assert_tris(
            &mesh,
            vec![
                ([ids[2], ids[1], ids[0]], 0),
                ([ids[1], ids[2], ids[3]], 0),
                ([ids[0], ids[3], ids[2]], 0),
                ([ids[3], ids[0], ids[1]], 0),
                ([ids[0], ids[1], ids[2]], 1),
                ([ids[3], ids[2], ids[1]], 0),
                ([ids[2], ids[3], ids[0]], 0),
                ([ids[1], ids[0], ids[3]], 0),
            ],
        );
        assert_tets(
            &mesh,
            vec![
                ([ids[0], ids[1], ids[3], ids[2]], 1),
                ([ids[0], ids[1], ids[2], ids[3]], 3),
            ],
        );
    }

    #[test]
    fn test_extend_tets() {
        let mut mesh = ComboMesh3::<usize, usize, usize, usize>::default();
        let ids = mesh.extend_vertices(vec![3, 6, 9, 2, 5, 8, 1, 4, 7]);

        let tets = vec![
            ([ids[0], ids[1], ids[2], ids[3]], 1),
            ([ids[1], ids[2], ids[3], ids[0]], 2),
            ([ids[0], ids[2], ids[3], ids[4]], 3),
            ([ids[2], ids[3], ids[4], ids[5]], 4),
            ([ids[6], ids[5], ids[4], ids[3]], 5),
            ([ids[6], ids[7], ids[4], ids[5]], 6),
        ];
        mesh.extend_tets(tets.clone(), || 0, || 0);
        assert_eq!(mesh.num_tris, 23);
        assert_tets(&mesh, tets);
    }

    #[test]
    fn test_remove_vertex() {
        let mut mesh = ComboMesh3::<usize, usize, usize, usize>::default();
        let ids = mesh.extend_vertices(vec![3, 6, 9, 2, 5]);
        let edges = vec![
            ([ids[0], ids[3]], 5),
            ([ids[1], ids[3]], 3),
            ([ids[3], ids[1]], 2),
            ([ids[2], ids[3]], 8),
            ([ids[1], ids[2]], 9),
        ];
        mesh.extend_edges(edges.clone());

        mesh.remove_vertex(ids[4]); // edgeless vertex
        assert_vertices(
            &mesh,
            vec![(ids[0], 3), (ids[1], 6), (ids[2], 9), (ids[3], 2)],
        );
        assert_edges(
            &mesh,
            vec![
                ([ids[0], ids[3]], 5),
                ([ids[1], ids[3]], 3),
                ([ids[3], ids[1]], 2),
                ([ids[2], ids[3]], 8),
                ([ids[1], ids[2]], 9),
            ],
        );

        mesh.remove_vertex(ids[1]); // vertex with edge
        assert_vertices(&mesh, vec![(ids[0], 3), (ids[2], 9), (ids[3], 2)]);
        assert_edges(&mesh, vec![([ids[0], ids[3]], 5), ([ids[2], ids[3]], 8)]);

        mesh.remove_vertex(ids[4]); // nonexistent vertex
        assert_vertices(&mesh, vec![(ids[0], 3), (ids[2], 9), (ids[3], 2)]);
        assert_edges(&mesh, vec![([ids[0], ids[3]], 5), ([ids[2], ids[3]], 8)]);
    }

    #[test]
    fn test_remove_add_vertex() {
        let mut mesh = ComboMesh3::<usize, usize, usize, usize>::default();
        let ids = mesh.extend_vertices(vec![3, 6, 9, 2]);
        let edges = vec![
            ([ids[0], ids[3]], 5),
            ([ids[1], ids[3]], 3),
            ([ids[3], ids[1]], 2),
            ([ids[2], ids[3]], 8),
            ([ids[1], ids[2]], 9),
        ];
        mesh.extend_edges(edges.clone());

        mesh.remove_vertex(ids[1]);
        assert_vertices(&mesh, vec![(ids[0], 3), (ids[2], 9), (ids[3], 2)]);
        assert_edges(&mesh, vec![([ids[0], ids[3]], 5), ([ids[2], ids[3]], 8)]);

        let id2 = mesh.add_vertex(6);
        assert_vertices(&mesh, vec![(ids[0], 3), (ids[2], 9), (ids[3], 2), (id2, 6)]);
        assert_edges(&mesh, vec![([ids[0], ids[3]], 5), ([ids[2], ids[3]], 8)]);
    }

    #[test]
    fn test_remove_edge() {
        let mut mesh = ComboMesh3::<usize, usize, usize, usize>::default();
        let ids = mesh.extend_vertices(vec![3, 6, 9, 2]);
        let edges = vec![
            ([ids[0], ids[3]], 5),
            ([ids[1], ids[3]], 3),
            ([ids[3], ids[1]], 2),
            ([ids[2], ids[3]], 8),
            ([ids[1], ids[2]], 9),
        ];
        mesh.extend_edges(edges.clone());

        mesh.remove_edge([ids[1], ids[3]]); // first outgoing edge from vertex
        assert_edges(
            &mesh,
            vec![
                ([ids[0], ids[3]], 5),
                ([ids[3], ids[1]], 2),
                ([ids[2], ids[3]], 8),
                ([ids[1], ids[2]], 9),
            ],
        );

        mesh.remove_edge([ids[1], ids[2]]); // last outgoing edge from vertex
        assert_edges(
            &mesh,
            vec![
                ([ids[0], ids[3]], 5),
                ([ids[3], ids[1]], 2),
                ([ids[2], ids[3]], 8),
            ],
        );

        mesh.remove_edge([ids[3], ids[0]]); // nonexistent edge
        assert_edges(
            &mesh,
            vec![
                ([ids[0], ids[3]], 5),
                ([ids[3], ids[1]], 2),
                ([ids[2], ids[3]], 8),
            ],
        );
    }

    #[test]
    fn test_remove_tri() {
        let mut mesh = ComboMesh3::<usize, usize, usize, usize>::default();
        let ids = mesh.extend_vertices(vec![3, 6, 9, 2, 5, 8, 1, 4]);
        let tris = vec![
            ([ids[0], ids[1], ids[2]], 1),
            ([ids[3], ids[1], ids[2]], 2),
            ([ids[4], ids[2], ids[1]], 3),
            ([ids[4], ids[1], ids[5]], 4),
            ([ids[5], ids[6], ids[4]], 5),
            ([ids[4], ids[6], ids[5]], 6),
        ];
        mesh.extend_tris(tris, || 0);

        assert_eq!(mesh.remove_tri([ids[0], ids[1], ids[2]]), Some(1)); // first tri with edge
        assert_edges(
            &mesh,
            vec![
                ([ids[1], ids[2]], 0),
                ([ids[3], ids[1]], 0),
                ([ids[2], ids[3]], 0),
                ([ids[4], ids[2]], 0),
                ([ids[2], ids[1]], 0),
                ([ids[1], ids[4]], 0),
                ([ids[4], ids[1]], 0),
                ([ids[1], ids[5]], 0),
                ([ids[5], ids[4]], 0),
                ([ids[5], ids[6]], 0),
                ([ids[6], ids[4]], 0),
                ([ids[4], ids[5]], 0),
                ([ids[4], ids[6]], 0),
                ([ids[6], ids[5]], 0),
            ],
        );
        assert_tris(
            &mesh,
            vec![
                ([ids[3], ids[1], ids[2]], 2),
                ([ids[4], ids[2], ids[1]], 3),
                ([ids[4], ids[1], ids[5]], 4),
                ([ids[5], ids[6], ids[4]], 5),
                ([ids[4], ids[6], ids[5]], 6),
            ],
        );

        assert_eq!(mesh.remove_tri([ids[3], ids[1], ids[2]]), Some(2)); // last tri with edge
        assert_edges(
            &mesh,
            vec![
                ([ids[4], ids[2]], 0),
                ([ids[2], ids[1]], 0),
                ([ids[1], ids[4]], 0),
                ([ids[4], ids[1]], 0),
                ([ids[1], ids[5]], 0),
                ([ids[5], ids[4]], 0),
                ([ids[5], ids[6]], 0),
                ([ids[6], ids[4]], 0),
                ([ids[4], ids[5]], 0),
                ([ids[4], ids[6]], 0),
                ([ids[6], ids[5]], 0),
            ],
        );
        assert_tris(
            &mesh,
            vec![
                ([ids[4], ids[2], ids[1]], 3),
                ([ids[4], ids[1], ids[5]], 4),
                ([ids[5], ids[6], ids[4]], 5),
                ([ids[4], ids[6], ids[5]], 6),
            ],
        );

        assert_eq!(mesh.remove_tri([ids[1], ids[2], ids[4]]), None); // nonexistent tri
        assert_edges(
            &mesh,
            vec![
                ([ids[4], ids[2]], 0),
                ([ids[2], ids[1]], 0),
                ([ids[1], ids[4]], 0),
                ([ids[4], ids[1]], 0),
                ([ids[1], ids[5]], 0),
                ([ids[5], ids[4]], 0),
                ([ids[5], ids[6]], 0),
                ([ids[6], ids[4]], 0),
                ([ids[4], ids[5]], 0),
                ([ids[4], ids[6]], 0),
                ([ids[6], ids[5]], 0),
            ],
        );
        assert_tris(
            &mesh,
            vec![
                ([ids[4], ids[2], ids[1]], 3),
                ([ids[4], ids[1], ids[5]], 4),
                ([ids[5], ids[6], ids[4]], 5),
                ([ids[4], ids[6], ids[5]], 6),
            ],
        );
    }

    #[test]
    fn test_remove_tet() {
        let mut mesh = ComboMesh3::<usize, usize, usize, usize>::default();
        let ids = mesh.extend_vertices(vec![3, 6, 9, 2, 5, 8, 1, 4, 7]);
        let tets = vec![
            ([ids[0], ids[1], ids[2], ids[3]], 1),
            ([ids[1], ids[2], ids[3], ids[0]], 2),
            ([ids[0], ids[2], ids[3], ids[4]], 3),
            ([ids[2], ids[3], ids[4], ids[5]], 4),
            ([ids[6], ids[5], ids[4], ids[3]], 5),
            ([ids[6], ids[7], ids[4], ids[5]], 6),
        ];
        mesh.extend_tets(tets.clone(), || 0, || 0);

        assert_eq!(mesh.remove_tet([ids[0], ids[1], ids[2], ids[3]]), Some(1));
        assert_eq!(mesh.num_tris, 20);
        assert_tets(
            &mesh,
            vec![
                ([ids[1], ids[2], ids[3], ids[0]], 2),
                ([ids[0], ids[2], ids[3], ids[4]], 3),
                ([ids[2], ids[3], ids[4], ids[5]], 4),
                ([ids[6], ids[5], ids[4], ids[3]], 5),
                ([ids[6], ids[7], ids[4], ids[5]], 6),
            ],
        );

        assert_eq!(mesh.remove_tet([ids[0], ids[1], ids[3], ids[2]]), Some(2));
        assert_eq!(mesh.num_tris, 16);
        assert_tets(
            &mesh,
            vec![
                ([ids[0], ids[2], ids[3], ids[4]], 3),
                ([ids[2], ids[3], ids[4], ids[5]], 4),
                ([ids[6], ids[5], ids[4], ids[3]], 5),
                ([ids[6], ids[7], ids[4], ids[5]], 6),
            ],
        );

        assert_eq!(mesh.remove_tet([ids[4], ids[1], ids[3], ids[2]]), None);
        assert_eq!(mesh.num_tris, 16);
        assert_tets(
            &mesh,
            vec![
                ([ids[0], ids[2], ids[3], ids[4]], 3),
                ([ids[2], ids[3], ids[4], ids[5]], 4),
                ([ids[6], ids[5], ids[4], ids[3]], 5),
                ([ids[6], ids[7], ids[4], ids[5]], 6),
            ],
        );
    }

    #[test]
    fn test_remove_add_tet() {
        let mut mesh = ComboMesh3::<usize, usize, usize, usize>::default();
        let ids = mesh.extend_vertices(vec![3, 6, 9, 2, 5, 8, 1, 4, 7]);
        let tets = vec![
            ([ids[0], ids[1], ids[2], ids[3]], 1),
            ([ids[1], ids[2], ids[3], ids[0]], 2),
            ([ids[0], ids[2], ids[3], ids[4]], 3),
            ([ids[2], ids[3], ids[4], ids[5]], 4),
            ([ids[6], ids[5], ids[4], ids[3]], 5),
            ([ids[6], ids[7], ids[4], ids[5]], 6),
        ];
        mesh.extend_tets(tets.clone(), || 0, || 0);

        assert_eq!(mesh.remove_tet([ids[0], ids[1], ids[2], ids[3]]), Some(1));
        assert_eq!(
            mesh.add_tet([ids[7], ids[6], ids[4], ids[8]], 7, || 0, || 0),
            None
        );
        assert_eq!(
            mesh.add_tet([ids[0], ids[1], ids[2], ids[3]], 8, || 0, || 0),
            None
        );
        assert_eq!(mesh.num_tris, 27);
        assert_tets(
            &mesh,
            vec![
                ([ids[0], ids[1], ids[2], ids[3]], 8),
                ([ids[1], ids[2], ids[3], ids[0]], 2),
                ([ids[0], ids[2], ids[3], ids[4]], 3),
                ([ids[2], ids[3], ids[4], ids[5]], 4),
                ([ids[6], ids[5], ids[4], ids[3]], 5),
                ([ids[6], ids[7], ids[4], ids[5]], 6),
                ([ids[7], ids[6], ids[4], ids[8]], 7),
            ],
        );
    }

    #[test]
    fn test_remove_add_edge() {
        let mut mesh = ComboMesh3::<usize, usize, usize, usize>::default();
        let ids = mesh.extend_vertices(vec![3, 6, 9, 2]);
        let edges = vec![
            ([ids[0], ids[3]], 5),
            ([ids[1], ids[3]], 3),
            ([ids[3], ids[1]], 2),
            ([ids[2], ids[3]], 8),
            ([ids[1], ids[2]], 9),
        ];
        mesh.extend_edges(edges.clone());

        mesh.remove_edge([ids[1], ids[3]]); // first outgoing edge from vertex
        assert_edges(
            &mesh,
            vec![
                ([ids[0], ids[3]], 5),
                ([ids[3], ids[1]], 2),
                ([ids[2], ids[3]], 8),
                ([ids[1], ids[2]], 9),
            ],
        );

        mesh.add_edge([ids[1], ids[0]], 4);
        mesh.add_edge([ids[1], ids[3]], 6);
        assert_edges(
            &mesh,
            vec![
                ([ids[0], ids[3]], 5),
                ([ids[3], ids[1]], 2),
                ([ids[2], ids[3]], 8),
                ([ids[1], ids[2]], 9),
                ([ids[1], ids[0]], 4),
                ([ids[1], ids[3]], 6),
            ],
        );
    }

    #[test]
    fn test_clear_vertices() {
        let mut mesh = ComboMesh3::<usize, usize, usize, usize>::default();
        let ids = mesh.extend_vertices(vec![3, 6, 9, 2, 5, 8, 1, 4]);
        let edges = vec![
            ([ids[0], ids[3]], 5),
            ([ids[1], ids[3]], 3),
            ([ids[3], ids[1]], 2),
            ([ids[2], ids[3]], 8),
            ([ids[1], ids[2]], 9),
        ];
        mesh.extend_edges(edges.clone());

        mesh.clear_vertices();
        assert_vertices(&mesh, vec![]);
        assert_edges(&mesh, vec![] as Vec<(EdgeId, _)>);
        assert_tris(&mesh, vec![] as Vec<(TriId, _)>);
        assert_tets(&mesh, vec![] as Vec<(TetId, _)>);
    }

    #[test]
    fn test_clear_edges() {
        let mut mesh = ComboMesh3::<usize, usize, usize, usize>::default();
        let ids = mesh.extend_vertices(vec![3, 6, 9, 2, 5, 8, 1, 4]);
        let tris = vec![
            ([ids[0], ids[1], ids[2]], 1),
            ([ids[3], ids[1], ids[2]], 2),
            ([ids[4], ids[2], ids[1]], 3),
            ([ids[4], ids[1], ids[5]], 4),
            ([ids[5], ids[6], ids[4]], 5),
            ([ids[4], ids[6], ids[5]], 6),
        ];
        mesh.extend_tris(tris, || 0);

        mesh.clear_edges();
        assert_vertices(
            &mesh,
            vec![
                (ids[0], 3),
                (ids[1], 6),
                (ids[2], 9),
                (ids[3], 2),
                (ids[4], 5),
                (ids[5], 8),
                (ids[6], 1),
                (ids[7], 4),
            ],
        );
        assert_edges(&mesh, vec![] as Vec<(EdgeId, _)>);
        assert_tris(&mesh, vec![] as Vec<(TriId, _)>);
        assert_tets(&mesh, vec![] as Vec<(TetId, _)>);
    }

    #[test]
    fn test_clear_tris() {
        let mut mesh = ComboMesh3::<usize, usize, usize, usize>::default();
        let ids = mesh.extend_vertices(vec![3, 6, 9, 2, 5, 8, 1, 4]);
        let tris = vec![
            ([ids[0], ids[1], ids[2]], 1),
            ([ids[3], ids[1], ids[2]], 2),
            ([ids[4], ids[2], ids[1]], 3),
            ([ids[4], ids[1], ids[5]], 4),
            ([ids[5], ids[6], ids[4]], 5),
            ([ids[4], ids[6], ids[5]], 6),
        ];
        mesh.extend_tris(tris, || 0);

        mesh.clear_tris();
        assert_vertices(
            &mesh,
            vec![
                (ids[0], 3),
                (ids[1], 6),
                (ids[2], 9),
                (ids[3], 2),
                (ids[4], 5),
                (ids[5], 8),
                (ids[6], 1),
                (ids[7], 4),
            ],
        );
        assert_edges(
            &mesh,
            vec![
                ([ids[0], ids[1]], 0),
                ([ids[1], ids[2]], 0),
                ([ids[2], ids[0]], 0),
                ([ids[3], ids[1]], 0),
                ([ids[2], ids[3]], 0),
                ([ids[4], ids[2]], 0),
                ([ids[2], ids[1]], 0),
                ([ids[1], ids[4]], 0),
                ([ids[4], ids[1]], 0),
                ([ids[1], ids[5]], 0),
                ([ids[5], ids[4]], 0),
                ([ids[5], ids[6]], 0),
                ([ids[6], ids[4]], 0),
                ([ids[4], ids[5]], 0),
                ([ids[4], ids[6]], 0),
                ([ids[6], ids[5]], 0),
            ],
        );
        assert_tris(&mesh, vec![] as Vec<(TriId, _)>);
        assert_tets(&mesh, vec![] as Vec<(TetId, _)>);
    }

    #[test]
    fn test_clear_tets() {
        let mut mesh = ComboMesh3::<usize, usize, usize, usize>::default();
        let ids = mesh.extend_vertices(vec![3, 6, 9, 2, 5, 8, 1, 4, 7]);
        let tets = vec![
            ([ids[0], ids[1], ids[2], ids[3]], 1),
            ([ids[1], ids[2], ids[3], ids[0]], 2),
            ([ids[0], ids[2], ids[3], ids[4]], 3),
            ([ids[2], ids[3], ids[4], ids[5]], 4),
            ([ids[6], ids[5], ids[4], ids[3]], 5),
            ([ids[6], ids[7], ids[4], ids[5]], 6),
        ];
        mesh.extend_tets(tets.clone(), || 0, || 0);

        mesh.clear_tets();
        assert_eq!(mesh.num_vertices(), 9);
        assert_ne!(mesh.num_edges(), 0); // Don't want to think about this number
        assert_eq!(mesh.num_tris(), 23);
        assert_tets(&mesh, vec![] as Vec<(TetId, _)>);
    }

    #[test]
    fn test_tet_walker() {
        let mut mesh = ComboMesh3::<usize, usize, usize, usize>::default();
        let ids = mesh.extend_vertices(vec![3, 6, 9, 2, 5, 8, 1, 4, 7]);
        let tets = vec![
            ([ids[0], ids[1], ids[2], ids[3]], 1),
            ([ids[1], ids[2], ids[3], ids[0]], 2),
            ([ids[0], ids[2], ids[3], ids[4]], 3),
            ([ids[2], ids[3], ids[4], ids[5]], 4),
            ([ids[6], ids[5], ids[4], ids[3]], 5),
            ([ids[6], ids[7], ids[4], ids[5]], 6),
        ];
        mesh.extend_tets(tets.clone(), || 0, || 0);
        mesh.add_tri([ids[6], ids[7], ids[8]], 7, || 0);

        assert!(mesh.tet_walker_from_tri([ids[6], ids[7], ids[8]]).is_none());

        let walker = mesh.tet_walker_from_edge_edge([ids[2], ids[3]], [ids[0], ids[1]]);
        assert_eq!(walker.edge(), EdgeId([ids[2], ids[3]]));
        assert_eq!(walker.opp_edge(), EdgeId([ids[0], ids[1]]));
        assert_eq!(walker.first(), ids[2]);
        assert_eq!(walker.second(), ids[3]);
        assert_eq!(walker.third(), ids[0]);
        assert_eq!(walker.fourth(), ids[1]);
        assert_eq!(
            walker.tri(),
            [ids[2], ids[3], ids[0]].try_into().ok().unwrap()
        );
        assert_eq!(
            walker.tet(),
            [ids[2], ids[3], ids[0], ids[1]].try_into().ok().unwrap()
        );

        let branch = walker.twin().unwrap(); // branch!
        assert_eq!(branch.edge(), EdgeId([ids[3], ids[2]]));
        assert_eq!(branch.opp_edge(), EdgeId([ids[0], ids[1]]));

        let walker = walker.next_edge();
        assert_eq!(walker.edge(), EdgeId([ids[3], ids[0]]));
        assert_eq!(walker.opp_edge(), EdgeId([ids[2], ids[1]]));

        let walker = walker.next_edge();
        assert_eq!(walker.edge(), EdgeId([ids[0], ids[2]]));
        assert_eq!(walker.opp_edge(), EdgeId([ids[3], ids[1]]));

        let walker = walker.next_edge();
        assert_eq!(walker.edge(), EdgeId([ids[2], ids[3]]));
        assert_eq!(walker.opp_edge(), EdgeId([ids[0], ids[1]]));

        let branch = walker.prev_edge(); // branch!
        assert_eq!(branch.edge(), EdgeId([ids[0], ids[2]]));
        assert_eq!(branch.opp_edge(), EdgeId([ids[3], ids[1]]));

        let walker = walker.next_tri();
        assert_eq!(walker.edge(), EdgeId([ids[1], ids[0]]));
        assert_eq!(walker.opp_edge(), EdgeId([ids[3], ids[2]]));

        let walker = walker.next_tri();
        assert_eq!(walker.edge(), EdgeId([ids[0], ids[1]]));
        assert_eq!(walker.opp_edge(), EdgeId([ids[2], ids[3]]));

        let walker = walker.next_tri();
        assert_eq!(walker.edge(), EdgeId([ids[3], ids[2]]));
        assert_eq!(walker.opp_edge(), EdgeId([ids[1], ids[0]]));

        let walker = walker.next_tri();
        assert_eq!(walker.edge(), EdgeId([ids[2], ids[3]]));
        assert_eq!(walker.opp_edge(), EdgeId([ids[0], ids[1]]));

        let branch = walker.prev_tri(); // branch!
        assert_eq!(branch.edge(), EdgeId([ids[3], ids[2]]));
        assert_eq!(branch.opp_edge(), EdgeId([ids[1], ids[0]]));

        let branch = branch.flip_tri(); // branch continue!
        assert_eq!(branch.edge(), EdgeId([ids[1], ids[0]]));
        assert_eq!(branch.opp_edge(), EdgeId([ids[3], ids[2]]));

        let walker = walker.next_opp();
        assert_eq!(walker.edge(), EdgeId([ids[2], ids[3]]));
        assert_eq!(walker.opp_edge(), EdgeId([ids[0], ids[4]]));

        let walker = walker.prev_opp();
        assert_eq!(walker.edge(), EdgeId([ids[2], ids[3]]));
        assert_eq!(walker.opp_edge(), EdgeId([ids[0], ids[1]]));

        let walker = mesh.tet_walker_from_edge_edge([ids[4], ids[3]], [ids[2], ids[0]]);
        let walker = walker.on_twin_tri().unwrap();
        assert_eq!(walker.edge(), EdgeId([ids[3], ids[4]]));
        assert_eq!(walker.opp_edge(), EdgeId([ids[2], ids[5]]));

        let walker = walker.prev_tri().on_twin_tri().unwrap();
        assert_eq!(walker.edge(), EdgeId([ids[3], ids[4]]));
        assert_eq!(walker.opp_edge(), EdgeId([ids[5], ids[6]]));

        assert!(walker.twin().is_none());
    }

    #[test]
    fn test_vertex_edges_out() {
        let mut mesh = ComboMesh3::<usize, usize, usize, usize>::default();
        let ids = mesh.extend_vertices(vec![3, 6, 9, 2, 5]);
        let edges = vec![
            ([ids[0], ids[3]], 5),
            ([ids[1], ids[3]], 3),
            ([ids[3], ids[1]], 2),
            ([ids[1], ids[2]], 9),
            ([ids[2], ids[3]], 8),
        ];
        mesh.extend_edges(edges.clone());

        let set = mesh.vertex_edges_out(ids[4]).collect::<FnvHashSet<_>>();
        let expected = vec![].into_iter().collect::<FnvHashSet<_>>();
        assert_eq!(set, expected);

        let set = mesh.vertex_edges_out(ids[2]).collect::<FnvHashSet<_>>();
        let expected = vec![EdgeId([ids[2], ids[3]])]
            .into_iter()
            .collect::<FnvHashSet<_>>();
        assert_eq!(set, expected);

        let set = mesh.vertex_edges_out(ids[1]).collect::<FnvHashSet<_>>();
        let expected = vec![EdgeId([ids[1], ids[3]]), EdgeId([ids[1], ids[2]])]
            .into_iter()
            .collect::<FnvHashSet<_>>();
        assert_eq!(set, expected);
    }

    #[test]
    fn test_vertex_edges_in() {
        let mut mesh = ComboMesh3::<usize, usize, usize, usize>::default();
        let ids = mesh.extend_vertices(vec![3, 6, 9, 2, 5]);
        let edges = vec![
            ([ids[0], ids[3]], 5),
            ([ids[1], ids[3]], 3),
            ([ids[3], ids[1]], 2),
            ([ids[1], ids[2]], 9),
        ];
        mesh.extend_edges(edges.clone());

        let set = mesh.vertex_edges_in(ids[4]).collect::<FnvHashSet<_>>();
        let expected = vec![].into_iter().collect::<FnvHashSet<_>>();
        assert_eq!(set, expected);

        let set = mesh.vertex_edges_in(ids[2]).collect::<FnvHashSet<_>>();
        let expected = vec![EdgeId([ids[1], ids[2]])]
            .into_iter()
            .collect::<FnvHashSet<_>>();
        assert_eq!(set, expected);

        let set = mesh.vertex_edges_in(ids[3]).collect::<FnvHashSet<_>>();
        let expected = vec![EdgeId([ids[0], ids[3]]), EdgeId([ids[1], ids[3]])]
            .into_iter()
            .collect::<FnvHashSet<_>>();
        assert_eq!(set, expected);
    }

    #[test]
    fn test_edge_tris() {
        let mut mesh = ComboMesh3::<usize, usize, usize, usize>::default();
        let ids = mesh.extend_vertices(vec![3, 6, 9, 2, 5, 8, 1, 4]);
        let tris = vec![
            ([ids[0], ids[1], ids[2]], 1),
            ([ids[3], ids[1], ids[2]], 2),
            ([ids[4], ids[2], ids[1]], 3),
            ([ids[4], ids[1], ids[5]], 4),
            ([ids[5], ids[6], ids[4]], 5),
            ([ids[4], ids[6], ids[5]], 6),
        ];
        mesh.extend_tris(tris, || 0);
        mesh.add_edge([ids[6], ids[7]], 1);

        let set = mesh.edge_tris([ids[6], ids[7]]).collect::<FnvHashSet<_>>();
        let expected = vec![].into_iter().collect::<FnvHashSet<_>>();
        assert_eq!(set, expected);

        let set = mesh.edge_tris([ids[0], ids[1]]).collect::<FnvHashSet<_>>();
        let expected = vec![TriId([ids[0], ids[1], ids[2]])]
            .into_iter()
            .collect::<FnvHashSet<_>>();
        assert_eq!(set, expected);

        let set = mesh.edge_tris([ids[1], ids[2]]).collect::<FnvHashSet<_>>();
        let expected = vec![
            TriId([ids[0], ids[1], ids[2]]),
            TriId([ids[1], ids[2], ids[3]]),
        ]
        .into_iter()
        .collect::<FnvHashSet<_>>();
        assert_eq!(set, expected);
    }

    #[test]
    fn test_vertex_tris() {
        let mut mesh = ComboMesh3::<usize, usize, usize, usize>::default();
        let ids = mesh.extend_vertices(vec![3, 6, 9, 2, 5, 8, 1, 4]);
        let tris = vec![
            ([ids[0], ids[1], ids[2]], 1),
            ([ids[3], ids[1], ids[2]], 2),
            ([ids[4], ids[2], ids[1]], 3),
            ([ids[4], ids[1], ids[5]], 4),
            ([ids[5], ids[6], ids[4]], 5),
        ];
        mesh.extend_tris(tris, || 0);
        mesh.add_edge([ids[6], ids[7]], 1);

        let set = mesh.vertex_tris(ids[7]).collect::<FnvHashSet<_>>();
        let expected = vec![].into_iter().collect::<FnvHashSet<_>>();
        assert_eq!(set, expected);

        let set = mesh.vertex_tris(ids[6]).collect::<FnvHashSet<_>>();
        let expected = vec![TriId([ids[4], ids[5], ids[6]])]
            .into_iter()
            .collect::<FnvHashSet<_>>();
        assert_eq!(set, expected);

        let set = mesh.vertex_tris(ids[1]).collect::<FnvHashSet<_>>();
        let expected = vec![
            TriId([ids[0], ids[1], ids[2]]),
            TriId([ids[1], ids[2], ids[3]]),
            TriId([ids[1], ids[4], ids[2]]),
            TriId([ids[1], ids[5], ids[4]]),
        ]
        .into_iter()
        .collect::<FnvHashSet<_>>();
        assert_eq!(set, expected);
    }

    #[test]
    fn test_tri_tets() {
        let mut mesh = ComboMesh3::<usize, usize, usize, usize>::default();
        let ids = mesh.extend_vertices(vec![3, 6, 9, 2, 5, 8, 1, 4, 7]);
        let tets = vec![
            ([ids[0], ids[1], ids[2], ids[3]], 1),
            ([ids[1], ids[2], ids[3], ids[0]], 2),
            ([ids[0], ids[2], ids[3], ids[4]], 3),
            ([ids[2], ids[3], ids[4], ids[5]], 4),
            ([ids[6], ids[5], ids[4], ids[3]], 5),
            ([ids[6], ids[7], ids[4], ids[5]], 6),
        ];
        mesh.extend_tets(tets.clone(), || 0, || 0);
        mesh.add_tri([ids[6], ids[7], ids[8]], 7, || 0);

        let set = mesh
            .tri_tets([ids[6], ids[7], ids[8]])
            .collect::<FnvHashSet<_>>();
        let expected = vec![].into_iter().collect::<FnvHashSet<_>>();
        assert_eq!(set, expected);

        let set = mesh
            .tri_tets([ids[4], ids[5], ids[6]])
            .collect::<FnvHashSet<_>>();
        let expected = vec![TetId([ids[4], ids[5], ids[6], ids[7]])]
            .into_iter()
            .collect::<FnvHashSet<_>>();
        assert_eq!(set, expected);

        let set = mesh
            .tri_tets([ids[0], ids[2], ids[3]])
            .collect::<FnvHashSet<_>>();
        let expected = vec![
            TetId([ids[0], ids[1], ids[2], ids[3]]),
            TetId([ids[0], ids[2], ids[3], ids[4]]),
        ]
        .into_iter()
        .collect::<FnvHashSet<_>>();
        assert_eq!(set, expected);
    }

    #[test]
    fn test_edge_tets() {
        let mut mesh = ComboMesh3::<usize, usize, usize, usize>::default();
        let ids = mesh.extend_vertices(vec![3, 6, 9, 2, 5, 8, 1, 4, 7]);
        let tets = vec![
            ([ids[0], ids[1], ids[2], ids[3]], 1),
            ([ids[1], ids[2], ids[3], ids[0]], 2),
            ([ids[0], ids[2], ids[3], ids[4]], 3),
            ([ids[2], ids[3], ids[4], ids[5]], 4),
            ([ids[6], ids[5], ids[4], ids[3]], 5),
            ([ids[6], ids[7], ids[4], ids[5]], 6),
        ];
        mesh.extend_tets(tets.clone(), || 0, || 0);
        mesh.add_tri([ids[6], ids[7], ids[8]], 7, || 0);

        let set = mesh.edge_tets([ids[7], ids[8]]).collect::<FnvHashSet<_>>();
        let expected = vec![].into_iter().collect::<FnvHashSet<_>>();
        assert_eq!(set, expected);

        let list = mesh.edge_tets([ids[6], ids[7]]).collect::<Vec<_>>();
        assert_eq!(list.len(), 1);
        let set = list.into_iter().collect::<FnvHashSet<_>>();
        let expected = vec![TetId([ids[4], ids[5], ids[6], ids[7]])]
            .into_iter()
            .collect::<FnvHashSet<_>>();
        assert_eq!(set, expected);

        let list = mesh.edge_tets([ids[3], ids[4]]).collect::<Vec<_>>();
        assert_eq!(list.len(), 3);
        let set = list.into_iter().collect::<FnvHashSet<_>>();
        let expected = vec![
            TetId([ids[0], ids[2], ids[3], ids[4]]),
            TetId([ids[2], ids[3], ids[4], ids[5]]),
            TetId([ids[3], ids[4], ids[5], ids[6]]),
        ]
        .into_iter()
        .collect::<FnvHashSet<_>>();
        assert_eq!(set, expected);
    }

    #[test]
    fn test_vertex_tets() {
        let mut mesh = ComboMesh3::<usize, usize, usize, usize>::default();
        let ids = mesh.extend_vertices(vec![3, 6, 9, 2, 5, 8, 1, 4, 7]);
        let tets = vec![
            ([ids[0], ids[1], ids[2], ids[3]], 1),
            ([ids[1], ids[2], ids[3], ids[0]], 2),
            ([ids[0], ids[2], ids[3], ids[4]], 3),
            ([ids[2], ids[3], ids[4], ids[5]], 4),
            ([ids[6], ids[5], ids[4], ids[3]], 5),
            ([ids[6], ids[7], ids[4], ids[5]], 6),
        ];
        mesh.extend_tets(tets.clone(), || 0, || 0);
        mesh.add_tri([ids[6], ids[7], ids[8]], 7, || 0);

        let set = mesh.vertex_tets(ids[8]).collect::<FnvHashSet<_>>();
        let expected = vec![].into_iter().collect::<FnvHashSet<_>>();
        assert_eq!(set, expected);

        let list = mesh.vertex_tets(ids[7]).collect::<Vec<_>>();
        assert_eq!(list.len(), 1);
        let set = list.into_iter().collect::<FnvHashSet<_>>();
        let expected = vec![TetId([ids[4], ids[5], ids[6], ids[7]])]
            .into_iter()
            .collect::<FnvHashSet<_>>();
        assert_eq!(set, expected);

        let list = mesh.vertex_tets(ids[4]).collect::<Vec<_>>();
        assert_eq!(list.len(), 4);
        let set = list.into_iter().collect::<FnvHashSet<_>>();
        let expected = vec![
            TetId([ids[0], ids[2], ids[3], ids[4]]),
            TetId([ids[2], ids[3], ids[4], ids[5]]),
            TetId([ids[3], ids[4], ids[5], ids[6]]),
            TetId([ids[4], ids[5], ids[6], ids[7]]),
        ]
        .into_iter()
        .collect::<FnvHashSet<_>>();
        assert_eq!(set, expected);
    }
}
