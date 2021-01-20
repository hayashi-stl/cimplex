use fnv::FnvHashMap;
use idmap::OrderedIdMap;
#[cfg(feature = "serde_")]
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use typenum::{U2, U3};

use crate::edge::{EdgeId, HasEdges};
use crate::mesh_1::internal::HigherVertex;
use crate::tri::{HasTris, TriId};
use crate::vertex::{HasVertices, VertexId};
use crate::VecN;

use internal::{HigherEdge, Tri};

/// A combinatorial simplicial 2-complex, containing only vertices, (oriented) edges, and (oriented) triangles.
/// Also known as an tri mesh.
/// Each vertex stores a value of type `V`.
/// Each edge stores its vertices and a value of type `E`.
/// Each triangle stores its vertices and a value of type `F`.
/// The edge manipulation methods can either be called with an array of 2 `VertexId`s
/// or an `EdgeId`.
/// The triangle manipulation methods can either be called with an array of 3 `VertexId`s
/// or an `TriId`.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde_", derive(Serialize, Deserialize))]
pub struct ComboMesh2<V, E, F> {
    vertices: OrderedIdMap<VertexId, HigherVertex<V>>,
    edges: FnvHashMap<EdgeId, HigherEdge<E>>,
    tris: FnvHashMap<TriId, Tri<F>>,
    next_vertex_id: u64,
    /// Keep separate track because edge twins may or may not exist
    num_edges: usize,
    num_tris: usize,
}
crate::impl_has_vertices!(ComboMesh2<V, E, F>, HigherVertex);
crate::impl_has_edges!(ComboMesh2<V, E, F>, HigherEdge);
crate::impl_has_tris!(ComboMesh2<V, E, F>, Tri);

impl<V, E, F> HasVertices for ComboMesh2<V, E, F> {}
impl<V, E, F> HasEdges for ComboMesh2<V, E, F> {}
impl<V, E, F> HasTris for ComboMesh2<V, E, F> {}

impl<V, E, F> Default for ComboMesh2<V, E, F> {
    fn default() -> Self {
        ComboMesh2 {
            vertices: OrderedIdMap::default(),
            edges: FnvHashMap::default(),
            tris: FnvHashMap::default(),
            next_vertex_id: 0,
            num_edges: 0,
            num_tris: 0,
        }
    }
}

impl<V, E, F> ComboMesh2<V, E, F> {
    /// Creates an empty tri mesh.
    pub fn new() -> Self {
        Self::default()
    }
}

/// A position-containing tri mesh
pub type Mesh2<V, E, F, D> = ComboMesh2<(VecN<D>, V), E, F>;

/// A 2D-position-containing tri mesh
pub type Mesh22<V, E, F> = Mesh2<V, E, F, U2>;

/// A 3D-position-containing tri mesh
pub type Mesh23<V, E, F> = Mesh2<V, E, F, U3>;

mod internal {
    use super::ComboMesh2;
    use crate::edge::internal::{ClearEdgesHigher, Link, RemoveEdgeHigher};
    use crate::edge::{EdgeId, HasEdges};
    use crate::tri::internal::{ClearTrisHigher, RemoveTriHigher};
    use crate::tri::{HasTris, TriId};
    use crate::vertex::internal::{ClearVerticesHigher, RemoveVertexHigher};
    use crate::vertex::VertexId;

    #[derive(Clone, Debug)]
    #[cfg_attr(feature = "serde_", derive(Serialize, Deserialize))]
    pub struct HigherEdge<E> {
        /// Outgoing targets from the same vertex, whether the edge actually exists or not
        link: Link<VertexId>,
        /// Some vertex opposite this edge in a triangle.
        /// This is the edge's first vertex if the edge is not part of a triangle.
        tri_opp: VertexId,
        /// The edge does not actually exist if the value is None;
        /// it is just there for the structural purpose of
        /// ensuring that every edge has a twin.
        value: Option<E>,
    }
    crate::impl_edge!(
        HigherEdge<E>,
        new | id,
        link,
        value | {
            HigherEdge {
                tri_opp: id,
                link,
                value,
            }
        }
    );
    crate::impl_higher_edge!(HigherEdge<E>);

    /// A triangle of an tri mesh
    #[derive(Clone, Debug)]
    #[doc(hidden)]
    #[cfg_attr(feature = "serde_", derive(Serialize, Deserialize))]
    pub struct Tri<F> {
        /// Targets from the same edge for each of the edges,
        /// whether the face actually exists or not
        links: [Link<VertexId>; 3],
        /// The triangle does not actually exist if the value is None;
        /// it is just there for the structural purpose of
        /// ensuring that every triangle has a twin.
        value: Option<F>,
    }
    crate::impl_tri!(Tri<F>, new | _id, links, value | Tri { links, value });

    impl<V, E, F> RemoveVertexHigher for ComboMesh2<V, E, F> {
        fn remove_vertex_higher(&mut self, vertex: VertexId) {
            self.remove_edges(
                self.vertex_edges_out(vertex)
                    .chain(self.vertex_edges_in(vertex))
                    .collect::<Vec<_>>(),
            );
        }
    }

    impl<V, E, F> ClearVerticesHigher for ComboMesh2<V, E, F> {
        fn clear_vertices_higher(&mut self) {
            self.tris.clear();
            self.num_tris = 0;
            self.edges.clear();
            self.num_edges = 0;
        }
    }

    impl<V, E, F> RemoveEdgeHigher for ComboMesh2<V, E, F> {
        fn remove_edge_higher(&mut self, edge: EdgeId) {
            self.remove_tris(self.edge_tris(edge).collect::<Vec<_>>());
        }
    }

    impl<V, E, F> ClearEdgesHigher for ComboMesh2<V, E, F> {
        fn clear_edges_higher(&mut self) {
            self.tris.clear();
            self.num_tris = 0;
        }
    }

    impl<V, E, F> RemoveTriHigher for ComboMesh2<V, E, F> {
        fn remove_tri_higher(&mut self, _: TriId) {}
    }

    impl<V, E, F> ClearTrisHigher for ComboMesh2<V, E, F> {
        fn clear_tris_higher(&mut self) {}
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
        I: IntoIterator<Item = (VertexId, V)>,
    >(
        mesh: &ComboMesh2<V, E, F>,
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
        I: IntoIterator<Item = (EI, E)>,
    >(
        mesh: &ComboMesh2<V, E, F>,
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
        I: IntoIterator<Item = (FI, F)>,
    >(
        mesh: &ComboMesh2<V, E, F>,
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

    #[test]
    fn test_default() {
        let mesh = ComboMesh2::<(), (), ()>::default();
        assert!(mesh.vertices.is_empty());
        assert!(mesh.edges.is_empty());
        assert!(mesh.tris.is_empty());
        assert_eq!(mesh.num_edges(), 0);
        assert_eq!(mesh.num_tris(), 0);
    }

    #[test]
    fn test_add_vertex() {
        let mut mesh = ComboMesh2::<usize, usize, usize>::default();
        let id = mesh.add_vertex(3);
        assert_eq!(mesh.vertex(id), Some(&3));

        let id2 = mesh.add_vertex(9);
        assert_eq!(mesh.vertex(id), Some(&3));
        assert_eq!(mesh.vertex(id2), Some(&9));
    }

    #[test]
    fn test_extend_vertices() {
        let mut mesh = ComboMesh2::<usize, usize, usize>::default();
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
        let mut mesh = ComboMesh2::<usize, usize, usize>::default();
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
        let mut mesh = ComboMesh2::<usize, usize, usize>::default();
        let ids = mesh.extend_vertices(vec![3, 6, 9, 2]);
        mesh.add_edge([ids[1], ids[1]], 4);
    }

    #[test]
    fn test_extend_edges() {
        let mut mesh = ComboMesh2::<usize, usize, usize>::default();
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
        let mut mesh = ComboMesh2::<usize, usize, usize>::default();
        let ids = mesh.extend_vertices(vec![3, 6, 9, 2]);
        let prev = mesh.add_edge([ids[1], ids[3]], 54);
    }

    #[test]
    fn test_remove_vertex() {
        let mut mesh = ComboMesh2::<usize, usize, usize>::default();
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
        let mut mesh = ComboMesh2::<usize, usize, usize>::default();
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
        let mut mesh = ComboMesh2::<usize, usize, usize>::default();
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
    fn test_remove_add_edge() {
        let mut mesh = ComboMesh2::<usize, usize, usize>::default();
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
        let mut mesh = ComboMesh2::<usize, usize, usize>::default();
        let ids = mesh.extend_vertices(vec![3, 6, 9, 2]);
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
    }

    #[test]
    fn test_clear_edges() {
        let mut mesh = ComboMesh2::<usize, usize, usize>::default();
        let ids = mesh.extend_vertices(vec![3, 6, 9, 2]);
        let edges = vec![
            ([ids[0], ids[3]], 5),
            ([ids[1], ids[3]], 3),
            ([ids[3], ids[1]], 2),
            ([ids[2], ids[3]], 8),
            ([ids[1], ids[2]], 9),
        ];
        mesh.extend_edges(edges.clone());

        mesh.clear_edges();
        assert_vertices(
            &mesh,
            vec![(ids[0], 3), (ids[1], 6), (ids[2], 9), (ids[3], 2)],
        );
        assert_edges(&mesh, vec![] as Vec<(EdgeId, _)>);
    }

    #[test]
    fn test_walker() {
        let mut mesh = ComboMesh2::<usize, usize, usize>::default();
        let ids = mesh.extend_vertices(vec![3, 6, 9, 2, 5]);
        let edges = vec![
            ([ids[0], ids[3]], 5),
            ([ids[1], ids[3]], 3),
            ([ids[3], ids[1]], 2),
            ([ids[1], ids[2]], 9),
            ([ids[2], ids[3]], 8),
        ];
        mesh.extend_edges(edges.clone());

        assert!(mesh.edge_walker_from_vertex(ids[4]).is_none());

        let walker = mesh.edge_walker_from_edge([ids[3], ids[1]]);
        assert_eq!(walker.edge(), EdgeId([ids[3], ids[1]]));
        assert_eq!(walker.vertex(), ids[3]);
        assert_eq!(walker.target(), ids[1]);

        let walker = walker.next();
        assert_eq!(walker.edge(), EdgeId([ids[3], ids[1]]));

        let walker = walker.prev();
        assert_eq!(walker.edge(), EdgeId([ids[3], ids[1]]));

        let walker = walker.twin().unwrap();
        assert_eq!(walker.edge(), EdgeId([ids[1], ids[3]]));

        let walker = walker.twin().unwrap();
        assert_eq!(walker.edge(), EdgeId([ids[3], ids[1]]));

        let walker = walker.forward().unwrap();
        assert_eq!(walker.edge(), EdgeId([ids[1], ids[2]]));

        let walker = walker.next();
        assert_eq!(walker.edge(), EdgeId([ids[1], ids[3]]));

        let walker = walker.prev();
        assert_eq!(walker.edge(), EdgeId([ids[1], ids[2]]));

        let walker = walker.forward().unwrap();
        assert_eq!(walker.edge(), EdgeId([ids[2], ids[3]]));

        let walker = walker.backward().unwrap();
        assert_eq!(walker.edge(), EdgeId([ids[1], ids[2]]));

        assert!(walker.twin().is_none());

        let walker = mesh.edge_walker_from_edge([ids[0], ids[3]]);
        assert!(walker.backward().is_none());

        let walker = walker.next_in();
        assert_ne!(walker.vertex(), ids[0]);
        assert_eq!(walker.target(), ids[3]);

        let walker = walker.prev_in();
        assert_eq!(walker.edge(), EdgeId([ids[0], ids[3]]));
    }

    #[test]
    fn test_vertex_edges_out() {
        let mut mesh = ComboMesh2::<usize, usize, usize>::default();
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
        let mut mesh = ComboMesh2::<usize, usize, usize>::default();
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
}
