use fnv::FnvHashMap;
use idmap::OrderedIdMap;
#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use typenum::{U2, U3, B0, B1};

use crate::{ComboMesh0, ComboMesh1, edge::{EdgeId, HasEdges}, mesh1::MwbComboMesh1};
use crate::mesh1::internal::HigherVertex;
use crate::tri::{HasTris, TriId};
use crate::vertex::{HasVertices, IdType, VertexId};
use crate::PtN;
use crate::private::Lock;

use internal::{HigherEdge, MwbTri, Tri};

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
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct ComboMesh2<V, E, F> {
    vertices: OrderedIdMap<VertexId, HigherVertex<V>>,
    edges: FnvHashMap<EdgeId, HigherEdge<E>>,
    tris: FnvHashMap<TriId, Tri<F>>,
    next_vertex_id: IdType,
}
crate::impl_index_vertex!(ComboMesh2<V, E, F>);
crate::impl_index_edge!(ComboMesh2<V, E, F>);
crate::impl_index_tri!(ComboMesh2<V, E, F>);

impl<V, E, F> HasVertices for ComboMesh2<V, E, F> {
    crate::impl_has_vertices!(HigherVertex<V>, Higher = B1);

    fn remove_vertex_higher<L: Lock>(&mut self, vertex: VertexId) {
        self.remove_edges(
            self.vertex_edges_out(vertex)
                .chain(self.vertex_edges_in(vertex))
                .collect::<Vec<_>>(),
        );
    }

    fn clear_vertices_higher<L: Lock>(&mut self) {
        self.tris.clear();
        self.edges.clear();
    }
}

impl<V, E, F> HasEdges for ComboMesh2<V, E, F> {
    crate::impl_has_edges!(HigherEdge<E>, Mwb = B0, Higher = B1);
    
    type WithoutEdges = ComboMesh0<V>;
    type WithMwbE = MwbComboMesh1<V, E>;
    type WithoutMwbE = ComboMesh1<V, E>;

    fn remove_edge_higher<L: Lock>(&mut self, edge: EdgeId) {
        self.remove_tris_keep_edges(self.edge_tris(edge).collect::<Vec<_>>());
    }

    fn clear_edges_higher<L: Lock>(&mut self) {
        self.tris.clear();
    }
}

impl<V, E, F> HasTris for ComboMesh2<V, E, F> {
    crate::impl_has_tris!(Tri<F>, Mwb = B0, Higher = B0);
    
    type WithoutTris = ComboMesh1<V, E>;
    type WithMwbF = MwbComboMesh2<V, E, F>;
    type WithoutMwbF = ComboMesh2<V, E, F>;

    fn remove_tri_higher<L: Lock>(&mut self, _: TriId) {}

    fn clear_tris_higher<L: Lock>(&mut self) {}
}

impl<V, E, F> Default for ComboMesh2<V, E, F> {
    fn default() -> Self {
        ComboMesh2 {
            vertices: OrderedIdMap::default(),
            edges: FnvHashMap::default(),
            tris: FnvHashMap::default(),
            next_vertex_id: 0,
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
pub type Mesh2<V, E, F, D> = ComboMesh2<(PtN<D>, V), E, F>;

/// A 2D-position-containing tri mesh
pub type Mesh22<V, E, F> = Mesh2<V, E, F, U2>;

/// A 3D-position-containing tri mesh
pub type Mesh23<V, E, F> = Mesh2<V, E, F, U3>;

/// A combinatorial simplicial 2-complex with the mwb property,
/// which forces every oriented edge to be part of at most 1 triangle.
/// Please don't call `add_edge` on this.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct MwbComboMesh2<V, E, F> {
    vertices: OrderedIdMap<VertexId, HigherVertex<V>>,
    edges: FnvHashMap<EdgeId, HigherEdge<E>>,
    tris: FnvHashMap<TriId, MwbTri<F>>,
    next_vertex_id: IdType,
}
crate::impl_index_vertex!(MwbComboMesh2<V, E, F>);
crate::impl_index_edge!(MwbComboMesh2<V, E, F>);
crate::impl_index_tri!(MwbComboMesh2<V, E, F>);

impl<V, E, F> HasVertices for MwbComboMesh2<V, E, F> {
    crate::impl_has_vertices!(HigherVertex<V>, Higher = B1);

    fn clear_vertices_higher<L: Lock>(&mut self) {
        self.tris.clear();
        self.edges.clear();
    }

    fn remove_vertex_higher<L: Lock>(&mut self, vertex: VertexId) {
        self.remove_edges(
            self.vertex_edges_out(vertex)
                .chain(self.vertex_edges_in(vertex))
                .collect::<Vec<_>>(),
        );
    }
}

impl<V, E, F> HasEdges for MwbComboMesh2<V, E, F> {
    crate::impl_has_edges!(HigherEdge<E>, Mwb = B0, Higher = B1);
    
    type WithoutEdges = ComboMesh0<V>;
    type WithMwbE = MwbComboMesh1<V, E>;
    type WithoutMwbE = ComboMesh1<V, E>;

    fn remove_edge_higher<L: Lock>(&mut self, edge: EdgeId) {
        self.edge_vertex_opp(edge).map(|opp| {
            self.remove_tri_keep_edges(TriId::from_valid([edge.0[0], edge.0[1], opp]));
            // Be careful not to remove `edge` as it will be removed after this function
            self.remove_edge(EdgeId([edge.0[1], opp]));
            self.remove_edge(EdgeId([opp, edge.0[0]]));
        });
    }

    fn clear_edges_higher<L: Lock>(&mut self) {
        self.tris.clear();
    }
}

impl<V, E, F> HasTris for MwbComboMesh2<V, E, F> {
    crate::impl_has_tris!(MwbTri<F>, Mwb = B1, Higher = B0);
    
    type WithoutTris = ComboMesh1<V, E>;
    type WithMwbF = MwbComboMesh2<V, E, F>;
    type WithoutMwbF = ComboMesh2<V, E, F>;

    fn remove_tri_higher<L: Lock>(&mut self, _: TriId) {}

    fn clear_tris_higher<L: Lock>(&mut self) {}
}

impl<V, E, F> Default for MwbComboMesh2<V, E, F> {
    fn default() -> Self {
        MwbComboMesh2 {
            vertices: OrderedIdMap::default(),
            edges: FnvHashMap::default(),
            tris: FnvHashMap::default(),
            next_vertex_id: 0,
        }
    }
}

impl<V, E, F> MwbComboMesh2<V, E, F> {
    /// Creates an empty tri mesh.
    pub fn new() -> Self {
        Self::default()
    }
}

pub(crate) mod internal {
    use crate::edge::Link;
    use crate::vertex::VertexId;
    #[cfg(feature = "serialize")]
    use serde::{Deserialize, Serialize};

    #[derive(Clone, Debug)]
    #[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
    pub struct HigherEdge<E> {
        /// Outgoing targets from the same vertex, whether the edge actually exists or not
        links: [Link<VertexId>; 2],
        /// Some vertex opposite this edge in a triangle.
        /// This is the edge's first vertex if the edge is not part of a triangle.
        tri_opp: VertexId,
        value: E,
    }
    #[rustfmt::skip]
    crate::impl_edge_higher!(
        HigherEdge<E>,
        new |id, links, value| {
            HigherEdge {
                tri_opp: id,
                links,
                value,
            }
        }
    );

    /// A triangle of an tri mesh
    #[derive(Clone, Debug)]
    #[doc(hidden)]
    #[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
    pub struct Tri<F> {
        /// Targets from the same edge for each of the edges,
        /// whether the triangle actually exists or not
        links: [Link<VertexId>; 3],
        value: F,
    }
    #[rustfmt::skip]
    crate::impl_tri!(Tri<F>, new |_id, links, value| Tri { links, value });

    /// A triangle of an tri mesh
    #[derive(Clone, Debug)]
    #[doc(hidden)]
    #[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
    pub struct MwbTri<F> {
        value: F,
    }
    #[rustfmt::skip]
    crate::impl_tri_mwb!(MwbTri<F>, new |_id, _links, value| MwbTri { value });
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

    #[track_caller]
    fn assert_vertices_m<
        V: Clone + Debug + Eq + Hash,
        E,
        F,
        I: IntoIterator<Item = (VertexId, V)>,
    >(
        mesh: &MwbComboMesh2<V, E, F>,
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
    fn assert_edges_m<
        V,
        E: Clone + Debug + Eq + Hash,
        EI: TryInto<EdgeId>,
        F,
        I: IntoIterator<Item = (EI, E)>,
    >(
        mesh: &MwbComboMesh2<V, E, F>,
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
    fn assert_tris_m<
        V,
        E,
        F: Clone + Debug + Eq + Hash,
        FI: TryInto<TriId>,
        I: IntoIterator<Item = (FI, F)>,
    >(
        mesh: &MwbComboMesh2<V, E, F>,
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
    fn test_add_tri() {
        let mut mesh = ComboMesh2::<usize, usize, usize>::default();
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
        let mut mesh = ComboMesh2::<usize, usize, usize>::default();
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
    fn test_remove_tri() {
        let mut mesh = ComboMesh2::<usize, usize, usize>::default();
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
    }

    #[test]
    fn test_clear_edges() {
        let mut mesh = ComboMesh2::<usize, usize, usize>::default();
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
    }

    #[test]
    fn test_clear_tris() {
        let mut mesh = ComboMesh2::<usize, usize, usize>::default();
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
    }

    #[test]
    fn test_tri_walker() {
        let mut mesh = ComboMesh2::<usize, usize, usize>::default();
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

        assert!(mesh.tri_walker_from_edge([ids[6], ids[7]]).is_none());

        let walker = mesh.tri_walker_from_edge_vertex([ids[0], ids[1]], ids[2]);
        assert_eq!(walker.edge(), EdgeId([ids[0], ids[1]]));
        assert_eq!(walker.first(), ids[0]);
        assert_eq!(walker.second(), ids[1]);
        assert_eq!(walker.third(), ids[2]);
        assert_eq!(
            walker.tri(),
            [ids[0], ids[1], ids[2]].try_into().ok().unwrap()
        );

        let walker = walker.next_edge();
        assert_eq!(walker.edge(), EdgeId([ids[1], ids[2]]));
        assert_eq!(walker.third(), ids[0]);

        let walker = walker.next_edge();
        assert_eq!(walker.edge(), EdgeId([ids[2], ids[0]]));
        assert_eq!(walker.third(), ids[1]);

        let walker = walker.prev_edge();
        assert_eq!(walker.edge(), EdgeId([ids[1], ids[2]]));
        assert_eq!(walker.third(), ids[0]);

        let walker = walker.next_opp();
        assert_eq!(walker.edge(), EdgeId([ids[1], ids[2]]));
        assert_eq!(walker.third(), ids[3]);

        let branch = walker.prev_opp(); // different name!
        assert_eq!(branch.edge(), EdgeId([ids[1], ids[2]]));
        assert_eq!(branch.third(), ids[0]);

        let walker = walker.on_twin_edge().unwrap();
        assert_eq!(walker.edge(), EdgeId([ids[2], ids[1]]));
        assert_eq!(walker.third(), ids[4]);

        assert!(walker.twin().is_none());

        let walker = walker.prev_edge();
        assert_eq!(walker.edge(), EdgeId([ids[4], ids[2]]));
        assert_eq!(walker.third(), ids[1]);

        assert!(walker.on_twin_edge().is_none());

        let walker = mesh.tri_walker_from_edge_vertex([ids[4], ids[6]], ids[5]);
        let walker = walker.twin().unwrap();
        assert_eq!(walker.edge(), EdgeId([ids[6], ids[4]]));
        assert_eq!(walker.third(), ids[5]);
    }

    #[test]
    fn test_edge_tris() {
        let mut mesh = ComboMesh2::<usize, usize, usize>::default();
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
        let mut mesh = ComboMesh2::<usize, usize, usize>::default();
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
    fn test_default_m() {
        let mesh = MwbComboMesh2::<(), (), ()>::default();
        assert!(mesh.vertices.is_empty());
        assert!(mesh.edges.is_empty());
        assert!(mesh.tris.is_empty());
        assert_eq!(mesh.num_edges(), 0);
        assert_eq!(mesh.num_tris(), 0);
    }

    #[test]
    fn test_add_tri_m() {
        let mut mesh = MwbComboMesh2::<usize, usize, usize>::default();
        let ids = mesh.extend_vertices(vec![3, 6, 9, 2]);
        assert_eq!(mesh.add_tri([ids[1], ids[0], ids[2]], 5, || 0), None);

        assert_edges_m(
            &mesh,
            vec![
                ([ids[1], ids[0]], 0),
                ([ids[0], ids[2]], 0),
                ([ids[2], ids[1]], 0),
            ],
        );
        assert_tris_m(&mesh, vec![([ids[0], ids[2], ids[1]], 5)]);

        // Add twin
        assert_eq!(mesh.add_tri([ids[1], ids[2], ids[0]], 6, || 0), None);
        assert_edges_m(
            &mesh,
            vec![
                ([ids[1], ids[0]], 0),
                ([ids[0], ids[2]], 0),
                ([ids[2], ids[1]], 0),
                ([ids[1], ids[2]], 0),
                ([ids[0], ids[1]], 0),
                ([ids[2], ids[0]], 0),
            ],
        );
        assert_tris_m(
            &mesh,
            vec![([ids[0], ids[2], ids[1]], 5), ([ids[0], ids[1], ids[2]], 6)],
        );

        // Modify tri
        assert_eq!(mesh.add_tri([ids[1], ids[2], ids[0]], 7, || 0), Some(6));
        assert_edges_m(
            &mesh,
            vec![
                ([ids[1], ids[0]], 0),
                ([ids[0], ids[2]], 0),
                ([ids[2], ids[1]], 0),
                ([ids[1], ids[2]], 0),
                ([ids[0], ids[1]], 0),
                ([ids[2], ids[0]], 0),
            ],
        );
        assert_tris_m(
            &mesh,
            vec![([ids[0], ids[2], ids[1]], 5), ([ids[0], ids[1], ids[2]], 7)],
        );
    }

    #[test]
    fn test_extend_tris_m() {
        let mut mesh = MwbComboMesh2::<usize, usize, usize>::default();
        let ids = mesh.extend_vertices(vec![3, 6, 9, 2, 5, 8, 1, 4]);

        let tris = vec![
            ([ids[0], ids[1], ids[2]], 1), // killed by 3-1-2
            ([ids[3], ids[1], ids[2]], 2),
            ([ids[4], ids[2], ids[1]], 3),
            ([ids[4], ids[1], ids[5]], 4),
            ([ids[5], ids[6], ids[4]], 5),
        ];
        mesh.extend_tris(tris.clone(), || 0);

        assert_edges_m(
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
            ],
        );
        assert_tris_m(
            &mesh,
            vec![
                ([ids[3], ids[1], ids[2]], 2),
                ([ids[4], ids[2], ids[1]], 3),
                ([ids[4], ids[1], ids[5]], 4),
                ([ids[5], ids[6], ids[4]], 5),
            ],
        );
    }

    #[test]
    fn test_remove_tri_m() {
        let mut mesh = MwbComboMesh2::<usize, usize, usize>::default();
        let ids = mesh.extend_vertices(vec![3, 6, 9, 2, 5, 8, 1, 4]);
        let tris = vec![
            ([ids[3], ids[1], ids[2]], 2),
            ([ids[4], ids[2], ids[1]], 3),
            ([ids[4], ids[1], ids[5]], 4),
            ([ids[5], ids[6], ids[4]], 5),
        ];
        mesh.extend_tris(tris, || 0);

        assert_eq!(mesh.remove_tri([ids[3], ids[1], ids[2]]), Some(2)); // last tri with edge
        assert_edges_m(
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
            ],
        );
        assert_tris_m(
            &mesh,
            vec![
                ([ids[4], ids[2], ids[1]], 3),
                ([ids[4], ids[1], ids[5]], 4),
                ([ids[5], ids[6], ids[4]], 5),
            ],
        );

        assert_eq!(mesh.remove_tri([ids[1], ids[2], ids[4]]), None); // nonexistent tri
        assert_edges_m(
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
            ],
        );
        assert_tris_m(
            &mesh,
            vec![
                ([ids[4], ids[2], ids[1]], 3),
                ([ids[4], ids[1], ids[5]], 4),
                ([ids[5], ids[6], ids[4]], 5),
            ],
        );
    }

    #[test]
    fn test_clear_vertices_m() {
        let mut mesh = MwbComboMesh2::<usize, usize, usize>::default();
        let ids = mesh.extend_vertices(vec![3, 6, 9, 2, 5, 8, 1, 4]);
        let tris = vec![
            ([ids[3], ids[1], ids[2]], 2),
            ([ids[4], ids[2], ids[1]], 3),
            ([ids[4], ids[1], ids[5]], 4),
            ([ids[5], ids[6], ids[4]], 5),
        ];
        mesh.extend_tris(tris, || 0);

        mesh.clear_vertices();
        assert_vertices_m(&mesh, vec![]);
        assert_edges_m(&mesh, vec![] as Vec<(EdgeId, _)>);
        assert_tris_m(&mesh, vec![] as Vec<(TriId, _)>);
    }

    #[test]
    fn test_clear_edges_m() {
        let mut mesh = MwbComboMesh2::<usize, usize, usize>::default();
        let ids = mesh.extend_vertices(vec![3, 6, 9, 2, 5, 8, 1, 4]);
        let tris = vec![
            ([ids[3], ids[1], ids[2]], 2),
            ([ids[4], ids[2], ids[1]], 3),
            ([ids[4], ids[1], ids[5]], 4),
            ([ids[5], ids[6], ids[4]], 5),
        ];
        mesh.extend_tris(tris, || 0);

        mesh.clear_edges();
        assert_vertices_m(
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
        assert_edges_m(&mesh, vec![] as Vec<(EdgeId, _)>);
        assert_tris_m(&mesh, vec![] as Vec<(TriId, _)>);
    }

    #[test]
    fn test_clear_tris_m() {
        let mut mesh = MwbComboMesh2::<usize, usize, usize>::default();
        let ids = mesh.extend_vertices(vec![3, 6, 9, 2, 5, 8, 1, 4]);
        let tris = vec![
            ([ids[3], ids[1], ids[2]], 2),
            ([ids[4], ids[2], ids[1]], 3),
            ([ids[4], ids[1], ids[5]], 4),
            ([ids[5], ids[6], ids[4]], 5),
        ];
        mesh.extend_tris(tris, || 0);

        mesh.clear_tris();
        assert_vertices_m(
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
        assert_edges_m(
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
            ],
        );
        assert_tris_m(&mesh, vec![] as Vec<(TriId, _)>);
    }

    #[test]
    fn test_tri_walker_m() {
        let mut mesh = MwbComboMesh2::<usize, usize, usize>::default();
        let ids = mesh.extend_vertices(vec![3, 6, 9, 2, 5, 8, 1, 4]);
        let tris = vec![
            ([ids[0], ids[1], ids[2]], 2),
            ([ids[4], ids[2], ids[1]], 3),
            ([ids[4], ids[1], ids[5]], 4),
            ([ids[5], ids[6], ids[4]], 5),
        ];
        mesh.extend_tris(tris, || 0);

        let walker = mesh.tri_walker_from_edge_vertex([ids[0], ids[1]], ids[2]);
        assert_eq!(walker.edge(), EdgeId([ids[0], ids[1]]));
        assert_eq!(walker.first(), ids[0]);
        assert_eq!(walker.second(), ids[1]);
        assert_eq!(walker.third(), ids[2]);
        assert_eq!(
            walker.tri(),
            [ids[0], ids[1], ids[2]].try_into().ok().unwrap()
        );

        let walker = walker.next_edge();
        assert_eq!(walker.edge(), EdgeId([ids[1], ids[2]]));
        assert_eq!(walker.third(), ids[0]);

        let walker = walker.next_edge();
        assert_eq!(walker.edge(), EdgeId([ids[2], ids[0]]));
        assert_eq!(walker.third(), ids[1]);

        let walker = walker.prev_edge();
        assert_eq!(walker.edge(), EdgeId([ids[1], ids[2]]));
        assert_eq!(walker.third(), ids[0]);

        let walker = walker.next_opp();
        assert_eq!(walker.edge(), EdgeId([ids[1], ids[2]]));
        assert_eq!(walker.third(), ids[0]);

        let walker = walker.prev_opp();
        assert_eq!(walker.edge(), EdgeId([ids[1], ids[2]]));
        assert_eq!(walker.third(), ids[0]);

        let walker = walker.on_twin_edge().unwrap();
        assert_eq!(walker.edge(), EdgeId([ids[2], ids[1]]));
        assert_eq!(walker.third(), ids[4]);

        assert!(walker.twin().is_none());

        let walker = walker.prev_edge();
        assert_eq!(walker.edge(), EdgeId([ids[4], ids[2]]));
        assert_eq!(walker.third(), ids[1]);

        assert!(walker.on_twin_edge().is_none());
    }

    #[test]
    fn test_edge_tris_m() {
        let mut mesh = MwbComboMesh2::<usize, usize, usize>::default();
        let ids = mesh.extend_vertices(vec![3, 6, 9, 2, 5, 8, 1, 4]);
        let tris = vec![
            ([ids[3], ids[1], ids[2]], 2),
            ([ids[4], ids[2], ids[1]], 3),
            ([ids[4], ids[1], ids[5]], 4),
            ([ids[5], ids[6], ids[4]], 5),
        ];
        mesh.extend_tris(tris, || 0);

        let set = mesh.edge_tris([ids[3], ids[1]]).collect::<FnvHashSet<_>>();
        let expected = vec![TriId([ids[1], ids[2], ids[3]])]
            .into_iter()
            .collect::<FnvHashSet<_>>();
        assert_eq!(set, expected);
    }

    #[test]
    fn test_vertex_tris_m() {
        let mut mesh = MwbComboMesh2::<usize, usize, usize>::default();
        let ids = mesh.extend_vertices(vec![3, 6, 9, 2, 5, 8, 1, 4]);
        let tris = vec![
            ([ids[3], ids[1], ids[2]], 2),
            ([ids[4], ids[2], ids[1]], 3),
            ([ids[4], ids[1], ids[5]], 4),
            ([ids[5], ids[6], ids[4]], 5),
        ];
        mesh.extend_tris(tris, || 0);

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
            TriId([ids[1], ids[2], ids[3]]),
            TriId([ids[1], ids[4], ids[2]]),
            TriId([ids[1], ids[5], ids[4]]),
        ]
        .into_iter()
        .collect::<FnvHashSet<_>>();
        assert_eq!(set, expected);
    }
}
