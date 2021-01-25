use edge::HasEdges;
use fnv::FnvHashMap;
use idmap::OrderedIdMap;
#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use typenum::{U2, U3};

use crate::vertex::VertexId;
use crate::{edge, vertex::HasVertices, PtN};
use crate::{edge::EdgeId, vertex::IdType};
use crate::private::Lock;

use internal::{Edge, HigherVertex, MwbEdge};

/// A combinatorial simplicial 1-complex, containing only vertices and (oriented) edges.
/// Also known as an edge mesh.
/// Each vertex stores a value of type `V`.
/// Each edge stores its vertices and a value of type `E`.
/// The edge manipulation methods can either be called with an array of 2 `VertexId`s
/// or an `EdgeId`.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct ComboMesh1<V, E> {
    vertices: OrderedIdMap<VertexId, HigherVertex<V>>,
    edges: FnvHashMap<EdgeId, Edge<E>>,
    next_vertex_id: IdType,
}
crate::impl_index_vertex!(ComboMesh1<V, E>);
crate::impl_index_edge!(ComboMesh1<V, E>);

impl<V, E> HasVertices for ComboMesh1<V, E> {
    crate::impl_has_vertices!(HigherVertex<V>);

    fn remove_vertex_higher<L: Lock>(&mut self, vertex: VertexId) {
        self.remove_edges(
            self.vertex_edges_out(vertex)
                .chain(self.vertex_edges_in(vertex))
                .collect::<Vec<_>>(),
        );
    }

    fn clear_vertices_higher<L: Lock>(&mut self) {
        self.edges.clear();
    }
}

impl<V, E> HasEdges for ComboMesh1<V, E> {
    crate::impl_has_edges!(Edge<E>);

    fn remove_edge_higher<L: Lock>(&mut self, _: EdgeId) {}

    fn clear_edges_higher<L: Lock>(&mut self) {}
}

impl<V, E> Default for ComboMesh1<V, E> {
    fn default() -> Self {
        ComboMesh1 {
            vertices: OrderedIdMap::default(),
            edges: FnvHashMap::default(),
            next_vertex_id: 0,
        }
    }
}

impl<V, E> ComboMesh1<V, E> {
    /// Creates an empty vertex mesh.
    pub fn new() -> Self {
        Self::default()
    }
}

/// A position-containing edge mesh
pub type Mesh1<V, E, D> = ComboMesh1<(PtN<D>, V), E>;

/// A 2D-position-containing edge mesh
pub type Mesh12<V, E> = Mesh1<V, E, U2>;

/// A 3D-position-containing edge mesh
pub type Mesh13<V, E> = Mesh1<V, E, U3>;

/// A combinatorial simplicial 1-complex with the mwb property,
/// which forces every vertex to be a source of at most 1 edge and a target of at most 1 edge.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct MwbComboMesh1<V, E> {
    vertices: OrderedIdMap<VertexId, HigherVertex<V>>,
    edges: FnvHashMap<EdgeId, MwbEdge<E>>,
    next_vertex_id: IdType,
}
crate::impl_index_vertex!(MwbComboMesh1<V, E>);
crate::impl_index_edge!(MwbComboMesh1<V, E>);

impl<V, E> HasVertices for MwbComboMesh1<V, E> {
    crate::impl_has_vertices!(HigherVertex<V>);

    fn remove_vertex_higher<L: Lock>(&mut self, vertex: VertexId) {
        self.vertex_edge_out(vertex)
            .map(|edge| self.remove_edge(edge));
        self.vertex_edge_in(vertex)
            .map(|edge| self.remove_edge(edge));
    }

    fn clear_vertices_higher<L: Lock>(&mut self) {
        self.edges.clear();
    }
}

impl<V, E> HasEdges for MwbComboMesh1<V, E> {
    crate::impl_has_edges!(MwbEdge<E>);

    fn remove_edge_higher<L: Lock>(&mut self, _: EdgeId) {}

    fn clear_edges_higher<L: Lock>(&mut self) {}
}

impl<V, E> Default for MwbComboMesh1<V, E> {
    fn default() -> Self {
        MwbComboMesh1 {
            vertices: OrderedIdMap::default(),
            edges: FnvHashMap::default(),
            next_vertex_id: 0,
        }
    }
}

impl<V, E> MwbComboMesh1<V, E> {
    /// Creates an empty vertex mesh.
    pub fn new() -> Self {
        Self::default()
    }
}

pub(crate) mod internal {
    use crate::edge::Link;
    use crate::vertex::VertexId;
    #[cfg(feature = "serialize")]
    use serde::{Deserialize, Serialize};

    //// A vertex of an edge mesh
    #[derive(Clone, Debug)]
    #[doc(hidden)]
    #[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
    pub struct HigherVertex<V> {
        /// `source` is this vertex's id if there is no source
        source: VertexId,
        /// `target` is this vertex's id if there is no target
        target: VertexId,
        value: V,
    }
    #[rustfmt::skip]
    crate::impl_vertex!(HigherVertex<V>, new |id, value| HigherVertex { source: id, target: id, value });
    crate::impl_higher_vertex!(HigherVertex<V>);

    /// An edge of an edge mesh
    #[derive(Clone, Debug)]
    #[doc(hidden)]
    #[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
    pub struct Edge<E> {
        /// Outgoing targets from the same vertex, whether the edge actually exists or not
        links: [Link<VertexId>; 2],
        value: E,
    }
    #[rustfmt::skip]
    crate::impl_edge!(Edge<E>, new |_id, links, value| Edge { links, value });

    /// An edge of a mwb edge mesh
    #[derive(Clone, Debug)]
    #[doc(hidden)]
    #[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
    pub struct MwbEdge<E> {
        value: E,
    }
    #[rustfmt::skip]
    crate::impl_edge_mwb!(MwbEdge<E>, new |_id, _links, value| MwbEdge { value });
}

#[cfg(test)]
mod tests {
    use super::*;
    use fnv::FnvHashSet;
    use std::convert::TryInto;
    use std::fmt::Debug;
    use std::hash::Hash;

    #[track_caller]
    fn assert_vertices<V: Clone + Debug + Eq + Hash, E, I: IntoIterator<Item = (VertexId, V)>>(
        mesh: &ComboMesh1<V, E>,
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
        I: IntoIterator<Item = (EI, E)>,
    >(
        mesh: &ComboMesh1<V, E>,
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
    fn assert_vertices_m<V: Clone + Debug + Eq + Hash, E, I: IntoIterator<Item = (VertexId, V)>>(
        mesh: &MwbComboMesh1<V, E>,
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
        I: IntoIterator<Item = (EI, E)>,
    >(
        mesh: &MwbComboMesh1<V, E>,
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

    #[test]
    fn test_default() {
        let mesh = ComboMesh1::<(), ()>::default();
        assert!(mesh.vertices.is_empty());
        assert!(mesh.edges.is_empty());
        assert_eq!(mesh.num_edges(), 0);
    }

    #[test]
    fn test_add_vertex() {
        let mut mesh = ComboMesh1::<usize, usize>::default();
        let id = mesh.add_vertex(3);
        assert_eq!(mesh.vertex(id), Some(&3));

        let id2 = mesh.add_vertex(9);
        assert_eq!(mesh.vertex(id), Some(&3));
        assert_eq!(mesh.vertex(id2), Some(&9));
    }

    #[test]
    fn test_extend_vertices() {
        let mut mesh = ComboMesh1::<usize, usize>::default();
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
        let mut mesh = ComboMesh1::<usize, usize>::default();
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
        let mut mesh = ComboMesh1::<usize, usize>::default();
        let ids = mesh.extend_vertices(vec![3, 6, 9, 2]);
        mesh.add_edge([ids[1], ids[1]], 4);
    }

    #[test]
    fn test_extend_edges() {
        let mut mesh = ComboMesh1::<usize, usize>::default();
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
    fn test_remove_vertex() {
        let mut mesh = ComboMesh1::<usize, usize>::default();
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
        let mut mesh = ComboMesh1::<usize, usize>::default();
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
        let mut mesh = ComboMesh1::<usize, usize>::default();
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
        let mut mesh = ComboMesh1::<usize, usize>::default();
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
        let mut mesh = ComboMesh1::<usize, usize>::default();
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
        let mut mesh = ComboMesh1::<usize, usize>::default();
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
        let mut mesh = ComboMesh1::<usize, usize>::default();
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
        assert_eq!(walker.first(), ids[3]);
        assert_eq!(walker.second(), ids[1]);

        let walker = walker.next();
        assert_eq!(walker.edge(), EdgeId([ids[3], ids[1]]));

        let walker = walker.prev();
        assert_eq!(walker.edge(), EdgeId([ids[3], ids[1]]));

        let walker = walker.twin().unwrap();
        assert_eq!(walker.edge(), EdgeId([ids[1], ids[3]]));

        let walker = walker.twin().unwrap();
        assert_eq!(walker.edge(), EdgeId([ids[3], ids[1]]));

        let walker = mesh.edge_walker_from_edge([ids[1], ids[2]]);
        assert_eq!(walker.edge(), EdgeId([ids[1], ids[2]]));

        let walker = walker.next();
        assert_eq!(walker.edge(), EdgeId([ids[1], ids[3]]));

        let walker = walker.prev();
        assert_eq!(walker.edge(), EdgeId([ids[1], ids[2]]));

        let walker = walker.target_out().unwrap();
        assert_eq!(walker.edge(), EdgeId([ids[2], ids[3]]));

        let walker = walker.source_in().unwrap();
        assert_eq!(walker.edge(), EdgeId([ids[1], ids[2]]));

        assert!(walker.twin().is_none());

        let walker = mesh.edge_walker_from_edge([ids[0], ids[3]]);
        assert!(walker.source_in().is_none());

        let walker = walker.next_in();
        assert_ne!(walker.first(), ids[0]);
        assert_eq!(walker.second(), ids[3]);

        let walker = walker.prev_in();
        assert_eq!(walker.edge(), EdgeId([ids[0], ids[3]]));
    }

    #[test]
    fn test_vertex_edges_out() {
        let mut mesh = ComboMesh1::<usize, usize>::default();
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
        let mut mesh = ComboMesh1::<usize, usize>::default();
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
    fn test_default_m() {
        let mesh = MwbComboMesh1::<(), ()>::default();
        assert!(mesh.vertices.is_empty());
        assert!(mesh.edges.is_empty());
        assert_eq!(mesh.num_edges(), 0);
    }

    #[test]
    fn test_add_vertex_m() {
        let mut mesh = MwbComboMesh1::<usize, usize>::default();
        let id = mesh.add_vertex(3);
        assert_eq!(mesh.vertex(id), Some(&3));

        let id2 = mesh.add_vertex(9);
        assert_eq!(mesh.vertex(id), Some(&3));
        assert_eq!(mesh.vertex(id2), Some(&9));
    }

    #[test]
    fn test_extend_vertices_m() {
        let mut mesh = MwbComboMesh1::<usize, usize>::default();
        let ids = mesh.extend_vertices(vec![3, 6, 9, 2]);
        assert_eq!(mesh.vertex(ids[0]), Some(&3));
        assert_eq!(mesh.vertex(ids[1]), Some(&6));
        assert_eq!(mesh.vertex(ids[2]), Some(&9));
        assert_eq!(mesh.vertex(ids[3]), Some(&2));

        let ids2 = mesh.extend_vertices(vec![5, 8]);
        assert_vertices_m(
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
    fn test_add_edge_m() {
        let mut mesh = MwbComboMesh1::<usize, usize>::default();
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
    fn test_add_edge_bad_m() {
        let mut mesh = MwbComboMesh1::<usize, usize>::default();
        let ids = mesh.extend_vertices(vec![3, 6, 9, 2]);
        mesh.add_edge([ids[1], ids[1]], 4);
    }

    #[test]
    fn test_extend_edges_m() {
        let mut mesh = MwbComboMesh1::<usize, usize>::default();
        let ids = mesh.extend_vertices(vec![3, 6, 9, 2]);
        let edges = vec![
            ([ids[0], ids[3]], 5), // killed by 1-3
            ([ids[1], ids[3]], 3), // killed by 2-3
            ([ids[3], ids[1]], 2),
            ([ids[2], ids[3]], 8),
            ([ids[1], ids[2]], 9),
        ];
        mesh.extend_edges(edges.clone());

        assert_edges_m(
            &mesh,
            vec![
                ([ids[3], ids[1]], 2),
                ([ids[2], ids[3]], 8),
                ([ids[1], ids[2]], 9),
            ],
        );
    }

    #[test]
    fn test_remove_vertex_m() {
        let mut mesh = MwbComboMesh1::<usize, usize>::default();
        let ids = mesh.extend_vertices(vec![3, 6, 9, 2, 5]);
        let edges = vec![
            ([ids[3], ids[1]], 2),
            ([ids[2], ids[3]], 8),
            ([ids[1], ids[2]], 9),
        ];
        mesh.extend_edges(edges.clone());

        mesh.remove_vertex(ids[4]); // edgeless vertex
        assert_vertices_m(
            &mesh,
            vec![(ids[0], 3), (ids[1], 6), (ids[2], 9), (ids[3], 2)],
        );
        assert_edges_m(
            &mesh,
            vec![
                ([ids[3], ids[1]], 2),
                ([ids[2], ids[3]], 8),
                ([ids[1], ids[2]], 9),
            ],
        );

        mesh.remove_vertex(ids[1]); // vertex with edge
        assert_vertices_m(&mesh, vec![(ids[0], 3), (ids[2], 9), (ids[3], 2)]);
        assert_edges_m(&mesh, vec![([ids[2], ids[3]], 8)]);

        mesh.remove_vertex(ids[4]); // nonexistent vertex
        assert_vertices_m(&mesh, vec![(ids[0], 3), (ids[2], 9), (ids[3], 2)]);
        assert_edges_m(&mesh, vec![([ids[2], ids[3]], 8)]);
    }

    #[test]
    fn test_remove_add_vertex_m() {
        let mut mesh = MwbComboMesh1::<usize, usize>::default();
        let ids = mesh.extend_vertices(vec![3, 6, 9, 2]);
        let edges = vec![
            ([ids[3], ids[1]], 2),
            ([ids[2], ids[3]], 8),
            ([ids[1], ids[2]], 9),
        ];
        mesh.extend_edges(edges.clone());

        mesh.remove_vertex(ids[1]);
        assert_vertices_m(&mesh, vec![(ids[0], 3), (ids[2], 9), (ids[3], 2)]);
        assert_edges_m(&mesh, vec![([ids[2], ids[3]], 8)]);

        let id2 = mesh.add_vertex(6);
        assert_vertices_m(&mesh, vec![(ids[0], 3), (ids[2], 9), (ids[3], 2), (id2, 6)]);
        assert_edges_m(&mesh, vec![([ids[2], ids[3]], 8)]);
    }

    #[test]
    fn test_remove_edge_m() {
        let mut mesh = MwbComboMesh1::<usize, usize>::default();
        let ids = mesh.extend_vertices(vec![3, 6, 9, 2]);
        let edges = vec![
            ([ids[3], ids[1]], 2),
            ([ids[2], ids[3]], 8),
            ([ids[1], ids[2]], 9),
        ];
        mesh.extend_edges(edges.clone());

        mesh.remove_edge([ids[1], ids[2]]); // last outgoing edge from vertex
        assert_edges_m(&mesh, vec![([ids[3], ids[1]], 2), ([ids[2], ids[3]], 8)]);

        mesh.remove_edge([ids[3], ids[2]]); // nonexistent edge
        assert_edges_m(&mesh, vec![([ids[3], ids[1]], 2), ([ids[2], ids[3]], 8)]);
    }

    #[test]
    fn test_remove_add_edge_m() {
        let mut mesh = MwbComboMesh1::<usize, usize>::default();
        let ids = mesh.extend_vertices(vec![3, 6, 9, 2]);
        let edges = vec![
            ([ids[3], ids[1]], 2),
            ([ids[2], ids[3]], 8),
            ([ids[1], ids[2]], 9),
        ];
        mesh.extend_edges(edges.clone());

        mesh.remove_edge([ids[1], ids[2]]);
        assert_edges_m(&mesh, vec![([ids[3], ids[1]], 2), ([ids[2], ids[3]], 8)]);

        mesh.add_edge([ids[1], ids[0]], 4); // killed by 1-3
        mesh.add_edge([ids[1], ids[3]], 6);
        assert_edges_m(&mesh, vec![([ids[3], ids[1]], 2), ([ids[1], ids[3]], 6)]);
    }

    #[test]
    fn test_clear_vertices_m() {
        let mut mesh = MwbComboMesh1::<usize, usize>::default();
        let ids = mesh.extend_vertices(vec![3, 6, 9, 2]);
        let edges = vec![
            ([ids[3], ids[1]], 2),
            ([ids[2], ids[3]], 8),
            ([ids[1], ids[2]], 9),
        ];
        mesh.extend_edges(edges.clone());

        mesh.clear_vertices();
        assert_vertices_m(&mesh, vec![]);
        assert_edges_m(&mesh, vec![] as Vec<(EdgeId, _)>);
    }

    #[test]
    fn test_clear_edges_m() {
        let mut mesh = MwbComboMesh1::<usize, usize>::default();
        let ids = mesh.extend_vertices(vec![3, 6, 9, 2]);
        let edges = vec![
            ([ids[3], ids[1]], 2),
            ([ids[2], ids[3]], 8),
            ([ids[1], ids[2]], 9),
        ];
        mesh.extend_edges(edges.clone());

        mesh.clear_edges();
        assert_vertices_m(
            &mesh,
            vec![(ids[0], 3), (ids[1], 6), (ids[2], 9), (ids[3], 2)],
        );
        assert_edges_m(&mesh, vec![] as Vec<(EdgeId, _)>);
    }

    #[test]
    fn test_walker_m() {
        let mut mesh = MwbComboMesh1::<usize, usize>::default();
        let ids = mesh.extend_vertices(vec![3, 6, 9, 2, 5]);
        let edges = vec![
            ([ids[3], ids[1]], 2),
            ([ids[1], ids[2]], 9),
            ([ids[2], ids[3]], 8),
        ];
        mesh.extend_edges(edges.clone());

        assert!(mesh.edge_walker_from_vertex(ids[4]).is_none());

        let walker = mesh.edge_walker_from_edge([ids[3], ids[1]]);
        assert_eq!(walker.edge(), EdgeId([ids[3], ids[1]]));
        assert_eq!(walker.first(), ids[3]);
        assert_eq!(walker.second(), ids[1]);

        let walker = walker.next();
        assert_eq!(walker.edge(), EdgeId([ids[3], ids[1]]));

        let walker = walker.prev();
        assert_eq!(walker.edge(), EdgeId([ids[3], ids[1]]));

        let walker = walker.next_in();
        assert_eq!(walker.edge(), EdgeId([ids[3], ids[1]]));

        let walker = walker.prev_in();
        assert_eq!(walker.edge(), EdgeId([ids[3], ids[1]]));

        assert!(walker.twin().is_none());

        let walker = walker.target_out().unwrap();
        assert_eq!(walker.edge(), EdgeId([ids[1], ids[2]]));

        let walker = walker.target_out().unwrap();
        assert_eq!(walker.edge(), EdgeId([ids[2], ids[3]]));

        let walker = walker.source_in().unwrap();
        assert_eq!(walker.edge(), EdgeId([ids[1], ids[2]]));
    }

    #[test]
    fn test_vertex_edges_out_m() {
        let mut mesh = MwbComboMesh1::<usize, usize>::default();
        let ids = mesh.extend_vertices(vec![3, 6, 9, 2, 5]);
        let edges = vec![
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
    }

    #[test]
    fn test_vertex_edges_in_m() {
        let mut mesh = MwbComboMesh1::<usize, usize>::default();
        let ids = mesh.extend_vertices(vec![3, 6, 9, 2, 5]);
        let edges = vec![([ids[3], ids[1]], 2), ([ids[1], ids[2]], 9)];
        mesh.extend_edges(edges.clone());

        let set = mesh.vertex_edges_in(ids[4]).collect::<FnvHashSet<_>>();
        let expected = vec![].into_iter().collect::<FnvHashSet<_>>();
        assert_eq!(set, expected);

        let set = mesh.vertex_edges_in(ids[2]).collect::<FnvHashSet<_>>();
        let expected = vec![EdgeId([ids[1], ids[2]])]
            .into_iter()
            .collect::<FnvHashSet<_>>();
        assert_eq!(set, expected);
    }
}
