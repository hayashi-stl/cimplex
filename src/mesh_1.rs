use edge::HasEdges;
use idmap::OrderedIdMap;
use typenum::{U2, U3};
use std::fmt::Debug;
use std::ops::{Index, IndexMut};
use fnv::FnvHashMap;
#[cfg(feature = "serde_")]
use serde::{Serialize, Deserialize};

use crate::{VecN, edge, vertex::HasVertices};
use crate::vertex::VertexId;
use crate::edge::EdgeId;

use internal::{HigherVertex, Edge};

/// A combinatorial simplicial 1-complex, containing only vertices and (oriented) edges.
/// Also known as an edge mesh.
/// Each vertex stores a value of type `V`.
/// Each edge stores its vertices and a value of type `E`.
/// The edge manipulation methods can either be called with an array of 2 `VertexId`s
/// or an `EdgeId`.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde_", derive(Serialize, Deserialize))]
pub struct ComboMesh1<V, E> {
    vertices: OrderedIdMap<VertexId, HigherVertex<V>>,
    edges: FnvHashMap<EdgeId, Edge<E>>,
    next_vertex_id: u64,
    /// Keep separate track because edge twins may or may not exist
    num_edges: usize,
}
crate::impl_has_vertices!(ComboMesh1<V, E>, HigherVertex);
crate::impl_has_edges!(ComboMesh1<V, E>, Edge);
crate::impl_index_vertex!(ComboMesh1<V, E>);
crate::impl_index_edge!(ComboMesh1<V, E>);

impl<V, E> HasVertices for ComboMesh1<V, E> {}
impl<V, E> HasEdges for ComboMesh1<V, E> {}

impl<V, E> Default for ComboMesh1<V, E> {
    fn default() -> Self {
        ComboMesh1 {
            vertices: OrderedIdMap::default(),
            edges: FnvHashMap::default(),
            next_vertex_id: 0,
            num_edges: 0,
        }
    }
}

impl<V, E> ComboMesh1<V, E> {
    /// Creates an empty vertex mesh.
    pub fn new() -> Self {
        Self::default()
    }
}

pub(crate) mod internal {
    use crate::ComboMesh1;
    use crate::vertex::internal::{RemoveVertexHigher, ClearVerticesHigher};
    use crate::edge::internal::{RemoveEdgeHigher, ClearEdgesHigher, Link};
    use crate::edge::{EdgeId, HasEdges};
    use crate::vertex::VertexId;
    
    //// A vertex of an edge mesh
    #[derive(Clone, Debug)]
    #[doc(hidden)]
    #[cfg_attr(feature = "serde_", derive(Serialize, Deserialize))]
    pub struct HigherVertex<V> {
        /// `target` is this vertex's id if there is no target
        target: VertexId,
        value: V,
    }
    crate::impl_vertex!(HigherVertex<V>, new |id, value| {
        HigherVertex {
            target: id,
            value,
        }
    });
    crate::impl_higher_vertex!(HigherVertex<V>);

    /// An edge of an edge mesh
    #[derive(Clone, Debug)]
    #[doc(hidden)]
    #[cfg_attr(feature = "serde_", derive(Serialize, Deserialize))]
    pub struct Edge<E> {
        /// Outgoing targets from the same vertex, whether the edge actually exists or not
        link: Link<VertexId>,
        /// The edge does not actually exist if the value is None;
        /// it is just there for the structural purpose of
        /// ensuring that every edge has a twin.
        value: Option<E>,
    }
    crate::impl_edge!(Edge<E>, new |_id, link, value| Edge { link, value });

    impl<V, E> RemoveVertexHigher for ComboMesh1<V, E> {
        fn remove_vertex_higher(&mut self, vertex: VertexId) {
            self.remove_edges(self.vertex_edges_out(vertex).chain(self.vertex_edges_in(vertex)).collect::<Vec<_>>());
        }
    }

    impl<V, E> ClearVerticesHigher for ComboMesh1<V, E> {
        fn clear_vertices_higher(&mut self) {
            self.edges.clear();
            self.num_edges = 0;
        }
    }

    impl<V, E> RemoveEdgeHigher for ComboMesh1<V, E> {
        fn remove_edge_higher(&mut self, _: EdgeId) {}
    }

    impl<V, E> ClearEdgesHigher for ComboMesh1<V, E> {
        fn clear_edges_higher(&mut self) {}
    }
}

/// A position-containing edge mesh
pub type Mesh1<V, E, D> = ComboMesh1<(VecN<D>, V), E>;

/// A 2D-position-containing edge mesh
pub type Mesh12<V, E> = Mesh1<V, E, U2>;

/// A 3D-position-containing edge mesh
pub type Mesh13<V, E> = Mesh1<V, E, U3>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt::Debug;
    use std::hash::Hash;
    use fnv::FnvHashSet;
    use std::convert::TryInto;

    #[track_caller]
    fn assert_vertices<V: Clone + Debug + Eq + Hash, E, I: IntoIterator<Item = (VertexId, V)>>(mesh: &ComboMesh1<V, E>, vertices: I) {
        let result = mesh.vertices().map(|(id, v)| (*id, v.clone()))
            .collect::<FnvHashSet<_>>();
        let expect = vertices.into_iter().collect::<FnvHashSet<_>>();

        assert_eq!(result, expect);
    }

    #[track_caller]
    fn assert_edges<V, E: Clone + Debug + Eq + Hash, EI: TryInto<EdgeId>, I: IntoIterator<Item = (EI, E)>>(mesh: &ComboMesh1<V, E>, edges: I) {
        let result = mesh.edges().map(|(id, e)| (*id, e.clone()))
            .collect::<FnvHashSet<_>>();
        let expect = edges.into_iter()
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
        assert_vertices(&mesh, vec![
            (ids[0], 3),
            (ids[1], 6),
            (ids[2], 9),
            (ids[3], 2),
            (ids2[0], 5),
            (ids2[1], 8),
        ]);
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
        assert_vertices(&mesh, vec![(ids[0], 3), (ids[1], 6), (ids[2], 9), (ids[3], 2)]);
        assert_edges(&mesh, vec![
            ([ids[0], ids[3]], 5),
            ([ids[1], ids[3]], 3),
            ([ids[3], ids[1]], 2),
            ([ids[2], ids[3]], 8),
            ([ids[1], ids[2]], 9),
        ]);

        mesh.remove_vertex(ids[1]); // vertex with edge
        assert_vertices(&mesh, vec![(ids[0], 3), (ids[2], 9), (ids[3], 2)]);
        assert_edges(&mesh, vec![
            ([ids[0], ids[3]], 5),
            ([ids[2], ids[3]], 8),
        ]);

        mesh.remove_vertex(ids[4]); // nonexistent vertex
        assert_vertices(&mesh, vec![(ids[0], 3), (ids[2], 9), (ids[3], 2)]);
        assert_edges(&mesh, vec![
            ([ids[0], ids[3]], 5),
            ([ids[2], ids[3]], 8),
        ]);
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
        assert_edges(&mesh, vec![
            ([ids[0], ids[3]], 5),
            ([ids[2], ids[3]], 8),
        ]);

        let id2 = mesh.add_vertex(6);
        assert_vertices(&mesh, vec![(ids[0], 3), (ids[2], 9), (ids[3], 2), (id2, 6)]);
        assert_edges(&mesh, vec![
            ([ids[0], ids[3]], 5),
            ([ids[2], ids[3]], 8),
        ]);
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
        assert_edges(&mesh, vec![
            ([ids[0], ids[3]], 5),
            ([ids[3], ids[1]], 2),
            ([ids[2], ids[3]], 8),
            ([ids[1], ids[2]], 9),
        ]);

        mesh.remove_edge([ids[1], ids[2]]); // last outgoing edge from vertex
        assert_edges(&mesh, vec![
            ([ids[0], ids[3]], 5),
            ([ids[3], ids[1]], 2),
            ([ids[2], ids[3]], 8),
        ]);

        mesh.remove_edge([ids[3], ids[0]]); // nonexistent edge
        assert_edges(&mesh, vec![
            ([ids[0], ids[3]], 5),
            ([ids[3], ids[1]], 2),
            ([ids[2], ids[3]], 8),
        ]);
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
        assert_edges(&mesh, vec![
            ([ids[0], ids[3]], 5),
            ([ids[3], ids[1]], 2),
            ([ids[2], ids[3]], 8),
            ([ids[1], ids[2]], 9),
        ]);

        mesh.add_edge([ids[1], ids[0]], 4);
        mesh.add_edge([ids[1], ids[3]], 6);
        assert_edges(&mesh, vec![
            ([ids[0], ids[3]], 5),
            ([ids[3], ids[1]], 2),
            ([ids[2], ids[3]], 8),
            ([ids[1], ids[2]], 9),
            ([ids[1], ids[0]], 4),
            ([ids[1], ids[3]], 6),
        ]);
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
        assert_vertices(&mesh, vec![(ids[0], 3), (ids[1], 6), (ids[2], 9), (ids[3], 2)]);
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
        let expected = vec![EdgeId([ids[2], ids[3]])].into_iter().collect::<FnvHashSet<_>>();
        assert_eq!(set, expected);

        let set = mesh.vertex_edges_out(ids[1]).collect::<FnvHashSet<_>>();
        let expected = vec![EdgeId([ids[1], ids[3]]), EdgeId([ids[1], ids[2]])].into_iter().collect::<FnvHashSet<_>>();
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
        let expected = vec![EdgeId([ids[1], ids[2]])].into_iter().collect::<FnvHashSet<_>>();
        assert_eq!(set, expected);

        let set = mesh.vertex_edges_in(ids[3]).collect::<FnvHashSet<_>>();
        let expected = vec![EdgeId([ids[0], ids[3]]), EdgeId([ids[1], ids[3]])].into_iter().collect::<FnvHashSet<_>>();
        assert_eq!(set, expected);
    }
}