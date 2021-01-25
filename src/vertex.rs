//! Traits and structs related to vertices

use idmap::table::DenseEntryTable;
use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, DimName, Point};
#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};
use std::iter::Map;
use alga::general::{MeetSemilattice, JoinSemilattice};
use nalgebra::dimension::U3;
use typenum::B1;

use internal::{ClearVerticesHigher, HasVertices as HasVerticesIntr, RemoveVertexHigher, Vertex};

use crate::{edge::{HasEdges, internal::{Edge, HigherEdge}}, tet::HasTets, tri::{HasTris, internal::{HigherTri, Tri}}};
use crate::edge::internal::HasEdges as HasEdgesIntr;
use crate::tri::internal::HasTris as HasTrisIntr;
use crate::tet::Tet;
use crate::private::Key;

use self::internal::HigherVertex;

pub(crate) type PositionDim<P> = <P as Position>::Dim;
pub(crate) type PositionPoint<P> = Point<f64, PositionDim<P>>;
pub(crate) type HasPositionDim<P> =
    <<<P as HasVerticesIntr>::Vertex as Vertex>::V as Position>::Dim;
pub(crate) type HasPositionPoint<P> = Point<f64, HasPositionDim<P>>;
pub(crate) type HasPositionRest<P> = <<<P as HasVerticesIntr>::Vertex as Vertex>::V as Position>::Rest;

/// For values that can represent a position.
pub trait Position
where
    DefaultAllocator: Allocator<f64, <Self as Position>::Dim>,
{
    /// The number of dimensions in the position
    type Dim: DimName;
    /// The rest of the type
    type Rest;

    /// The actual position represented
    fn position(&self) -> PositionPoint<Self>;

    fn with_position(point: PositionPoint<Self>, rest: Self::Rest) -> Self;
}

/// Not a blanket implementation because
/// I also want to implement this for tuples
impl<D: DimName> Position for Point<f64, D>
where
    DefaultAllocator: Allocator<f64, D>,
{
    type Dim = D;
    type Rest = ();

    fn position(&self) -> PositionPoint<Self> {
        self.clone()
    }

    fn with_position(point: PositionPoint<Self>, rest: Self::Rest) -> Self {
        point
    }
}

impl<D: DimName, V> Position for (Point<f64, D>, V)
where
    DefaultAllocator: Allocator<f64, D>,
{
    type Dim = D;
    type Rest = V;

    fn position(&self) -> PositionPoint<Self> {
        self.0.clone()
    }

    fn with_position(point: PositionPoint<Self>, rest: Self::Rest) -> Self {
        (point, rest)
    }
}

/// The integer type used for a vertex
pub type IdType = u32;

/// An index to a vertex of a mesh.
/// Will not be invalidated unless the vertex gets removed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct VertexId(IdType);
crate::impl_integer_id!(VertexId(IdType));

impl VertexId {
    pub(crate) fn dummy() -> Self {
        Self(0)
    }
}

/// Iterator over the vertex ids of a mesh.
pub type VertexIds<'a, VT> = idmap::Keys<'a, VertexId, VT, DenseEntryTable<VertexId, VT>>;

/// Iterator over the vertices of a mesh.
pub type IntoVertices<VT> = Map<
    <idmap::OrderedIdMap<VertexId, VT> as IntoIterator>::IntoIter,
    fn((VertexId, VT)) -> (VertexId, <VT as Vertex>::V),
>;

/// Iterator over the vertices of a mesh.
pub type Vertices<'a, VT> = Map<
    idmap::Iter<'a, VertexId, VT, DenseEntryTable<VertexId, VT>>,
    for<'b> fn((&'b VertexId, &'b VT)) -> (&'b VertexId, &'b <VT as Vertex>::V),
>;

/// Iterator over the vertices of a mesh mutably.
pub type VerticesMut<'a, VT> = Map<
    idmap::IterMut<'a, VertexId, VT, DenseEntryTable<VertexId, VT>>,
    for<'b> fn((&'b VertexId, &'b mut VT)) -> (&'b VertexId, &'b mut <VT as Vertex>::V),
>;

macro_rules! V {
    () => {
        <Self::Vertex as Vertex>::V
    };
}

/// For simplicial complexes that can have vertices, that is, all of them
pub trait HasVertices: internal::HasVertices + RemoveVertexHigher + ClearVerticesHigher {
    /// Gets the number of vertices.
    fn num_vertices(&self) -> usize {
        self.vertices_r().len()
    }

    /// Iterates over the vertex ids of this mesh.
    fn vertex_ids(&self) -> VertexIds<Self::Vertex> {
        self.vertices_r().keys()
    }

    /// Iterates over the vertices of this mesh.
    /// Gives (id, value) pairs
    fn vertices(&self) -> Vertices<Self::Vertex> {
        self.vertices_r().iter().map(|(id, v)| (id, v.value()))
    }

    /// Iterates mutably over the vertices of this mesh.
    /// Gives (id, value) pairs
    fn vertices_mut(&mut self) -> VerticesMut<Self::Vertex> {
        self.vertices_r_mut()
            .iter_mut()
            .map(|(id, v)| (id, v.value_mut()))
    }

    /// Gets the value of the vertex at a specific id.
    /// Returns None if not found.
    fn vertex(&self, id: VertexId) -> Option<&V!()> {
        self.vertices_r().get(id).map(|v| v.value())
    }

    /// Gets the value of the vertex at a specific id mutably.
    /// Returns None if not found.
    fn vertex_mut(&mut self, id: VertexId) -> Option<&mut V!()> {
        self.vertices_r_mut().get_mut(id).map(|v| v.value_mut())
    }

    /// Adds a vertex to the mesh and returns the id.
    fn add_vertex(&mut self, value: V!()) -> VertexId {
        let id = VertexId(self.next_vertex_id());
        *self.next_vertex_id_mut() += 1;
        debug_assert!(self
            .vertices_r_mut()
            .insert(id, <Self::Vertex as Vertex>::new(id, value))
            .is_none());
        id
    }

    /// Adds a vertex with a specific id.
    /// Returns the value that was previously there, if any
    fn add_vertex_with_id(&mut self, id: VertexId, value: V!()) -> Option<V!()> {
        *self.next_vertex_id_mut() = (id.0 + 1).max(self.next_vertex_id());
        self.vertices_r_mut().insert(id, <Self::Vertex as Vertex>::new(id, value))
            .map(|vertex| vertex.to_value())
    }

    /// Extends the vertex list with an iterator and returns a `Vec`
    /// of the vertex ids that are created in order.
    fn extend_vertices<I: IntoIterator<Item = V!()>>(&mut self, iter: I) -> Vec<VertexId> {
        iter.into_iter()
            .map(|value| self.add_vertex(value))
            .collect()
    }

    /// Extends the vertex list with an iterator over (id, value) pairs
    fn extend_vertices_with_ids<I: IntoIterator<Item = (VertexId, V!())>>(&mut self, iter: I) {
        iter.into_iter()
            .for_each(|(id, value)| {
                self.add_vertex_with_id(id, value);
            });
    }

    /// Removes a vertex from the mesh.
    /// Returns the value of the vertex that was there or None if none was there,
    fn remove_vertex(&mut self, id: VertexId) -> Option<V!()> {
        if self.vertex(id).is_some() {
            self.remove_vertex_higher(id);
        }
        self.vertices_r_mut().remove(id).map(|v| v.to_value())
    }

    /// Removes a list of vertices.
    fn remove_vertices<I: IntoIterator<Item = VertexId>>(&mut self, iter: I) {
        iter.into_iter().for_each(|id| {
            self.remove_vertex(id);
        })
    }

    /// Keeps only the vertices that satisfy a predicate
    fn retain_vertices<P: FnMut(VertexId, &V!()) -> bool>(&mut self, mut predicate: P) {
        let to_remove = self
            .vertices()
            .filter(|(id, v)| !predicate(**id, *v))
            .map(|(id, _)| *id)
            .collect::<Vec<_>>();
        self.remove_vertices(to_remove);
    }

    /// Removes all vertices from the mesh.
    fn clear_vertices(&mut self) {
        self.clear_vertices_higher();
        self.vertices_r_mut().clear();
    }
}

/// For concrete simplicial complexes
pub trait HasPosition: HasVertices
where
    <Self::Vertex as Vertex>::V: Position,
    DefaultAllocator: Allocator<f64, HasPositionDim<Self>>,
{
    /// Gets the position of a vertex. Assumes the vertex exists.
    fn position(&self, vertex: VertexId) -> HasPositionPoint<Self> {
        self.vertex(vertex).unwrap().position()
    }

    /// Adds a vertex with some position and the rest of the vertex value
    fn add_with_position(&mut self, position: HasPositionPoint<Self>, rest: HasPositionRest<Self>) -> VertexId {
        self.add_vertex(<<Self::Vertex as Vertex>::V as Position>::with_position(position, rest))
    }

    /// Gets the bounding box of the mesh
    /// as the array [mininum coordinates, maximum coordinates]
    fn bounding_box(&self) -> Option<[HasPositionPoint<Self>; 2]> {
        self.vertex_ids().map(|v| self.position(*v))
            .fold(None, |acc, pos| {
                if let Some([min, max]) = acc {
                    let (min, max) = (min.meet(&pos), max.join(&pos));
                    Some([min, max])
                } else {
                    Some([pos.clone(), pos])
                }
            })
    }
}

impl<M> HasPosition for M
where
    M: HasVertices,
    <Self::Vertex as Vertex>::V: Position,
    DefaultAllocator: Allocator<f64, HasPositionDim<Self>>,
{}

/// For 3D concrete simplicial complexes
pub trait HasPosition3D: HasPosition
where
    <Self::Vertex as Vertex>::V: Position<Dim = U3>,
    DefaultAllocator: Allocator<f64, HasPositionDim<Self>>,
{
    /// Turns this mesh into a Delaunay tetrahedralization of its vertices
    fn delaunay_tets<M>(self,
        tet_value_fn: impl Fn() -> <<M as HasTets>::Tet as Tet>::T,
        tri_value_fn: impl Fn() -> <<M as HasTrisIntr>::Tri as Tri>::F + Clone,
        edge_value_fn: impl Fn() -> <<M as HasEdgesIntr>::Edge as Edge>::E + Clone,
        v_rest_fn: impl Fn() -> <<Self::Vertex as Vertex>::V as Position>::Rest,
    ) -> M where
        Self: Sized,
        M: HasVertices,
        <M as HasVerticesIntr>::Vertex: Vertex<V = V!()> + HigherVertex,
        M: HasEdges,
        <M as HasEdgesIntr>::Edge: HigherEdge,
        M: HasTris,
        <M as HasTrisIntr>::Tri: HigherTri,
        M: HasTets,
        <M as HasTets>::Tet: Tet<Mwb = B1>,
    {
        let mesh = M::from_veft_r::<_, _, _, _, _, _, Key>(self.into_v_r(), vec![], vec![], vec![],
            tri_value_fn.clone(), edge_value_fn.clone());

        crate::tetrahedralize::delaunay_tets(mesh, tet_value_fn, tri_value_fn, edge_value_fn, v_rest_fn)
    }
}

impl<M> HasPosition3D for M
where
    M: HasVertices,
    <Self::Vertex as Vertex>::V: Position<Dim = U3>,
    DefaultAllocator: Allocator<f64, HasPositionDim<Self>>,
{}

#[macro_export]
#[doc(hidden)]
macro_rules! impl_index_vertex {
    ($name:ident<$v:ident $(, $args:ident)*>) => {
        impl<$v $(, $args)*> std::ops::Index<crate::vertex::VertexId> for $name<$v $(, $args)*> {
            type Output = $v;

            fn index(&self, index: crate::vertex::VertexId) -> &Self::Output {
                self.vertex(index).unwrap()
            }
        }

        impl<$v $(, $args)*> std::ops::IndexMut<crate::vertex::VertexId> for $name<$v $(, $args)*> {
            fn index_mut(&mut self, index: crate::vertex::VertexId) -> &mut Self::Output {
                self.vertex_mut(index).unwrap()
            }
        }
    };
}

pub(crate) mod internal {
    use super::{IdType, IntoVertices, VertexId};
    use idmap::OrderedIdMap;

    #[macro_export]
    #[doc(hidden)]
    macro_rules! impl_vertex {
        ($name:ident<$v:ident>, new |$id:ident, $value:ident| $new:expr) => {
            impl<$v> crate::vertex::internal::Vertex for $name<$v> {
                type V = $v;

                fn new($id: crate::vertex::VertexId, $value: Self::V) -> Self {
                    $new
                }

                fn to_value(self) -> Self::V {
                    self.value
                }

                fn value(&self) -> &Self::V {
                    &self.value
                }

                fn value_mut(&mut self) -> &mut Self::V {
                    &mut self.value
                }
            }
        };
    }

    #[macro_export]
    #[doc(hidden)]
    macro_rules! impl_higher_vertex {
        ($name:ident<$v:ident>) => {
            impl<$v> crate::vertex::internal::HigherVertex for $name<$v> {
                fn source(&self) -> VertexId {
                    self.source
                }

                fn source_mut(&mut self) -> &mut crate::vertex::VertexId {
                    &mut self.source
                }

                fn target(&self) -> VertexId {
                    self.target
                }

                fn target_mut(&mut self) -> &mut crate::vertex::VertexId {
                    &mut self.target
                }
            }
        };
    }

    #[macro_export]
    #[doc(hidden)]
    macro_rules! impl_has_vertices {
        ($name:ident<$v:ident $(, $args:ident)*>, $vertex:ident) => {
            impl<$v $(, $args)*> crate::vertex::internal::HasVertices for $name<$v $(, $args)*> {
                type Vertex = $vertex<$v>;

                fn from_v_r<
                    VI: IntoIterator<Item = (crate::vertex::VertexId, <Self::Vertex as crate::vertex::internal::Vertex>::V)>
                >(vertices: VI) -> Self {
                    let mut mesh = Self::default();
                    mesh.extend_vertices_with_ids(vertices);
                    mesh
                }

                fn into_v_r(self) -> crate::vertex::IntoVertices<Self::Vertex> {
                    use crate::vertex::internal::Vertex;
                    self.vertices.into_iter().map(|(id, v)| (id, v.to_value()))
                }

                fn vertices_r(&self) -> &idmap::OrderedIdMap<crate::vertex::VertexId, Self::Vertex> {
                    &self.vertices
                }

                fn vertices_r_mut(&mut self) -> &mut idmap::OrderedIdMap<crate::vertex::VertexId, Self::Vertex> {
                    &mut self.vertices
                }

                fn next_vertex_id(&self) -> crate::vertex::IdType {
                    self.next_vertex_id
                }

                fn next_vertex_id_mut(&mut self) -> &mut crate::vertex::IdType {
                    &mut self.next_vertex_id
                }
            }
        }
    }

    /// Storage for a vertex
    pub trait Vertex {
        type V;

        fn new(id: VertexId, value: Self::V) -> Self;

        fn to_value(self) -> Self::V;

        fn value(&self) -> &Self::V;

        fn value_mut(&mut self) -> &mut Self::V;
    }

    /// Extra storage for a vertex in a mesh that contains edges
    pub trait HigherVertex: Vertex {
        fn source(&self) -> VertexId;

        fn source_mut(&mut self) -> &mut VertexId;

        fn target(&self) -> VertexId;

        fn target_mut(&mut self) -> &mut VertexId;
    }

    pub trait HasVertices {
        type Vertex: Vertex;

        fn from_v_r<VI: IntoIterator<Item = (VertexId, <Self::Vertex as Vertex>::V)>>(vertices: VI) -> Self;

        fn into_v_r(self) -> IntoVertices<Self::Vertex>;

        fn vertices_r(&self) -> &OrderedIdMap<VertexId, Self::Vertex>;

        fn vertices_r_mut(&mut self) -> &mut OrderedIdMap<VertexId, Self::Vertex>;

        fn next_vertex_id(&self) -> IdType;

        fn next_vertex_id_mut(&mut self) -> &mut IdType;
    }

    /// Removes higher-order simplexes that contain some vertex
    pub trait RemoveVertexHigher: HasVertices {
        fn remove_vertex_higher(&mut self, vertex: VertexId);
    }

    /// Clears higher-order simplexes
    pub trait ClearVerticesHigher: HasVertices {
        fn clear_vertices_higher(&mut self);
    }
}
