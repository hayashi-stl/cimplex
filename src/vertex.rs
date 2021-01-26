//! Traits and structs related to vertices

use alga::general::{JoinSemilattice, MeetSemilattice};
use idmap::{table::DenseEntryTable, OrderedIdMap};
use nalgebra::allocator::Allocator;
use nalgebra::dimension::U3;
use nalgebra::{DefaultAllocator, DimName, Point};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::iter::Map;
use typenum::{Bit, B1, B0};

use crate::private::{Key, Lock};
use crate::{
    tet::{HasTets, WithTets},
};

pub(crate) type PositionDim<P> = <P as Position>::Dim;
pub(crate) type PositionPoint<P> = Point<f64, PositionDim<P>>;
pub(crate) type HasPositionDim<P> = <<<P as HasVertices>::Vertex as Vertex>::V as Position>::Dim;
pub(crate) type HasPositionPoint<P> = Point<f64, HasPositionDim<P>>;
pub(crate) type HasPositionRest<P> = <<<P as HasVertices>::Vertex as Vertex>::V as Position>::Rest;

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

    /// Set the position of `self`
    fn with_position(self, point: PositionPoint<Self>) -> Self;

    /// Construct a new object with a point and a rest of the type
    fn with_position_rest(point: PositionPoint<Self>, rest: Self::Rest) -> Self;
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

    fn with_position(self, point: PositionPoint<Self>) -> Self {
        point
    }

    fn with_position_rest(point: PositionPoint<Self>, _rest: Self::Rest) -> Self {
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

    fn with_position(self, point: PositionPoint<Self>) -> Self {
        (point, self.1)
    }

    fn with_position_rest(point: PositionPoint<Self>, rest: Self::Rest) -> Self {
        (point, rest)
    }
}

/// The integer type used for a vertex
pub type IdType = u32;

/// An index to a vertex of a mesh.
/// Will not be invalidated unless the vertex gets removed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct VertexId(pub(crate) IdType);
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

/// Vertex attributes
pub trait Vertex {
    type V;
    type Higher: Bit;

    #[doc(hidden)]
    fn new<L: Lock>(id: VertexId, value: Self::V) -> Self;

    #[doc(hidden)]
    fn to_value<L: Lock>(self) -> Self::V;

    #[doc(hidden)]
    fn value<L: Lock>(&self) -> &Self::V;

    #[doc(hidden)]
    fn value_mut<L: Lock>(&mut self) -> &mut Self::V;

    #[doc(hidden)]
    fn source<L: Lock>(&self) -> VertexId
    where
        Self: Vertex<Higher = B1>;

    #[doc(hidden)]
    fn source_mut<L: Lock>(&mut self) -> &mut VertexId
    where
        Self: Vertex<Higher = B1>;

    #[doc(hidden)]
    fn target<L: Lock>(&self) -> VertexId
    where
        Self: Vertex<Higher = B1>;

    #[doc(hidden)]
    fn target_mut<L: Lock>(&mut self) -> &mut VertexId
    where
        Self: Vertex<Higher = B1>;
}

/// For simplicial complexes that can have vertices, that is, all of them
pub trait HasVertices {
    type Vertex: Vertex<V = Self::V, Higher = Self::HigherV>;
    type V;
    type HigherV: Bit;

    #[doc(hidden)]
    fn from_v_r<VI: IntoIterator<Item = (VertexId, Self::V)>, L: Lock>(vertices: VI, default_v: fn() -> Self::V) -> Self
        where Self: HasVertices<HigherV = B0>;

    #[doc(hidden)]
    fn into_v_r<L: Lock>(self) -> IntoVertices<Self::Vertex>;

    #[doc(hidden)]
    fn vertices_r<L: Lock>(&self) -> &OrderedIdMap<VertexId, Self::Vertex>;

    #[doc(hidden)]
    fn vertices_r_mut<L: Lock>(&mut self) -> &mut OrderedIdMap<VertexId, Self::Vertex>;

    #[doc(hidden)]
    fn next_vertex_id<L: Lock>(&self) -> IdType;

    #[doc(hidden)]
    fn next_vertex_id_mut<L: Lock>(&mut self) -> &mut IdType;

    #[doc(hidden)]
    fn remove_vertex_higher<L: Lock>(&mut self, vertex: VertexId);

    #[doc(hidden)]
    fn clear_vertices_higher<L: Lock>(&mut self);

    #[doc(hidden)]
    fn default_v_r<L: Lock>(&self) -> fn() -> Self::V;

    /// Gets the default value of a vertex.
    fn default_vertex(&self) -> Self::V {
        self.default_v_r::<Key>()()
    }

    /// Gets the number of vertices.
    fn num_vertices(&self) -> usize {
        self.vertices_r::<Key>().len()
    }

    /// Iterates over the vertex ids of this mesh.
    fn vertex_ids(&self) -> VertexIds<Self::Vertex> {
        self.vertices_r::<Key>().keys()
    }

    /// Iterates over the vertices of this mesh.
    /// Gives (id, value) pairs
    fn vertices(&self) -> Vertices<Self::Vertex> {
        self.vertices_r::<Key>()
            .iter()
            .map(|(id, v)| (id, v.value::<Key>()))
    }

    /// Iterates mutably over the vertices of this mesh.
    /// Gives (id, value) pairs
    fn vertices_mut(&mut self) -> VerticesMut<Self::Vertex> {
        self.vertices_r_mut::<Key>()
            .iter_mut()
            .map(|(id, v)| (id, v.value_mut::<Key>()))
    }

    /// Gets the value of the vertex at a specific id.
    /// Returns None if not found.
    fn vertex(&self, id: VertexId) -> Option<&Self::V> {
        self.vertices_r::<Key>().get(id).map(|v| v.value::<Key>())
    }

    /// Gets the value of the vertex at a specific id mutably.
    /// Returns None if not found.
    fn vertex_mut(&mut self, id: VertexId) -> Option<&mut Self::V> {
        self.vertices_r_mut::<Key>()
            .get_mut(id)
            .map(|v| v.value_mut::<Key>())
    }

    /// Adds a vertex to the mesh and returns the id.
    fn add_vertex(&mut self, value: Self::V) -> VertexId {
        let id = VertexId(self.next_vertex_id::<Key>());
        *self.next_vertex_id_mut::<Key>() += 1;
        debug_assert!(self
            .vertices_r_mut::<Key>()
            .insert(id, <Self::Vertex as Vertex>::new::<Key>(id, value))
            .is_none());
        id
    }

    /// Adds a vertex with a specific id.
    /// Returns the value that was previously there, if any
    fn add_vertex_with_id(&mut self, id: VertexId, value: Self::V) -> Option<Self::V> {
        *self.next_vertex_id_mut::<Key>() = (id.0 + 1).max(self.next_vertex_id::<Key>());
        self.vertices_r_mut::<Key>()
            .insert(id, <Self::Vertex as Vertex>::new::<Key>(id, value))
            .map(|vertex| vertex.to_value::<Key>())
    }

    /// Extends the vertex list with an iterator and returns a `Vec`
    /// of the vertex ids that are created in order.
    fn extend_vertices<I: IntoIterator<Item = Self::V>>(&mut self, iter: I) -> Vec<VertexId> {
        iter.into_iter()
            .map(|value| self.add_vertex(value))
            .collect()
    }

    /// Extends the vertex list with an iterator over (id, value) pairs
    fn extend_vertices_with_ids<I: IntoIterator<Item = (VertexId, Self::V)>>(&mut self, iter: I) {
        iter.into_iter().for_each(|(id, value)| {
            self.add_vertex_with_id(id, value);
        });
    }

    /// Removes a vertex from the mesh.
    /// Returns the value of the vertex that was there or None if none was there,
    fn remove_vertex(&mut self, id: VertexId) -> Option<Self::V> {
        if self.vertex(id).is_some() {
            self.remove_vertex_higher::<Key>(id);
        }
        self.vertices_r_mut::<Key>()
            .remove(id)
            .map(|v| v.to_value::<Key>())
    }

    /// Removes a list of vertices.
    fn remove_vertices<I: IntoIterator<Item = VertexId>>(&mut self, iter: I) {
        iter.into_iter().for_each(|id| {
            self.remove_vertex(id);
        })
    }

    /// Keeps only the vertices that satisfy a predicate
    fn retain_vertices<P: FnMut(VertexId, &Self::V) -> bool>(&mut self, mut predicate: P) {
        let to_remove = self
            .vertices()
            .filter(|(id, v)| !predicate(**id, *v))
            .map(|(id, _)| *id)
            .collect::<Vec<_>>();
        self.remove_vertices(to_remove);
    }

    /// Removes all vertices from the mesh.
    fn clear_vertices(&mut self) {
        self.clear_vertices_higher::<Key>();
        self.vertices_r_mut::<Key>().clear();
    }
}

/// For concrete simplicial complexes
pub trait HasPosition: HasVertices
where
    Self::V: Position,
    DefaultAllocator: Allocator<f64, HasPositionDim<Self>>,
{
    /// Gets the position of a vertex. Assumes the vertex exists.
    fn position(&self, vertex: VertexId) -> HasPositionPoint<Self> {
        self.vertex(vertex).unwrap().position()
    }

    fn distance(&self, v0: VertexId, v1: VertexId) -> f64 {
        (self.position(v0) - self.position(v1)).norm()
    }

    fn distance_squared(&self, v0: VertexId, v1: VertexId) -> f64 {
        (self.position(v0) - self.position(v1)).norm_squared()
    }

    /// Adds a vertex with some position and the default rest of the vertex value
    fn add_with_position(
        &mut self,
        position: HasPositionPoint<Self>,
    ) -> VertexId {
        self.add_vertex(self.default_vertex().with_position(position))
    }

    /// Adds a vertex with some position and the rest of the vertex value
    fn add_with_position_rest(
        &mut self,
        position: HasPositionPoint<Self>,
        rest: HasPositionRest<Self>,
    ) -> VertexId {
        self.add_vertex(<Self::V as Position>::with_position_rest(position, rest))
    }

    /// Gets the bounding box of the mesh
    /// as the array [mininum coordinates, maximum coordinates]
    fn bounding_box(&self) -> Option<[HasPositionPoint<Self>; 2]> {
        self.vertex_ids()
            .map(|v| self.position(*v))
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
    Self::V: Position,
    DefaultAllocator: Allocator<f64, HasPositionDim<Self>>,
{
}

/// For 3D concrete simplicial complexes
pub trait HasPosition3D: HasPosition
where
    Self::V: Position<Dim = U3>,
    DefaultAllocator: Allocator<f64, HasPositionDim<Self>>,
{
    /// Turns this mesh into a Delaunay tetrahedralization of its vertices
    fn delaunay_tets<E, F, T>(
        self,
        default_edge: fn() -> E,
        default_tri: fn() -> F,
        default_tet: fn() -> T,
    ) -> <Self::WithTets as HasTets>::WithMwbT
    where
        Self: Sized,
        Self: WithTets<<Self as HasVertices>::V, E, F, T>,
    {
        let default_v = self.default_v_r::<Key>();
        let mesh = <Self::WithTets as HasTets>::WithMwbT::from_veft_r::<_, _, _, _, Key>(
            self.into_v_r::<Key>(),
            vec![],
            vec![],
            vec![],
            default_v,
            default_edge,
            default_tri,
            default_tet,
        );

        crate::tetrahedralize::delaunay_tets(mesh)
    }
}

impl<M> HasPosition3D for M
where
    M: HasVertices,
    Self::V: Position<Dim = U3>,
    DefaultAllocator: Allocator<f64, HasPositionDim<Self>>,
{
}

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

#[macro_export]
#[doc(hidden)]
macro_rules! impl_vertex {
    ($name:ident<$v:ident>, new |$id:ident, $value:ident| $new:expr) => {
        impl<$v> crate::vertex::Vertex for $name<$v> {
            type V = $v;
            type Higher = typenum::B0;

            fn new<L: crate::private::Lock>($id: crate::vertex::VertexId, $value: Self::V) -> Self {
                $new
            }

            fn to_value<L: crate::private::Lock>(self) -> Self::V {
                self.value
            }

            fn value<L: crate::private::Lock>(&self) -> &Self::V {
                &self.value
            }

            fn value_mut<L: crate::private::Lock>(&mut self) -> &mut Self::V {
                &mut self.value
            }

            fn source<L: crate::private::Lock>(&self) -> crate::vertex::VertexId
            where
                Self: crate::vertex::Vertex<Higher = typenum::B1>,
            {
                unreachable!()
            }

            fn source_mut<L: crate::private::Lock>(&mut self) -> &mut crate::vertex::VertexId
            where
                Self: crate::vertex::Vertex<Higher = typenum::B1>,
            {
                unreachable!()
            }

            fn target<L: crate::private::Lock>(&self) -> crate::vertex::VertexId
            where
                Self: crate::vertex::Vertex<Higher = typenum::B1>,
            {
                unreachable!()
            }

            fn target_mut<L: crate::private::Lock>(&mut self) -> &mut crate::vertex::VertexId
            where
                Self: crate::vertex::Vertex<Higher = typenum::B1>,
            {
                unreachable!()
            }
        }
    };
}

#[macro_export]
#[doc(hidden)]
macro_rules! impl_vertex_higher {
    ($name:ident<$v:ident>, new |$id:ident, $value:ident| $new:expr) => {
        impl<$v> crate::vertex::Vertex for $name<$v> {
            type V = $v;
            type Higher = typenum::B1;

            fn new<L: crate::private::Lock>($id: crate::vertex::VertexId, $value: Self::V) -> Self {
                $new
            }

            fn to_value<L: crate::private::Lock>(self) -> Self::V {
                self.value
            }

            fn value<L: crate::private::Lock>(&self) -> &Self::V {
                &self.value
            }

            fn value_mut<L: crate::private::Lock>(&mut self) -> &mut Self::V {
                &mut self.value
            }

            fn source<L: crate::private::Lock>(&self) -> crate::vertex::VertexId
            where
                Self: crate::vertex::Vertex<Higher = typenum::B1>,
            {
                self.source
            }

            fn source_mut<L: crate::private::Lock>(&mut self) -> &mut crate::vertex::VertexId
            where
                Self: crate::vertex::Vertex<Higher = typenum::B1>,
            {
                &mut self.source
            }

            fn target<L: crate::private::Lock>(&self) -> crate::vertex::VertexId
            where
                Self: crate::vertex::Vertex<Higher = typenum::B1>,
            {
                self.target
            }

            fn target_mut<L: crate::private::Lock>(&mut self) -> &mut crate::vertex::VertexId
            where
                Self: crate::vertex::Vertex<Higher = typenum::B1>,
            {
                &mut self.target
            }
        }
    };
}

#[macro_export]
#[doc(hidden)]
macro_rules! impl_has_vertices {
    ($vertex:ident<$v:ident> $($z:ident)*, Higher = $higher:ty) => {
        type Vertex = $vertex<$v>;
        type V = $v;
        type HigherV = $higher;

        fn from_v_r<
            VI: IntoIterator<
                Item = (
                    crate::vertex::VertexId,
                    <Self::Vertex as crate::vertex::Vertex>::V,
                ),
            >,
            L: crate::private::Lock,
        >(
            vertices: VI,
            default_v: fn() -> Self::V,
        ) -> Self {
            use typenum::Bit;
            if <$higher>::BOOL {
                unreachable!()
            }
            // The code below will not be executed if the value is invalid.
            #[allow(invalid_value)]
            let mut mesh = Self::with_defaults(default_v $(, unsafe { std::mem::$z() })*);
            mesh.extend_vertices_with_ids(vertices);
            mesh
        }

        fn into_v_r<L: crate::private::Lock>(self) -> crate::vertex::IntoVertices<Self::Vertex> {
            use crate::vertex::Vertex;
            self.vertices
                .into_iter()
                .map(|(id, v)| (id, v.to_value::<crate::private::Key>()))
        }

        fn vertices_r<L: crate::private::Lock>(
            &self,
        ) -> &idmap::OrderedIdMap<crate::vertex::VertexId, Self::Vertex> {
            &self.vertices
        }

        fn vertices_r_mut<L: crate::private::Lock>(
            &mut self,
        ) -> &mut idmap::OrderedIdMap<crate::vertex::VertexId, Self::Vertex> {
            &mut self.vertices
        }

        fn next_vertex_id<L: crate::private::Lock>(&self) -> crate::vertex::IdType {
            self.next_vertex_id
        }

        fn next_vertex_id_mut<L: crate::private::Lock>(&mut self) -> &mut crate::vertex::IdType {
            &mut self.next_vertex_id
        }

        fn default_v_r<L: crate::private::Lock>(&self) -> fn() -> Self::V {
            self.default_v
        }
    };
}
