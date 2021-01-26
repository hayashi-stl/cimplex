use crate::{edge::Edge, tet::{HasTets, Tet}, tri::Tri, vertex::{HasPosition3D, Position, Vertex}};
use typenum::B1;
use nalgebra::{Point3, dimension::U3};

pub(crate) fn delaunay_tets<M>(mut mesh: M,
    tet_value_fn: impl Fn() -> M::T,
    tri_value_fn: impl Fn() -> M::F + Clone,
    edge_value_fn: impl Fn() -> M::E + Clone,
    v_rest_fn: impl Fn() -> <M::V as Position>::Rest,
) -> M where
    M: HasTets<MwbT = B1> + HasPosition3D,
    M::V: Position<Dim = U3>,
{
    // It takes 4 vertices to make a tet
    if mesh.num_vertices() < 4 {
        return mesh;
    }

    let [min, max] = mesh.bounding_box().unwrap();

    mesh.add_with_position(Point3::new(0.0, 0.0, 0.0), v_rest_fn());
    mesh
}