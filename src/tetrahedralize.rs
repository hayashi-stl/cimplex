use crate::{edge::Edge, tet::{HasTets, Tet}, tri::Tri, vertex::{HasPosition3D, Position, Vertex}};
use typenum::B1;
use nalgebra::{Point3, dimension::U3, Vector3};

pub(crate) fn delaunay_tets<M>(mut mesh: M,
    tet_t: impl Fn() -> M::T,
    tri_f: impl Fn() -> M::F + Clone,
    edge_e: impl Fn() -> M::E + Clone,
    v_rest: impl Fn() -> <M::V as Position>::Rest,
) -> M where
    M: HasTets<MwbT = B1> + HasPosition3D,
    M::V: Position<Dim = U3>,
{
    // It takes 4 vertices to make a tet
    if mesh.num_vertices() < 4 {
        return mesh;
    }

    let [mut min, max] = mesh.bounding_box().unwrap();
    let offset = (max - min) * 3.0 + Vector3::new(2.0, 2.0, 2.0);
    min += Vector3::new(-1.0, -1.0, -1.0);

    // Add bounding tetrahedron
    let b0 = mesh.add_with_position(min, v_rest());
    let b1 = mesh.add_with_position(Point3::new(min.x + offset.x, min.y, min.z), v_rest());
    let b2 = mesh.add_with_position(Point3::new(min.x, min.y + offset.y, min.z), v_rest());
    let b3 = mesh.add_with_position(Point3::new(min.x, min.y, min.z + offset.z), v_rest());
    mesh.add_tet([b0, b1, b3, b2], tet_t(), tri_f.clone(), edge_e.clone());
    mesh
}