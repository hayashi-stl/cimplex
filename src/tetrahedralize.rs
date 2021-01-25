use crate::{edge::{HasEdges, internal::{Edge, HigherEdge}}, tet::{HasTets, Tet}, tri::{HasTris, HigherTri, Tri}, vertex::{HasPosition3D, HasVertices, Position, internal::{HigherVertex, Vertex}}};
use typenum::B1;
use nalgebra::{Point3, dimension::U3};

pub(crate) fn delaunay_tets<M>(mut mesh: M,
    tet_value_fn: impl Fn() -> <M::Tet as Tet>::T,
    tri_value_fn: impl Fn() -> <M::Tri as Tri>::F + Clone,
    edge_value_fn: impl Fn() -> <M::Edge as Edge>::E + Clone,
    v_rest_fn: impl Fn() -> <<M::Vertex as Vertex>::V as Position>::Rest,
) -> M where
    M: HasTets + HasPosition3D,
    M::Vertex: HigherVertex,
    M::Edge: HigherEdge,
    M::Tri: HigherTri,
    M::Tet: Tet<Mwb = B1>,
    <M::Vertex as Vertex>::V: Position<Dim = U3>
{
    let [min, max] = match mesh.bounding_box() {
        Some(box_) => box_,
        None => return mesh, // no points.
    };

    mesh.add_with_position(Point3::new(0.0, 0.0, 0.0), v_rest_fn());
    mesh
}