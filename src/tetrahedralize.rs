use crate::{edge::Edge, tet::{HasTets, Tet, TetId}, tri::Tri, vertex::{HasPosition3D, Position, Vertex, VertexId}};
use fnv::FnvHashMap;
use nalgebra::{dimension::U3, Point1, Point3, Vector3};
use simplicity as sim;
use typenum::B1;
use float_ord::FloatOrd;

fn index_fn<M>(mesh: &M, i: VertexId) -> Vector3<f64>
where
    M: HasPosition3D,
    M::V: Position<Dim = U3>
{
    mesh.position(i).coords
}

/// Modified in-sphere test to deal with the ghost vertex.
/// `m` is the point to test the in-sphere of; it cannot be the ghost.
fn in_sphere_with_ghosts<M>(mesh: &M, tet: TetId, m: VertexId, ghost: VertexId) -> bool
where
    M: HasTets<MwbT = B1> + HasPosition3D,
    M::V: Position<Dim = U3>
{
    if tet.contains_vertex(ghost) {
        let tri = tet.opp_tri(ghost);
        sim::orient_3d(mesh, index_fn, tri.0[0], tri.0[1], tri.0[2], m)
    } else {
        sim::in_sphere(mesh, index_fn, tet.0[0], tet.0[1], tet.0[2], tet.0[3], m)
    }
}

fn find_tet_to_delete<M>(mesh: &M, new_vertex: VertexId, ghost: VertexId) -> TetId
where
    M: HasTets<MwbT = B1> + HasPosition3D,
    M::V: Position<Dim = U3>
{
    // Look for closest vertex to the new vertex to add
    let mut vertex = mesh.tets().next().unwrap().0.0[0];
    while let Some(closer) = mesh.vertex_targets(vertex)
        .filter(|target| mesh.distance_squared(*target, new_vertex) < mesh.distance_squared(vertex, new_vertex))
        .min_by_key(|target| FloatOrd(mesh.distance_squared(*target, new_vertex)))
    {
        vertex = closer;
    }

    // The new vertex is in the circumsphere of some tet on that vertex.
    // If not, there's a floating-point error and we search further.

    // TODO: BFS
    mesh.vertex_tets(vertex)
        .filter(|tet| in_sphere_with_ghosts(mesh, *tet, new_vertex, ghost))
        .next()
        .unwrap()
}

/// Implementation of the Bowyer-Watson algorithm,
/// with ghost triangles 👻 (https://people.eecs.berkeley.edu/~jrs/meshpapers/delnotes.pdf, section 3.4)
/// to avoid the concave tetrahedralization problem that happens with a super tet.
pub(crate) fn delaunay_tets<M>(
    mut mesh: M,
    tet_t: impl Fn() -> M::T,
    tri_f: impl Fn() -> M::F + Clone,
    edge_e: impl Fn() -> M::E + Clone,
    v_rest: impl Fn() -> <M::V as Position>::Rest,
) -> M
where
    M: HasTets<MwbT = B1> + HasPosition3D,
    M::V: Position<Dim = U3>,
{
    // It takes 4 vertices to make a tet
    if mesh.num_vertices() < 4 {
        return mesh;
    }

    let mut v_ids = mesh.vertex_ids().copied().collect::<Vec<_>>();

    // Ghost vertex
    let g = mesh.add_with_position(Point1::new(f64::INFINITY).xxx(), v_rest());

    // First tet
    let v0 = v_ids.pop().unwrap();
    let v1 = v_ids.pop().unwrap();
    let mut v2 = v_ids.pop().unwrap();
    let mut v3 = v_ids.pop().unwrap();
    if !sim::orient_3d(&mesh, index_fn, v0, v1, v2, v3) {
        std::mem::swap(&mut v2, &mut v3);
    }
    let first = TetId::from_valid([v0, v1, v2, v3]);
    mesh.add_tet([v0, v1, v2, v3], tet_t(), tri_f.clone(), edge_e.clone());

    // Ghost tets
    for tri in &first.tris() {
        mesh.add_tet([tri.0[0], tri.0[2], tri.0[1], g], tet_t(), tri_f.clone(), edge_e.clone());
    }

    mesh
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{mesh3::MwbComboMesh3, vertex::HasVertices};

    #[test]
    fn test_in_sphere_ghost() {
        let mut mesh = MwbComboMesh3::<Point3<f64>, (), (), ()>::default();
        let ids = mesh.extend_vertices(vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
            Point3::new(0.5, 0.3, 0.6),
        ]);
        let tet = TetId::from_valid([ids[0], ids[1], ids[3], ids[2]]);
        mesh.add_tet(tet, (), || (), || ());

        assert!(!in_sphere_with_ghosts(&mesh, tet, ids[4], ids[0]));
        assert!(in_sphere_with_ghosts(&mesh, tet, ids[4], ids[1]));
        assert!(in_sphere_with_ghosts(&mesh, tet, ids[4], ids[2]));
        assert!(in_sphere_with_ghosts(&mesh, tet, ids[4], ids[3]));
    }

    #[test]
    fn test_in_sphere_no_ghost() {
        let mut mesh = MwbComboMesh3::<Point3<f64>, (), (), ()>::default();
        let ids = mesh.extend_vertices(vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
            Point3::new(0.5, 0.3, 0.6),
        ]);
        let tet = TetId::from_valid([ids[0], ids[1], ids[3], ids[2]]);
        mesh.add_tet(tet, (), || (), || ());

        assert!(in_sphere_with_ghosts(&mesh, tet, ids[4], VertexId(5)));
    }
}