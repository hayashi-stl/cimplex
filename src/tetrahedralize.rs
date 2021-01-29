use crate::{edge::HasEdges, iter, tri::{HasTris, TriId}};
use crate::{
    tet::{HasTets, TetId},
    vertex::{HasPosition3D, Position, VertexId},
};
use float_ord::FloatOrd;
use fnv::FnvHashSet;
use nalgebra::{dimension::U3, Point1, Vector3};
use simplicity as sim;
use typenum::B1;

pub(crate) fn index_fn<M>(mesh: &M, i: VertexId) -> Vector3<f64>
where
    M: HasPosition3D,
    M::V: Position<Dim = U3>,
{
    mesh.position(i).coords
}

/// Modified in-sphere test to deal with the ghost vertex.
/// `m` is the point to test the in-sphere of; it cannot be the ghost.
fn in_sphere_with_ghosts<M>(mesh: &M, tet: TetId, m: VertexId, ghost: VertexId) -> bool
where
    M: HasTets<MwbT = B1> + HasPosition3D,
    M::V: Position<Dim = U3>,
{
    if tet.contains_vertex(ghost) {
        let tri = tet.opp_tri(ghost);
        sim::orient_3d(mesh, index_fn, tri.0[0], tri.0[1], tri.0[2], m)
    } else {
        sim::in_sphere(mesh, index_fn, tet.0[0], tet.0[1], tet.0[2], tet.0[3], m)
    }
}

/// Whether some tri intersects some edge, both given
/// by vertex ids in case they don't exist.
pub(crate) 
fn tri_intersects_edge<M>(mesh: &M, v1: VertexId, v2: VertexId, v3: VertexId, vp: VertexId, vn: VertexId) -> bool
where
    M: HasPosition3D,
    M::V: Position<Dim = U3>,
{
    let keep = sim::orient_3d(mesh, index_fn, v1, v2, v3, vp);
    sim::orient_3d(mesh, index_fn, v3, v2, v1, vn) == keep &&
    sim::orient_3d(mesh, index_fn, v1, v2, vn, vp) == keep &&
    sim::orient_3d(mesh, index_fn, v2, v3, vn, vp) == keep &&
    sim::orient_3d(mesh, index_fn, v3, v1, vn, vp) == keep
}

fn find_tet_to_delete<M>(mesh: &M, new_vertex: VertexId, ghost: VertexId) -> TetId
where
    M: HasTets<MwbT = B1> + HasPosition3D,
    M::V: Position<Dim = U3>,
{
    // Look for closest vertex to the new vertex to add
    let mut vertex = (mesh.tets().next().unwrap().0).0[0];
    while let Some(closer) = mesh
        .vertex_targets(vertex)
        .filter(|target| {
            mesh.distance_squared(*target, new_vertex) < mesh.distance_squared(vertex, new_vertex)
        })
        .min_by_key(|target| FloatOrd(mesh.distance_squared(*target, new_vertex)))
    {
        vertex = closer;
    }

    // The new vertex is in the circumsphere of some tet on that vertex.
    // If not, there's a floating-point error and we search further.

    iter::bfs(
        mesh.vertex_tets(vertex),
        |tet| mesh.adjacent_tets(*tet),
        |_| true,
    )
    .find(|tet| in_sphere_with_ghosts(mesh, *tet, new_vertex, ghost))
    .unwrap()
}

fn tets_to_delete<'a, M>(
    mesh: &'a M,
    new_vertex: VertexId,
    ghost: VertexId,
) -> impl Iterator<Item = TetId> + 'a
where
    M: HasTets<MwbT = B1> + HasPosition3D,
    M::V: Position<Dim = U3>,
{
    iter::bfs(
        std::iter::once(find_tet_to_delete(mesh, new_vertex, ghost)),
        move |tet| mesh.adjacent_tets(*tet),
        move |tet| in_sphere_with_ghosts(mesh, *tet, new_vertex, ghost),
    )
}

/// Implementation of the Bowyer-Watson algorithm,
/// with ghost tetrahedrons 👻 (https://people.eecs.berkeley.edu/~jrs/meshpapers/delnotes.pdf, section 3.4)
/// to avoid the concave tetrahedralization problem that happens with a super tet.
pub(crate) fn delaunay_tets<M>(mut mesh: M) -> M
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
    let ghost = mesh.add_with_position(Point1::new(f64::INFINITY).xxx());

    // First tet
    let v0 = v_ids.pop().unwrap();
    let v1 = v_ids.pop().unwrap();
    let mut v2 = v_ids.pop().unwrap();
    let mut v3 = v_ids.pop().unwrap();
    if !sim::orient_3d(&mesh, index_fn, v0, v1, v2, v3) {
        std::mem::swap(&mut v2, &mut v3);
    }
    let first = TetId::from_valid([v0, v1, v2, v3]);
    mesh.add_tet([v0, v1, v2, v3], mesh.default_tet());

    // Ghost tets
    for tri in &first.tris() {
        mesh.add_tet([tri.0[0], tri.0[2], tri.0[1], ghost], mesh.default_tet());
    }

    while let Some(vertex) = v_ids.pop() {
        let to_delete = tets_to_delete(&mesh, vertex, ghost).collect::<Vec<_>>();

        // Get boundary
        let tris = to_delete.iter().flat_map(|tet| tet.tris().to_vec()).collect::<FnvHashSet<_>>();
        let boundary = tris
            .iter()
            .copied()
            .filter(|tri| !tris.contains(&tri.twin()))
            .collect::<Vec<_>>();

        // Retetrahedralize region
        mesh.remove_tets(to_delete);
        mesh.extend_tets(
            boundary.into_iter()
                .map(|tri| {
                    (
                        TetId::from_valid([tri.0[0], tri.0[1], tri.0[2], vertex]),
                        mesh.default_tet(),
                    )
                })
                .collect::<Vec<_>>(),
        );
    }

    mesh.remove_vertex(ghost);
    mesh
}

/// Recover as many edges as possible in the tetrahedralization.
pub(crate) fn recover_edges<M, EM>(mut mesh: M, edge_mesh: &EM) -> M
where
    M: HasTets<MwbT = B1> + HasPosition3D,
    M::V: Position<Dim = U3> + Clone,
    M::E: Clone,
    M::F: Clone,
    M::T: Clone,
    M::WithoutTets: HasTris<HigherF = typenum::B0>,
    EM: HasEdges + HasPosition3D,
    EM::V: Position<Dim = U3>,
{
    let mut num_missing_edges = 0;

    for depth in 1..4 {
        let mut to_recover = 
            edge_mesh.edge_ids().copied().filter(|e| !mesh.contains_edge(*e)).map(|e| e.undirected()).collect::<FnvHashSet<_>>()
                .into_iter().collect::<Vec<_>>();
        num_missing_edges = 0;

        while let Some(edge) = to_recover.pop() {
            let mut interfering_tris: Vec<TriId> = vec![];

            while !mesh.contains_edge(edge) {
                // Look for next triangle that intersects the edge to recover
                if let Some(tri) = interfering_tris.last() {
                    let twin = tri.twin();
                    let vertex = mesh.tri_vertex_opp(twin).unwrap();
                    if vertex == edge.0[1] {
                        // Could not recover edge
                        num_missing_edges += 1;
                        break;
                    }
                    interfering_tris.push(twin.edges()
                        .iter()
                        .map(|edge| TriId::from_valid([edge.0[1], edge.0[0], vertex]))
                        .find(|tri| tri_intersects_edge(&mesh, tri.0[0], tri.0[1], tri.0[2], edge.0[0], edge.0[1]))
                        .unwrap());
                } else {
                    interfering_tris.push(mesh.vertex_tri_opps(edge.0[0])
                        .find(|tri| tri_intersects_edge(&mesh, tri.0[0], tri.0[1], tri.0[2], edge.0[0], edge.0[1]))
                        .unwrap()); // It must exist
                }

                // Try to perform flips back to the source vertex of the edge to recover
                while let Some(interfering_tri) = interfering_tris.pop() {
                    if mesh.contains_tri(interfering_tri) && !mesh.remove_tri_via_flips(interfering_tri, depth,
                        |_, edge| !edge_mesh.contains_edge(edge),
                        |_, _| true,
                        |_, _, _| true,
                        |m, v0, v1, v2|
                            edge.contains_vertex(v0) || edge.contains_vertex(v1) || edge.contains_vertex(v2) ||
                            !tri_intersects_edge(m, v0, v1, v2, edge.0[0], edge.0[1]),
                        &mut vec![]
                    ) {
                        interfering_tris.push(interfering_tri);
                        break;
                    }
                }
            }
        }
    }

    println!("Missing {} out of {} edges.", num_missing_edges, edge_mesh.num_edges() / 2);
    
    mesh
}

#[cfg(test)]
mod tests {
    use std::convert::TryInto;
    use std::fmt::Debug;
    use std::hash::Hash;

    use fnv::FnvHashSet;
    use nalgebra::Point3;

    use super::*;
    use crate::{ComboMesh0, mesh3::MwbComboMesh3};
    use crate::vertex::HasVertices;

    #[track_caller]
    fn assert_tets_m<
        V,
        E,
        F,
        T: Clone + Debug + Eq + Hash,
        TI: TryInto<TetId>,
        I: IntoIterator<Item = (TI, T)>,
    >(
        mesh: &MwbComboMesh3<V, E, F, T>,
        tets: I,
    ) {
        let result = mesh
            .tets()
            .map(|(id, f)| (*id, f.clone()))
            .collect::<FnvHashSet<_>>();
        let expect = tets
            .into_iter()
            .map(|(vertices, f)| (vertices.try_into().ok().unwrap(), f))
            .collect::<FnvHashSet<_>>();

        assert_eq!(result, expect);
        assert_eq!(mesh.num_tets(), expect.len());
    }

    #[test]
    fn test_in_sphere_ghost() {
        let mut mesh = MwbComboMesh3::<Point3<f64>, (), (), ()>::with_defaults(
            || Point3::origin(),
            || (),
            || (),
            || (),
        );
        let ids = mesh.extend_vertices(vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
            Point3::new(0.5, 0.3, 0.6),
        ]);
        let tet = TetId::from_valid([ids[0], ids[1], ids[3], ids[2]]);
        mesh.add_tet(tet, ());

        assert!(!in_sphere_with_ghosts(&mesh, tet, ids[4], ids[0]));
        assert!(in_sphere_with_ghosts(&mesh, tet, ids[4], ids[1]));
        assert!(in_sphere_with_ghosts(&mesh, tet, ids[4], ids[2]));
        assert!(in_sphere_with_ghosts(&mesh, tet, ids[4], ids[3]));
    }

    #[test]
    fn test_in_sphere_no_ghost() {
        let mut mesh = MwbComboMesh3::<Point3<f64>, (), (), ()>::with_defaults(
            || Point3::origin(),
            || (),
            || (),
            || (),
        );
        let ids = mesh.extend_vertices(vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
            Point3::new(0.5, 0.3, 0.6),
        ]);
        let tet = TetId::from_valid([ids[0], ids[1], ids[3], ids[2]]);
        mesh.add_tet(tet, ());

        assert!(in_sphere_with_ghosts(&mesh, tet, ids[4], VertexId(5)));
    }

    #[test]
    fn test_tets_to_delete() {
        let mut mesh = MwbComboMesh3::<Point3<f64>, (), (), ()>::with_defaults(
            || Point3::origin(),
            || (),
            || (),
            || (),
        );
        let ids = mesh.extend_vertices(vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
            Point3::new(1.0, 1.0, 1.0),
            Point1::new(f64::INFINITY).xxx(),
            Point3::new(0.1, 0.1, 0.1),
            Point3::new(0.1, -0.1, 0.1),
            Point3::new(-1.0, -1.0, 0.1),
        ]);
        mesh.extend_tets(vec![
            ([ids[0], ids[1], ids[3], ids[2]], ()),
            ([ids[4], ids[1], ids[2], ids[3]], ()),
            ([ids[5], ids[0], ids[1], ids[3]], ()),
            ([ids[5], ids[3], ids[2], ids[0]], ()),
            ([ids[5], ids[1], ids[0], ids[2]], ()),
            ([ids[5], ids[4], ids[1], ids[2]], ()),
            ([ids[5], ids[2], ids[3], ids[4]], ()),
            ([ids[5], ids[1], ids[4], ids[3]], ()),
        ]);

        // In convex hull
        let result = tets_to_delete(&mesh, ids[6], ids[5]).collect::<FnvHashSet<_>>();
        assert_eq!(
            result,
            vec![
                TetId::from_valid([ids[0], ids[1], ids[3], ids[2]]),
                TetId::from_valid([ids[4], ids[1], ids[2], ids[3]]),
            ]
            .into_iter()
            .collect::<FnvHashSet<_>>()
        );

        // Remove both solid tetrahedrons and ghost tetrahedrons
        let result = tets_to_delete(&mesh, ids[7], ids[5]).collect::<FnvHashSet<_>>();
        assert_eq!(
            result,
            vec![
                TetId::from_valid([ids[0], ids[1], ids[3], ids[2]]),
                TetId::from_valid([ids[4], ids[1], ids[2], ids[3]]),
                TetId::from_valid([ids[5], ids[0], ids[1], ids[3]]),
            ]
            .into_iter()
            .collect::<FnvHashSet<_>>()
        );

        // Remove only ghost tetrahedrons
        let result = tets_to_delete(&mesh, ids[8], ids[5]).collect::<FnvHashSet<_>>();
        assert_eq!(
            result,
            vec![
                TetId::from_valid([ids[5], ids[0], ids[1], ids[3]]),
                TetId::from_valid([ids[5], ids[3], ids[2], ids[0]]),
            ]
            .into_iter()
            .collect::<FnvHashSet<_>>()
        );
    }

    #[test]
    fn test_delaunay_tets_empty() {
        let mut mesh = ComboMesh0::<Point3<f64>>::with_defaults(|| Point3::origin());
        mesh.extend_vertices(vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ]);

        let result = mesh.clone().delaunay_tets(|| (), || (), || ());
        assert_eq!(
            mesh.vertex_ids().collect::<FnvHashSet<_>>(),
            result.vertex_ids().collect::<FnvHashSet<_>>(),
        );
        assert_tets_m(&result, vec![] as Vec<(TetId, ())>);
    }

    #[test]
    fn test_delaunay_tets_single() {
        let mut mesh = ComboMesh0::<Point3<f64>>::with_defaults(|| Point3::origin());
        let ids = mesh.extend_vertices(vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
        ]);

        let result = mesh.clone().delaunay_tets(|| (), || (), || ());
        assert_eq!(
            mesh.vertex_ids().collect::<FnvHashSet<_>>(),
            result.vertex_ids().collect::<FnvHashSet<_>>(),
        );
        assert_tets_m(&result, vec![
            (TetId::from_valid([ids[0], ids[1], ids[3], ids[2]]), ()),
        ]);
    }

    #[test]
    fn test_delaunay_tets_multiple() {
        let mut mesh = ComboMesh0::<Point3<f64>>::with_defaults(|| Point3::origin());
        let ids = mesh.extend_vertices(vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
            Point3::new(1.5, 1.5, 1.0),
            Point3::new(0.5, 0.5, 0.5),
        ]);

        let result = mesh.clone().delaunay_tets(|| (), || (), || ());
        assert_eq!(
            mesh.vertex_ids().collect::<FnvHashSet<_>>(),
            result.vertex_ids().collect::<FnvHashSet<_>>(),
        );
        assert_tets_m(&result, vec![
            (TetId::from_valid([ids[0], ids[3], ids[2], ids[5]]), ()),
            (TetId::from_valid([ids[0], ids[1], ids[3], ids[5]]), ()),
            (TetId::from_valid([ids[1], ids[0], ids[2], ids[5]]), ()),
            (TetId::from_valid([ids[1], ids[4], ids[3], ids[5]]), ()),
            (TetId::from_valid([ids[3], ids[4], ids[2], ids[5]]), ()),
            (TetId::from_valid([ids[2], ids[4], ids[1], ids[5]]), ()),
        ]);
    }

    #[test]
    fn test_delaunay_tets_same_position() {
        // Simulation of simplicity is used. This should be perfectly fine.

        let mut mesh = ComboMesh0::<Point3<f64>>::with_defaults(|| Point3::origin());
        mesh.extend_vertices(vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
        ]);
        mesh.delaunay_tets(|| (), || (), || ());
    }

    #[test]
    fn test_tri_intersects_edge_true() {
        let mut mesh = ComboMesh0::<Point3<f64>>::with_defaults(|| Point3::origin());
        let ids = mesh.extend_vertices(vec![
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(-1.0, -1.0, 0.0),
            Point3::new(-1.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
            Point3::new(0.0, 0.0, -1.0),
        ]);

        assert!(tri_intersects_edge(&mesh, ids[0], ids[1], ids[2], ids[3], ids[4]));
        assert!(tri_intersects_edge(&mesh, ids[0], ids[1], ids[2], ids[4], ids[3]));
    }

    #[test]
    fn test_tri_intersects_edge_false_off_triangle() {
        let mut mesh = ComboMesh0::<Point3<f64>>::with_defaults(|| Point3::origin());
        let ids = mesh.extend_vertices(vec![
            Point3::new(2.0, 0.0, 0.0),
            Point3::new(1.0, -1.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
            Point3::new(0.0, 0.0, -1.0),
        ]);

        assert!(!tri_intersects_edge(&mesh, ids[0], ids[1], ids[2], ids[3], ids[4]));
        assert!(!tri_intersects_edge(&mesh, ids[1], ids[2], ids[0], ids[3], ids[4]));
        assert!(!tri_intersects_edge(&mesh, ids[2], ids[0], ids[1], ids[3], ids[4]));
        assert!(!tri_intersects_edge(&mesh, ids[0], ids[1], ids[2], ids[4], ids[3]));
        assert!(!tri_intersects_edge(&mesh, ids[1], ids[2], ids[0], ids[4], ids[3]));
        assert!(!tri_intersects_edge(&mesh, ids[2], ids[0], ids[1], ids[4], ids[3]));
    }

    #[test]
    fn test_tri_intersects_edge_false_off_plane() {
        let mut mesh = ComboMesh0::<Point3<f64>>::with_defaults(|| Point3::origin());
        let ids = mesh.extend_vertices(vec![
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(-1.0, -1.0, 0.0),
            Point3::new(-1.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, -1.0),
            Point3::new(0.0, 0.0, -2.0),
        ]);

        assert!(!tri_intersects_edge(&mesh, ids[0], ids[2], ids[1], ids[3], ids[4]));
        assert!(!tri_intersects_edge(&mesh, ids[0], ids[2], ids[1], ids[4], ids[3]));
    }

    //#[test]
    //fn test_export() {
    //    let mut mesh = ComboMesh0::<Point3<f64>>::with_defaults(|| Point3::origin());
    //    mesh.extend_vertices((0..3).flat_map(|x| (0..3).flat_map(move |y| (0..3).map(move |z| 
    //        Point3::new(x as f64, y as f64, z as f64)))));

    //    let result = mesh.delaunay_tets(|| (), || (), || ());
    //    result.to_separate_tets().write_obj("assets/grid.obj").unwrap();
    //}

    //#[test]
    //fn test_import() {
    //    use crate::mesh2::ComboMesh2;
    //    let mesh = ComboMesh2::read_obj("assets/ybg.obj", || Point3::origin(), || (), || ()).unwrap()
    //        .delaunay_tets(|| (), || (), || ());

    //    mesh.to_separate_tets().write_obj("assets/ybg_out.obj").unwrap();
    //}

    #[test]
    fn test_import() {
        use crate::mesh2::ComboMesh2;
        let mesh = ComboMesh2::read_obj("assets/ybg.obj", || Point3::origin(), || (), || ()).unwrap();
        let mesh = recover_edges(mesh.clone().delaunay_tets(|| (), || (), || ()),
            &mesh);

        mesh.to_separate_tets().write_obj("assets/ybg_out.obj").unwrap();
    }
}
