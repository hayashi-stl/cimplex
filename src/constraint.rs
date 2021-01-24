use crate::mesh_3::internal::{ManifoldTet, Tet};
use crate::tet::internal::Tet as TetIntr;

/// The "manifold" constraint forces (d - 1)-simplexes
/// to be attached to at most 1 d-simplex.
/// This is not the full manifold constraint, even if boundary is allowed.
pub trait ManifoldFlag<T> {
    type Tet: TetIntr<T = T>;
}

/// No manifold-like restriction
pub struct NonManifold;

/// (d - 1)-simplexes must be attached to at most 1 d-simplex.
pub struct Manifold;

impl<T> ManifoldFlag<T> for NonManifold {
    #[doc(hidden)]
    type Tet = Tet<T>;
}

impl<T> ManifoldFlag<T> for Manifold {
    #[doc(hidden)]
    type Tet = ManifoldTet<T>;
}
