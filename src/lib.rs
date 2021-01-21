//! Cimplex (prononuced KIM-plex) is a simplicial complex library.

use std::convert::TryInto;

use nalgebra::{Vector1, Vector2, Vector3, VectorN};
type Vec1 = Vector1<f64>;
type Vec2 = Vector2<f64>;
type Vec3 = Vector3<f64>;
type VecN<D> = VectorN<f64, D>;

pub mod edge;
pub mod mesh_0;
pub mod mesh_1;
pub mod mesh_2;
pub mod mesh_3;
pub mod tri;
pub mod tet;
pub mod vertex;
mod iter;

pub use mesh_0::{ComboMesh0, Mesh0, Mesh02, Mesh03};
pub use mesh_1::{ComboMesh1, Mesh1, Mesh12, Mesh13};
pub use mesh_2::{ComboMesh2, Mesh2, Mesh22, Mesh23};
pub use mesh_3::{ComboMesh3, Mesh3, Mesh32, Mesh33};

/// Marker trait to aid converting
/// lists of vertices into simplex ids.
/// This trait exists to simplify the need for
/// the error type in `TryFrom` to be `Debug`,
/// avoiding complicating the bounds of functions.
pub trait IntoId<I> {
    /// Turns `self` into an id. Is allowed to panic.
    fn into_id(self) -> I;

    /// Turns `self` into an id without checking for inequality of the vertices.
    fn into_id_unchecked(self) -> I;
}

impl<I> IntoId<I> for I {
    fn into_id(self) -> I {
        self
    }

    fn into_id_unchecked(self) -> I {
        self
    }
}

#[macro_export]
macro_rules! impl_integer_id {
    ($type:tt) => {
        impl idmap::IntegerId for $type {
            fn id(&self) -> u64 {
                self.0
            }

            fn id32(&self) -> u32 {
                self.0 as u32
            }

            fn from_id(id: u64) -> Self {
                $type(id)
            }
        }
    };
}
