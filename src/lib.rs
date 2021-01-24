pub mod edge;
mod iter;
pub mod mesh0;
pub mod mesh1;
pub mod mesh2;
pub mod mesh3;
pub mod tet;
pub mod tri;
pub mod vertex;

use nalgebra::Point;
type PtN<D> = Point<f64, D>;

pub use mesh0::{ComboMesh0, Mesh0, Mesh02, Mesh03};
pub use mesh1::{ComboMesh1, Mesh1, Mesh12, Mesh13};
pub use mesh2::{ComboMesh2, Mesh2, Mesh22, Mesh23};
pub use mesh3::{ComboMesh3, Mesh3, Mesh32, Mesh33};

#[macro_export]
macro_rules! impl_integer_id {
    ($type:tt($int:ty)) => {
        impl idmap::IntegerId for $type {
            fn id(&self) -> u64 {
                self.0 as u64
            }

            fn id32(&self) -> u32 {
                self.0 as u32
            }

            fn from_id(id: u64) -> Self {
                $type(id as $int)
            }
        }
    };
}
