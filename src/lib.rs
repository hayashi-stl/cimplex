//! Cimplex (prononuced KIM-plex) is a simplicial complex library.

use nalgebra::{Vector1, Vector2, Vector3, VectorN};
type Vec1 = Vector1<f64>;
type Vec2 = Vector2<f64>;
type Vec3 = Vector3<f64>;
type VecN<D> = VectorN<f64, D>;

pub mod mesh_0;
pub mod mesh_1;

pub use mesh_0::{ComboMesh0, Mesh0, Mesh02, Mesh03};
pub use mesh_1::ComboMesh1;

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
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
