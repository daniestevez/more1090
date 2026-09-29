//! Reproducible random number generation.
//!
//! This module contains code for reproducible random number generation to be
//! used for simulation generation. The current implementation uses the
//! [`Xoshiro256PlusPlus`] reproducible portable generator from [`rand`].

use rand::{Rng, SeedableRng, rngs::Xoshiro256PlusPlus};

/// Creates an RNG from a seed.
///
/// This function returns an RNG initialized with a 64-bit seed. The RNG is
/// guaranteed to give reproducible results.
///
/// An `impl Rng` is returned to allow changing the underlying RNG
/// implementation in the future without having to change the API.
pub fn rng_from_seed(seed: u64) -> impl Rng {
    Xoshiro256PlusPlus::seed_from_u64(seed)
}

#[cfg(test)]
mod test {
    use super::*;
    use rand::RngExt;

    #[test]
    fn reproducibility() {
        let seed = 42;
        let mut rng = rng_from_seed(seed);
        let output: [u64; 16] = rng.random();
        let expected = [
            15021278609987233951,
            5881210131331364753,
            18149643915985481100,
            12933668939759105464,
            14637574242682825331,
            10848501901068131965,
            2312344417745909078,
            11162538943635311430,
            3831705504650218695,
            17217215411128672468,
            10321681451779520834,
            15680282660304795149,
            12543905331768826776,
            1282610804685344189,
            7435390023275438269,
            10071993084810367336,
        ];
        assert_eq!(output, expected);
    }
}
