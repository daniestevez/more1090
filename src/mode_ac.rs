//! Mode-A/C routines.

use anyhow::Result;

/// Mode-A/C reply message.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ModeACReply {
    // ABCD data. Only 12 bits.
    data: u16,
    spi: bool,
}

impl ModeACReply {
    /// Creates a new Mode-A/C reply message.
    ///
    /// An error is returned if the data exceeds the 12-bit code range.
    pub fn new(data: u16, spi: bool) -> Result<ModeACReply> {
        anyhow::ensure!(data < 4096);
        Ok(ModeACReply { data, spi })
    }

    /// Creates a new Mode-A/C reply message from its A, B, C, D octal digits.
    ///
    /// An error is returned if any of the digits exceeds the 3-bit octal digit
    /// range.
    pub fn from_abcd(a: u8, b: u8, c: u8, d: u8, spi: bool) -> Result<ModeACReply> {
        anyhow::ensure!(a < 8);
        anyhow::ensure!(b < 8);
        anyhow::ensure!(c < 8);
        anyhow::ensure!(d < 8);
        let data = (u16::from(a) << 9) | (u16::from(b) << 6) | (u16::from(c) << 3) | u16::from(d);
        ModeACReply::new(data, spi)
    }

    /// Returns the 12-bit data of the reply message.
    pub fn data(&self) -> u16 {
        self.data
    }

    /// Returns the SPI flag of the reply message.
    pub fn spi(&self) -> bool {
        self.spi
    }

    /// Returns the A digit of the reply message.
    pub fn a(&self) -> u8 {
        ((self.data >> 9) & 7) as u8
    }

    /// Returns the B digit of the reply message.
    pub fn b(&self) -> u8 {
        ((self.data >> 6) & 7) as u8
    }

    /// Returns the C digit of the reply message.
    pub fn c(&self) -> u8 {
        ((self.data >> 3) & 7) as u8
    }

    /// Returns the D digit of the reply message.
    pub fn d(&self) -> u8 {
        (self.data & 7) as u8
    }

    /// Returns the reply message as a tuple of (A, B, C, D) digits.
    pub fn abcd(&self) -> (u8, u8, u8, u8) {
        (self.a(), self.b(), self.c(), self.d())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn mode_ac_new_accepts_valid(data in 0_u16..4096, spi: bool) {
            assert!(ModeACReply::new(data, spi).is_ok());
        }
    }

    proptest! {
        #[test]
        fn mode_ac_new_rejects_invalid(data in 4096_u16.., spi: bool) {
            assert!(ModeACReply::new(data, spi).is_err());
        }
    }

    #[test]
    fn mode_ac_1200() {
        let message = ModeACReply::new(0o1200, false).unwrap();
        assert_eq!(message.abcd(), (1, 2, 0, 0));
        assert!(!message.spi());
    }

    #[test]
    fn mode_ac_7136() {
        let message = ModeACReply::new(0o7136, true).unwrap();
        assert_eq!(message.abcd(), (7, 1, 3, 6));
        assert!(message.spi());
    }

    proptest! {
        #[test]
        fn mode_ac_abcd(a in 0u8..8, b in 0u8..8, c in 0u8..8, d in 0u8..8, spi: bool) {
            let message = ModeACReply::from_abcd(a, b, c, d, spi).unwrap();
            assert_eq!(message.abcd(), (a, b, c, d));
            assert_eq!(message.spi(), spi);
        }
    }
}
