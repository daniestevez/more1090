//! Mode-S routines.

use crate::constants::mode_s::{
    MODE_S_LONG_BITS, MODE_S_LONG_BYTES, MODE_S_SHORT_BITS, MODE_S_SHORT_BYTES,
};
use std::sync::OnceLock;

/// Mode-S message.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum ModeSMessage {
    /// Short (56 bit) Mode-S message, packed as 8 bits per byte.
    Short([u8; MODE_S_SHORT_BYTES]),
    /// Long (112 bit) Mode-S message, packed as 8 bits per byte.
    Long([u8; MODE_S_LONG_BYTES]),
}

impl ModeSMessage {
    /// Returns the message as a slice of bytes.
    pub fn as_slice(&self) -> &[u8] {
        match self {
            ModeSMessage::Short(a) => &a[..],
            ModeSMessage::Long(a) => &a[..],
        }
    }

    /// Returns the message as a mutable slice of bytes.
    pub fn as_slice_mut(&mut self) -> &mut [u8] {
        match self {
            ModeSMessage::Short(a) => &mut a[..],
            ModeSMessage::Long(a) => &mut a[..],
        }
    }

    /// Returns the number of bits in the message.
    pub fn num_bits(&self) -> usize {
        match self {
            ModeSMessage::Short(_) => MODE_S_SHORT_BITS,
            ModeSMessage::Long(_) => MODE_S_LONG_BITS,
        }
    }
}

/// Mode-S CRC-24 algorithm.
pub const CRC24_ALGORITHM: crc::Algorithm<u32> = crc::Algorithm {
    width: 24,
    poly: 0xFFF409,
    init: 0,
    refin: false,
    refout: false,
    xorout: 0,
    check: 0x54268,
    residue: 0,
};

/// Mode-S CRC-24.
pub const CRC24: crc::Crc<u32> = crc::Crc::<u32>::new(&CRC24_ALGORITHM);

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
struct SingleBitErrorTable {
    // The CRC-24 is in the 24 MSBs, and the error position in the 8 LSBs.
    crc_and_error_position: [u32; MODE_S_LONG_BITS],
}

impl Default for SingleBitErrorTable {
    fn default() -> SingleBitErrorTable {
        SingleBitErrorTable::new()
    }
}

impl SingleBitErrorTable {
    fn new() -> SingleBitErrorTable {
        let mut entries = (0..MODE_S_LONG_BITS)
            .map(|j| {
                let mut message = [0; MODE_S_LONG_BYTES];
                message[j / 8] = 1 << (7 - (j % 8));
                let crc = CRC24.checksum(&message);
                (crc << 8) | u32::try_from(j).unwrap()
            })
            .collect::<Vec<_>>();
        entries.sort_unstable();
        SingleBitErrorTable {
            crc_and_error_position: entries.try_into().unwrap(),
        }
    }

    fn lookup(&self, crc: u32) -> Option<u8> {
        let idx = self
            .crc_and_error_position
            .binary_search_by(|x| (x >> 8).cmp(&crc))
            .ok()?;
        Some(u8::try_from(self.crc_and_error_position[idx] & 0xff).unwrap())
    }
}

/// Looks up a single-bit error position by using the CRC-24.
///
/// If the CRC corresponds to the CRC of an Mode-S long message with a single
/// bit error, this function returns the location of the bit error. Otherwise
/// the function returns `None`.
pub fn single_bit_error_from_crc24(crc: u32) -> Option<u8> {
    static TABLE: OnceLock<SingleBitErrorTable> = OnceLock::new();
    TABLE.get_or_init(SingleBitErrorTable::new).lookup(crc)
}

const NUM_DOUBLE_BIT_ERROR_PATTERNS: usize = MODE_S_LONG_BITS * (MODE_S_LONG_BITS - 1) / 2;

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
struct DoubleBitErrorTable {
    // The CRC-24 is in the 24 MSBs, and the first error position in the 8 LSBs.
    crc_and_first_error_position: [u32; NUM_DOUBLE_BIT_ERROR_PATTERNS],
    second_error_position: [u8; NUM_DOUBLE_BIT_ERROR_PATTERNS],
}

impl Default for DoubleBitErrorTable {
    fn default() -> DoubleBitErrorTable {
        DoubleBitErrorTable::new()
    }
}

impl DoubleBitErrorTable {
    fn new() -> DoubleBitErrorTable {
        let mut entries = (0..MODE_S_LONG_BITS)
            .flat_map(|j| {
                (j + 1..MODE_S_LONG_BITS).map(move |k| {
                    let mut message = [0; MODE_S_LONG_BYTES];
                    message[j / 8] = 1 << (7 - (j % 8));
                    message[k / 8] ^= 1 << (7 - (k % 8));
                    let crc = CRC24.checksum(&message);
                    let a = (crc << 8) | u32::try_from(j).unwrap();
                    (a, u8::try_from(k).unwrap())
                })
            })
            .collect::<Vec<_>>();
        entries.sort_unstable_by_key(|(a, _)| *a);
        DoubleBitErrorTable {
            crc_and_first_error_position: entries
                .iter()
                .map(|(a, _)| *a)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
            second_error_position: entries
                .iter()
                .map(|(_, b)| *b)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
        }
    }

    fn lookup(&self, crc: u32) -> Option<(u8, u8)> {
        let idx = self
            .crc_and_first_error_position
            .binary_search_by(|x| (x >> 8).cmp(&crc))
            .ok()?;
        let first_pos = u8::try_from(self.crc_and_first_error_position[idx] & 0xff).unwrap();
        let second_pos = self.second_error_position[idx];
        Some((first_pos, second_pos))
    }
}

/// Looks up a double-bit error position by using the CRC-24.
///
/// If the CRC corresponds to the CRC of a Mode-S long message with exactly two
/// bit errors, this function returns the location of the two bit errors as
/// tuple. Otherwise the function returns `None`.
pub fn double_bit_error_from_crc24(crc: u32) -> Option<(u8, u8)> {
    static TABLE: OnceLock<DoubleBitErrorTable> = OnceLock::new();
    TABLE.get_or_init(DoubleBitErrorTable::new).lookup(crc)
}

/// Formats a Mode-S message as an AVR message.
///
/// An AVR message is a string of the form `*8D3C5EE69901BD9540078D37335F;`
/// (without a terminating new line).
pub fn format_avr(message: &[u8]) -> String {
    format!("*{:X};", const_hex::display(message))
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn crc24() {
        // Example from https://mode-s.org/1090mhz/content/ads-b/8-error-control.html
        assert_eq!(
            CRC24.checksum(&const_hex::decode("8D406B902015A678D4D220").unwrap()),
            0xAA4BDA
        );
        assert_eq!(
            CRC24.checksum(&const_hex::decode("8D406B902015A678D4D220AA4BDA").unwrap()),
            0
        );
    }

    /// Checks that, for single-bit error patterns in a Mode-S long message, the
    /// CRC-24 determines the bit error position univocally.
    #[test]
    fn crc24_single_bit_error_univocal() {
        let table = SingleBitErrorTable::new();
        for (a, b) in table
            .crc_and_error_position
            .iter()
            .zip(table.crc_and_error_position[1..].iter())
        {
            assert!(a >> 8 != b >> 8);
            assert!(a >> 8 != 0);
            assert!(b >> 8 != 0);
        }
    }

    #[test]
    fn single_bit_error() {
        for j in 0..MODE_S_LONG_BITS {
            let mut message = [0; MODE_S_LONG_BYTES];
            message[j / 8] = 1 << (7 - (j % 8));
            let crc = CRC24.checksum(&message);
            assert_eq!(
                single_bit_error_from_crc24(crc).unwrap(),
                u8::try_from(j).unwrap()
            );
        }
    }

    /// Checks that, for single-bit or double error patterns in a Mode-S long
    /// message, the CRC-24 determines the bit error position univocally.
    #[test]
    fn crc24_single_or_double_bit_error_univocal() {
        let single_table = SingleBitErrorTable::new();
        let double_table = DoubleBitErrorTable::new();
        let mut crcs = single_table
            .crc_and_error_position
            .iter()
            .chain(double_table.crc_and_first_error_position.iter())
            .map(|x| x >> 8)
            .collect::<Vec<_>>();
        crcs.sort_unstable();
        for (&a, &b) in crcs.iter().zip(crcs[1..].iter()) {
            assert!(a != b);
            assert!(a != 0);
            assert!(b != 0);
        }
    }

    #[test]
    fn double_bit_error() {
        for j in 0..MODE_S_LONG_BITS {
            for k in j + 1..MODE_S_LONG_BITS {
                let mut message = [0; MODE_S_LONG_BYTES];
                message[j / 8] = 1 << (7 - (j % 8));
                message[k / 8] ^= 1 << (7 - (k % 8));
                let crc = CRC24.checksum(&message);
                assert_eq!(
                    double_bit_error_from_crc24(crc).unwrap(),
                    (u8::try_from(j).unwrap(), u8::try_from(k).unwrap())
                );
            }
        }
    }

    #[test]
    fn format_avr() {
        let message = [
            0x8D, 0x3C, 0x5E, 0xE6, 0x99, 0x01, 0xBD, 0x95, 0x40, 0x07, 0x8D, 0x37, 0x33, 0x5F,
        ];
        let expected = "*8D3C5EE69901BD9540078D37335F;";
        assert_eq!(super::format_avr(&message), expected);
    }
}
