//! SigMF format.
//!
//! This module contains utilities to deal with the SigMF format. It is not a
//! complete implementation. Just the minimum that is needed here.

use crate::{
    constants::dsp::{CARRIER_FREQUENCY, MODE_S_LONG_SAMPLES},
    receiver::ADSBPacket,
};
use anyhow::{Context, Result};
use num::complex::Complex32;
use serde_json::{Map, Value};
use std::{
    ffi::OsStr,
    fs::File,
    path::{Path, PathBuf},
};

const SIGMF_VERSION: &str = "1.2.6";

/// Checks if a path has a SigMF extension.
///
/// This function returns `true` if the path extension is `.sigmf-data` or
/// `.sigmf-meta`.
pub fn has_sigmf_extension<P: AsRef<Path>>(path: P) -> bool {
    let path = path.as_ref();
    path.extension() == Some(OsStr::new("sigmf-data"))
        || path.extension() == Some(OsStr::new("sigmf-meta"))
}

/// Returns the corresponding .sigmf-data path for a path.
///
/// This function returns the path of the `.sigmf-data` file corresponding to a
/// given path. The path must have a SigMF extension, or an error is returned.
pub fn data_path<P: AsRef<Path>>(path: P) -> Result<PathBuf> {
    anyhow::ensure!(
        has_sigmf_extension(&path),
        "{} does not have a SigMF extension",
        path.as_ref().display()
    );
    Ok(path.as_ref().with_extension("sigmf-data"))
}

/// Returns the corresponding .sigmf-meta path for a path.
///
/// This function returns the path of the `.sigmf-meta` file corresponding to a
/// given path. The path must have a SigMF extension, or an error is returned.
pub fn meta_path<P: AsRef<Path>>(path: P) -> Result<PathBuf> {
    anyhow::ensure!(
        has_sigmf_extension(&path),
        "{} does not have a SigMF extension",
        path.as_ref().display()
    );
    Ok(path.as_ref().with_extension("sigmf-meta"))
}

/// SigMF meta.
///
/// This struct contains SigMF metadata.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SigMFMeta {
    global: Map<String, Value>,
    captures: Vec<Map<String, Value>>,
    annotations: Vec<Annotation>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct Annotation {
    sample_start: u64,
    sample_count: u64,
    metadata: Map<String, Value>,
}

fn get_section_from_meta<'a>(meta: &'a Map<String, Value>, key: &str) -> Result<&'a Value> {
    meta.get(key)
        .ok_or_else(|| anyhow::anyhow!("key {key} missing from sigmf-meta"))
}

fn to_vec_of_maps(v: &[Value]) -> Result<Vec<Map<String, Value>>> {
    let mut ret = Vec::with_capacity(v.len());
    for a in v.iter() {
        let Value::Object(a) = a else {
            anyhow::bail!("{} is not a JSON object", a);
        };
        ret.push(a.clone());
    }
    Ok(ret)
}

fn to_vec_of_annotations(v: &[Value]) -> Result<Vec<Annotation>> {
    let mut ret = Vec::with_capacity(v.len());
    for a in v.iter().cloned() {
        let Value::Object(mut a) = a else {
            anyhow::bail!("{} is not a JSON object", a);
        };
        let Value::Number(sample_start) = a
            .remove(keys::SAMPLE_START_KEY)
            .ok_or_else(|| anyhow::anyhow!("annotation is missing {}", keys::SAMPLE_START_KEY))?
        else {
            anyhow::bail!("{} is not a number", keys::SAMPLE_START_KEY);
        };
        let Value::Number(sample_count) =
            a.remove(keys::SAMPLE_COUNT_KEY).unwrap_or_else(|| 0.into())
        else {
            anyhow::bail!("{} is not a number", keys::SAMPLE_COUNT_KEY);
        };
        ret.push(Annotation {
            sample_start: try_as_u64(&sample_start)?,
            sample_count: try_as_u64(&sample_count)?,
            metadata: a,
        });
    }
    Ok(ret)
}

fn get_capture_key<'a>(capture: &'a Map<String, Value>, key: &str) -> Result<&'a Value> {
    capture.get(key).ok_or_else(|| {
        anyhow::anyhow!(
            "key {key} missing from capture {}",
            serde_json::to_string(capture).unwrap_or_else(|_| "<invalid JSON>".to_string()),
        )
    })
}

fn try_as_f64(number: &serde_json::Number) -> Result<f64> {
    number
        .as_f64()
        .ok_or_else(|| anyhow::anyhow!("cannot convert {number} to f64"))
}

fn try_as_u64(number: &serde_json::Number) -> Result<u64> {
    number
        .as_u64()
        .ok_or_else(|| anyhow::anyhow!("cannot convert {number} to u64"))
}

fn try_as_usize(number: &serde_json::Number) -> Result<usize> {
    let number = try_as_u64(number)?;
    number
        .try_into()
        .map_err(|err| anyhow::anyhow!("cannot convert {number} to usize: {err}"))
}

impl SigMFMeta {
    /// Reads and parses a SigMF meta file.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<SigMFMeta> {
        let path = path.as_ref();
        let f = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
        let Value::Object(meta) =
            serde_json::from_reader(f).context("failed to parse sigmf-meta")?
        else {
            anyhow::bail!("sigmf-meta is not a JSON object");
        };
        let Value::Object(global) = get_section_from_meta(&meta, keys::GLOBAL_KEY)? else {
            anyhow::bail!("{} is not a JSON object", keys::GLOBAL_KEY);
        };
        let Value::Array(captures) = get_section_from_meta(&meta, keys::CAPTURES_KEY)? else {
            anyhow::bail!("{} is not a JSON array", keys::CAPTURES_KEY);
        };
        let Value::Array(annotations) = get_section_from_meta(&meta, keys::ANNOTATIONS_KEY)? else {
            anyhow::bail!("{} is not a JSON array", keys::ANNOTATIONS_KEY);
        };
        let meta = SigMFMeta {
            global: global.clone(),
            captures: to_vec_of_maps(captures)?,
            annotations: to_vec_of_annotations(annotations)?,
        };
        Ok(meta)
    }

    fn map(&self) -> Map<String, Value> {
        let mut meta = Map::new();
        meta.insert(keys::GLOBAL_KEY.to_string(), self.global.clone().into());
        meta.insert(
            keys::CAPTURES_KEY.to_string(),
            self.captures
                .iter()
                .cloned()
                .map(|v| v.into())
                .collect::<Vec<Value>>()
                .into(),
        );
        let mut annotations = self.annotations.clone();
        annotations.sort_unstable_by_key(|annotation| annotation.sample_start);
        meta.insert(
            keys::ANNOTATIONS_KEY.to_string(),
            annotations
                .into_iter()
                .map(|mut annotation| {
                    annotation.metadata.insert(
                        keys::SAMPLE_START_KEY.to_string(),
                        annotation.sample_start.into(),
                    );
                    annotation.metadata.insert(
                        keys::SAMPLE_COUNT_KEY.to_string(),
                        annotation.sample_count.into(),
                    );
                    annotation.metadata.into()
                })
                .collect::<Vec<Value>>()
                .into(),
        );
        meta
    }

    /// Formats and writes the SigMF meta to a file.
    pub fn to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();
        let f =
            File::create(path).with_context(|| format!("failed to create {}", path.display()))?;
        serde_json::to_writer_pretty(f, &self.map())?;
        Ok(())
    }

    /// Formats and writes the SigMF meta to a string.
    pub fn to_string(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(&self.map())?)
    }

    /// Returns the datatype.
    pub fn datatype(&self) -> Result<&str> {
        let Value::String(datatype) = self.get_global_key(keys::DATATYPE_KEY)? else {
            anyhow::bail!("{} is not a string", keys::DATATYPE_KEY);
        };
        Ok(datatype)
    }

    /// Returns the sample rate.
    pub fn sample_rate(&self) -> Result<f64> {
        let Value::Number(sample_rate) = self.get_global_key(keys::SAMPLE_RATE_KEY)? else {
            anyhow::bail!("{} is not a number", keys::SAMPLE_RATE_KEY);
        };
        try_as_f64(sample_rate)
    }

    /// Returns the RF frequency.
    ///
    /// This function checks that all the captures have the same
    /// `"core:frequency"` and if so it returns that frequency. Otherwise it returns an error.
    pub fn frequency(&self) -> Result<f64> {
        let mut freq = None;
        for capture in self.captures.iter() {
            let f = get_capture_key(capture, keys::FREQUENCY_KEY)?;
            let Value::Number(f) = f else {
                anyhow::bail!("{f} is not a number");
            };
            let f = try_as_f64(f)?;
            if let Some(freq) = freq
                && f != freq
            {
                anyhow::bail!("captures contain different frequencies");
            }
            freq = Some(f);
        }
        freq.ok_or_else(|| anyhow::anyhow!("no captures"))
    }

    /// Returns the number of channels.
    ///
    /// If "core:num_channels" is missing, this function returns 1, which is the
    /// implied default.
    pub fn num_channels(&self) -> Result<usize> {
        let Ok(num_channels) = self.get_global_key(keys::NUM_CHANNELS_KEY) else {
            return Ok(1);
        };
        let Value::Number(num_channels) = num_channels else {
            anyhow::bail!("{} is not a number", keys::NUM_CHANNELS_KEY);
        };
        try_as_usize(num_channels)
    }

    /// Returns a key from the global section.
    pub fn get_global_key(&self, key: &str) -> Result<&Value> {
        self.global
            .get(key)
            .ok_or_else(|| anyhow::anyhow!("global key {key} missing from sigmf-meta"))
    }

    /// Clears all the annotations.
    pub fn clear_annotations(&mut self) {
        self.annotations.clear();
    }

    /// Clears all the annotations made by a given generator.
    pub fn clear_annotations_generator(&mut self, generator: &str) {
        self.annotations.retain(|annotation| {
            annotation
                .metadata
                .get(keys::GENERATOR_KEY)
                .map(|g| g == generator)
                .unwrap_or(true)
        });
    }

    /// Adds an annotation.
    pub fn add_annotation(
        &mut self,
        sample_start: u64,
        sample_count: u64,
        metadata: Map<String, Value>,
    ) {
        for key in [keys::SAMPLE_START_KEY, keys::SAMPLE_COUNT_KEY] {
            if metadata.contains_key(key) {
                panic!("{key} already present in metadata");
            }
        }
        self.annotations.push(Annotation {
            sample_start,
            sample_count,
            metadata,
        });
    }

    /// Adds an annotation corresponding to an ADS-B packet.
    pub fn add_adsb_packet_annotation(&mut self, packet: &ADSBPacket, label: &str) {
        let metadata = adsb_packet_metadata(packet, label);
        let sample_start = packet
            .decode
            .as_ref()
            .map(|decode| decode.sample_index)
            .unwrap_or(packet.detection.sample_index);
        let sample_count = u64::try_from(MODE_S_LONG_SAMPLES).unwrap();
        self.add_annotation(sample_start, sample_count, metadata);
    }
}

/// Returns an ADS-B packet formatted as SigMF metadata.
pub fn adsb_packet_metadata(packet: &ADSBPacket, label: &str) -> Map<String, Value> {
    let mut map = Map::new();
    const ANNOTATION_BANDWIDTH: f64 = 4e6;
    let f_center = CARRIER_FREQUENCY + packet.detection.carrier_frequency_offset;
    let f_high = f_center + 0.5 * ANNOTATION_BANDWIDTH;
    let f_low = f_center - 0.5 * ANNOTATION_BANDWIDTH;
    map.insert(keys::FREQ_UPPER_KEY.to_string(), f_high.into());
    map.insert(keys::FREQ_LOWER_KEY.to_string(), f_low.into());
    if !label.is_empty() {
        map.insert(keys::LABEL_KEY.to_string(), label.into());
    }
    map.insert(keys::GENERATOR_KEY.to_string(), GENERATOR.into());
    map.insert(
        keys::CARRIER_POWER_KEY.to_string(),
        packet.detection.carrier_power.into(),
    );
    map.insert(
        keys::FLOOR_POWER_KEY.to_string(),
        packet.detection.floor_power.into(),
    );
    let mut comment = String::new();
    if let Some(decode) = &packet.decode {
        let message_hex = const_hex::encode(decode.message);
        comment.push_str(&message_hex);
        map.insert(keys::MODE_S_HEX_KEY.to_string(), message_hex.into());
        let bit_errors = decode.corrected_errors.num_errors();
        comment.push_str(&format!("\nBit errors: {bit_errors}"));
        map.insert(keys::BIT_ERRORS_KEY.to_string(), bit_errors.into());
        comment.push_str(&format!("\nDemodulation: {}", decode.demodulation));
        map.insert(
            keys::DEMODULATION_KEY.to_string(),
            decode.demodulation.to_string().into(),
        );
        map.insert(
            keys::SAMPLE_START_FRACTIONAL_KEY.to_string(),
            decode.sample_index_frac.into(),
        );
        comment.push_str(&format!("\nCN0: {:.2} dB·Hz", decode.cn0_db));
        map.insert(keys::CN0_KEY.to_string(), decode.cn0_db.into());
    }
    if !comment.is_empty() {
        map.insert(keys::COMMENT_KEY.to_string(), comment.into());
    }
    map
}

/// Creates a [`SigMFMeta`] object for the simulator output.
pub fn simulation_meta() -> SigMFMeta {
    let mut global = Map::new();
    global.insert(keys::DATATYPE_KEY.to_string(), "cf32_le".into());
    global.insert(keys::NUM_CHANNELS_KEY.to_string(), 1.into());
    global.insert(keys::OFFSET_KEY.to_string(), 0.into());
    global.insert(
        keys::SAMPLE_RATE_KEY.to_string(),
        crate::constants::dsp::SAMPLE_RATE_F64.into(),
    );
    global.insert(keys::VERSION_KEY.to_string(), SIGMF_VERSION.into());
    global.insert(
        keys::DESCRIPTION_KEY.to_string(),
        "more1090-simulator scenario".into(),
    );
    global.insert(keys::RECORDER_KEY.to_string(), "more1090-simulator".into());
    let mut captures = Vec::new();
    let mut capture = Map::new();
    capture.insert(
        keys::FREQUENCY_KEY.to_string(),
        crate::constants::dsp::CARRIER_FREQUENCY.into(),
    );
    capture.insert(keys::SAMPLE_START_KEY.to_string(), 0.into());
    captures.push(capture);
    let annotations = Vec::new();
    SigMFMeta {
        global,
        captures,
        annotations,
    }
}

fn ci16le_to_complex32(bytes: &[u8; 4]) -> Complex32 {
    let re = i16::from_le_bytes(bytes[..2].try_into().unwrap());
    let im = i16::from_le_bytes(bytes[2..].try_into().unwrap());
    const SCALE: f32 = (1 << 15) as f32;
    Complex32::new(re as f32 / SCALE, im as f32 / SCALE)
}

fn cf32le_to_complex32(bytes: &[u8; 8]) -> Complex32 {
    let re = f32::from_le_bytes(bytes[..4].try_into().unwrap());
    let im = f32::from_le_bytes(bytes[4..].try_into().unwrap());
    Complex32::new(re, im)
}

macro_rules! data_reader {
    ($reader_name:ident, $iq_sample_size:expr, $convert:expr, $doc:expr) => {
        #[derive(Debug, Clone)]
        #[doc = $doc]
        pub struct $reader_name<R> {
            byte_buffer: crate::A<[u8; crate::CHUNK_LEN * $iq_sample_size]>,
            iq_buffer: crate::A<[Complex32; crate::CHUNK_LEN]>,
            to_read: u64,
            read: R,
        }

        impl<R> $reader_name<R> {
            /// Creates a new .sigmf-data reader.
            ///
            /// The `read` argument is the object to read from. The `to_read`
            /// argument indicates the number of bytes that we expect to read
            /// from it.
            ///
            /// This function returns an error if `to_read` is not a multiple of
            /// the IQ sample size.
            pub fn new(read: R, to_read: u64) -> Result<Self> {
                anyhow::ensure!(
                    to_read.is_multiple_of($iq_sample_size),
                    "data length {} is not a multiple of the sample size {}",
                    to_read,
                    $iq_sample_size
                );
                Ok(Self {
                    byte_buffer: aligned::Aligned([0_u8; _]),
                    iq_buffer: aligned::Aligned([Complex32::default(); _]),
                    to_read,
                    read,
                })
            }
        }

        impl<R: std::io::Read> $reader_name<R> {
            /// Reads a new chunk of IQ data.
            ///
            /// This function reads a chunk of IQ data from the reader, converts
            /// it to an array of [`Complex32`] data, and returns a reference to
            /// that array. `None` is returned if we have reached the end of the
            /// reader (as indicated by the `to_read` argument passed to
            /// [`new`](Self::new)).
            pub fn read(&mut self) -> Option<Result<&[Complex32; crate::CHUNK_LEN]>> {
                let chunk_bytes = u64::try_from(crate::CHUNK_LEN * $iq_sample_size).unwrap();
                if self.to_read < chunk_bytes {
                    return None;
                }
                if let Err(err) = self.read.read_exact(&mut self.byte_buffer[..]) {
                    return Some(Err(err.into()));
                }
                self.to_read -= chunk_bytes;
                for (a, z) in self
                    .byte_buffer
                    .as_chunks::<$iq_sample_size>()
                    .0
                    .iter()
                    .zip(self.iq_buffer.iter_mut())
                {
                    *z = $convert(a);
                }
                Some(Ok(&self.iq_buffer))
            }
        }
    };
}

data_reader!(CI16LEReader, 4, ci16le_to_complex32, "ci16_le IQ reader.");

data_reader!(CF32LEReader, 8, cf32le_to_complex32, "cf32_le IQ reader.");

/// "core:generator" value used by this software.
pub const GENERATOR: &str = "more1090";

/// SigMF keys.
pub mod keys {
    /// "global" key.
    pub const GLOBAL_KEY: &str = "global";
    /// "annotations" key.
    pub const ANNOTATIONS_KEY: &str = "annotations";
    /// "captures" key.
    pub const CAPTURES_KEY: &str = "captures";
    /// "core:datatype" key.
    pub const DATATYPE_KEY: &str = "core:datatype";
    /// "core:num_channels" key.
    pub const NUM_CHANNELS_KEY: &str = "core:num_channels";
    /// "core:offset" key.
    pub const OFFSET_KEY: &str = "core:offset";
    /// "core:sample_rate" key.
    pub const SAMPLE_RATE_KEY: &str = "core:sample_rate";
    /// "core:sha512" key.
    pub const SHA512_KEY: &str = "core:sha512";
    /// "core:version" key.
    pub const VERSION_KEY: &str = "core:version";
    /// "core:description" key.
    pub const DESCRIPTION_KEY: &str = "core:description";
    /// "core:recorder" key.
    pub const RECORDER_KEY: &str = "core:recorder";
    /// "core:frequency" key.
    pub const FREQUENCY_KEY: &str = "core:frequency";
    /// "core:datetime" key.
    pub const DATETIME_KEY: &str = "core:datetime";
    /// "core:freq_lower_edge" key.
    pub const FREQ_LOWER_KEY: &str = "core:freq_lower_edge";
    /// "core:freq_upper_edge" key.
    pub const FREQ_UPPER_KEY: &str = "core:freq_upper_edge";
    /// "core:generator" key.
    pub const GENERATOR_KEY: &str = "core:generator";
    /// "core:label" key.
    pub const LABEL_KEY: &str = "core:label";
    /// "core:sample_start" key.
    pub const SAMPLE_START_KEY: &str = "core:sample_start";
    /// "core:sample_count" key.
    pub const SAMPLE_COUNT_KEY: &str = "core:sample_count";
    /// "core:comment" key.
    pub const COMMENT_KEY: &str = "core:comment";

    /// "more1090:carrier_power" key.
    pub const CARRIER_POWER_KEY: &str = "more1090:carrier_power";
    /// "more1090:floor_power" key.
    pub const FLOOR_POWER_KEY: &str = "more1090:floor_power";
    /// "more1090:mode_s_hex" key.
    pub const MODE_S_HEX_KEY: &str = "more1090:mode_s_hex";
    /// "more1090:bit_errors" key.
    pub const BIT_ERRORS_KEY: &str = "more1090:bit_errors";
    /// "more1090:demodulation" key.
    pub const DEMODULATION_KEY: &str = "more1090:demodulation";
    /// "more1090:sample_start_fractional" key.
    pub const SAMPLE_START_FRACTIONAL_KEY: &str = "more1090:sample_start_fractional";
    /// "more1090:cn0" key.
    pub const CN0_KEY: &str = "more1090:cn0";
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::receiver::{ADSBDecode, ADSBDetection, BitErrors, DemodulationType};
    use proptest::prelude::*;
    use std::io::Write;

    #[test]
    fn has_sigmf_extension() {
        assert!(super::has_sigmf_extension("file.sigmf-data"));
        assert!(super::has_sigmf_extension("file.sigmf-meta"));
        assert!(super::has_sigmf_extension("file.0.sigmf-data"));
        assert!(super::has_sigmf_extension("file.0.sigmf-meta"));
        assert!(super::has_sigmf_extension("file.0.1.sigmf-data"));
        assert!(super::has_sigmf_extension("file.0.1.sigmf-meta"));
        assert!(super::has_sigmf_extension("a/b/file.sigmf-data"));
        assert!(super::has_sigmf_extension("a/b/file.sigmf-meta"));
        assert!(super::has_sigmf_extension("a/b/file.0.sigmf-data"));
        assert!(super::has_sigmf_extension("a/b/file.0.sigmf-meta"));
        assert!(super::has_sigmf_extension("a/b/file.0.1.sigmf-data"));
        assert!(super::has_sigmf_extension("a/b/file.0.1.sigmf-meta"));
        assert!(!super::has_sigmf_extension("file.sigmf"));
        assert!(!super::has_sigmf_extension("file.tar.xz"));
        assert!(!super::has_sigmf_extension("a/b/file.sigmf"));
        assert!(!super::has_sigmf_extension("a/b/file.tar.xz"));
    }

    #[test]
    fn sigmf_data_path() {
        assert_eq!(
            data_path("file.sigmf-data").unwrap(),
            PathBuf::from("file.sigmf-data")
        );
        assert_eq!(
            data_path("file.sigmf-meta").unwrap(),
            PathBuf::from("file.sigmf-data")
        );
        assert_eq!(
            data_path("file.0.sigmf-data").unwrap(),
            PathBuf::from("file.0.sigmf-data")
        );
        assert_eq!(
            data_path("file.0.sigmf-meta").unwrap(),
            PathBuf::from("file.0.sigmf-data")
        );
        assert_eq!(
            data_path("a/b/file.sigmf-data").unwrap(),
            PathBuf::from("a/b/file.sigmf-data")
        );
        assert_eq!(
            data_path("a/b/file.sigmf-meta").unwrap(),
            PathBuf::from("a/b/file.sigmf-data")
        );
        assert_eq!(
            data_path("a/b/file.0.sigmf-data").unwrap(),
            PathBuf::from("a/b/file.0.sigmf-data")
        );
        assert_eq!(
            data_path("a/b/file.0.sigmf-meta").unwrap(),
            PathBuf::from("a/b/file.0.sigmf-data")
        );
        assert!(data_path("file.sigmf").is_err());
        assert!(data_path("file.tar.xz").is_err());
        assert!(data_path("a/b/file.sigmf").is_err());
        assert!(data_path("a/b/file.tar.xz").is_err());
    }

    #[test]
    fn sigmf_meta_path() {
        assert_eq!(
            meta_path("file.sigmf-data").unwrap(),
            PathBuf::from("file.sigmf-meta")
        );
        assert_eq!(
            meta_path("file.sigmf-meta").unwrap(),
            PathBuf::from("file.sigmf-meta")
        );
        assert_eq!(
            meta_path("file.0.sigmf-data").unwrap(),
            PathBuf::from("file.0.sigmf-meta")
        );
        assert_eq!(
            meta_path("file.0.sigmf-meta").unwrap(),
            PathBuf::from("file.0.sigmf-meta")
        );
        assert_eq!(
            meta_path("a/b/file.sigmf-data").unwrap(),
            PathBuf::from("a/b/file.sigmf-meta")
        );
        assert_eq!(
            meta_path("a/b/file.sigmf-meta").unwrap(),
            PathBuf::from("a/b/file.sigmf-meta")
        );
        assert_eq!(
            meta_path("a/b/file.0.sigmf-data").unwrap(),
            PathBuf::from("a/b/file.0.sigmf-meta")
        );
        assert_eq!(
            meta_path("a/b/file.0.sigmf-meta").unwrap(),
            PathBuf::from("a/b/file.0.sigmf-meta")
        );
        assert!(meta_path("file.sigmf").is_err());
        assert!(meta_path("file.tar.xz").is_err());
        assert!(meta_path("a/b/file.sigmf").is_err());
        assert!(meta_path("a/b/file.tar.xz").is_err());
    }

    const SAMPLE_SIGMF: &str = r#"{
 "annotations": [
    {
      "core:comment": "8da4c34cea3ca864013c081c17f0\nBit errors: 0\nDemodulation: coherent\nCN0: 93.83 dB·Hz",
      "core:freq_lower_edge": 1087986623.0263891,
      "core:freq_upper_edge": 1091986623.0263891,
      "core:generator": "more1090",
      "core:label": "ADS-B packet 0",
      "core:sample_count": 896,
      "core:sample_start": 8055,
      "more1090:bit_errors": 0,
      "more1090:carrier_power": 5174.94970703125,
      "more1090:cn0": 93.82881927490234,
      "more1090:demodulation": "coherent",
      "more1090:floor_power": 16086.732421875,
      "more1090:mode_s_hex": "8da4c34cea3ca864013c081c17f0",
      "more1090:sample_start_fractional": -0.2884511947631836
    }
 ],
 "captures": [
    {
      "core:datetime": "2026-08-27T08:36:43.1544015625Z",
      "core:frequency": 1090000000.0,
      "core:sample_start": 0
    }
  ],
  "global": {
    "core:datatype": "ci16_le",
    "core:num_channels": 1,
    "core:offset": 0,
    "core:sample_rate": 8000000.0,
    "core:sha512": "939dea8d509f276b706873a880ea93879c6b156bcea155f64f4301954a9c8f5256c8d462bb9146a5dfcbc8b3e60355bd6d388cec5fb02dc37a560ca3168effa0",
    "core:version": "1.2.6"
  }
}
"#;

    fn sample_sigmf_parsed() -> SigMFMeta {
        let mut global = Map::new();
        global.insert(keys::DATATYPE_KEY.to_string(), "ci16_le".into());
        global.insert(keys::NUM_CHANNELS_KEY.to_string(), 1.into());
        global.insert(keys::OFFSET_KEY.to_string(), 0.into());
        global.insert(keys::SAMPLE_RATE_KEY.to_string(), 8_000_000.0.into());
        let sha = "939dea8d509f276b706873a880ea93879c6b156bcea155f64f4301954a9c8f5256c8d462bb9146a5dfcbc8b3e60355bd6d388cec5fb02dc37a560ca3168effa0";
        global.insert(keys::SHA512_KEY.to_string(), sha.into());
        global.insert(keys::VERSION_KEY.to_string(), "1.2.6".into());
        let mut captures = Vec::new();
        let mut capture = Map::new();
        capture.insert(
            keys::DATETIME_KEY.to_string(),
            "2026-08-27T08:36:43.1544015625Z".into(),
        );
        capture.insert(keys::FREQUENCY_KEY.to_string(), 1_090_000_000.0.into());
        capture.insert(keys::SAMPLE_START_KEY.to_string(), 0.into());
        captures.push(capture);
        let mut annotations = Vec::new();
        let mut annotation = Annotation {
            sample_start: 8055,
            sample_count: 896,
            metadata: Map::new(),
        };
        annotation.metadata.insert(
            keys::COMMENT_KEY.to_string(),
            "8da4c34cea3ca864013c081c17f0\nBit errors: 0\nDemodulation: coherent\nCN0: 93.83 dB·Hz"
                .into(),
        );
        annotation
            .metadata
            .insert(keys::FREQ_LOWER_KEY.to_string(), 1087986623.0263891.into());
        annotation
            .metadata
            .insert(keys::FREQ_UPPER_KEY.to_string(), 1091986623.0263891.into());
        annotation
            .metadata
            .insert(keys::GENERATOR_KEY.to_string(), "more1090".into());
        annotation
            .metadata
            .insert(keys::LABEL_KEY.to_string(), "ADS-B packet 0".into());
        annotation
            .metadata
            .insert(keys::BIT_ERRORS_KEY.to_string(), 0.into());
        annotation
            .metadata
            .insert(keys::CARRIER_POWER_KEY.to_string(), 5174.94970703125.into());
        annotation
            .metadata
            .insert(keys::CN0_KEY.to_string(), 93.82881927490234.into());
        annotation
            .metadata
            .insert(keys::DEMODULATION_KEY.to_string(), "coherent".into());
        annotation
            .metadata
            .insert(keys::FLOOR_POWER_KEY.to_string(), 16086.732421875.into());
        annotation.metadata.insert(
            keys::MODE_S_HEX_KEY.to_string(),
            "8da4c34cea3ca864013c081c17f0".into(),
        );
        annotation.metadata.insert(
            keys::SAMPLE_START_FRACTIONAL_KEY.into(),
            (-0.2884511947631836).into(),
        );
        annotations.push(annotation);
        SigMFMeta {
            global,
            captures,
            annotations,
        }
    }

    #[test]
    fn sigmf_from_file() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        write!(f, "{}", SAMPLE_SIGMF).unwrap();
        f.flush().unwrap();
        let sigmf = SigMFMeta::from_file(f.path()).unwrap();
        assert_eq!(sigmf, sample_sigmf_parsed());
    }

    #[test]
    fn sigmf_to_file() {
        let f = tempfile::NamedTempFile::new().unwrap();
        sample_sigmf_parsed().to_file(f.path()).unwrap();
        let sigmf = SigMFMeta::from_file(f.path()).unwrap();
        assert_eq!(sigmf, sample_sigmf_parsed());
    }

    #[test]
    fn get_datatype() {
        assert_eq!(sample_sigmf_parsed().datatype().unwrap(), "ci16_le");
    }

    #[test]
    fn get_sample_rate() {
        assert_eq!(sample_sigmf_parsed().sample_rate().unwrap(), 8_000_000.0);
    }

    #[test]
    fn get_frequency() {
        assert_eq!(sample_sigmf_parsed().frequency().unwrap(), 1_090_000_000.0);
    }

    #[test]
    fn get_global_key() {
        assert_eq!(
            sample_sigmf_parsed()
                .get_global_key(keys::VERSION_KEY)
                .unwrap(),
            &Value::from("1.2.6")
        );
    }

    #[test]
    fn clear_annotations() {
        let mut sigmf = sample_sigmf_parsed();
        assert!(!sigmf.annotations.is_empty());
        sigmf.clear_annotations();
        assert!(sigmf.annotations.is_empty());
    }

    #[test]
    fn add_annotation() {
        let mut sigmf = sample_sigmf_parsed();
        sigmf.clear_annotations();
        let annotation = sample_sigmf_parsed().annotations[0].clone();
        sigmf.add_annotation(
            annotation.sample_start,
            annotation.sample_count,
            annotation.metadata,
        );
        assert_eq!(sigmf, sample_sigmf_parsed());
    }

    #[test]
    fn add_adsb_packet_annotation() {
        let mut sigmf = sample_sigmf_parsed();
        sigmf.clear_annotations();
        let packet = ADSBPacket {
            detection: ADSBDetection {
                sample_index: 8064,
                carrier_frequency_offset: -13376.97361087799,
                carrier_power: 5_174.949_7,
                floor_power: 16_086.732,
            },
            decode: Some(ADSBDecode {
                sample_index: 8055,
                sample_index_frac: -0.288_451_2,
                message: const_hex::decode_to_array("8da4c34cea3ca864013c081c17f0").unwrap(),
                corrected_errors: BitErrors::NoErrors,
                demodulation: DemodulationType::Coherent,
                cn0_db: 93.828_82,
            }),
        };
        let label = "ADS-B packet 0";
        sigmf.add_adsb_packet_annotation(&packet, label);
        assert_eq!(sigmf, sample_sigmf_parsed());
    }

    #[test]
    fn simulation_meta_json() {
        let sigmf = simulation_meta();
        let json = sigmf.to_string().unwrap();
        let expected = r#"{
  "annotations": [],
  "captures": [
    {
      "core:frequency": 1090000000.0,
      "core:sample_start": 0
    }
  ],
  "global": {
    "core:datatype": "cf32_le",
    "core:description": "more1090-simulator scenario",
    "core:num_channels": 1,
    "core:offset": 0,
    "core:recorder": "more1090-simulator",
    "core:sample_rate": 8000000.0,
    "core:version": "1.2.6"
  }
}"#;
        assert_eq!(json, expected);
    }

    proptest! {
        #[test]
        fn ci16_le_reader(mut data in proptest::collection::vec(any::<i16>(), 0..=8192)) {
            let mut v = Vec::new();
            if !data.len().is_multiple_of(2) {
                data.pop();
            }
            for d in data.iter() {
                v.extend_from_slice(&d.to_le_bytes());
            }
            let scalar_size = std::mem::size_of::<i16>();
            let to_read = u64::try_from(data.len() * scalar_size).unwrap();
            let mut reader = CI16LEReader::new(&v[..], to_read).unwrap();
            let mut chunks = Vec::new();
            while let Some(chunk) = reader.read() {
                let chunk = chunk.unwrap();
                chunks.extend_from_slice(&chunk[..]);
            }
            let expected = data.as_chunks::<2>()
                .0.
                iter()
                .map(|[a, b]| {
                    let scale = (1 << 15) as f32;
                    Complex32::new(*a as f32 / scale, *b as f32 / scale)
                })
                .take(data.len() / (crate::CHUNK_LEN * 2) * crate::CHUNK_LEN)
                .collect::<Vec<_>>();
            assert_eq!(chunks, expected);
        }
    }

    proptest! {
        #[test]
        fn cf32_le_reader(mut data in proptest::collection::vec(any::<f32>(), 0..=8192)) {
            let mut v = Vec::new();
            if !data.len().is_multiple_of(2) {
                data.pop();
            }
            for d in data.iter() {
                v.extend_from_slice(&d.to_le_bytes());
            }
            let scalar_size = std::mem::size_of::<f32>();
            let to_read = u64::try_from(data.len() * scalar_size).unwrap();
            let mut reader = CF32LEReader::new(&v[..], to_read).unwrap();
            let mut chunks = Vec::new();
            while let Some(chunk) = reader.read() {
                let chunk = chunk.unwrap();
                chunks.extend_from_slice(&chunk[..]);
            }
            let expected = data.as_chunks::<2>()
                .0.
                iter()
                .map(|[a, b]| Complex32::new(*a, *b))
                .take(data.len() / (crate::CHUNK_LEN * 2) * crate::CHUNK_LEN)
                .collect::<Vec<_>>();
            assert_eq!(chunks, expected);
        }
    }
}
