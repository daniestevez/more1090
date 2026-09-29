//! Input/output routines.

use num::complex::Complex32;
use std::io::{BufWriter, IntoInnerError, Result, Write};

/// [`Complex32`] batch writer.
///
/// This trait models an object that is capable of writing [`Complex32`]
/// elements in batches. Examples of implementors of this trait are a [`Vec`] to
/// which the elements are appended and a [`Write`] object to which the data is
/// written in `cf32_le` format in batches.
pub trait WriteComplex32 {
    /// Writes samples to the object.
    fn write_complex32(&mut self, samples: &[Complex32]) -> Result<()>;
}

/// cf32_le writer.
///
/// This struct wraps an object that implements [`Write`] and provides a
/// [`WriteComplex32`] implementation that writes in cf32_le format to the
/// `Write` object. Since the [`WriteComplex32`] implementation calls
/// `write_all` for each `f32` scalar, internally a [`BufWriter`] is used for
/// good performance.
///
/// As in the case of a [`BufWriter`], it is critical to call
/// [`flush`](CF32LEWriter) before the `CF32LEWriter<W>` is dropped.
#[derive(Debug)]
pub struct CF32LEWriter<W: Write> {
    inner: BufWriter<W>,
}

impl<W: Write> CF32LEWriter<W> {
    /// Creates a new cf32_le writer.
    ///
    /// This function creates a new cf32_le writer by wrapping `inner` with a
    /// [`BufWriter`].
    pub fn new(inner: W) -> CF32LEWriter<W> {
        CF32LEWriter {
            inner: BufWriter::new(inner),
        }
    }

    /// Flushes the underlying [`BufWriter`].
    pub fn flush(&mut self) -> Result<()> {
        self.inner.flush()
    }

    /// Returns the underlying writer.
    ///
    /// Buffered data is writen out before returning the writer.
    pub fn into_inner(self) -> std::result::Result<W, IntoInnerError<BufWriter<W>>> {
        self.inner.into_inner()
    }
}

impl WriteComplex32 for Vec<Complex32> {
    fn write_complex32(&mut self, samples: &[Complex32]) -> Result<()> {
        self.extend_from_slice(samples);
        Ok(())
    }
}

impl<W: Write> WriteComplex32 for CF32LEWriter<W> {
    fn write_complex32(&mut self, samples: &[Complex32]) -> Result<()> {
        for z in samples {
            self.inner.write_all(&z.re.to_le_bytes())?;
            self.inner.write_all(&z.im.to_le_bytes())?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn vec_write_complex32() {
        let mut v = Vec::<Complex32>::new();
        let a = [
            Complex32::new(1.0, -3.5),
            Complex32::new(7.2, 4.3),
            Complex32::new(0.0, 0.0),
        ];
        let b = [Complex32::new(-100.27, 0.0), Complex32::new(0.0, -0.0072)];
        v.write_complex32(&a).unwrap();
        v.write_complex32(&b).unwrap();
        assert_eq!(v.len(), 5);
        assert_eq!(v[..3], a);
        assert_eq!(v[3..], b);
    }

    #[test]
    fn vec_cf32lewriter() {
        let a = [
            Complex32::new(1.0, -0.5),
            Complex32::new(0.0, 0.0),
            Complex32::new(0.1, -0.0037),
        ];
        let mut writer = CF32LEWriter::new(Vec::new());
        writer.write_complex32(&a).unwrap();
        let v = writer.into_inner().unwrap();
        assert_eq!(v.len(), a.len() * 8);
        let mut w = Vec::new();
        for chunk in v.as_chunks::<8>().0 {
            let re = f32::from_le_bytes(chunk[..4].try_into().unwrap());
            let im = f32::from_le_bytes(chunk[4..].try_into().unwrap());
            w.push(Complex32::new(re, im));
        }
        assert_eq!(w, a);
    }
}
