//! Circular buffer.
//!
//! This module implements a circular buffer that is intended to hold a history
//! of previous samples. The circular buffer has a size that is a power of two
//! for efficient access, has a notion of absolute time, in terms of total
//! number of samples written to the buffer, and allows access to ranges past
//! samples still in the buffer as two disjoint slices.

use generic_array::{ArrayLength, GenericArray, sequence::GenericSequence, typenum::PowerOfTwo};

/// Circular buffer.
///
/// This circular buffer holds a history of the previous `N` samples of type `T`
/// written to the buffer. `N` should be a power of two. The buffer allows
/// reading previously written samples that are still in the buffer. Samples are
/// indexed by their absolute sample number, according to all writes done to the
/// buffer. A range of samples can be read as a pair of disjoint slices that
/// correspond to the slice of samples before the buffer wrap around and after
/// the wrap around (the second slice is empty if the range does not wrap around
/// the buffer).
#[derive(Clone, Eq, PartialEq, Hash)]
pub struct CircBuf<T, N: ArrayLength> {
    samples: Box<GenericArray<T, N>>,
    samples_written: u64,
}

impl<T: std::fmt::Debug, N: ArrayLength> std::fmt::Debug for CircBuf<T, N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        f.debug_struct("CircBuf")
            .field("samples", &self.samples)
            .field("samples_written", &self.samples_written)
            .finish()
    }
}

impl<T: Default, N: ArrayLength> Default for CircBuf<T, N> {
    /// Creates a new empty circular buffer.
    ///
    /// See [`CircBuf::new`].
    fn default() -> Self {
        CircBuf {
            samples: Box::new(GenericArray::default()),
            samples_written: 0,
        }
    }
}

impl<T: Default, N: ArrayLength> CircBuf<T, N> {
    /// Creates a new empty circular buffer.
    ///
    /// For simplicity, the contents of the buffer are initialized to
    /// `T::default()`, so it is required that `T` implements [`Default`]. These
    /// contents, are, however, not accessible.
    pub fn new() -> Self {
        Self::default()
    }
}

impl<T, N: ArrayLength> CircBuf<T, N> {
    /// Capacity of the circular buffer.
    ///
    /// The capacity is simply the number `N`.
    const CAPACITY: usize = N::USIZE;

    /// Returns the capacity of the circular buffer.
    ///
    /// The capacity is simply the number `N`.
    pub fn capacity(&self) -> usize {
        Self::CAPACITY
    }

    /// Returns the number of samples that have been written into the buffer.
    ///
    /// This is the cumulative count of all the samples that have been written
    /// into the buffer since its construction.
    pub fn samples_written(&self) -> u64 {
        self.samples_written
    }
}

impl<T, N: ArrayLength + PowerOfTwo> CircBuf<T, N> {
    const MASK: u64 = N::U64 - 1;

    fn abs_index_to_rel(&self, abs_index: u64) -> Option<usize> {
        if abs_index >= self.samples_written {
            // index in the future
            return None;
        }
        if abs_index + N::U64 < self.samples_written {
            // index already overwritten
            return None;
        }
        Some(usize::try_from(abs_index & Self::MASK).unwrap())
    }

    // This is like abs_index_to_rel, but the abs_index is allowed to be one
    // past the last sample written. The function is to be used for the end of
    // an exclusive range.
    fn abs_index_to_rel_exclusive(&self, abs_index: u64) -> Option<usize> {
        if abs_index > self.samples_written {
            // index in the future
            return None;
        }
        if abs_index + N::U64 <= self.samples_written {
            // index already overwritten
            return None;
        }
        Some(usize::try_from(abs_index & Self::MASK).unwrap())
    }

    /// Returns a reference to a sample or subslice of samples depending on the
    /// index type.
    ///
    /// - If given a position, indicated by a `u64` that corresponds to the
    ///   absolute index of the sample, this returns a reference to the sample
    ///   at that position or `None` if out of bounds.
    ///
    /// - If given a range of the form `a..b` or `a..=b`, this returns the
    ///   subslices corresponding to this range of samples according to the
    ///   absolute indices `a` and `b`, or `None` if part of range or all the
    ///   range is out of bounds. A pair of slices is returned to handle the
    ///   case when the range wraps around the circular buffer.
    ///
    /// Ranges of the form `..`, `..b`, `..=b`, and `a..` are not supported.
    pub fn get<'a, I>(&'a self, index: I) -> Option<<I as Index<'a, CircBuf<T, N>>>::Output>
    where
        I: Index<'a, CircBuf<T, N>>,
    {
        index.get(self)
    }

    /// Returns a mutable reference to a sample or subslice of samples depending
    /// on the index type (see [`get`](CircBuf::get)) or `None` if the index is
    /// out of bounds.
    pub fn get_mut<'a, I>(
        &'a mut self,
        index: I,
    ) -> Option<<I as Index<'a, CircBuf<T, N>>>::OutputMut>
    where
        I: Index<'a, CircBuf<T, N>>,
    {
        index.get_mut(self)
    }

    /// Writes a single sample to the circular buffer.
    pub fn push(&mut self, value: T) {
        self.samples[usize::try_from(self.samples_written & Self::MASK).unwrap()] = value;
        self.samples_written += 1;
    }

    /// Writes a sample in-place by returning a mutable reference to that sample.
    ///
    /// This function returns a mutable reference to a sample that is to be
    /// written with a new sample. The number of samples is written to the
    /// buffer is incremented by one after this function returns, and the
    /// caller is responsible for writing the sample value through the
    /// reference before dropping it.
    #[must_use]
    pub fn writer_ref(&mut self) -> &mut T {
        let r = &mut self.samples[usize::try_from(self.samples_written & Self::MASK).unwrap()];
        self.samples_written += 1;
        r
    }

    /// Writes samples in-place by returning a pair of slices to which the
    /// samples can be written.
    ///
    /// Given a number of samples `num_samples`, this function returns two
    /// mutable slices where those many samples can be written in the circular
    /// buffer. The second slice handles wraps around the circular buffer and
    /// may be empty if no wrap around is needed. The number of samples written
    /// to the buffer is incremented by `num_samples` after this function
    /// returns, and the caller is responsible for writing the sample values in
    /// the slices before dropping them.
    ///
    /// # Panics
    ///
    /// This function panics if the number of samples is larger than the
    /// capacity of the circular buffer.
    #[must_use]
    pub fn writer_slices(&mut self, num_samples: usize) -> (&mut [T], &mut [T]) {
        assert!(num_samples <= N::USIZE);
        if num_samples == 0 {
            // Special case of an empty write.
            return (&mut [], &mut []);
        }
        let i0 = usize::try_from(self.samples_written & Self::MASK).unwrap();
        let i1 = usize::try_from(
            (self.samples_written + u64::try_from(num_samples).unwrap()) & Self::MASK,
        )
        .unwrap();
        self.samples_written += u64::try_from(num_samples).unwrap();
        self.slice_mut_pair_from_idxs(i0, i1)
    }

    fn slice_mut_pair_from_idxs(&mut self, i0: usize, i1: usize) -> (&mut [T], &mut [T]) {
        debug_assert!(i0 < N::USIZE);
        debug_assert!(i1 < N::USIZE);
        if i0 < i1 {
            // No wrap around.
            (&mut self.samples[i0..i1], &mut [])
        } else {
            // Wrap around. We need to use split_at to avoid borrowing cirbuf.samples
            // mutably twice.
            let (a, b) = self.samples.split_at_mut(i1);
            (&mut b[i0 - i1..], a)
        }
    }
}

impl<T: Clone, N: ArrayLength + PowerOfTwo> CircBuf<T, N> {
    /// Creates a new empty circular buffer filled with a given element.
    ///
    /// The contents of the buffer are initialized to clones of `value`, so
    /// unlike for [`new`](CircBuf::new), it is not required that `T` implements
    /// [`Default`]. These contents, are, however, not accessible, so the value
    /// supplied is irrelevant in general.
    pub fn new_filled_with(value: &T) -> Self {
        CircBuf {
            samples: Box::new(GenericArray::generate(|_| value.clone())),
            samples_written: 0,
        }
    }

    /// Writes the contents of a slice to the circular buffer.
    ///
    /// The length of the slice must be smaller than or equal to the capacity of
    /// the circular buffer, so that all the samples in the slice can be
    /// written. The elements are written by cloning them.
    ///
    /// # Panics
    ///
    /// This function panics if the the length of `other` is larger than the
    /// capacity of the circular buffer.
    pub fn extend_from_slice(&mut self, other: &[T]) {
        assert!(other.len() <= N::USIZE);
        let a = usize::try_from(self.samples_written & Self::MASK).unwrap();
        let b = (a + other.len()).min(N::USIZE);
        let n = b - a;
        self.samples[a..b].clone_from_slice(&other[..n]);
        if n < other.len() {
            let c = other.len() - n;
            self.samples[..c].clone_from_slice(&other[n..]);
        }
        self.samples_written += u64::try_from(other.len()).unwrap();
    }
}

impl<T, N: ArrayLength + PowerOfTwo> Extend<T> for CircBuf<T, N> {
    /// Writes the elements of an iterator sequentially into the circular buffer.
    ///
    /// This function calls [`push`](CircBuf::push) for each element produced by
    /// the iterator to write it into the buffer. It is allowed that the length
    /// of the iterator is larger than the capacity of the buffer. In this case
    /// older items will be overwritten by newer ones as the write loops around
    /// the circular buffer.
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = T>,
    {
        for x in iter {
            self.push(x);
        }
    }
}

/// Helper trait used for [`CircBuf`] indexing operations.
///
/// This trait is implemented by types that can be used to index a [`CircBuf`]
/// with the [`get`](CircBuf::get) and [`get_mut`](CircBuf::get_mut) methods.
pub trait Index<'a, T>
where
    T: ?Sized,
{
    /// Output type for immutable output ([`get`](CircBuf::get) output).
    type Output: 'a;
    /// Output type for mutable output ([`get_mut`](CircBuf::get_mut) output).
    type OutputMut: 'a;
    /// Implementation of [`get`](CircBuf::get) for this index type.
    fn get(self, buf: &'a T) -> Option<Self::Output>;
    /// Implementation of [`get_mut`](CircBuf::get_mut) for this index type.
    fn get_mut(self, buf: &'a mut T) -> Option<Self::OutputMut>;
}

impl<'a, T: 'a, N: ArrayLength + PowerOfTwo> Index<'a, CircBuf<T, N>> for u64 {
    type Output = &'a T;
    type OutputMut = &'a mut T;

    fn get(self, circbuf: &CircBuf<T, N>) -> Option<&T> {
        let index = circbuf.abs_index_to_rel(self)?;
        Some(&circbuf.samples[index])
    }

    fn get_mut(self, circbuf: &mut CircBuf<T, N>) -> Option<&mut T> {
        let index = circbuf.abs_index_to_rel(self)?;
        Some(&mut circbuf.samples[index])
    }
}

impl<'a, T: 'a, N: ArrayLength + PowerOfTwo> Index<'a, CircBuf<T, N>> for std::ops::Range<u64> {
    type Output = (&'a [T], &'a [T]);
    type OutputMut = (&'a mut [T], &'a mut [T]);

    fn get(self, circbuf: &CircBuf<T, N>) -> Option<(&[T], &[T])> {
        let len = self.end.saturating_sub(self.start);
        if len > N::U64 {
            // range is longer than buffer
            return None;
        }
        if len == 0 {
            // special case of empty range
            return Some((&[], &[]));
        }
        let i0 = circbuf.abs_index_to_rel(self.start)?;
        let i1 = circbuf.abs_index_to_rel_exclusive(self.end)?;
        if i0 < i1 {
            // no wrap around
            return Some((&circbuf.samples[i0..i1], &[]));
        }
        // wrap around
        Some((&circbuf.samples[i0..], &circbuf.samples[..i1]))
    }

    fn get_mut(self, circbuf: &mut CircBuf<T, N>) -> Option<(&mut [T], &mut [T])> {
        let len = self.end.saturating_sub(self.start);
        if len > N::U64 {
            // range is longer than buffer
            return None;
        }
        if len == 0 {
            // special case of empty range
            return Some((&mut [], &mut []));
        }
        let i0 = circbuf.abs_index_to_rel(self.start)?;
        let i1 = circbuf.abs_index_to_rel_exclusive(self.end)?;
        Some(circbuf.slice_mut_pair_from_idxs(i0, i1))
    }
}

impl<'a, T: 'a, N: ArrayLength + PowerOfTwo> Index<'a, CircBuf<T, N>>
    for std::ops::RangeInclusive<u64>
{
    type Output = (&'a [T], &'a [T]);
    type OutputMut = (&'a mut [T], &'a mut [T]);

    fn get(self, circbuf: &CircBuf<T, N>) -> Option<(&[T], &[T])> {
        circbuf.get(*self.start()..*self.end() + 1)
    }

    fn get_mut(self, circbuf: &mut CircBuf<T, N>) -> Option<(&mut [T], &mut [T])> {
        circbuf.get_mut(*self.start()..*self.end() + 1)
    }
}

impl<T, N: ArrayLength + PowerOfTwo> std::ops::Index<u64> for CircBuf<T, N> {
    type Output = T;

    fn index(&self, index: u64) -> &T {
        self.get(index).unwrap()
    }
}

impl<T, N: ArrayLength + PowerOfTwo> std::ops::IndexMut<u64> for CircBuf<T, N> {
    fn index_mut(&mut self, index: u64) -> &mut T {
        self.get_mut(index).unwrap()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use generic_array::typenum::consts::U32;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn push(num_integers in 0usize..=1024) {
            let mut buf = CircBuf::<usize, U32>::new();
            for n in 0..num_integers {
                buf.push(n);
            }
            assert_eq!(buf.samples_written(), num_integers as u64);
            if num_integers < buf.capacity() {
                // samples not written yet should be zeros
                for x in &buf.samples[num_integers..] {
                    assert_eq!(*x, 0);
                }
            }
            let start = num_integers.saturating_sub(buf.capacity());
            let end = num_integers;
            for n in start..end {
                assert_eq!(buf.samples[n % buf.capacity()], n);
            }
        }
    }

    proptest! {
        #[test]
        fn extend_equals_push_loop(num_integers in 0usize..=1024) {
            let mut buf0 = CircBuf::<usize, U32>::new();
            let mut buf1 = CircBuf::<usize, U32>::new();
            buf0.extend(0..num_integers);
            for n in 0..num_integers {
                buf1.push(n);
            }
            assert_eq!(buf0, buf1);
        }
    }

    #[test]
    #[should_panic]
    fn extend_from_slice_too_large_panics() {
        let mut buf = CircBuf::<usize, U32>::new();
        buf.extend_from_slice(&[0; 33]);
    }

    proptest! {
        #[test]
        fn extend_from_slice(
            num_written in 0usize..=1024,
            len in 0usize..=32,
        ) {
            let mut buf = CircBuf::<usize, U32>::new();
            // Write usize::MAX num_written times, which is a different value
            // compared to all the integers we will write in the slice.
            buf.extend(std::iter::repeat_n(usize::MAX, num_written));
            let v = (0..len).collect::<Vec<usize>>();
            buf.extend_from_slice(&v);
            assert_eq!(buf.samples_written(), u64::try_from(num_written + len).unwrap());
            for n in 0..buf.capacity() {
                let idx = (n + num_written) % buf.capacity();
                let x = buf.samples[idx];
                let expected = if n < len {
                    // value from extend_from_slice()
                    n
                } else if idx >= num_written {
                    // initial buffer value, never overwritten
                    0
                } else {
                    // value from extend()
                    usize::MAX
                };
                assert_eq!(x, expected);
            }
        }
    }

    proptest! {
        #[test]
        fn get(
            num_written in 0u64..=1024,
        ) {
            let mut buf = CircBuf::<u64, U32>::new();
            buf.extend(0..num_written);
            for n in 0..2048 {
                let a = buf.get(n);
                if n >= num_written {
                    // still not written
                    assert_eq!(a, None);
                } else if n < num_written.saturating_sub(buf.capacity() as u64) {
                    // already overwritten
                    assert_eq!(a, None);
                } else {
                    assert_eq!(*a.unwrap(), n);
                    // test also the [] operator
                    assert_eq!(buf[n], n);
                }
            }
        }
    }

    proptest! {
        #[test]
        fn get_mut(
            num_written in 0u64..=1024,
        ) {
            let mut buf = CircBuf::<u64, U32>::new();
            let capacity = u64::try_from(buf.capacity()).unwrap();
            buf.extend(0..num_written);
            for n in 0..2048 {
                let a = buf.get_mut(n);
                if n >= num_written {
                    // still not written
                    assert_eq!(a, None);
                } else if n < num_written.saturating_sub(capacity) {
                    // already overwritten
                    assert_eq!(a, None);
                } else {
                    assert!(a.is_some());
                    assert_eq!(**a.as_ref().unwrap(), n);
                    *a.unwrap() = u64::MAX - n;
                    assert_eq!(buf[n], u64::MAX - n);
                    // test also the [] operator for writing
                    buf[n] = u64::MAX - 3 * n;
                    assert_eq!(buf[n], u64::MAX - 3 * n);
                }
            }
        }
    }

    proptest! {
        #[test]
        fn get_range(
            num_written in 0u64..=1024,
            a in 0u64..=2048,
            b in 0u64..=2048,
        ) {
            let mut buf = CircBuf::<u64, U32>::new();
            let capacity = u64::try_from(buf.capacity()).unwrap();
            buf.extend(0..num_written);
            let out = buf.get(a..b);
            if b <= a {
                // empty range
                let expected: Option<(&[u64], &[u64])> = Some((&[], &[]));
                assert_eq!(out, expected);
            } else if b > num_written {
                // end not written yet
                assert_eq!(out, None);
            } else if a < num_written.saturating_sub(capacity) {
                // beginning already overwritten
                assert_eq!(out, None);
            } else {
                let (x, y) = out.unwrap();
                let xy = [x, y].concat();
                assert_eq!(xy, (a..b).collect::<Vec<_>>());
            }
        }
    }

    proptest! {
        #[test]
        fn get_range_inclusive(
            num_written in 0u64..=1024,
            a in 0u64..=2048,
            b in 0u64..=2048,
        ) {
            let mut buf = CircBuf::<u64, U32>::new();
            let capacity = u64::try_from(buf.capacity()).unwrap();
            buf.extend(0..num_written);
            let out = buf.get(a..=b);
            if b < a {
                // empty range
                let expected: Option<(&[u64], &[u64])> = Some((&[], &[]));
                assert_eq!(out, expected);
            } else if b >= num_written {
                // end not written yet
                assert_eq!(out, None);
            } else if a < num_written.saturating_sub(capacity) {
                // beginning already overwritten
                assert_eq!(out, None);
            } else {
                let (x, y) = out.unwrap();
                let xy = [x, y].concat();
                assert_eq!(xy, (a..=b).collect::<Vec<_>>());
            }
        }
    }

    proptest! {
        #[test]
        fn get_range_mut(
            num_written in 0u64..=1024,
            a in 0u64..=2048,
            b in 0u64..=2048,
        ) {
            let mut buf = CircBuf::<u64, U32>::new();
            let capacity = u64::try_from(buf.capacity()).unwrap();
            buf.extend(0..num_written);
            let out = buf.get_mut(a..b);
            if b <= a {
                // empty range
                let expected: Option<(&mut [u64], &mut [u64])> = Some((&mut [], &mut []));
                assert_eq!(out, expected);
            } else if b > num_written {
                // end not written yet
                assert_eq!(out, None);
            } else if a < num_written.saturating_sub(capacity) {
                // beginning already overwritten
                assert_eq!(out, None);
            } else {
                let (x, y) = out.unwrap();
                let xy = [x.to_vec(), y.to_vec()].concat();
                assert_eq!(xy, (a..b).collect::<Vec<_>>());
                let new = (a..b).map(|n| u64::MAX - n).collect::<Vec<_>>();
                x.copy_from_slice(&new[..x.len()]);
                y.copy_from_slice(&new[x.len()..]);
                let (x_new, y_new) = buf.get(a..b).unwrap();
                let xy_new = [x_new, y_new].concat();
                assert_eq!(xy_new, new);

            }
        }
    }

    proptest! {
        #[test]
        fn get_range_mut_inclusive(
            num_written in 0u64..=1024,
            a in 0u64..=2048,
            b in 0u64..=2048,
        ) {
            let mut buf = CircBuf::<u64, U32>::new();
            let capacity = u64::try_from(buf.capacity()).unwrap();
            buf.extend(0..num_written);
            let out = buf.get_mut(a..=b);
            if b < a {
                // empty range
                let expected: Option<(&mut [u64], &mut [u64])> = Some((&mut [], &mut []));
                assert_eq!(out, expected);
            } else if b >= num_written {
                // end not written yet
                assert_eq!(out, None);
            } else if a < num_written.saturating_sub(capacity) {
                // beginning already overwritten
                assert_eq!(out, None);
            } else {
                let (x, y) = out.unwrap();
                let xy = [x.to_vec(), y.to_vec()].concat();
                assert_eq!(xy, (a..=b).collect::<Vec<_>>());
                let new = (a..=b).map(|n| u64::MAX - n).collect::<Vec<_>>();
                x.copy_from_slice(&new[..x.len()]);
                y.copy_from_slice(&new[x.len()..]);
                let (x_new, y_new) = buf.get(a..=b).unwrap();
                let xy_new = [x_new, y_new].concat();
                assert_eq!(xy_new, new);

            }
        }
    }

    proptest! {
        #[test]
        fn writer_ref(
            num_written in 0u64..=1024,
        ) {
            let mut buf = CircBuf::<u64, U32>::new();
            let capacity = u64::try_from(buf.capacity()).unwrap();
            buf.extend(0..num_written);
            let x = buf.writer_ref();
            *x = num_written;
            assert_eq!(buf.samples_written(), num_written + 1);
            let start = buf.samples_written().saturating_sub(capacity);
            let end = buf.samples_written();
            let (x, y) = buf.get(start..end).unwrap();
            let xy = [x, y].concat();
            assert_eq!(xy, (start..end).collect::<Vec<_>>());
        }
    }

    proptest! {
        #[test]
        fn writer_slices(
            num_written in 0u64..=1024,
            num_samples in 0usize..=32,
        ) {
            let mut buf = CircBuf::<u64, U32>::new();
            let capacity = u64::try_from(buf.capacity()).unwrap();
            buf.extend(0..num_written);
            let (a, b) = buf.writer_slices(num_samples);
            for (aa, n) in a.iter_mut().zip(num_written..) {
                *aa = n;
            }
            for (bb, n) in b.iter_mut().zip(num_written + u64::try_from(a.len()).unwrap()..) {
                *bb = n;
            }
            assert_eq!(buf.samples_written(), num_written + u64::try_from(num_samples).unwrap());
            let start = buf.samples_written().saturating_sub(capacity);
            let end = buf.samples_written();
            let (x, y) = buf.get(start..end).unwrap();
            let xy = [x, y].concat();
            assert_eq!(xy, (start..end).collect::<Vec<_>>());
        }
    }
}
