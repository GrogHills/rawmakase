//! One HEVC picture decoded by rust_h265 (pure Rust, without `unsafe`): the
//! parameter sets of a HEIF item's `hvcC`, then the item's length-prefixed
//! NAL units.
//!
//! rust_h265 is young and may panic on malformed data rather than return an
//! error. Being Rust, the panic unwinds, and [`decode`] catches it, so a
//! corrupt file is an error, not a crash. Its own limits refuse a picture
//! over 35.7 MP (HEVC level 6.2's largest) or 512 MiB of state before
//! allocating it.
use anyhow::{Result, anyhow, bail};
use rust_h265::{Decoder, Frame, PixelData, parse_hvcc, parse_nal};

/// What an `hvcC` (ISO/IEC 14496-15 HEVCDecoderConfigurationRecord) says.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Config {
    /// Bytes before each NAL unit giving its length: 1, 2 or 4.
    length_size: u8,
    /// Bits per luma sample.
    pub(super) depth: u8,
    /// The VPS, SPS, PPS and SEI NAL units, in order.
    parameter_sets: Vec<Vec<u8>>,
}

impl Config {
    /// The record in `hvcc`; `None` when it is cut off or not version 1.
    pub(super) fn read(hvcc: &[u8]) -> Option<Self> {
        if hvcc.len() < 23 || hvcc[0] != 1 {
            return None;
        }
        let length_size = (hvcc[21] & 3) + 1;
        if length_size == 3 {
            return None;
        }
        let depth = (hvcc[17] & 7) + 8;
        let mut rest = &hvcc[23..];
        let mut take = |n: usize| {
            let (head, tail) = rest.split_at_checked(n)?;
            rest = tail;
            Some(head)
        };
        let mut parameter_sets = Vec::new();
        for _ in 0..hvcc[22] {
            take(1)?; // completeness and NAL unit type
            let count = u16::from_be_bytes(take(2)?.try_into().ok()?);
            for _ in 0..count {
                let len = u16::from_be_bytes(take(2)?.try_into().ok()?);
                parameter_sets.push(take(len as usize)?.to_vec());
            }
        }
        Some(Self {
            length_size,
            depth,
            parameter_sets,
        })
    }
}

/// A decoded picture: luma, then two chroma planes each `chroma_width` wide,
/// rows packed, 8-bit samples or deeper ones.
pub(super) struct Picture {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) depth: u8,
    /// 4:2:0 (subsampled both ways) rather than 4:4:4.
    pub(super) subsampled: bool,
    pub(super) planes: Planes,
}

pub(super) enum Planes {
    Bytes([Vec<u8>; 3]),
    Words([Vec<u16>; 3]),
}

impl Picture {
    /// The width of a chroma plane.
    pub(super) fn chroma_width(&self) -> u32 {
        if self.subsampled {
            self.width.div_ceil(2)
        } else {
            self.width
        }
    }
    /// The height of a chroma plane.
    pub(super) fn chroma_height(&self) -> u32 {
        if self.subsampled {
            self.height.div_ceil(2)
        } else {
            self.height
        }
    }
}

/// The picture coded in `data`, an item's NAL units, after `config`'s
/// parameter sets.
pub(super) fn decode(config: &Config, data: &[u8]) -> Result<Picture> {
    let frame = std::panic::catch_unwind(|| decode_frame(config, data))
        .map_err(|_| anyhow!("The HEVC image data is corrupt"))??;
    picture(frame)
}

fn decode_frame(config: &Config, data: &[u8]) -> Result<Frame> {
    let failed = |e: rust_h265::DecodeError| anyhow!("Could not decode the HEVC image: {e}");
    let mut decoder = Decoder::new();
    for nal in config.parameter_sets.iter().filter_map(|n| parse_nal(n)) {
        decoder.decode_nal(&nal).map_err(failed)?;
    }
    for nal in parse_hvcc(data, config.length_size) {
        // A still is one picture: the first one finished is it.
        if let Some(frame) = decoder.decode_nal(&nal).map_err(failed)? {
            return Ok(frame);
        }
    }
    decoder
        .flush()
        .ok_or_else(|| anyhow!("The HEVC image data is missing or cut off"))
}

/// `frame` checked: 4:2:0 or 4:4:4, of 8 to 12 bits, each plane as large as
/// its size says.
fn picture(frame: Frame) -> Result<Picture> {
    let subsampled = match frame.chroma_format_idc {
        1 => true,
        3 => false,
        n => bail!("HEVC images in chroma format {n} can't be read"),
    };
    if !matches!(frame.bit_depth, 8 | 10 | 12) {
        bail!("{}-bit HEVC images are not supported", frame.bit_depth);
    }
    let Frame {
        y,
        u,
        v,
        width,
        height,
        bit_depth,
        ..
    } = frame;
    let planes = match (y, u, v) {
        (PixelData::U8(y), PixelData::U8(u), PixelData::U8(v)) if bit_depth == 8 => {
            Planes::Bytes([y, u, v])
        }
        (PixelData::U16(y), PixelData::U16(u), PixelData::U16(v)) if bit_depth > 8 => {
            Planes::Words([y, u, v])
        }
        _ => bail!("The HEVC image's samples are not as deep as it says"),
    };
    let picture = Picture {
        width,
        height,
        depth: bit_depth,
        subsampled,
        planes,
    };
    let luma = width as usize * height as usize;
    let chroma = picture.chroma_width() as usize * picture.chroma_height() as usize;
    let lengths = match &picture.planes {
        Planes::Bytes(p) => p.each_ref().map(Vec::len),
        Planes::Words(p) => p.each_ref().map(Vec::len),
    };
    if lengths != [luma, chroma, chroma] {
        bail!("The HEVC image is not the size it says");
    }
    Ok(picture)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An `hvcC` of 8-bit 4:2:0 with 4-byte lengths and one array of `nals`.
    fn hvcc(nals: &[&[u8]]) -> Vec<u8> {
        let mut record = vec![1, 1, 0x60, 0, 0, 0, 0x90, 0, 0, 0, 0, 0, 0x5a];
        record.extend([0xf0, 0, 0xfc, 0xfd, 0xf8, 0xf8, 0, 0, 0x0f, 1]);
        record.extend([0xa0, 0, nals.len() as u8]);
        for nal in nals {
            record.extend((nal.len() as u16).to_be_bytes());
            record.extend(*nal);
        }
        record
    }

    #[test]
    fn a_configuration_records_parameter_sets_and_sizes_are_read() {
        let config = Config::read(&hvcc(&[&[0x40, 1, 2], &[0x42, 1]])).unwrap();
        assert_eq!(config.length_size, 4);
        assert_eq!(config.depth, 8);
        assert_eq!(config.parameter_sets, [vec![0x40, 1, 2], vec![0x42, 1]]);
        let mut ten_bit = hvcc(&[]);
        ten_bit[17] = 0xfa;
        assert_eq!(Config::read(&ten_bit).unwrap().depth, 10);
    }

    #[test]
    fn a_configuration_record_cut_off_or_of_another_version_is_not_read() {
        let record = hvcc(&[&[0x40, 1, 2]]);
        assert_eq!(Config::read(&record[..record.len() - 1]), None);
        assert_eq!(Config::read(&record[..20]), None);
        let mut version = record.clone();
        version[0] = 2;
        assert_eq!(Config::read(&version), None);
        // A length of 3 bytes is not one HEVC allows.
        let mut three = record;
        three[21] = 0x0e;
        assert_eq!(Config::read(&three), None);
    }

    #[test]
    fn data_without_a_picture_is_an_error() {
        let config = Config::read(&hvcc(&[])).unwrap();
        let error = decode(&config, &[0, 0, 0, 2, 0x4e, 1]).err().unwrap();
        assert!(error.to_string().contains("missing"), "{error}");
        assert!(decode(&config, &[]).is_err());
    }
}
