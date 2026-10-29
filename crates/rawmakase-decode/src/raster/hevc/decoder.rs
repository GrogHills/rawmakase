//! One HEVC picture decoded by rust_h265 (pure Rust, without `unsafe`): the
//! parameter sets of a HEIF item's `hvcC`, then the item's length-prefixed
//! NAL units.
//!
//! rust_h265 is young and may panic on malformed data rather than return an
//! error. Being Rust, the panic unwinds, and [`decode`] catches it, so a
//! corrupt file is an error, not a crash. Its own limits refuse a picture
//! over 35.7 MP (HEVC level 6.2's largest) or 512 MiB of state before
//! allocating it.
use super::Config;
use anyhow::{Result, anyhow, bail};
use rust_h265::{Decoder, Frame, PixelData, parse_hvcc, parse_nal};

/// A decoded picture: luma, then two chroma planes each `chroma_width` wide,
/// rows packed, 8-bit samples or deeper ones.
pub(in crate::raster) struct Picture {
    pub(in crate::raster) width: u32,
    pub(in crate::raster) height: u32,
    pub(in crate::raster) depth: u8,
    /// 4:2:0 (subsampled both ways) rather than 4:4:4.
    pub(in crate::raster) subsampled: bool,
    pub(in crate::raster) planes: Planes,
}

pub(in crate::raster) enum Planes {
    Bytes([Vec<u8>; 3]),
    Words([Vec<u16>; 3]),
}

impl Picture {
    /// The width of a chroma plane.
    pub(in crate::raster) fn chroma_width(&self) -> u32 {
        if self.subsampled {
            self.width.div_ceil(2)
        } else {
            self.width
        }
    }
    /// The height of a chroma plane.
    pub(in crate::raster) fn chroma_height(&self) -> u32 {
        if self.subsampled {
            self.height.div_ceil(2)
        } else {
            self.height
        }
    }
}

/// The picture coded in `data`, an item's NAL units, after `config`'s
/// parameter sets.
pub(in crate::raster) fn decode(config: &Config, data: &[u8]) -> Result<Picture> {
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
    use super::super::tests::hvcc;
    use super::*;

    #[test]
    fn data_without_a_picture_is_an_error() {
        let config = Config::read(&hvcc(&[])).unwrap();
        let error = decode(&config, &[0, 0, 0, 2, 0x4e, 1]).err().unwrap();
        assert!(error.to_string().contains("missing"), "{error}");
        assert!(decode(&config, &[]).is_err());
    }
}
