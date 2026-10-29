//! A HEIC (a HEIF of HEVC) still: our reading of its container
//! ([`rawmakase_interop::isobmff`]), its HEVC decoded by rust_h265
//! ([`super::hevc`]) and turned into RGB with the matrix and range it says.
//!
//! A phone's photo is a grid (`grid`) of tiles, 512×512 on an iPhone, each its
//! own HEVC picture. The tiles are decoded in parallel, a few at a time so a
//! large grid never holds them all, and laid into one picture before it is
//! turned into RGB, so subsampled chroma is interpolated across their edges as
//! within them: bilinearly, as an AVIF's is (libheif repeats each sample
//! instead, so the two differ along sharp colour edges, and elsewhere agree
//! to within one level). A clean aperture (`clap`) then crops the
//! picture, before its rotation and mirror, the order MIAF requires.
//!
//! The colour comes from the `colr` boxes of the image, or of its first tile
//! when the image has none of that kind (an iPhone's tiles carry the `nclx`):
//! an ICC profile, or one built from the CICP code points; a HEIC with neither
//! is taken as sRGB. The matrix and range come from the `nclx` box; without
//! one they are taken as BT.601 and full range, what phones and libheif write
//! (the HEVC header's own are not read). Alpha, depth maps, gain maps and thumbnails are not read.
use super::avif::{self, Nclx};
use super::{Unsupported, hevc};
use anyhow::{Result, anyhow, bail, ensure};
use image::ImageDecoder;
use image::metadata::Orientation;
use rawmakase_interop::isobmff::{Heif, Property};
use rayon::prelude::*;
use std::path::Path;
use yuv::{YuvPlanarImage, YuvRange};

/// The most pixels the decoded picture may have before its crop, as many as
/// `prepare` lets an image have after it.
const MAX_PIXELS: u64 = 150_000_000;

/// Where the image is in the decoded picture: its clean aperture.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Crop {
    left: u32,
    top: u32,
    width: u32,
    height: u32,
}

/// The pictures an image is decoded from: one, or a grid's tiles.
struct Layout {
    /// The whole picture's size, before its crop.
    width: u32,
    height: u32,
    /// The pictures, row by row, `columns` to a row: each item and its HEVC
    /// configuration.
    tiles: Vec<(u32, hevc::Config)>,
    columns: u32,
    /// Each picture's size.
    tile: (u32, u32),
}

pub(super) struct HeicFile {
    crop: Crop,
    /// More than 8 bits per sample, from `hvcC`.
    deep: bool,
    /// The file and its boxes, to read the pictures from when decoding.
    file: std::fs::File,
    heif: Heif,
    layout: Layout,
    icc: Option<Vec<u8>>,
    nclx: Option<Nclx>,
    orientation: Orientation,
    refused: Option<Unsupported>,
}

impl HeicFile {
    pub(super) fn open(path: &Path) -> Result<Self> {
        let mut file = std::fs::File::open(path)?;
        let heif = Heif::read(&mut file).ok_or_else(|| anyhow!("Could not read the HEIF file"))?;
        let primary = heif
            .item(heif.primary)
            .ok_or_else(|| anyhow!("The HEIF file has no image"))?;
        if let Some(kind) = heif.unread_essential(heif.primary) {
            let kind = String::from_utf8_lossy(&kind);
            bail!("The HEIF image needs `{kind}`, which RAWmakase can't read");
        }
        let layout = match &primary.kind {
            b"hvc1" => {
                let (config, (width, height)) = picture(&heif, heif.primary)?;
                Layout {
                    width,
                    height,
                    tiles: vec![(heif.primary, config)],
                    columns: 1,
                    tile: (width, height),
                }
            }
            b"grid" => grid(&heif, &mut file)?,
            _ => bail!("The HEIF image is not HEVC"),
        };
        ensure!(
            u64::from(layout.width) * u64::from(layout.height) <= MAX_PIXELS,
            "Image too large to preview"
        );
        let props = heif.properties(heif.primary);
        let crop = crop(&props, layout.width, layout.height)?;
        let (icc, nclx) = avif::colour(&props);
        let (tile_icc, tile_nclx) = avif::colour(&heif.properties(layout.tiles[0].0));
        let (icc, nclx) = (icc.or(tile_icc), nclx.or(tile_nclx));
        let depth = layout.tiles[0].1.depth;
        ensure!(
            layout.tiles.iter().all(|(_, c)| c.depth == depth),
            "The HEIF image's tiles differ in depth"
        );
        let animated = [b"msf1", b"hevc", b"hevx", b"hevm", b"hevs"];
        let refused = if nclx.is_some_and(Nclx::hdr) {
            Some(Unsupported("HDR".into()))
        } else if heif.brands.iter().any(|b| animated.contains(&b)) {
            Some(Unsupported("animated".into()))
        } else {
            None
        };
        Ok(Self {
            crop,
            deep: depth > 8,
            file,
            orientation: avif::transforms(&props),
            heif,
            layout,
            icc,
            nclx,
            refused,
        })
    }

    pub(super) fn refusal(&self) -> Option<Unsupported> {
        self.refused.as_ref().map(|r| Unsupported(r.0.clone()))
    }

    /// The whole picture: each tile decoded and laid in its place.
    fn decode(mut self) -> Result<hevc::Picture> {
        let Layout {
            width,
            height,
            tiles,
            columns,
            tile,
        } = &self.layout;
        let mut canvas: Option<hevc::Picture> = None;
        let batch = rayon::current_num_threads().max(1);
        for (b, chunk) in tiles.chunks(batch).enumerate() {
            let data = chunk
                .iter()
                .map(|(id, _)| {
                    self.heif
                        .data(&mut self.file, *id)
                        .ok_or_else(|| anyhow!("The HEIF image data is missing or cut off"))
                })
                .collect::<Result<Vec<_>>>()?;
            let pictures: Vec<_> = chunk
                .par_iter()
                .zip(&data)
                .map(|((_, config), data)| hevc::decode(config, data))
                .collect();
            for (i, picture) in pictures.into_iter().enumerate() {
                let picture = picture?;
                ensure!(
                    (picture.width, picture.height) == *tile,
                    "The HEIF image is not the size it says"
                );
                if tiles.len() == 1 {
                    return Ok(picture);
                }
                let at = (b * batch + i) as u32;
                let canvas = canvas.get_or_insert_with(|| blank(&picture, *width, *height));
                place(
                    canvas,
                    &picture,
                    at % columns * tile.0,
                    at / columns * tile.1,
                )?;
            }
        }
        canvas.ok_or_else(|| anyhow!("The HEIF image has no tiles"))
    }
}

/// An `hvc1` item's configuration and size; the item may be a grid's tile.
fn picture(heif: &Heif, id: u32) -> Result<(hevc::Config, (u32, u32))> {
    let item = heif
        .item(id)
        .ok_or_else(|| anyhow!("The HEIF image is missing a tile"))?;
    ensure!(&item.kind == b"hvc1", "The HEIF image is not HEVC");
    if let Some(kind) = heif.unread_essential(id) {
        let kind = String::from_utf8_lossy(&kind);
        bail!("The HEIF image needs `{kind}`, which RAWmakase can't read");
    }
    let props = heif.properties(id);
    let size = props
        .iter()
        .find_map(|p| match p {
            Property::Size(w, h) => Some((*w, *h)),
            _ => None,
        })
        .filter(|(w, h)| *w > 0 && *h > 0)
        .ok_or_else(|| anyhow!("The HEIF image has no size"))?;
    let config = props
        .iter()
        .find_map(|p| match p {
            Property::HevcConfig(c) => hevc::Config::read(c),
            _ => None,
        })
        .ok_or_else(|| anyhow!("The HEIF image has no HEVC configuration"))?;
    Ok((config, size))
}

/// A `grid` item (ISO/IEC 23008-12 6.6.2.3): its size, and its tiles, all of
/// one size, which cover it without a row or column left over.
fn grid(heif: &Heif, file: &mut std::fs::File) -> Result<Layout> {
    let bad = || anyhow!("The HEIF image's grid can't be read");
    let data = heif.data_up_to(file, heif.primary, 64).ok_or_else(bad)?;
    let (rows, columns) = match data.get(..4) {
        Some([0, _, rows, columns]) => (u32::from(*rows) + 1, u32::from(*columns) + 1),
        _ => return Err(bad()),
    };
    let (width, height) = if data[1] & 1 == 0 {
        let size = data.get(4..8).ok_or_else(bad)?;
        (
            u32::from(u16::from_be_bytes([size[0], size[1]])),
            u32::from(u16::from_be_bytes([size[2], size[3]])),
        )
    } else {
        let size = data.get(4..12).ok_or_else(bad)?;
        (
            u32::from_be_bytes(size[..4].try_into()?),
            u32::from_be_bytes(size[4..].try_into()?),
        )
    };
    let ids = heif.referenced(*b"dimg", heif.primary);
    ensure!(
        ids.len() as u32 == rows * columns,
        "The HEIF image's grid has {} tiles, not {}",
        ids.len(),
        rows * columns
    );
    let tiles = ids
        .iter()
        .map(|id| picture(heif, *id).map(|(config, size)| (*id, config, size)))
        .collect::<Result<Vec<_>>>()?;
    let tile = tiles[0].2;
    ensure!(
        tiles.iter().all(|(_, _, size)| *size == tile),
        "The HEIF image's tiles differ in size"
    );
    let covers = |count: u32, size: u32, full: u32| {
        full > 0
            && u64::from(count) * u64::from(size) >= u64::from(full)
            && u64::from(count - 1) * u64::from(size) < u64::from(full)
    };
    ensure!(
        covers(columns, tile.0, width) && covers(rows, tile.1, height),
        "The HEIF image's tiles don't cover it"
    );
    Ok(Layout {
        width,
        height,
        tiles: tiles
            .into_iter()
            .map(|(id, config, _)| (id, config))
            .collect(),
        columns,
        tile,
    })
}

/// The part of a `width` × `height` picture a `clap` keeps: its size rounded,
/// its left and top edges rounded down, as libheif crops; the whole picture
/// without one.
fn crop(props: &[&Property], width: u32, height: u32) -> Result<Crop> {
    let Some(clap) = props.iter().find_map(|p| match p {
        Property::CleanAperture(c) => Some(*c),
        _ => None,
    }) else {
        return Ok(Crop {
            left: 0,
            top: 0,
            width,
            height,
        });
    };
    let round = |(n, d): (u32, u32)| (i128::from(n) + i128::from(d) / 2) / i128::from(d);
    // The centre's offset, plus half of what the crop leaves out:
    // offset + (full - size) / 2, over the denominator 2 × od × sd.
    let edge = |(on, od): (i32, u32), full: u32, (sn, sd): (u32, u32)| {
        let (on, od, full) = (i128::from(on), i128::from(od), i128::from(full));
        let (sn, sd) = (i128::from(sn), i128::from(sd));
        (2 * on * sd + full * od * sd - sn * od).div_euclid(2 * od * sd)
    };
    let (w, h) = (round(clap.width), round(clap.height));
    let left = edge(clap.horizontal_offset, width, clap.width);
    let top = edge(clap.vertical_offset, height, clap.height);
    ensure!(
        w > 0
            && h > 0
            && left >= 0
            && top >= 0
            && left + w <= width.into()
            && top + h <= height.into(),
        "The HEIF image's crop is outside it"
    );
    // Each is within `width` or `height`, so fits.
    Ok(Crop {
        left: left as u32,
        top: top as u32,
        width: w as u32,
        height: h as u32,
    })
}

/// A `width` × `height` picture of `like`'s depth and chroma, all zero.
fn blank(like: &hevc::Picture, width: u32, height: u32) -> hevc::Picture {
    let mut picture = hevc::Picture {
        width,
        height,
        depth: like.depth,
        subsampled: like.subsampled,
        planes: hevc::Planes::Bytes(Default::default()),
    };
    let luma = width as usize * height as usize;
    let chroma = picture.chroma_width() as usize * picture.chroma_height() as usize;
    let sizes = [luma, chroma, chroma];
    picture.planes = match like.planes {
        hevc::Planes::Bytes(_) => hevc::Planes::Bytes(sizes.map(|n| vec![0; n])),
        hevc::Planes::Words(_) => hevc::Planes::Words(sizes.map(|n| vec![0; n])),
    };
    picture
}

/// `tile` copied into `canvas` with its top left corner at (`x`, `y`), as
/// much of it as fits.
fn place(canvas: &mut hevc::Picture, tile: &hevc::Picture, x: u32, y: u32) -> Result<()> {
    ensure!(
        (tile.depth, tile.subsampled) == (canvas.depth, canvas.subsampled),
        "The HEIF image's tiles differ in depth or chroma"
    );
    // A subsampled tile's chroma starts at an even pixel.
    ensure!(
        !canvas.subsampled || (x.is_multiple_of(2) && y.is_multiple_of(2)),
        "The HEIF image's tiles are an odd size"
    );
    let (cx, cy) = if canvas.subsampled {
        (x / 2, y / 2)
    } else {
        (x, y)
    };
    // The tile's width and height, the canvas's, and where the tile goes: of
    // the luma plane, then of the chroma ones.
    let sizes = [
        [tile.width, tile.height, canvas.width, canvas.height, x, y],
        [
            tile.chroma_width(),
            tile.chroma_height(),
            canvas.chroma_width(),
            canvas.chroma_height(),
            cx,
            cy,
        ],
    ]
    .map(|s| s.map(|v| v as usize));
    fn copy<T: Copy>(to: &mut [T], from: &[T], [w, h, cw, ch, x, y]: [usize; 6]) {
        let (cols, rows) = (w.min(cw.saturating_sub(x)), h.min(ch.saturating_sub(y)));
        for row in 0..rows {
            to[(y + row) * cw + x..][..cols].copy_from_slice(&from[row * w..][..cols]);
        }
    }
    match (&mut canvas.planes, &tile.planes) {
        (hevc::Planes::Bytes(to), hevc::Planes::Bytes(from)) => {
            for (plane, (to, from)) in to.iter_mut().zip(from).enumerate() {
                copy(to, from, sizes[plane.min(1)]);
            }
        }
        (hevc::Planes::Words(to), hevc::Planes::Words(from)) => {
            for (plane, (to, from)) in to.iter_mut().zip(from).enumerate() {
                copy(to, from, sizes[plane.min(1)]);
            }
        }
        _ => bail!("The HEIF image's tiles differ in depth"),
    }
    Ok(())
}

/// The picture's pixels as interleaved RGB: 8-bit samples, or 16-bit ones
/// (native endian) scaled up from its depth.
fn to_rgb(picture: &hevc::Picture, nclx: Option<Nclx>) -> Result<Vec<u8>> {
    let (code, full) = match nclx {
        Some(n) => (u8::try_from(n.matrix).unwrap_or(2), n.full_range),
        None => (6, true),
    };
    // BT.2020 constant luminance (10) and chroma-derived constant luminance
    // (13) need a reconstruction no matrix gives.
    ensure!(
        !matches!(code, 10 | 13),
        "HEIF images in constant luminance can't be read"
    );
    let (w, h) = (picture.width, picture.height);
    let (cw, depth) = (picture.chroma_width(), picture.depth);
    // Identity (GBR): the planes are green, blue and red.
    if code == 0 {
        ensure!(!picture.subsampled, "A GBR HEIF image must be 4:4:4");
        return Ok(match &picture.planes {
            hevc::Planes::Bytes([g, b, r]) => interleave(r, g, b),
            hevc::Planes::Words([g, b, r]) => avif::widen(&interleave(r, g, b), depth),
        });
    }
    let range = if full {
        YuvRange::Full
    } else {
        YuvRange::Limited
    };
    let m = avif::matrix(code);
    let stride = w * 3;
    let failed = |e: yuv::YuvError| anyhow!("Could not convert the HEIF image: {e:?}");
    fn planar<'a, T: Copy + std::fmt::Debug>(
        planes: &'a [Vec<T>; 3],
        w: u32,
        h: u32,
        cw: u32,
    ) -> YuvPlanarImage<'a, T> {
        YuvPlanarImage {
            y_plane: &planes[0],
            y_stride: w,
            u_plane: &planes[1],
            u_stride: cw,
            v_plane: &planes[2],
            v_stride: cw,
            width: w,
            height: h,
        }
    }
    match &picture.planes {
        hevc::Planes::Bytes(planes) => {
            let mut rgb = vec![0u8; stride as usize * h as usize];
            let image = planar(planes, w, h, cw);
            if picture.subsampled {
                yuv::yuv420_to_rgb_bilinear(&image, &mut rgb, stride, range, m)
            } else {
                yuv::yuv444_to_rgb(&image, &mut rgb, stride, range, m)
            }
            .map_err(failed)?;
            Ok(rgb)
        }
        hevc::Planes::Words(planes) => {
            let mut rgb = vec![0u16; stride as usize * h as usize];
            let image = planar(planes, w, h, cw);
            match (depth, picture.subsampled) {
                (10, true) => yuv::i010_to_rgb10_bilinear(&image, &mut rgb, stride, range, m),
                (10, false) => yuv::i410_to_rgb10(&image, &mut rgb, stride, range, m),
                (12, true) => yuv::i012_to_rgb12_bilinear(&image, &mut rgb, stride, range, m),
                (12, false) => yuv::i412_to_rgb12(&image, &mut rgb, stride, range, m),
                _ => bail!("{depth}-bit HEIF images are not supported"),
            }
            .map_err(failed)?;
            Ok(avif::widen(&rgb, depth))
        }
    }
}

/// Three planes as interleaved samples.
fn interleave<T: Copy>(r: &[T], g: &[T], b: &[T]) -> Vec<T> {
    r.iter()
        .zip(g)
        .zip(b)
        .flat_map(|((r, g), b)| [*r, *g, *b])
        .collect()
}

impl ImageDecoder for HeicFile {
    fn dimensions(&self) -> (u32, u32) {
        (self.crop.width, self.crop.height)
    }
    fn color_type(&self) -> image::ColorType {
        if self.deep {
            image::ColorType::Rgb16
        } else {
            image::ColorType::Rgb8
        }
    }
    fn icc_profile(&mut self) -> image::ImageResult<Option<Vec<u8>>> {
        Ok(avif::profile(self.icc.as_ref(), self.nclx))
    }
    fn orientation(&mut self) -> image::ImageResult<Orientation> {
        Ok(self.orientation)
    }
    fn read_image(self, buf: &mut [u8]) -> image::ImageResult<()> {
        let failed = |e: anyhow::Error| {
            image::ImageError::Decoding(image::error::DecodingError::new(
                image::error::ImageFormatHint::Name("HEIF".into()),
                e.to_string(),
            ))
        };
        let (crop, deep, nclx) = (self.crop, self.deep, self.nclx);
        let picture = self.decode().map_err(failed)?;
        if (picture.depth > 8) != deep {
            return Err(failed(anyhow!("The HEIF image is not the depth it says")));
        }
        let rgb = to_rgb(&picture, nclx).map_err(failed)?;
        // Bytes per pixel: three samples of one or two bytes.
        let pixel = if deep { 6 } else { 3 };
        let (row, kept) = (picture.width as usize * pixel, crop.width as usize * pixel);
        if kept * crop.height as usize != buf.len() {
            return Err(failed(anyhow!("The HEIF image is not the size it says")));
        }
        for (y, out) in buf.chunks_exact_mut(kept).enumerate() {
            let start = (crop.top as usize + y) * row + crop.left as usize * pixel;
            out.copy_from_slice(&rgb[start..start + kept]);
        }
        Ok(())
    }
    fn read_image_boxed(self: Box<Self>, buf: &mut [u8]) -> image::ImageResult<()> {
        (*self).read_image(buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rawmakase_interop::isobmff::CleanAperture;

    fn clap(width: (u32, u32), height: (u32, u32), h: (i32, u32), v: (i32, u32)) -> Property {
        Property::CleanAperture(CleanAperture {
            width,
            height,
            horizontal_offset: h,
            vertical_offset: v,
        })
    }

    #[test]
    fn a_clean_aperture_crops_around_its_centre() {
        let crop_of = |p: Property, w, h| crop(&[&p], w, h);
        // libheif's 16x12 coded as 64x64: centred, offset half the difference.
        assert_eq!(
            crop_of(clap((16, 1), (12, 1), (-24, 1), (-26, 1)), 64, 64).unwrap(),
            Crop {
                left: 0,
                top: 0,
                width: 16,
                height: 12
            }
        );
        // avifenc's --crop 2,2,8,8 of 16x12: a centre 2 left of the picture's.
        assert_eq!(
            crop_of(clap((8, 1), (8, 1), (-4, 2), (0, 1)), 16, 12).unwrap(),
            Crop {
                left: 2,
                top: 2,
                width: 8,
                height: 8
            }
        );
        // An odd difference: the left edge rounds down.
        assert_eq!(
            crop_of(clap((3, 1), (4, 1), (0, 1), (0, 1)), 4, 4)
                .unwrap()
                .left,
            0
        );
        // Without a clap, the whole picture.
        assert_eq!(
            crop(&[], 5, 7).unwrap(),
            Crop {
                left: 0,
                top: 0,
                width: 5,
                height: 7
            }
        );
    }

    #[test]
    fn a_clean_aperture_outside_the_picture_is_refused() {
        for p in [
            clap((17, 1), (12, 1), (0, 1), (0, 1)),
            clap((8, 1), (8, 1), (5, 1), (0, 1)),
            clap((8, 1), (8, 1), (0, 1), (-3, 1)),
            clap((0, 1), (8, 1), (0, 1), (0, 1)),
        ] {
            assert!(crop(&[&p], 16, 12).is_err(), "{p:?}");
        }
    }

    /// A picture of `w` × `h` whose samples are `value`.
    fn flat(w: u32, h: u32, subsampled: bool, value: u8) -> hevc::Picture {
        let like = hevc::Picture {
            width: 0,
            height: 0,
            depth: 8,
            subsampled,
            planes: hevc::Planes::Bytes(Default::default()),
        };
        let mut picture = blank(&like, w, h);
        if let hevc::Planes::Bytes(planes) = &mut picture.planes {
            planes.iter_mut().for_each(|p| p.fill(value));
        }
        picture
    }

    #[test]
    fn tiles_are_laid_in_place_and_cut_at_the_edge() {
        let mut canvas = flat(6, 5, true, 0);
        place(&mut canvas, &flat(4, 4, true, 1), 0, 0).unwrap();
        place(&mut canvas, &flat(4, 4, true, 2), 4, 0).unwrap();
        place(&mut canvas, &flat(4, 4, true, 3), 0, 4).unwrap();
        place(&mut canvas, &flat(4, 4, true, 4), 4, 4).unwrap();
        let hevc::Planes::Bytes([y, u, _]) = &canvas.planes else {
            unreachable!()
        };
        assert_eq!(&y[..6], [1, 1, 1, 1, 2, 2]);
        assert_eq!(&y[24..], [3, 3, 3, 3, 4, 4]);
        // Chroma is 3 × 3.
        assert_eq!(u, &[1, 1, 2, 1, 1, 2, 3, 3, 4]);
        // A subsampled tile at an odd column is refused.
        assert!(place(&mut canvas, &flat(3, 4, true, 1), 3, 0).is_err());
        // As is a tile of other chroma.
        assert!(place(&mut canvas, &flat(4, 4, false, 1), 0, 0).is_err());
    }
}
