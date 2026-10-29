//! A HEIC's pixels decoded by macOS's ImageIO, whose HEVC decoder is the
//! system's, so RAWmakase ships none there.
//!
//! ImageIO decodes the file's primary image, grid and all. It is drawn into a
//! bitmap of the colour space the header says (its ICC profile, one built
//! from its CICP, or sRGB) at the depth the header says, so the pixels are
//! what [`HeicFile`]'s colour type and profile promise whatever colour ImageIO
//! itself makes of the file: CoreGraphics converts them when the two differ.
//! An HDR file's CICP (PQ or HLG, BT.2020) is drawn into CoreGraphics' own
//! space of that name, the one it knows the transfer of. A profile is used
//! only when moxcms, which reads it for the rest of RAWmakase, finds it an RGB
//! one, and CoreGraphics reads it as RGB too; otherwise the pixels are drawn
//! as sRGB, as RAWmakase takes such a profile. An image with alpha comes out
//! composited over black, as CoreGraphics premultiplies it (RAWmakase's own
//! decoder ignores alpha and keeps the stored colour); photos rarely have
//! any.
//!
//! The image ImageIO makes is not turned (RAWmakase turns it by the header's
//! `irot` and `imir`); whether it is cropped by the `clap` or not, the caller
//! crops what it was not. It all runs in an autorelease pool, so nothing the
//! frameworks autorelease outlives the call on a long-lived worker thread.
use super::super::avif;
use super::{HeicFile, Pixels};
use anyhow::{Result, anyhow, ensure};
use moxcms::{ColorProfile, DataColorSpace};
use objc2::rc::autoreleasepool;
use objc2_core_foundation::{CFData, CFRetained, CFString, CFURL, CGPoint, CGRect, CGSize};
use objc2_core_graphics::{
    CGBitmapContextCreate, CGColorSpace, CGColorSpaceModel, CGContext, CGImage, CGImageAlphaInfo,
    CGImageByteOrderInfo, CGInterpolationQuality, kCGColorSpaceITUR_2100_HLG,
    kCGColorSpaceITUR_2100_PQ, kCGColorSpaceSRGB,
};
use objc2_image_io::CGImageSource;

pub(super) fn pixels(file: HeicFile) -> Result<Pixels> {
    autoreleasepool(|_| decode(&file))
}

/// The colour space the header says: CoreGraphics' own for an HDR CICP in
/// BT.2020, else the header's profile, and sRGB for none, or one that moxcms
/// or CoreGraphics can't read as RGB.
fn space(file: &HeicFile) -> Option<CFRetained<CGColorSpace>> {
    let named = |name: &CFString| CGColorSpace::with_name(Some(name));
    // SAFETY: constants CoreGraphics defines.
    let (pq, hlg, srgb) = unsafe {
        (
            kCGColorSpaceITUR_2100_PQ,
            kCGColorSpaceITUR_2100_HLG,
            kCGColorSpaceSRGB,
        )
    };
    let space = match (&file.icc, file.nclx.map(|n| (n.primaries, n.transfer))) {
        (None, Some((9, 16))) => named(pq),
        (None, Some((9, 18))) => named(hlg),
        _ => avif::profile(file.icc.as_ref(), file.nclx)
            .filter(|icc| {
                ColorProfile::new_from_slice(icc)
                    .is_ok_and(|profile| profile.color_space == DataColorSpace::Rgb)
            })
            .and_then(|icc| {
                // SAFETY: the data is a CFData, as the function takes.
                unsafe { CGColorSpace::with_icc_data(Some(&CFData::from_bytes(&icc))) }
            }),
    };
    space
        .filter(|space| CGColorSpace::model(Some(space)) == CGColorSpaceModel::RGB)
        .or_else(|| named(srgb))
}

fn decode(file: &HeicFile) -> Result<Pixels> {
    let failed = || anyhow!("macOS could not decode the HEIF image");
    let url = CFURL::from_file_path(&file.path).ok_or_else(failed)?;
    // SAFETY: no options are passed.
    let source = unsafe { CGImageSource::with_url(&url, None) }.ok_or_else(failed)?;
    // SAFETY: the index is the source's own, and no options are passed.
    let image =
        unsafe { source.image_at_index(source.primary_image_index(), None) }.ok_or_else(failed)?;
    let (width, height) = (CGImage::width(Some(&image)), CGImage::height(Some(&image)));
    ensure!(
        file.expects(u32::try_from(width)?, u32::try_from(height)?),
        "The HEIF image is not the size it says"
    );
    let space = space(file).ok_or_else(|| anyhow!("macOS can't make an sRGB colour space"))?;
    // RGB with an unused fourth sample, as CoreGraphics draws no packed RGB:
    // 8-bit, or 16-bit in the Mac's own (little-endian) order.
    let (bits, order) = if file.deep {
        (16, CGImageByteOrderInfo::Order16Little)
    } else {
        (8, CGImageByteOrderInfo::OrderDefault)
    };
    let sample = bits / 8;
    let row = width * 4 * sample;
    let mut rgbx = vec![0u8; row * height];
    // SAFETY: `rgbx` holds `height` rows of `row` bytes, outlives the
    // context, and nothing else uses it while the context draws.
    let context = unsafe {
        CGBitmapContextCreate(
            rgbx.as_mut_ptr().cast(),
            width,
            height,
            bits,
            row,
            Some(&space),
            CGImageAlphaInfo::NoneSkipLast.0 | order.0,
        )
    }
    .ok_or_else(failed)?;
    CGContext::set_interpolation_quality(Some(&context), CGInterpolationQuality::None);
    let size = CGSize::new(width as f64, height as f64);
    CGContext::draw_image(
        Some(&context),
        CGRect::new(CGPoint::new(0., 0.), size),
        Some(&image),
    );
    drop(context);
    let rgb = rgbx
        .chunks_exact(4 * sample)
        .flat_map(|pixel| &pixel[..3 * sample])
        .copied()
        .collect();
    Ok(Pixels {
        width: u32::try_from(width)?,
        height: u32::try_from(height)?,
        rgb,
    })
}
