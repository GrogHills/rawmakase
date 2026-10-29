//! A HEIC's pixels decoded by macOS's ImageIO, whose HEVC decoder is the
//! system's, so RAWmakase ships none there.
//!
//! ImageIO decodes the file's primary image, grid and all. It is drawn into a
//! bitmap of the colour space the header says (its ICC profile, one built
//! from its CICP, or sRGB) at the depth the header says, so the pixels are
//! what [`HeicFile`]'s colour type and profile promise whatever colour ImageIO
//! itself makes of the file: CoreGraphics converts them when the two differ.
//! The image ImageIO makes is not turned (RAWmakase turns it by the header's
//! `irot` and `imir`); whether it is cropped by the `clap` or not, the caller
//! crops what it was not.
use super::super::avif;
use super::{HeicFile, Pixels};
use anyhow::{Result, anyhow, ensure};
use objc2_core_foundation::{CFData, CFURL, CGPoint, CGRect, CGSize};
use objc2_core_graphics::{
    CGBitmapContextCreate, CGColorSpace, CGContext, CGImage, CGImageAlphaInfo,
    CGImageByteOrderInfo, CGInterpolationQuality, kCGColorSpaceSRGB,
};
use objc2_image_io::CGImageSource;

pub(super) fn pixels(file: HeicFile) -> Result<Pixels> {
    let failed = || anyhow!("macOS could not decode the HEIF image");
    let url = CFURL::from_file_path(&file.path).ok_or_else(failed)?;
    // SAFETY: no options are passed.
    let source = unsafe { CGImageSource::with_url(&url, None) }.ok_or_else(failed)?;
    // SAFETY: the index is the source's own, and no options are passed.
    let image =
        unsafe { source.image_at_index(source.primary_image_index(), None) }.ok_or_else(failed)?;
    let (width, height) = (CGImage::width(Some(&image)), CGImage::height(Some(&image)));
    ensure!(
        width > 0 && height > 0,
        "macOS decoded the HEIF image empty"
    );
    let space = match avif::profile(file.icc.as_ref(), file.nclx) {
        // SAFETY: the data is a CFData, as the function takes.
        Some(icc) => unsafe { CGColorSpace::with_icc_data(Some(&CFData::from_bytes(&icc))) },
        // SAFETY: kCGColorSpaceSRGB is a constant CoreGraphics defines.
        None => CGColorSpace::with_name(Some(unsafe { kCGColorSpaceSRGB })),
    }
    .ok_or_else(|| anyhow!("macOS can't use the HEIF image's colour profile"))?;
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
