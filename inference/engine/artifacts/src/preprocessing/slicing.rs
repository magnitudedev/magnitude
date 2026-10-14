//! Aspect-preserving overview and detail crops, with a bounded slice grid.
use super::{ImageProcessor, resize};
use image::{GenericImageView, Rgb32FImage};

fn divisible(value: f64, factor: usize) -> usize {
    ((value / factor as f64).round_ties_even() as usize).max(1) * factor
}

pub(super) fn best_size(
    height: f64,
    width: f64,
    resolution: usize,
    patch: usize,
    upscale: bool,
) -> (usize, usize) {
    let (height, width) = if height * width > (resolution * resolution) as f64 || upscale {
        let ratio = width / height;
        let height = (resolution as f64 / ratio.sqrt()).floor();
        (height, (height * ratio).floor())
    } else {
        (height, width)
    };
    (divisible(height, patch * 4), divisible(width, patch * 4))
}

pub(super) fn grid(height: usize, width: usize, resolution: usize, maximum: usize) -> [usize; 2] {
    let multiple = ((height as f64 * width as f64 / (resolution as f64 * resolution as f64)).ceil()
        as usize)
        .min(maximum);
    if multiple <= 1 {
        return [0, 0];
    }
    let target = (width as f64 / height as f64).ln();
    let (mut best, mut error) = ([1, 1], f64::INFINITY);
    for count in multiple - 1..=multiple + 1 {
        if count == 1 || count > maximum {
            continue;
        }
        for rows in 1..=count {
            if count % rows != 0 {
                continue;
            }
            let columns = count / rows;
            let candidate = (target - (columns as f64 / rows as f64).ln()).abs();
            if candidate < error || (candidate == error && rows > best[0]) {
                best = [rows, columns];
                error = candidate;
            }
        }
    }
    best
}

impl ImageProcessor {
    pub(super) fn slices(
        &self,
        source: &Rgb32FImage,
        resolution: usize,
        maximum: usize,
        prior_rows: usize,
    ) -> Result<([usize; 2], Vec<Rgb32FImage>), String> {
        let (height, width) = (source.height() as usize, source.width() as usize);
        // The same source-domain checks apply to the overview and every detail crop.
        self.resized_size(height, width)?;
        let grid = grid(height, width, resolution, maximum);
        let (h, w) = best_size(
            height as f64,
            width as f64,
            resolution,
            self.config.patch,
            grid == [0, 0],
        );
        let overview_rows = self.prepared_rows(prior_rows, h, w)?;
        let detail = if grid != [0, 0] {
            let [rows, columns] = grid;
            let h = divisible(height as f64, rows) as f64 / rows as f64;
            let w = divisible(width as f64, columns) as f64 / columns as f64;
            let (h, w) = best_size(h, w, resolution, self.config.patch, true);
            let full_h = h
                .checked_mul(rows)
                .ok_or("refined image height overflows")?;
            let full_w = w
                .checked_mul(columns)
                .ok_or("refined image width overflows")?;
            self.prepared_rows(overview_rows, full_h, full_w)?;
            Some((h, w, full_h, full_w))
        } else {
            None
        };
        let mut crops = vec![resize(source, w, h, self.config.resampling)?];
        if let Some((h, w, full_h, full_w)) = detail {
            let [rows, columns] = grid;
            let refined = resize(source, full_w, full_h, self.config.resampling)?;
            for row in 0..rows {
                for column in 0..columns {
                    crops.push(
                        refined
                            .view((column * w) as u32, (row * h) as u32, w as u32, h as u32)
                            .to_image(),
                    );
                }
            }
        }
        Ok((grid, crops))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn overview_and_detail_grids_follow_the_reference_rounding() {
        assert_eq!(grid(448, 448, 448, 9), [0, 0]);
        assert_eq!(grid(1000, 2000, 448, 9), [2, 4]);
        assert_eq!(best_size(1000., 2000., 448, 14, false), (336, 616));
        assert_eq!(best_size(100., 200., 448, 14, true), (336, 616));
        assert_eq!(best_size(448., 448., 448, 14, true), (448, 448));
        assert_eq!(grid(896, 896, 448, 9), [2, 2]);
    }
}
