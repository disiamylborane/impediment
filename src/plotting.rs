
extern crate vecmath;
extern crate cairo;

use crate::imped_math::*;


/// Ensure the point is inside the bounds 
pub fn update_bounds(val: V2, bounds: &mut Bounds) {
    if bounds.min.x > val.x {
        bounds.min.x = val.x;
    }
    if bounds.min.y > val.y {
        bounds.min.y = val.y;
    }
    if bounds.max.x < val.x {
        bounds.max.x = val.x;
    }
    if bounds.max.y < val.y {
        bounds.max.y = val.y;
    }
}

#[derive(PartialEq, Clone, Copy)]
pub enum DisplayMarker{Ball, Line}

/// Convert a datapoint coordinate to a DrawingArea coordinate
fn coords(value: V2, viewport: Bounds, bounds: Bounds) -> V2 {
    let x = {
        let _xdif = bounds.max.x-bounds.min.x;
        let xdif = if _xdif != 0.0 {_xdif} else {1.0};
        let xfrac = (value.x-bounds.min.x) / xdif;
        xfrac*(viewport.max.x - viewport.min.x) + viewport.min.x
    };

    let y = {
        let _ydif = bounds.max.y-bounds.min.y;
        let ydif = if _ydif != 0.0 {_ydif} else {1.0};
        let yfrac = (value.y-bounds.min.y) / ydif;
        viewport.max.y - yfrac*(viewport.max.y - viewport.min.y)
    };

    V2{x, y}
}

/// Plot a single datarow on a DrawingArea
#[allow(dead_code)]
pub fn display(
    data: &mut dyn Iterator<Item=V2>,
    ctx: &cairo::Context,
    area_size: V2,
    bounds: Bounds,
    marker: DisplayMarker
)
{
    let border = 5.;

    // Setup axes
    let minx = format!("{:.1e}", bounds.min.x);
    let maxx = format!("{:.1e}", bounds.max.x);

    let extents = ctx.text_extents(&maxx);

    let miny = format!("{:.1e}", bounds.min.y);
    let maxy = format!("{:.1e}", bounds.max.y);

    let miny_size = ctx.text_extents(&miny);
    let maxy_size = ctx.text_extents(&maxy);

    let yticks_size = miny_size.width .max (maxy_size.width);

    ctx.move_to(0., miny_size.height + border);
    ctx.set_source_rgb(0.0, 0.0, 0.0);
    ctx.show_text(&maxy);

    ctx.move_to(0., area_size.y - 1. - extents.height - border);
    ctx.set_source_rgb(0.0, 0.0, 0.0);
    ctx.show_text(&miny);

    ctx.move_to(yticks_size + border, area_size.y-1.);
    ctx.set_source_rgb(0.0, 0.0, 0.0);
    ctx.show_text(&minx);

    ctx.move_to(area_size.x - border - extents.width, area_size.y-1.);
    ctx.set_source_rgb(0.0, 0.0, 0.0);
    ctx.show_text(&maxx);

    let viewport = Bounds{min: V2{x: yticks_size + border, y: border}, max: V2{x: area_size.x-border, y: area_size.y-extents.height-border}};

    ctx.rectangle(viewport.min.x, viewport.min.y, viewport.max.x-viewport.min.x, viewport.max.y-viewport.min.y);

    ctx.set_source_rgb(0.0, 0.0, 0.0);
    ctx.set_line_width(1.);
    ctx.stroke();

    let draw_circle = |point: V2| {
        ctx.move_to(point.x, point.y);
        ctx.arc(point.x, point.y, 3., 0., 2.*std::f64::consts::PI);
    };

    // Move to the place of the first
    {
        let mt = data.take(1);
        for d in mt {
            let point = coords(d, viewport, bounds);
            match marker {
                DisplayMarker::Ball => {draw_circle(point);}
                DisplayMarker::Line => {ctx.move_to(point.x, point.y);}
            }
        }
    }

    // Draw all the points elementwise
    for d in data {
        let point = coords(d, viewport, bounds);
        match marker {
            DisplayMarker::Ball => { draw_circle(point); }
            DisplayMarker::Line => { ctx.line_to(point.x, point.y); }
        }
    }

    match marker {
        DisplayMarker::Ball => {
            ctx.set_source_rgb(0.4, 0.4, 0.0);
            ctx.fill();
        }
        DisplayMarker::Line => {
            ctx.set_source_rgb(0.0, 0.0, 1.0);
            ctx.set_line_width(2.);
            ctx.stroke();
        }
    };
}


pub struct NyquistExtractor {}
pub struct BodeAmpExtractor {}
pub struct BodePhaseExtractor {}

/// The converter from DataPiece{angular freq, impedance} to V2{x,y}
pub trait DataExtractor {
    fn extract(val: DataPiece) -> V2;
}

impl DataExtractor for NyquistExtractor{
    fn extract(val: DataPiece) -> V2 {
        V2{x: val.imp.re, y: -val.imp.im}
    }
}

impl DataExtractor for BodeAmpExtractor{
    fn extract(val: DataPiece) -> V2 {
        V2{x: val.omega.log10(), y: (val.imp.re*val.imp.re + val.imp.im*val.imp.im).sqrt()}
    }
}

impl DataExtractor for BodePhaseExtractor{
    fn extract(val: DataPiece) -> V2 {
        V2{x: val.omega.log10(), y: val.imp.re.atan2(val.imp.im)}
    }
}

/// Converts Iterator<DataPiece> to Iterator<V2> using `DataExtractor` as a helper
pub struct DataIter<'a, T>
{
    pub stored: &'a mut dyn Iterator<Item=DataPiece>,
    pub _d: std::marker::PhantomData<T>,
}

impl<T> Iterator for DataIter<'_, T> 
    where T: DataExtractor
{
    type Item = V2;
    
    fn next(&mut self) -> Option<V2> {
        let new_next = self.stored.next();

        match new_next {
            Some(val) => {Some(T::extract(val))}
            None => None
        }
    }
}


/// Plot a spectrum of a model and experimental data points
/// using `DataExtractor` as a helper
pub fn plot_model<'a, EXT: DataExtractor>(
    model: Model,
    ctx: &cairo::Context,
    area_size: V2,
    bounds: Option<Bounds>,
    experimental : Option<&'a [DataPiece]>,
)
{
    let m1 = Model{..model};
    let m2 = Model{..model};
    let bounds = match bounds {
        Some(bounds) => bounds,
        None => {
            let mut points : Box<dyn Iterator<Item=f64>> = match experimental {
                Some(data) => {Box::new(data.iter().map(|x| x.omega))}
                None => {Box::new(geomspace(0.1, 10000.0, 100))}
            };
            let mut chbounds = Bounds{min: V2{x: std::f64::INFINITY, y:std::f64::INFINITY}, max: V2{x: std::f64::NEG_INFINITY, y:std::f64::NEG_INFINITY}};
            let data = DataIter::<EXT>{stored: &mut ModelIter{model : m1, points: &mut points}, _d: std::marker::PhantomData{}};
            for d in data {
                update_bounds(d,&mut chbounds);
            }

            if let Some(e) = experimental {
                let data = DataIter::<EXT>{stored: &mut e.iter().map(|x| x.clone()), _d: std::marker::PhantomData{}};
                for d in data {
                    update_bounds(d,&mut chbounds);
                }
            }

            chbounds
        }
    };

    let mut points : Box<dyn Iterator<Item=f64>> = match experimental {
        Some(data) => {Box::new(data.iter().map(|x| x.omega))}
        None => {Box::new(geomspace(0.1, 10000.0, 100))}
    };
    let mut miter = ModelIter{model : m2, points: &mut points};
    let mut data = DataIter::<EXT>{stored: &mut miter, _d: std::marker::PhantomData{}};


    if let Some(exper) = experimental {
        let mut realvals = DataIter::<EXT>{stored: &mut exper.iter().map(|x| x.clone()), _d: std::marker::PhantomData{}};

        ctx.set_source_rgb(0.4, 0.4, 0.0);
        display(&mut realvals, ctx, area_size, bounds, DisplayMarker::Ball);
        ctx.fill();
    }

    ctx.set_source_rgb(0.0, 0.0, 1.0);
    display(&mut data, ctx, area_size, bounds, DisplayMarker::Line);
    ctx.stroke();
}



#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_update_bounds() {
        let mut bounds = Bounds{min: V2{x: 0.0, y: 0.0}, max: V2{x: 0.0, y: 0.0}};

        update_bounds(V2{x: 25.0, y: 25.0}, &mut bounds);
        assert_eq!(bounds, Bounds{min: V2{x: 0.0, y: 0.0}, max: V2{x: 25.0, y: 25.0}});

        update_bounds(V2{x: 12.0, y: 12.0}, &mut bounds);
        assert_eq!(bounds, Bounds{min: V2{x: 0.0, y: 0.0}, max: V2{x: 25.0, y: 25.0}});

        update_bounds(V2{x: 12.0, y: 25.0}, &mut bounds);
        assert_eq!(bounds, Bounds{min: V2{x: 0.0, y: 0.0}, max: V2{x: 25.0, y: 25.0}});

        update_bounds(V2{x: 70.0, y: 25.0}, &mut bounds);
        assert_eq!(bounds, Bounds{min: V2{x: 0.0, y: 0.0}, max: V2{x: 70.0, y: 25.0}});

        update_bounds(V2{x: -30.0, y: 45.0}, &mut bounds);
        assert_eq!(bounds, Bounds{min: V2{x: -30.0, y: 0.0}, max: V2{x: 70.0, y: 45.0}});

        update_bounds(V2{x: -70.0, y: -70.0}, &mut bounds);
        assert_eq!(bounds, Bounds{min: V2{x: -70.0, y: -70.0}, max: V2{x: 70.0, y: 45.0}});
    }
}
