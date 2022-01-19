
extern crate num;
use std::{f64::consts::PI, vec};
use eframe::egui::{Color32, Pos2, Rect, Stroke, pos2, vec2};

use crate::{Cplx, ParameterDesc, ParameterEditability};

const I: Cplx = Cplx{ re: 0.0, im: 1.0 };

#[derive(Clone, Copy, Debug)]
pub enum Element {
    Resistor,
    Capacitor,
    Inductor,
    Warburg,
    Cpe,
}

impl Element {
    fn exchange_param<'a>(self, element: Self, idx: usize, paramlist: impl Iterator<Item = &'a mut ParameterDesc>, editability: &mut Vec<ParameterEditability>) {
        for pdesc in paramlist {
            match self {
                Element::Resistor => {pdesc.remove(idx);},
                Element::Capacitor => {pdesc.remove(idx);},
                Element::Inductor => {pdesc.remove(idx);},
                Element::Warburg => {pdesc.remove(idx);},
                Element::Cpe => {pdesc.remove(idx); pdesc.remove(idx);},
            }
            match element {
                Element::Resistor => {pdesc.insert(idx, (100., (1., 100000.)))},
                Element::Capacitor => {pdesc.insert(idx, (0.000001, (0.000000001, 0.1)))},
                Element::Inductor => {pdesc.insert(idx, (0.000001, (0.000000001, 0.1)))},
                Element::Warburg => {pdesc.insert(idx, (10., (1., 100000.)))},
                Element::Cpe => {
                    pdesc.insert(idx, (0.8, (0.01, 0.99)));
                    pdesc.insert(idx, (0.000001, (0.000000001, 0.1)));
                },
            }
        }
        match self {
            Element::Resistor => editability.remove(idx),
            Element::Capacitor => editability.remove(idx),
            Element::Inductor => editability.remove(idx),
            Element::Warburg => editability.remove(idx),
            Element::Cpe => {
                editability.remove(idx);
                editability.remove(idx)
            }
        };
        match element {
            Element::Resistor => editability.insert(idx, ParameterEditability::Plural),
            Element::Capacitor => editability.insert(idx, ParameterEditability::Plural),
            Element::Inductor => editability.insert(idx, ParameterEditability::Plural),
            Element::Warburg => editability.insert(idx, ParameterEditability::Plural),
            Element::Cpe => {
                editability.insert(idx, ParameterEditability::Plural);
                editability.insert(idx, ParameterEditability::Plural);
            }
        }
    }
    fn remove_param<'a>(self, idx: usize, paramlist: impl Iterator<Item = &'a mut ParameterDesc>, editability: &mut Vec<ParameterEditability>) {
        for pdesc in paramlist {
                match self {
                Element::Resistor => {pdesc.remove(idx);},
                Element::Capacitor => {pdesc.remove(idx);},
                Element::Inductor => {pdesc.remove(idx);},
                Element::Warburg => {pdesc.remove(idx);},
                Element::Cpe => {pdesc.remove(idx); pdesc.remove(idx);},
            }
        }
        match self {
            Element::Resistor => editability.remove(idx),
            Element::Capacitor => editability.remove(idx),
            Element::Inductor => editability.remove(idx),
            Element::Warburg => editability.remove(idx),
            Element::Cpe => {
                editability.remove(idx);
                editability.remove(idx)
            }
        };
    }
    fn insert_param<'a>(self, idx: usize, paramlist: impl Iterator<Item = &'a mut ParameterDesc>, editability: &mut Vec<ParameterEditability>) {
        for pdesc in paramlist {
            match self {
                Element::Resistor => {pdesc.insert(idx, (100., (1., 100000.)))},
                Element::Capacitor => {pdesc.insert(idx, (0.000001, (0.000000001, 0.1)))},
                Element::Inductor => {pdesc.insert(idx, (0.000001, (0.000000001, 0.1)))},
                Element::Warburg => {pdesc.insert(idx, (10., (1., 100000.)))},
                Element::Cpe => {
                    pdesc.insert(idx, (0.8, (0.01, 0.99)));
                    pdesc.insert(idx, (0.000001, (0.000000001, 0.1)));
                },
            }
        }

        match self {
            Element::Resistor => editability.insert(idx, ParameterEditability::Plural),
            Element::Capacitor => editability.insert(idx, ParameterEditability::Plural),
            Element::Inductor => editability.insert(idx, ParameterEditability::Plural),
            Element::Warburg => editability.insert(idx, ParameterEditability::Plural),
            Element::Cpe => {
                editability.insert(idx, ParameterEditability::Plural);
                editability.insert(idx, ParameterEditability::Plural);
            }
        }
    }
}

#[derive(Clone, Debug)]
pub enum Circuit {
    Element(Element),
    Series(Vec<Circuit>),
    Parallel(Vec<Circuit>),
}


fn series_index_by_block(series: &[Circuit], block: (u16, u16)) -> Option<(usize, u16)> {
    let mut start_x = 0_u16;
    for (i, el) in &mut series.iter().enumerate() {
        let elemsize = el.painted_size();
        if start_x + elemsize.0 > block.0 {
            if block.1 < elemsize.1 {
                return Some((i, start_x));
            }
            return None;
        }
        start_x += elemsize.0;
    }
    None
}

fn parallel_painted_size(elems: &[Circuit]) -> (u16, u16) {
    let s = elems.iter()
                 .map(|x| x.painted_size())
                 .fold((0,0), |a, b| (std::cmp::max(a.0,b.0), a.1+b.1));
    (s.0 + 2, s.1)
}


enum ParallelBlockPicked {
    Child(usize, u16, u16),
    This,
    None
}


fn parallel_index_by_block(parallel: &[Circuit], block: (u16, u16)) -> ParallelBlockPicked {
        let wsize = parallel_painted_size(parallel);
        if block.1 < wsize.1 && (block.0 == 0 || block.0 == wsize.0-1) {
            return ParallelBlockPicked::This;
        }

        let xsize = parallel_painted_size(parallel).0;

        let mut start_coord_y = 0_u16;
        for (i,el) in &mut parallel.iter().enumerate() {
            let elemsize = el.painted_size();
            if start_coord_y + elemsize.1 > block.1 {
                let psize = el.painted_size().0;
                let elemblock = (xsize-2-psize)/2 + 1;

                if block.0 >= elemblock && block.0 < elemblock+psize {
                    return ParallelBlockPicked::Child(i, elemblock, start_coord_y);
                }
                return ParallelBlockPicked::This;
            }
            start_coord_y += elemsize.1;
        }
        ParallelBlockPicked::None
}


/// Paint a sign under the element
fn sign_element(ctx: &eframe::egui::Painter, text: &str, pos: Pos2, blocksize: f32, color: Color32) {
    ctx.text(pos+vec2(blocksize, blocksize*3./2.), eframe::egui::Align2::CENTER_CENTER, text, eframe::egui::TextStyle::Body, color);
}


impl Circuit {
    pub fn generate_new_params(&self) -> ParameterDesc {
        match self {
            Circuit::Element(element) => {
                let pd = match element {
                    Element::Resistor => {(100., (1., 100000.))},
                    Element::Capacitor => {(0.000001, (0.000000001, 0.1))},
                    Element::Inductor => {(0.000001, (0.000000001, 0.1))},
                    Element::Warburg => {(10., (1., 100000.))},
                    Element::Cpe => {
                        return ParameterDesc{ vals: vec![0.8, 1e-6], bounds: vec![(0.01, 0.99), (0.000000001, 0.1)] };
                    },
                };
                ParameterDesc{ vals: vec![pd.0], bounds: vec![pd.1] }
            }
            Circuit::Series(s)|Circuit::Parallel(s) => {
                let mut evals = vec![];
                let mut ebounds = vec![];
                for ParameterDesc{mut vals, mut bounds} in s.iter().map(|x| x.generate_new_params()) {
                    evals.append(&mut vals);
                    ebounds.append(&mut bounds);
                }
                ParameterDesc{vals: evals, bounds: ebounds}
            },
        }
    }

    pub fn param_names(&self) -> Vec<String> {
        self.param_names_rec(0).0
    }

    pub fn param_names_rec(&self, mut start_index: usize) -> (Vec<String>, usize) {
        match self {
            Circuit::Element(Element::Resistor) => (vec![format!("R{}", start_index)], start_index+1),
            Circuit::Element(Element::Capacitor) => (vec![format!("C{}", start_index)], start_index+1),
            Circuit::Element(Element::Inductor) => (vec![format!("L{}", start_index)], start_index+1),
            Circuit::Element(Element::Warburg) => (vec![format!("W{}", start_index)], start_index+1),
            Circuit::Element(Element::Cpe) => (vec![format!("Q{}", start_index), format!("n{}", start_index)], start_index+1),
            Circuit::Series(elems)|Circuit::Parallel(elems) => {
                let mut out = vec![];
                for e in elems {
                    let ps = e.param_names_rec(start_index);
                    out.extend(ps.0);
                    start_index = ps.1;
                }
                (out, start_index)
            }
        }
    }

    pub fn painted_size(&self) -> (u16, u16) {
        match self {
            Circuit::Element(_) => (2,2),

            Circuit::Series(elems) => 
                elems.iter()
                     .map(|x| x.painted_size())
                     .fold((0,0), |a, b| (a.0+b.0, std::cmp::max(a.1,b.1))),

            Circuit::Parallel(elems) =>  {
                let s = elems.iter()
                             .map(|x| x.painted_size())
                             .fold((0,0), |a, b| (std::cmp::max(a.0,b.0), a.1+b.1));
                (s.0 + 2, s.1)
            }
        }
    }

    pub fn paint(&self, pos: eframe::egui::Pos2, blocksize: f32, painter: &eframe::egui::Painter, start_index: usize, color: Color32)->usize {
        let stroke = Stroke::new(1., color);
        match self {
            Circuit::Element(Element::Resistor) => {
                painter.line_segment([pos+vec2(0., blocksize/2.), pos+vec2(blocksize/4., blocksize/2.)], stroke);
                painter.rect_stroke(Rect::from_min_size(pos+vec2(blocksize/4., blocksize/4.), vec2(blocksize*3./2., blocksize/2.)), 0., stroke);
                painter.line_segment([pos+vec2(blocksize*7./4., blocksize/2.), pos+vec2(blocksize*2., blocksize/2.)], stroke);
                sign_element(painter, &format!("R{}", start_index), pos, blocksize, color);
                start_index+1
            }
            Circuit::Element(Element::Capacitor) => {
                painter.line_segment([pos+vec2(0., blocksize/2.), pos+vec2(blocksize*3./4., blocksize/2.)], stroke);
                painter.line_segment([pos+vec2(blocksize*3./4., 0.), pos+vec2(blocksize*3./4., blocksize)], stroke);
                painter.line_segment([pos+vec2(blocksize*5./4., 0.), pos+vec2(blocksize*5./4., blocksize)], stroke);
                painter.line_segment([pos+vec2(blocksize*5./4., blocksize/2.), pos+vec2(blocksize*2., blocksize/2.)], stroke);
                sign_element(painter, &format!("C{}", start_index), pos, blocksize, color);
                start_index+1
            },
            Circuit::Element(Element::Inductor) => {
                let mut spos = vec2(0., blocksize/2.);

                let mut lineto = |newpos| {
                    painter.line_segment([pos+spos, pos+newpos], stroke);
                    spos = newpos;
                };

                lineto(vec2(blocksize/4., blocksize/2.));
                    lineto(vec2(blocksize*5./12., blocksize/4.));
                    lineto(vec2(blocksize*7./12., blocksize/4.));
                lineto(vec2(blocksize*9./12., blocksize/2.));
                    lineto(vec2(blocksize*11./12., blocksize/4.));
                    lineto(vec2(blocksize*13./12., blocksize/4.));
                lineto(vec2(blocksize*15./12., blocksize/2.));
                    lineto(vec2(blocksize*17./12., blocksize/4.));
                    lineto(vec2(blocksize*19./12., blocksize/4.));
                lineto(vec2(blocksize*21./12., blocksize/2.));
                lineto(vec2(blocksize*2., blocksize/2.));

                sign_element(painter, &format!("L{}", start_index), pos, blocksize, color);
                start_index+1
            }

            Circuit::Element(Element::Warburg) => {
                let line = |p1,p2| {
                    painter.line_segment([pos+p1*blocksize, pos+p2*blocksize], stroke);
                };

                line(vec2(0.0, 0.5), vec2(0.75, 0.5));
                line(vec2(0.75, 0.0), vec2(0.75, 1.0));
                line(vec2(1.5, 0.0), vec2(1.25, 0.5));
                line(vec2(1.25, 0.5), vec2(1.5, 1.0));
                line(vec2(1.25, 0.5), vec2(2.0, 0.5));

                sign_element(painter, &format!("W{}", start_index), pos, blocksize, color);
                start_index+1
            }

            Circuit::Element(Element::Cpe) => {
                let line = |p1,p2| {
                    painter.line_segment([pos+p1*blocksize, pos+p2*blocksize], stroke);
                };

                line(vec2(0.0, 0.5), vec2(0.75, 0.5));
                line(vec2(0.75, 0.5), vec2(1.0, 0.0));
                line(vec2(0.75, 0.5), vec2(1.0, 1.0));
                line(vec2(1.5, 0.0), vec2(1.25, 0.5));
                line(vec2(1.25, 0.5), vec2(1.5, 1.0));
                line(vec2(1.25, 0.5), vec2(2.0, 0.5));

                sign_element(painter, &format!("Z{}", start_index), pos, blocksize, color);
                start_index+1
            }

            Circuit::Series(elems) => {
                let mut index = elems[0].paint(pos, blocksize, painter, start_index, color);
                let mut pos = pos + vec2(elems[0].painted_size().0 as f32 * blocksize, 0.);

                for c in elems[1..].iter() {
                    index = c.paint(pos, blocksize, painter, index, color);
                    pos += vec2(c.painted_size().0 as f32 * blocksize, 0.);
                }
                index
            }

            Circuit::Parallel(elems) => {
                let mut index = start_index;
                let mut y = pos.y;
                let xsize = self.painted_size().0;

                painter.line_segment([pos+vec2(0., blocksize/2.), pos+vec2(blocksize, blocksize/2.)], stroke);
                painter.line_segment([pos+vec2((xsize-1) as f32 * blocksize, blocksize/2.), pos+vec2(((xsize-1) as f32).mul_add(blocksize, blocksize), blocksize/2.)], stroke);

                for c in elems.iter() {
                    let psize = c.painted_size().0;
                    let drawend =  (xsize as f32).mul_add(blocksize, pos.x);
                    let elemblock = (xsize-2-psize)/2 + 1;
                    let elemstart = (elemblock as f32).mul_add(blocksize, pos.x);
                    let elemend = (psize as f32).mul_add(blocksize, elemstart);
                    index = c.paint(pos2(elemstart, y), blocksize, painter, index, color);

                    painter.line_segment([pos2(pos.x + blocksize/2., y+blocksize/2.), pos2(elemstart, y+blocksize/2.)], stroke);
                    painter.line_segment([pos2(elemend, y+blocksize/2.), pos2((xsize as f32).mul_add(blocksize, pos.x) - blocksize/2., y+blocksize/2.)], stroke);

                    painter.line_segment([pos+vec2(blocksize/2., blocksize/2.), pos2(pos.x + blocksize/2., y+blocksize/2.)], stroke);

                    painter.line_segment([pos2(drawend - blocksize/2., pos.y+blocksize/2.), pos2(drawend - blocksize/2., y+blocksize/2.)], stroke);
                    
                    y += c.painted_size().1 as f32 * blocksize;
                }
                index
            }
        }
    }

    pub fn paramlen(&self) -> usize {
        match self {
            Circuit::Element(Element::Resistor) => 1,
            Circuit::Element(Element::Capacitor) => 1,
            Circuit::Element(Element::Inductor) => 1,
            Circuit::Element(Element::Warburg) => 1,
            Circuit::Element(Element::Cpe) => 2,
            Circuit::Series(elems)|Circuit::Parallel(elems) => 
                elems.iter().map(|x| x.paramlen()).sum(),
        }
    }

    fn _impedance(&self, omega: f64, params: &[f64]) -> Cplx{
        match self {
            Circuit::Element(Element::Resistor) => Cplx::new(params[0], 0.0),

            Circuit::Element(Element::Capacitor) => 1.0 / (I * omega * params[0]),

            Circuit::Element(Element::Inductor) => I * omega * params[0],

            Circuit::Element(Element::Warburg) => (1.0 - I) * params[0] / omega.sqrt(),

            Circuit::Element(Element::Cpe) => {
                let q = params[0];
                let n = params[1];
                let numer = (-I*PI/2.0*n).exp();
                let denom = q * omega.powf(n);
                numer/denom
            }

            Circuit::Series(elems) => {
                let mut cval = 0;
                let mut imped = Cplx::new(0.0, 0.0);
                for c in elems.iter() {
                    let cend = cval + c.paramlen();
                    let slc = &params[cval..cend];
                    let ipd = c.impedance(omega, slc);
                    imped += ipd;
                    cval = cend;
                }
                imped
            }

            Circuit::Parallel(elems) => {
                let mut cval = 0;
                let mut admit = Cplx::new(0.0, 0.0);
                for c in elems.iter() {
                    let cend = cval + c.paramlen();
                    let slc = &params[cval..cend];
                    let ipd = c.impedance(omega, slc);
                    admit += 1.0/ipd;
                    cval = cend;
                }
                1.0/admit
            }
        }
    }

    pub fn impedance(&self, omega: f64, params: &[f64]) -> Cplx {
        assert!(params.len() == self.paramlen());
        self._impedance(omega, params)
    }

    pub fn _d_impedance(&self, omega: f64, params: &[f64], dparam: usize) -> Cplx {
        if dparam >= self.paramlen() {
            //panic!("Gradient requested for {} parameter of {}", dparam, self)
            panic!()
        }
        match self {
            Circuit::Element(Element::Resistor) => Cplx::new(1.0, 0.0),

            Circuit::Element(Element::Capacitor) => I / (omega * params[0].powi(2)),

            Circuit::Element(Element::Inductor) => I * omega,

            Circuit::Element(Element::Warburg) => ((1.0 - I)*params[0])/(omega.sqrt()),

            Circuit::Element(Element::Cpe) => {
                let q = params[0];
                let n = params[1];
                let numer = (-I*PI/2.0*n).exp();
                let denom = q * omega.powf(n);
                match dparam {
                    0 => {-numer/denom/q}
                    1 => {-(omega.ln() + I*PI/2.0)*numer/denom}
                    _ => unreachable!()
                }
            }

            Circuit::Series(elems) => {
                let mut cval = 0;
                for c in elems.iter() {
                    let cend = cval + c.paramlen();
                    let slc = &params[cval..cend];
                    if cval <= dparam && dparam<cend {
                        return c._d_impedance(omega, slc, dparam-cval);
                    }
                    cval = cend;
                }
                unreachable!()
            }

            Circuit::Parallel(elems) => {
                let mut cval = 0;
                for c in elems.iter() {
                    let cend = cval + c.paramlen();
                    let slc = &params[cval..cend];
                    if cval <= dparam && dparam<cend {
                        let denommed = self._impedance(omega, params);
                        return denommed.powi(2) / c.impedance(omega, slc).powi(2) * c._d_impedance(omega, slc, dparam-cval);
                    }
                    cval = cend;
                }
                unreachable!()
            }
        }
    }


    pub fn _replace<'a>(&mut self, 
                coord: (u16, u16), 
                element: Element, 
                paramlist: impl Iterator<Item = &'a mut ParameterDesc>, 
                param_idx: usize,
                editability: &mut Vec<ParameterEditability>) {
        match self {
            Circuit::Element(e) => {
                e.exchange_param(element, param_idx, paramlist, editability);
                editability[param_idx] = ParameterEditability::Plural;
                *self = Self::Element(element);
            },

            Circuit::Series(elems) => {
                assert!(elems.len() > 1);
                if let Some((i, start_x)) = series_index_by_block(elems, coord) {
                    let new_pidx = param_idx + (0..i).map(|el| elems[el].paramlen()).sum::<usize>();
                    let el = &mut elems[i];
                    el._replace((coord.0 - start_x, coord.1), element, paramlist, new_pidx, editability)
                }
            }

            Circuit::Parallel(elems) => {
                assert!(elems.len() > 1);
                let ib = parallel_index_by_block(elems, coord);

                match ib {
                    ParallelBlockPicked::This => {
                        // todo!()
                    },
                    ParallelBlockPicked::None => {
                    },
                    ParallelBlockPicked::Child(i, elemblock, start_y) => {
                        let new_pidx = param_idx + (0..i).map(|el| elems[el].paramlen()).sum::<usize>();
                        let el = &mut elems[i];
                        el._replace((coord.0 - elemblock, coord.1 - start_y), element, paramlist, new_pidx, editability);
                    }
                }
            }
        }
    }
 
    pub fn replace(&mut self, coord: (u16, u16), element: Element, paramlist: std::slice::IterMut<'_, ParameterDesc>, editability: &mut Vec<ParameterEditability>) {
        self._replace(coord, element, paramlist, 0, editability)
    }

    pub fn _add_series(&mut self, coord: (u16, u16), element: Element, paramlist: std::slice::IterMut<'_, ParameterDesc>, param_idx: usize, editability: &mut Vec<ParameterEditability>){
        match self {
            Circuit::Element(e) => {
                let new = Self::Series(vec![Self::Element(*e), Self::Element(element)]);
                let paramlen = self.paramlen();
                element.insert_param(param_idx+paramlen, paramlist, editability);
                *self = new;
            },

            Circuit::Series(elems) => {
                assert!(elems.len() > 1);
                if let Some((i, start_x)) = series_index_by_block(elems, coord) {
                    if let Circuit::Element(_) = elems[i] {
                        let new_pidx = param_idx + (0..=i).map(|el| elems[el].paramlen()).sum::<usize>();
                        elems.insert(i+1, Self::Element(element));
                        element.insert_param(new_pidx, paramlist, editability);
                    }
                    else {
                        let new_pidx = param_idx + (0..i).map(|el| elems[el].paramlen()).sum::<usize>();
                        elems[i]._add_series((coord.0 - start_x, coord.1), element, paramlist, new_pidx, editability)
                    }
                }
            }

            Circuit::Parallel(elems) => {
                assert!(elems.len() > 1);
                let ib = parallel_index_by_block(elems, coord);

                if let ParallelBlockPicked::Child(i, elemblock, start_y) = ib {
                    let new_pidx = param_idx + (0..i).map(|el| elems[el].paramlen()).sum::<usize>();
                    elems[i]._add_series((coord.0-elemblock, coord.1-start_y), element, paramlist, new_pidx, editability);
                }
            }
        }
    }

    pub fn _add_parallel(&mut self, coord: (u16, u16), element: Element, paramlist: std::slice::IterMut<'_, ParameterDesc>, param_idx: usize, editability: &mut Vec<ParameterEditability>){
        match self {
            Self::Element(e) => {
                let new = Self::Parallel(vec![Self::Element(*e), Self::Element(element)]);
                let paramlen = self.paramlen();
                element.insert_param(param_idx+paramlen, paramlist, editability);
                *self = new;
            },

            Self::Series(elems) => {
                assert!(elems.len() > 1);
                if let Some((i, start_x)) = series_index_by_block(elems, coord) {
                    let new_pidx = param_idx + (0..i).map(|el| elems[el].paramlen()).sum::<usize>();
                    elems[i]._add_parallel((coord.0 - start_x, coord.1), element, paramlist, new_pidx, editability)
                }
            }

            Self::Parallel(elems) => {
                assert!(elems.len() > 1);
                let ib = parallel_index_by_block(elems, coord);

                if let ParallelBlockPicked::Child(i, elemblock, start_y) = ib {
                    if let Self::Element(_) = elems[i] {
                        let new_pidx = param_idx + (0..=i).map(|el| elems[el].paramlen()).sum::<usize>();
                        elems.insert(i+1, Self::Element(element));
                        element.insert_param(new_pidx, paramlist, editability);
                    }
                    else {
                        let new_pidx = param_idx + (0..i).map(|el| elems[el].paramlen()).sum::<usize>();
                        elems[i]._add_parallel((coord.0-elemblock, coord.1-start_y), element, paramlist, new_pidx, editability);
                    }
                }
            }
        }
    }


    pub fn _remove(&mut self, coord: (u16, u16), paramlist: std::slice::IterMut<'_, ParameterDesc>, param_idx: usize, editability: &mut Vec<ParameterEditability>) {
        match self {
            Circuit::Element(_) => {},

            Circuit::Series(elems) => {
                assert!(elems.len() > 1);
                
                if let Some((i,start_x)) = series_index_by_block(elems, coord) {
                    let new_pidx = param_idx + (0..i).map(|el| elems[el].paramlen()).sum::<usize>();
                    let elemlen = elems.len();
                    if let Circuit::Element(el) = &mut elems[i] {
                        el.remove_param(new_pidx, paramlist, editability);
                        if elemlen == 2 {
                            *self = elems[if i == 0 {1} else {0}].clone();
                        }
                        else {
                            elems.remove(i);
                        }
                    }
                    else {
                        elems[i]._remove((coord.0 - start_x, coord.1), paramlist, new_pidx, editability);
                    }
                }
            }

            Circuit::Parallel(elems) => {
                assert!(elems.len() > 1);
                let ib = parallel_index_by_block(elems, coord);

                match ib {
                    ParallelBlockPicked::Child(i, elemblock, start_y) => {
                        let new_pidx = param_idx + (0..i).map(|el| elems[el].paramlen()).sum::<usize>();
                        let elemlen = elems.len();
                        if let Circuit::Element(el) = &mut elems[i] {
                            el.remove_param(new_pidx, paramlist, editability);
                            if elemlen == 2 {
                                *self = elems[if i == 0 {1} else {0}].clone();
                            }
                            else {
                                elems.remove(i);
                            }
                        }
                        else {
                            elems[i]._remove((coord.0-elemblock, coord.1-start_y), paramlist, new_pidx, editability);
                        }
                    }
                    ParallelBlockPicked::This => {
                        
                    }
                    ParallelBlockPicked::None => {
                        
                    }
                }
            }
        }
    }
}



impl std::fmt::Display for Circuit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Circuit::Element(Element::Resistor) => write!(f, "R"),
            Circuit::Element(Element::Capacitor) => write!(f, "C"),
            Circuit::Element(Element::Inductor) => write!(f, "L"),
            Circuit::Element(Element::Warburg) => write!(f, "W"),
            Circuit::Element(Element::Cpe) => write!(f, "Q"),
            Circuit::Series(elems) => {
                write!(f, "[")?;
                for e in elems { e.fmt(f)?; }
                write!(f, "]")
            }
            Circuit::Parallel(elems) => {
                write!(f, "{{")?;
                for e in elems { e.fmt(f)?; }
                write!(f, "}}")
            }
        }
    }
}

enum CircuitParseElement {Circ(Circuit), Series, Parallel}
impl std::str::FromStr for Circuit {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut stack : Vec<CircuitParseElement> = vec![];

        for c in s.chars() {
            match c {
                'R' => {stack.push(CircuitParseElement::Circ(Self::Element(Element::Resistor)));}
                'C' => {stack.push(CircuitParseElement::Circ(Self::Element(Element::Capacitor)));}
                'L' => {stack.push(CircuitParseElement::Circ(Self::Element(Element::Inductor)));}
                'W' => {stack.push(CircuitParseElement::Circ(Self::Element(Element::Warburg)));}
                'Q' => {stack.push(CircuitParseElement::Circ(Self::Element(Element::Cpe)));}
                '[' => {stack.push(CircuitParseElement::Series);}
                '{' => {stack.push(CircuitParseElement::Parallel);}

                ']' => {
                    let mut elements : Vec<Self> = vec![];

                    while let Some(v) = stack.pop() {
                        match v {
                            CircuitParseElement::Circ(c) => {elements.insert(0, c);}
                            CircuitParseElement::Series => {break;}
                            CircuitParseElement::Parallel => {return Err(());}
                        }
                    }

                    stack.push(CircuitParseElement::Circ(Self::Series(elements)));
                }

                '}' => {
                    let mut elements : Vec<Self> = vec![];

                    while let Some(v) = stack.pop() {
                        match v {
                            CircuitParseElement::Circ(c) => {elements.insert(0, c);}
                            CircuitParseElement::Series => {return Err(());}
                            CircuitParseElement::Parallel => {break;}
                        }
                    }

                    stack.push(CircuitParseElement::Circ(Self::Parallel(elements)));
                }

                _ => {return Err(())}
            }
        }

        if stack.len() == 1 {
            if let Some(CircuitParseElement::Circ(ret)) = stack.pop() {
                return Ok(ret);
            }
        }
        Err(())
    }
}
