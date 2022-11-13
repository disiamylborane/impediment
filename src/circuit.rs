#![allow(dead_code)]

extern crate num;
use std::{f64::consts::PI, vec};
use eframe::egui::{Color32, Pos2, Rect, Stroke, pos2, vec2};

use crate::{Cplx, project::ModelVariable};

#[derive(Debug, Clone)]
pub struct ParameterDesc {
    pub vals : Vec<f64>,
    pub bounds : Vec<(f64, f64)>
}

const I: Cplx = Cplx{ re: 0.0, im: 1.0 };

#[derive(PartialEq, Eq, Clone, Copy, Debug, Default)]
pub enum Element {
    #[default]
    Resistor,
    Capacitor,
    Inductor,
    Warburg,
    Cpe,
}
use Element::{Capacitor, Inductor, Resistor, Warburg, Cpe};


impl Element {
    pub const fn param_count(self) -> usize {
        match self {
            Resistor|Capacitor|Inductor|Warburg => 1,
            Cpe => 2,
        }
    }

    pub fn gen_individual_params(self) -> Vec<ModelVariable> {
        match self {
            Resistor => vec![ModelVariable{ val: 100., bounds: (1.0, 10000.0), enabled: true }],
            Capacitor => vec![ModelVariable{ val: 100., bounds: (1e-6, 1.0), enabled: true }],
            Inductor => vec![ModelVariable{ val: 1e-4, bounds: (1e-6, 1.0), enabled: true }],
            Warburg => vec![ModelVariable{ val: 1e-4, bounds: (1e-6, 1.0), enabled: true }],
            Cpe => vec![
                ModelVariable{ val: 100., bounds: (1e-6, 1.0), enabled: true },
                ModelVariable{ val: 0.8, bounds: (0.0, 1.0), enabled: true },
            ],
        }
    }
}

#[derive(Clone, Debug)]
pub enum Circuit {
    Element(Element),
    Series(Vec<Self>),
    Parallel(Vec<Self>),
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
                 .map(Circuit::painted_size)
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

        let g_size = parallel_painted_size(parallel).0;

        let mut start_coord_y = 0_u16;
        for (i,el) in &mut parallel.iter().enumerate() {
            let elemsize = el.painted_size();
            if start_coord_y + elemsize.1 > block.1 {
                let p_size = el.painted_size().0;
                let elemblock = (g_size-2-p_size)/2 + 1;

                if block.0 >= elemblock && block.0 < elemblock+p_size {
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
    ctx.text(pos+vec2(blocksize, blocksize*3./2.), eframe::egui::Align2::CENTER_CENTER, text, eframe::egui::FontId{size: 12.0, family: eframe::egui::FontFamily::Proportional}, color);
}


impl Circuit {
    pub fn element_count(&self) -> usize {
        match self {
            Self::Element(_) => 1,
            Self::Series(x)|Self::Parallel(x) => x.iter().map(Self::element_count).sum(),
        }
    }

    pub fn param_count(&self) -> usize {
        match self {
            Self::Element(e) => e.param_count(),
            Self::Series(x) | Self::Parallel(x) => x.iter().map(Self::param_count).sum(),
        }
    }

    fn _pick_block(&self, block: (u16, u16), add: (usize, usize)) -> Option<(usize, std::ops::Range<usize>)> {
        match self {
            Self::Element(e) => {
                if block == (0,0) || block == (1,0) {
                    Some((add.0, (add.1)..(e.param_count()+ add.1) ))
                }
                else {None}
            }
            Self::Series(s) => {
                let mut start_x = 0;
                let mut start_element = 0;
                let mut start_param = 0;
                for circ in s {
                    let sz = circ.painted_size();

                    if block.0 < start_x + sz.0 {
                        if block.1 < sz.1 {
                            return circ._pick_block((block.0 - start_x, block.1), (start_element, start_param));
                        }
                        return None;
                    } 
                    start_x += sz.0;
                    start_element += circ.element_count();
                    start_param += circ.param_count();
                }
                None
            }
            Self::Parallel(p) => {
                let wsize = parallel_painted_size(p);
                let g_size = wsize.0;

                if block.0 == 0 || block.0 >= wsize.0-1 || block.1 >= wsize.1 {
                    return None;
                }

                let mut start_coord_y = 0_u16;
                let mut start_element = 0;
                let mut start_param = 0;
                for circ in p {
                    let elemsize = circ.painted_size();
                    if start_coord_y + elemsize.1 > block.1 {
                        let p_size = circ.painted_size().0;
                        let elemblock = (g_size-2-p_size)/2 + 1;
                        if block.0 >= elemblock && block.0 < elemblock+p_size {
                            return circ._pick_block((block.0-elemblock, block.1-start_coord_y), (start_element,start_param));// ParallelBlockPicked::Child(i, elemblock, start_coord_y);
                        }
                        return None;
                    }

                    start_coord_y += elemsize.1;
                    start_element += circ.element_count();
                    start_param += circ.param_count();
                }

                None
            }
        }
    }


    pub fn pick_block(&self, block: (u16, u16)) -> Option<(usize, std::ops::Range<usize>)> {
        self._pick_block(block, (0,0))
    }


    pub fn get_element_mut(&mut self, idx: usize) -> Option<&mut Element> {
        match self {
            Self::Element(e) => {
                if idx == 0 { Some(e) } else { None }
            }
            Self::Series(list) | Self::Parallel(list) => {
                let mut curr_element = 0;
                for child in list.iter_mut() {
                    let ch_len = child.element_count();
                    if idx < curr_element + ch_len {
                        match child {
                            Self::Element(ech) => {
                                return Some(ech);
                            }
                            Self::Series(_)|Self::Parallel(_) => {
                                return child.get_element_mut(idx-curr_element);
                            }
                        }
                    }
                    curr_element += ch_len;
                }
                None
            }
        }
    }

    pub fn simplify(&mut self) {
        match self {
            Self::Element(_) => {}
            Self::Series(s) => {
                for s in s.iter_mut() {
                    s.simplify();
                }

                let mut i = 0;
                while i < s.len() {
                    if let Self::Series(_) = &s[i] {
                        let Self::Series(x) = s.remove(i) else {unreachable!()};
                        for e in x.into_iter().rev() {
                            s.insert(i, e);
                        }
                    }
                    i += 1;
                }

                if s.len() == 1 {
                    *self = s[0].clone();
                }
            }
            Self::Parallel(s) => {
                for s in s.iter_mut() {
                    s.simplify();
                }

                let mut i = 0;
                while i < s.len() {
                    if let Self::Parallel(_) = &s[i] {
                        let Self::Parallel(x) = s.remove(i) else {unreachable!()};
                        for e in x.into_iter().rev() {
                            s.insert(i, e);
                        }
                    }
                    i += 1;
                }

                if s.len() == 1 {
                    *self = s[0].clone();
                }
            }
        }
    }

    // Param name + component index
    pub fn param_letters(&self) -> Vec<(&'static str, usize)> {
        self._param_letters(0)
    }

    pub fn _param_letters(&self, start_index: usize) -> Vec<(&'static str, usize)> {
        match self {
            Self::Element(el) => {
                match el {
                    Resistor => { vec![("R", start_index)] }
                    Capacitor => { vec![("C", start_index)] }
                    Inductor => { vec![("L", start_index)] }
                    Warburg => { vec![("W", start_index)] }
                    Cpe => { vec![("Q", start_index), ("n", start_index)] }
                }
            }
            Self::Series(elems) | Self::Parallel(elems) => {
                let mut out = vec![];
                let mut ni = start_index;
                for el in elems {
                    let mut children = el._param_letters(ni);
                    ni += children.len();
                    out.append(&mut children);
                }
                out
            }
        }
    }

    pub fn param_names(&self) -> Vec<String> {
        self.param_names_rec(0).0
    }

    pub fn param_names_rec(&self, mut start_index: usize) -> (Vec<String>, usize) {
        match self {
            Self::Element(Element::Resistor) => (vec![format!("R{start_index}")], start_index+1),
            Self::Element(Element::Capacitor) => (vec![format!("C{start_index}")], start_index+1),
            Self::Element(Element::Inductor) => (vec![format!("L{start_index}")], start_index+1),
            Self::Element(Element::Warburg) => (vec![format!("W{start_index}")], start_index+1),
            Self::Element(Element::Cpe) => (vec![format!("Q{start_index}"), format!("n{start_index}")], start_index+1),
            Self::Series(elems)|Self::Parallel(elems) => {
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
            Self::Element(_) => (2,2),

            Self::Series(elems) => 
                elems.iter()
                     .map(Self::painted_size)
                     .fold((0,0), |a, b| (a.0+b.0, std::cmp::max(a.1,b.1))),

            Self::Parallel(elems) =>  {
                let s = elems.iter()
                             .map(Self::painted_size)
                             .fold((0,0), |a, b| (std::cmp::max(a.0,b.0), a.1+b.1));
                (s.0 + 2, s.1)
            }
        }
    }

    pub fn paint(&self, pos: eframe::egui::Pos2, blocksize: f32, painter: &eframe::egui::Painter, start_index: usize, color: Color32, names: &[String])->usize {
        let stroke = Stroke::new(1., color);
        match self {
            Self::Element(Element::Resistor) => {
                painter.line_segment([pos+vec2(0., blocksize/2.), pos+vec2(blocksize/4., blocksize/2.)], stroke);
                painter.rect_stroke(Rect::from_min_size(pos+vec2(blocksize/4., blocksize/4.), vec2(blocksize*3./2., blocksize/2.)), 0., stroke);
                painter.line_segment([pos+vec2(blocksize*7./4., blocksize/2.), pos+vec2(blocksize*2., blocksize/2.)], stroke);
                sign_element(painter, &format!("R{}", &names[start_index]), pos, blocksize, color);
                start_index+1
            }
            Self::Element(Element::Capacitor) => {
                painter.line_segment([pos+vec2(0., blocksize/2.), pos+vec2(blocksize*3./4., blocksize/2.)], stroke);
                painter.line_segment([pos+vec2(blocksize*3./4., 0.), pos+vec2(blocksize*3./4., blocksize)], stroke);
                painter.line_segment([pos+vec2(blocksize*5./4., 0.), pos+vec2(blocksize*5./4., blocksize)], stroke);
                painter.line_segment([pos+vec2(blocksize*5./4., blocksize/2.), pos+vec2(blocksize*2., blocksize/2.)], stroke);
                sign_element(painter, &format!("C{}", &names[start_index]), pos, blocksize, color);
                start_index+1
            },
            Self::Element(Element::Inductor) => {
                let mut curr_pos = vec2(0., blocksize/2.);

                let mut lineto = |newpos| {
                    painter.line_segment([pos+curr_pos, pos+newpos], stroke);
                    curr_pos = newpos;
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

                sign_element(painter, &format!("L{}", &names[start_index]), pos, blocksize, color);
                start_index+1
            }

            Self::Element(Element::Warburg) => {
                let line = |p1,p2| {
                    painter.line_segment([pos+p1*blocksize, pos+p2*blocksize], stroke);
                };

                line(vec2(0.0, 0.5), vec2(0.75, 0.5));
                line(vec2(0.75, 0.0), vec2(0.75, 1.0));
                line(vec2(1.5, 0.0), vec2(1.25, 0.5));
                line(vec2(1.25, 0.5), vec2(1.5, 1.0));
                line(vec2(1.25, 0.5), vec2(2.0, 0.5));

                sign_element(painter, &format!("W{}", &names[start_index]), pos, blocksize, color);
                start_index+1
            }

            Self::Element(Element::Cpe) => {
                let line = |p1,p2| {
                    painter.line_segment([pos+p1*blocksize, pos+p2*blocksize], stroke);
                };

                line(vec2(0.0, 0.5), vec2(0.75, 0.5));
                line(vec2(0.75, 0.5), vec2(1.0, 0.0));
                line(vec2(0.75, 0.5), vec2(1.0, 1.0));
                line(vec2(1.5, 0.0), vec2(1.25, 0.5));
                line(vec2(1.25, 0.5), vec2(1.5, 1.0));
                line(vec2(1.25, 0.5), vec2(2.0, 0.5));

                sign_element(painter, &format!("Z{}", &names[start_index]), pos, blocksize, color);
                start_index+1
            }

            Self::Series(elems) => {
                let mut index = elems[0].paint(pos, blocksize, painter, start_index, color, names);
                let mut pos = pos + vec2(f32::from(elems[0].painted_size().0) * blocksize, 0.);

                for c in elems[1..].iter() {
                    index = c.paint(pos, blocksize, painter, index, color, names);
                    pos += vec2(f32::from(c.painted_size().0) * blocksize, 0.);
                }
                index
            }

            Self::Parallel(elems) => {
                let mut index = start_index;
                let mut y = pos.y;
                let xsize = self.painted_size().0;

                painter.line_segment([pos+vec2(0., blocksize/2.), pos+vec2(blocksize, blocksize/2.)], stroke);
                painter.line_segment([pos+vec2(f32::from(xsize-1) * blocksize, blocksize/2.), pos+vec2((f32::from(xsize-1)).mul_add(blocksize, blocksize), blocksize/2.)], stroke);

                for c in elems.iter() {
                    let painted_size = c.painted_size().0;
                    let drawend = f32::from(xsize).mul_add(blocksize, pos.x);
                    let elemblock = (xsize-2-painted_size)/2 + 1;
                    let elemstart = f32::from(elemblock).mul_add(blocksize, pos.x);
                    let elemend = f32::from(painted_size).mul_add(blocksize, elemstart);
                    index = c.paint(pos2(elemstart, y), blocksize, painter, index, color, names);

                    painter.line_segment([pos2(pos.x + blocksize/2., y+blocksize/2.), pos2(elemstart, y+blocksize/2.)], stroke);
                    painter.line_segment([pos2(elemend, y+blocksize/2.), pos2(f32::from(xsize).mul_add(blocksize, pos.x) - blocksize/2., y+blocksize/2.)], stroke);

                    painter.line_segment([pos+vec2(blocksize/2., blocksize/2.), pos2(pos.x + blocksize/2., y+blocksize/2.)], stroke);

                    painter.line_segment([pos2(drawend - blocksize/2., pos.y+blocksize/2.), pos2(drawend - blocksize/2., y+blocksize/2.)], stroke);
                    
                    y += f32::from(c.painted_size().1) * blocksize;
                }
                index
            }
        }
    }

    pub fn paramlen(&self) -> usize {
        match self {
            Self::Element(Resistor|Capacitor|Inductor|Warburg) => 1,
            Self::Element(Cpe) => 2,
            Self::Series(elems) | Self::Parallel(elems) => 
                elems.iter().map(Self::paramlen).sum(),
        }
    }

    fn _impedance(&self, omega: f64, params: &[f64]) -> Cplx{
        match self {
            Self::Element(Element::Resistor) => Cplx::new(params[0], 0.0),

            Self::Element(Element::Capacitor) => 1.0 / (I * omega * params[0]),

            Self::Element(Element::Inductor) => I * omega * params[0],

            Self::Element(Element::Warburg) => (1.0 - I) * params[0] / omega.sqrt(),

            Self::Element(Element::Cpe) => {
                let q = params[0];
                let n = params[1];
                let numer = (-I*PI/2.0*n).exp();
                let denom = q * omega.powf(n);
                numer/denom
            }

            Self::Series(elems) => {
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

            Self::Parallel(elems) => {
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
            Self::Element(Element::Resistor) => Cplx::new(1.0, 0.0),

            Self::Element(Element::Capacitor) => I / (omega * params[0].powi(2)),

            Self::Element(Element::Inductor) => I * omega,

            Self::Element(Element::Warburg) => ((1.0 - I)*params[0])/(omega.sqrt()),

            Self::Element(Element::Cpe) => {
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

            Self::Series(elems) => {
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

            Self::Parallel(elems) => {
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
}


use std::ops::Range;
pub struct ParameterRanger {
    param_start: usize,
    param_old_count: usize,
}


impl Circuit {
    pub fn replace_element(&mut self, coord: (u16, u16), new_element: Element) -> Option<Range<usize>> {
        self._replace_element(coord, new_element, 0)
    }

    pub fn _replace_element(&mut self, coord: (u16, u16), new_element: Element, prev_param: usize) -> Option<Range<usize>> {
        match self {
            &mut Circuit::Element(e) => {
                *self = Self::Element(new_element);
                (prev_param..(prev_param+e.param_count())).into()
            }
            Circuit::Series(elems) => {
                assert!(elems.len() > 1);
                if let Some((i, start_x)) = series_index_by_block(elems, coord) {
                    let new_pidx = prev_param + (0..i).map(|el| elems[el].paramlen()).sum::<usize>();
                    let el = &mut elems[i];
                    el._replace_element((coord.0 - start_x, coord.1), new_element, new_pidx)
                }
                else {None}
            }
            Circuit::Parallel(elems) => {
                assert!(elems.len() > 1);
                let ib = parallel_index_by_block(elems, coord);

                match ib {
                    ParallelBlockPicked::This | ParallelBlockPicked::None => {
                        None
                    },
                    ParallelBlockPicked::Child(i, elemblock, start_y) => {
                        let new_pidx = prev_param + (0..i).map(|el| elems[el].paramlen()).sum::<usize>();
                        let el = &mut elems[i];
                        el._replace_element((coord.0 - elemblock, coord.1 - start_y), new_element, new_pidx)
                    }
                }
            }
        }
    }


    pub fn add_series_element(&mut self, coord: (u16, u16), new_element: Element) -> Option<(usize, usize)> {
        self._add_series_element(coord, new_element, 0, 0)
    }

    pub fn _add_series_element(&mut self, coord: (u16, u16), new_element: Element, prev_component: usize, prev_param: usize) -> Option<(usize, usize)> {
        let out: Option<(usize, usize)> = match self {
            Self::Element(e) => {
                let new_cmp = prev_component + 1;
                let new_param = prev_param+e.param_count();
                *self = Self::Series(vec![Self::Element(*e), Self::Element(new_element)]);
                (new_cmp, new_param).into()
            },

            Self::Series(elems) => {
                assert!(elems.len() > 1);
                if let Some((i, start_x)) = series_index_by_block(elems, coord) {
                    {
                        let new_cmp = prev_component+i;
                        let new_pidx = prev_param + (0..i).map(|el| elems[el].paramlen()).sum::<usize>();
                        elems[i]._add_series_element((coord.0 - start_x, coord.1), new_element, new_cmp, new_pidx)
                    }
                } else {None}
            }

            Self::Parallel(elems) => {
                assert!(elems.len() > 1);
                let ib = parallel_index_by_block(elems, coord);

                if let ParallelBlockPicked::Child(i, elemblock, start_y) = ib {
                    let new_cmp = prev_component+i;
                    let new_pidx = prev_param + (0..i).map(|el| elems[el].paramlen()).sum::<usize>();
                    elems[i]._add_series_element((coord.0-elemblock, coord.1-start_y), new_element, new_cmp, new_pidx)
                } else {None}
            }
        };

        self.simplify();

        out
    }


    pub fn add_parallel_element(&mut self, coord: (u16, u16), new_element: Element) -> Option<(usize, usize)> {
        self._add_parallel_element(coord, new_element, 0, 0)
    }

    pub fn _add_parallel_element(&mut self, coord: (u16, u16), new_element: Element, prev_component: usize, prev_param: usize) -> Option<(usize, usize)> {
        let out = match self {
            Self::Element(e) => {
                let new_cmp = prev_component + 1;
                let new_param = prev_param+e.param_count();
                *self = Self::Parallel(vec![Self::Element(*e), Self::Element(new_element)]);
                (new_cmp, new_param).into()
            },

            Self::Series(elems) => {
                assert!(elems.len() > 1);
                if let Some((i, start_x)) = series_index_by_block(elems, coord) {
                    {
                        let new_cmp = prev_component+i;
                        let new_pidx = prev_param + (0..i).map(|el| elems[el].paramlen()).sum::<usize>();
                        elems[i]._add_parallel_element((coord.0 - start_x, coord.1), new_element, new_cmp, new_pidx)
                    }
                } else {None}
            }

            Self::Parallel(elems) => {
                assert!(elems.len() > 1);
                let ib = parallel_index_by_block(elems, coord);

                if let ParallelBlockPicked::Child(i, elemblock, start_y) = ib {
                    let new_cmp = prev_component+i;
                    let new_pidx = prev_param + (0..i).map(|el| elems[el].paramlen()).sum::<usize>();
                    elems[i]._add_parallel_element((coord.0-elemblock, coord.1-start_y), new_element, new_cmp, new_pidx)
                } else {None}
            }
        };

        self.simplify();

        out
    }



    pub fn delete_element(&mut self, coord: (u16, u16)) -> Option<Range<usize>> {
        self._delete_element(coord, 0)
    }

    pub fn _delete_element(&mut self, coord: (u16, u16), prev_param: usize) -> Option<Range<usize>> {
        let out = match self {
            Circuit::Element(_) => {None}
            Circuit::Series(elems) => {
                assert!(elems.len() > 1);

                if let Some((i, start_x)) = series_index_by_block(elems, coord) {
                    let new_pidx = prev_param + (0..i).map(|el| elems[el].paramlen()).sum::<usize>();

                    match &elems[i] {
                        &Circuit::Element(e) => {elems.remove(i); (new_pidx..(new_pidx+e.param_count())).into()},
                        Circuit::Series(_)|Circuit::Parallel(_) => {
                            elems[i]._delete_element((coord.0 - start_x, coord.1), new_pidx)
                        }
                    }

                } else {None}

            }
            Circuit::Parallel(elems) => {
                assert!(elems.len() > 1);
                let ib = parallel_index_by_block(elems, coord);
                
                if let ParallelBlockPicked::Child(i, elemblock, start_y) = ib {
                    let new_pidx = prev_param + (0..i).map(|el| elems[el].paramlen()).sum::<usize>();

                    match &elems[i] {
                        &Circuit::Element(e) => {elems.remove(i); (new_pidx..(new_pidx+e.param_count())).into()},
                        Circuit::Series(_)|Circuit::Parallel(_) => {
                            elems[i]._delete_element((coord.0-elemblock, coord.1-start_y), new_pidx)
                        }
                    }

                } else {None}
            }
        };
        self.simplify();
        out
    }
}


impl std::fmt::Display for Circuit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Element(Element::Resistor) => write!(f, "R"),
            Self::Element(Element::Capacitor) => write!(f, "C"),
            Self::Element(Element::Inductor) => write!(f, "L"),
            Self::Element(Element::Warburg) => write!(f, "W"),
            Self::Element(Element::Cpe) => write!(f, "Q"),
            Self::Series(elems) => {
                write!(f, "[")?;
                for e in elems { e.fmt(f)?; }
                write!(f, "]")
            }
            Self::Parallel(elems) => {
                write!(f, "{{")?;
                for e in elems { e.fmt(f)?; }
                write!(f, "}}")
            }
        }
    }
}

enum SelfParseElement {Circ(Circuit), Series, Parallel}
impl std::str::FromStr for Circuit {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut stack : Vec<SelfParseElement> = vec![];

        for c in s.chars() {
            match c {
                'R' => {stack.push(SelfParseElement::Circ(Self::Element(Element::Resistor)));}
                'C' => {stack.push(SelfParseElement::Circ(Self::Element(Element::Capacitor)));}
                'L' => {stack.push(SelfParseElement::Circ(Self::Element(Element::Inductor)));}
                'W' => {stack.push(SelfParseElement::Circ(Self::Element(Element::Warburg)));}
                'Q' => {stack.push(SelfParseElement::Circ(Self::Element(Element::Cpe)));}
                '[' => {stack.push(SelfParseElement::Series);}
                '{' => {stack.push(SelfParseElement::Parallel);}

                ']' => {
                    let mut elements : Vec<Self> = vec![];

                    while let Some(v) = stack.pop() {
                        match v {
                            SelfParseElement::Circ(c) => {elements.insert(0, c);}
                            SelfParseElement::Series => {break;}
                            SelfParseElement::Parallel => {return Err(());}
                        }
                    }

                    stack.push(SelfParseElement::Circ(Self::Series(elements)));
                }

                '}' => {
                    let mut elements : Vec<Self> = vec![];

                    while let Some(v) = stack.pop() {
                        match v {
                            SelfParseElement::Circ(c) => {elements.insert(0, c);}
                            SelfParseElement::Series => {return Err(());}
                            SelfParseElement::Parallel => {break;}
                        }
                    }

                    stack.push(SelfParseElement::Circ(Self::Parallel(elements)));
                }

                _ => {return Err(())}
            }
        }

        if stack.len() == 1 {
            if let Some(SelfParseElement::Circ(ret)) = stack.pop() {
                return Ok(ret);
            }
        }
        Err(())
    }
}
