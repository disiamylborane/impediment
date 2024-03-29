
extern crate num;
use std::{f64::consts::PI, vec};
use eframe::egui::{Color32, Pos2, Rect, Stroke, pos2, vec2};

use crate::{Cplx, ParameterDesc, StringParameterDesc, ParameterEditability};

const I: Cplx = Cplx{ re: 0.0, im: 1.0 };

// Types of elements used in equivalent circuit
#[derive(Clone, Copy, Debug)]
pub enum Element {
    Resistor,
    Capacitor,
    Inductor,
    Warburg,
    Cpe,
}
use Element::{Capacitor, Inductor, Resistor, Warburg, Cpe};


impl Element {
    // set the parameter list attached to the element and return the previous ones
    fn exchange_param<'a>(self, element: Self, idx: usize, paramlist: impl Iterator<Item = &'a mut StringParameterDesc>, editability: &mut Vec<ParameterEditability>) {
        for pdesc in paramlist {
            match self {
                Resistor|Capacitor|Inductor|Warburg => {pdesc.remove(idx);},
                Cpe => {pdesc.remove(idx); pdesc.remove(idx);},
            }
            match element {
                Resistor => {pdesc.insert(idx, (100., (1., 1e5)))},
                Capacitor|Inductor => {pdesc.insert(idx, (1e-6, (1e-9, 0.1)))},
                Warburg => {pdesc.insert(idx, (10., (1., 1e5)))},
                Cpe => {
                    pdesc.insert(idx, (0.8, (0.01, 0.99)));
                    pdesc.insert(idx, (1e-6, (1e-9, 0.1)));
                },
            }
        }
        match self {
            Resistor|Capacitor|Inductor|Warburg => editability.remove(idx),
            Cpe => {
                editability.remove(idx);
                editability.remove(idx)
            }
        };
        match element {
            Resistor|Capacitor|Inductor|Warburg => editability.insert(idx, ParameterEditability::Plural),
            Cpe => {
                editability.insert(idx, ParameterEditability::Plural);
                editability.insert(idx, ParameterEditability::Plural);
            }
        }
    }

    // Remove parameters when this element is replaced from the circuit
    fn remove_param<'a>(self, idx: usize, paramlist: impl Iterator<Item = &'a mut StringParameterDesc>, editability: &mut Vec<ParameterEditability>) {
        for pdesc in paramlist {
                match self {
                Resistor|Capacitor|Inductor|Warburg => {pdesc.remove(idx);},
                Cpe => {pdesc.remove(idx); pdesc.remove(idx);},
            }
        }
        match self {
            Resistor|Capacitor|Inductor|Warburg => editability.remove(idx),
            Cpe => {
                editability.remove(idx);
                editability.remove(idx)
            }
        };
    }

    // Add parameters when this element is placed to the circuit
    fn insert_param<'a>(self, idx: usize, paramlist: impl Iterator<Item = &'a mut StringParameterDesc>, editability: &mut Vec<ParameterEditability>) {
        for pdesc in paramlist {
            // @Refactor: Duplicate
            match self {
                Resistor => {pdesc.insert(idx, (100., (1., 1e5)))},
                Capacitor|Inductor => {pdesc.insert(idx, (1e-6, (1e-9, 0.1)))},
                Warburg => {pdesc.insert(idx, (10., (1., 1e5)))},
                Cpe => {
                    pdesc.insert(idx, (0.8, (0.01, 0.99)));
                    pdesc.insert(idx, (1e-6, (1e-9, 0.1)));
                },
            }
        }

        match self {
            Resistor|Capacitor|Inductor|Warburg => editability.insert(idx, ParameterEditability::Plural),
            Cpe => {
                editability.insert(idx, ParameterEditability::Plural);
                editability.insert(idx, ParameterEditability::Plural);
            }
        }
    }
}

// An equivalent circuit
#[derive(Clone, Debug)]
pub enum Circuit {
    Element(Element),
    Series(Vec<Circuit>),
    Parallel(Vec<Circuit>),
}

// Helper function: get subcircuit index and the x coordinate of the subcircuit
// for series circuit
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

// Helper function: calculate the painted area size for Parallel circuit
fn parallel_painted_size(elems: &[Circuit]) -> (u16, u16) {
    let s = elems.iter()
                 .map(Circuit::painted_size)
                 .fold((0,0), |a, b| (std::cmp::max(a.0,b.0), a.1+b.1));
    (s.0 + 2, s.1)
}

// What is picked inside Parallel circuit when it's clicked?
enum ParallelBlockPicked {
    Child(usize, u16, u16),
    This,
    None
}

// Helper function: get subcircuit index and the x coordinate of the subcircuit
// for parallel circuit
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


/// Paint a text label under the element
fn sign_element(ctx: &eframe::egui::Painter, text: &str, pos: Pos2, blocksize: f32, color: Color32) {
    ctx.text(
        pos+vec2(blocksize, blocksize*3./2.),
        eframe::egui::Align2::CENTER_CENTER,
        text,
        eframe::egui::FontId{size: 12.0, family: eframe::egui::FontFamily::Proportional},
        color);
}


impl Circuit {
    // Create a new param list for the circuit
    pub fn generate_new_params(&self) -> ParameterDesc {
        match self {
            Self::Element(element) => {
                let pd = match element {
                    Resistor => {(100., (1., 1e5))},
                    Capacitor|Inductor => {(1e-6, (1e-9, 0.1))},
                    Warburg => {(10., (1., 1e5))},
                    Cpe => {
                        return ParameterDesc{ vals: vec![0.8, 1e-6], bounds: vec![(0.01, 0.99), (1e-9, 0.1)] };
                    },
                };
                ParameterDesc{ vals: vec![pd.0], bounds: vec![pd.1] }
            }
            Self::Series(s) | Self::Parallel(s) => {
                let mut evals = vec![];
                let mut ebounds = vec![];
                for ParameterDesc{mut vals, mut bounds} in s.iter().map(Self::generate_new_params) {
                    evals.append(&mut vals);
                    ebounds.append(&mut bounds);
                }
                ParameterDesc{vals: evals, bounds: ebounds}
            },
        }
    }

    // Generate and list all the parameter names
    pub fn param_names(&self) -> Vec<String> {
        self.param_names_rec(0).0
    }

    // Recursive helper: generate all the parameter names
    pub fn param_names_rec(&self, mut start_index: usize) -> (Vec<String>, usize) {
        match self {
            Self::Element(Element::Resistor) => (vec![format!("R{start_index}")], start_index+1),
            Self::Element(Element::Capacitor) => (vec![format!("C{start_index}")], start_index+1),
            Self::Element(Element::Inductor) => (vec![format!("L{start_index}")], start_index+1),
            Self::Element(Element::Warburg) => (vec![format!("W{start_index}")], start_index+1),
            Self::Element(Element::Cpe) => (vec![format!("Q{start_index}"), format!("n{start_index}")], start_index+1),
            Self::Series(elems) | Self::Parallel(elems) => {
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

    // Rectangular area size needed for the circuit in blocks
    // (one element = 2*1 block)
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

    // Draw the circuit starting from "start_index" element
    pub fn paint(&self, pos: eframe::egui::Pos2, blocksize: f32, painter: &eframe::egui::Painter, start_index: usize, color: Color32)->usize {
        let stroke = Stroke::new(1., color);
        match self {
            Self::Element(Element::Resistor) => {
                painter.line_segment([pos+vec2(0., blocksize/2.), pos+vec2(blocksize/4., blocksize/2.)], stroke);
                painter.rect_stroke(Rect::from_min_size(pos+vec2(blocksize/4., blocksize/4.), vec2(blocksize*3./2., blocksize/2.)), 0., stroke);
                painter.line_segment([pos+vec2(blocksize*7./4., blocksize/2.), pos+vec2(blocksize*2., blocksize/2.)], stroke);
                sign_element(painter, &format!("R{start_index}"), pos, blocksize, color);
                start_index+1
            }
            Self::Element(Element::Capacitor) => {
                painter.line_segment([pos+vec2(0., blocksize/2.), pos+vec2(blocksize*3./4., blocksize/2.)], stroke);
                painter.line_segment([pos+vec2(blocksize*3./4., 0.), pos+vec2(blocksize*3./4., blocksize)], stroke);
                painter.line_segment([pos+vec2(blocksize*5./4., 0.), pos+vec2(blocksize*5./4., blocksize)], stroke);
                painter.line_segment([pos+vec2(blocksize*5./4., blocksize/2.), pos+vec2(blocksize*2., blocksize/2.)], stroke);
                sign_element(painter, &format!("C{start_index}"), pos, blocksize, color);
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

                sign_element(painter, &format!("L{start_index}"), pos, blocksize, color);
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

                sign_element(painter, &format!("W{start_index}"), pos, blocksize, color);
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

                sign_element(painter, &format!("Z{start_index}"), pos, blocksize, color);
                start_index+1
            }

            Self::Series(elems) => {
                let mut index = elems[0].paint(pos, blocksize, painter, start_index, color);
                let mut pos = pos + vec2(f32::from(elems[0].painted_size().0) * blocksize, 0.);

                for c in elems[1..].iter() {
                    index = c.paint(pos, blocksize, painter, index, color);
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
                    index = c.paint(pos2(elemstart, y), blocksize, painter, index, color);

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

    // Amount of parameters of the circuit
    pub fn paramlen(&self) -> usize {
        match self {
            Self::Element(Resistor|Capacitor|Inductor|Warburg) => 1,
            Self::Element(Cpe) => 2,
            Self::Series(elems)|Self::Parallel(elems) => 
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

    // Calculate the impedance at a given angular frequency
    pub fn impedance(&self, omega: f64, params: &[f64]) -> Cplx {
        assert!(params.len() == self.paramlen());
        self._impedance(omega, params)
    }

    // Calculate the impedance derivative of parameter no. dparam at a given angular frequency
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


    pub fn _replace<'a>(&mut self, 
                coord: (u16, u16), 
                element: Element, 
                paramlist: impl Iterator<Item = &'a mut StringParameterDesc>, 
                param_idx: usize,
                editability: &mut Vec<ParameterEditability>) {
        match self {
            Self::Element(e) => {
                e.exchange_param(element, param_idx, paramlist, editability);
                editability[param_idx] = ParameterEditability::Plural;
                *self = Self::Element(element);
            },

            Self::Series(elems) => {
                assert!(elems.len() > 1);
                if let Some((i, start_x)) = series_index_by_block(elems, coord) {
                    let new_pidx = param_idx + (0..i).map(|el| elems[el].paramlen()).sum::<usize>();
                    let el = &mut elems[i];
                    el._replace((coord.0 - start_x, coord.1), element, paramlist, new_pidx, editability);
                }
            }

            Self::Parallel(elems) => {
                assert!(elems.len() > 1);
                let ib = parallel_index_by_block(elems, coord);

                match ib {
                    ParallelBlockPicked::This | ParallelBlockPicked::None => {
                        // todo!()
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
 
    // Circuit editing: replace the clicked element with a new one
    pub fn replace(&mut self, coord: (u16, u16), element: Element, paramlist: std::slice::IterMut<'_, StringParameterDesc>, editability: &mut Vec<ParameterEditability>) {
        self._replace(coord, element, paramlist, 0, editability);
    }

    // Circuit editing: add the element in series to the clicked one
    pub fn _add_series(&mut self, coord: (u16, u16), element: Element, paramlist: std::slice::IterMut<'_, StringParameterDesc>, param_idx: usize, editability: &mut Vec<ParameterEditability>){
        match self {
            Self::Element(e) => {
                let new = Self::Series(vec![Self::Element(*e), Self::Element(element)]);
                let paramlen = self.paramlen();
                element.insert_param(param_idx+paramlen, paramlist, editability);
                *self = new;
            },

            Self::Series(elems) => {
                assert!(elems.len() > 1);
                if let Some((i, start_x)) = series_index_by_block(elems, coord) {
                    if let Self::Element(_) = elems[i] {
                        let new_pidx = param_idx + (0..=i).map(|el| elems[el].paramlen()).sum::<usize>();
                        elems.insert(i+1, Self::Element(element));
                        element.insert_param(new_pidx, paramlist, editability);
                    }
                    else {
                        let new_pidx = param_idx + (0..i).map(|el| elems[el].paramlen()).sum::<usize>();
                        elems[i]._add_series((coord.0 - start_x, coord.1), element, paramlist, new_pidx, editability);
                    }
                }
            }

            Self::Parallel(elems) => {
                assert!(elems.len() > 1);
                let ib = parallel_index_by_block(elems, coord);

                if let ParallelBlockPicked::Child(i, elemblock, start_y) = ib {
                    let new_pidx = param_idx + (0..i).map(|el| elems[el].paramlen()).sum::<usize>();
                    elems[i]._add_series((coord.0-elemblock, coord.1-start_y), element, paramlist, new_pidx, editability);
                }
            }
        }
    }

    // Circuit editing: add the element parallel to the clicked one
    pub fn _add_parallel(&mut self, coord: (u16, u16), element: Element, paramlist: std::slice::IterMut<'_, StringParameterDesc>, param_idx: usize, editability: &mut Vec<ParameterEditability>){
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
                    elems[i]._add_parallel((coord.0 - start_x, coord.1), element, paramlist, new_pidx, editability);
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


    // Circuit editing: remove the clicked element
    pub fn _remove(&mut self, coord: (u16, u16), paramlist: std::slice::IterMut<'_, StringParameterDesc>, param_idx: usize, editability: &mut Vec<ParameterEditability>) {
        match self {
            Self::Element(_) => {},

            Self::Series(elems) => {
                assert!(elems.len() > 1);
                
                if let Some((i,start_x)) = series_index_by_block(elems, coord) {
                    let new_pidx = param_idx + (0..i).map(|el| elems[el].paramlen()).sum::<usize>();
                    let elemlen = elems.len();
                    if let Self::Element(el) = &mut elems[i] {
                        el.remove_param(new_pidx, paramlist, editability);
                        if elemlen == 2 {
                            *self = elems[usize::from(i == 0)].clone();
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

            Self::Parallel(elems) => {
                assert!(elems.len() > 1);
                let ib = parallel_index_by_block(elems, coord);

                match ib {
                    ParallelBlockPicked::Child(i, elemblock, start_y) => {
                        let new_pidx = param_idx + (0..i).map(|el| elems[el].paramlen()).sum::<usize>();
                        let elemlen = elems.len();
                        if let Self::Element(el) = &mut elems[i] {
                            el.remove_param(new_pidx, paramlist, editability);
                            if elemlen == 2 {
                                *self = elems[usize::from(i == 0)].clone();
                            }
                            else {
                                elems.remove(i);
                            }
                        }
                        else {
                            elems[i]._remove((coord.0-elemblock, coord.1-start_y), paramlist, new_pidx, editability);
                        }
                    }
                    ParallelBlockPicked::This | ParallelBlockPicked::None => {
                        
                    }
                }
            }
        }
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
