
extern crate num;
extern crate cairo;

pub type Cplx = num::complex::Complex<f64>;

use std::f64::consts::PI;
const I: Cplx = Cplx{ re: 0.0, im: 1.0 };

/// A helper structure representing a parameter type of circuit elements
#[derive(Clone)]
pub struct ParameterBase {
    pub letter : char,
    pub limits : (f64, f64),
    pub default : f64,
}


pub const RESISTANCE: ParameterBase = ParameterBase {letter: 'R', limits: (0.0, std::f64::INFINITY), default:100.0};
pub const CAPACITY:   ParameterBase = ParameterBase {letter: 'C', limits: (0.0, std::f64::INFINITY), default:0.001};
pub const WARBURG_A:  ParameterBase = ParameterBase {letter: 'A', limits: (0.0, std::f64::INFINITY), default:1000.0};
pub const INDUCTANCE: ParameterBase = ParameterBase {letter: 'L', limits: (0.0, std::f64::INFINITY), default:0.001};
pub const CPE_Q:      ParameterBase = ParameterBase {letter: 'Q', limits: (0.0, std::f64::INFINITY), default:0.001};
pub const CPE_N:      ParameterBase = ParameterBase {letter: 'n', limits: (0.0, 1.0), default:0.5};

#[derive(Clone, Debug)]
pub enum Element {
    Resistor,
    Capacitor,
    Inductor,
    Warburg,
    CPE,
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
fn sign_element(ctx: &cairo::Context, text: &str, pos: (f64, f64), blocksize: f64) {
    ctx.move_to(pos.0 + blocksize*3./4., pos.1 + blocksize*3./2.);
    ctx.set_font_size(blocksize*3./4.);
    ctx.show_text(text);
}


pub enum RemoveAction {
    DoNothing,
    Remove,
    ChangeTo(Circuit),
}
/*
fn strslice_to_subcircuits(s: &str) -> Result<Circuit, ()> {
    let buffer : Vec<Circuit> = vec![];
    for c in s.chars() {
        match c {
            'R' => {buffer.push(Circuit::Element(Element::Resistor));}
            'C' => {buffer.push(Circuit::Element(Element::Capacitor));}
            'L' => {buffer.push(Circuit::Element(Element::Inductor));}
            'W' => {buffer.push(Circuit::Element(Element::Warburg));}
            'Q' => {buffer.push(Circuit::Element(Element::CPE));}
        }
    }
}

enum CircuitParseElement {Series, Parallel}
*/
enum CircuitParseElement {Circ(Circuit), Series, Parallel}

impl std::str::FromStr for Circuit {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut stack : Vec<CircuitParseElement> = vec![];

        for c in s.chars() {
            match c {
                'R' => {stack.push(CircuitParseElement::Circ(Circuit::Element(Element::Resistor)));}
                'C' => {stack.push(CircuitParseElement::Circ(Circuit::Element(Element::Capacitor)));}
                'L' => {stack.push(CircuitParseElement::Circ(Circuit::Element(Element::Inductor)));}
                'W' => {stack.push(CircuitParseElement::Circ(Circuit::Element(Element::Warburg)));}
                'Q' => {stack.push(CircuitParseElement::Circ(Circuit::Element(Element::CPE)));}
                '[' => {stack.push(CircuitParseElement::Series);}
                '{' => {stack.push(CircuitParseElement::Parallel);}

                ']' => {
                    let mut elements : Vec<Circuit> = vec![];

                    while let Some(v) = stack.pop() {
                        match v {
                            CircuitParseElement::Circ(c) => {elements.insert(0, c);}
                            CircuitParseElement::Series => {break;}
                            CircuitParseElement::Parallel => {return Err(());}
                        }
                    }

                    stack.push(CircuitParseElement::Circ(Circuit::Series(elements)));
                }

                '}' => {
                    let mut elements : Vec<Circuit> = vec![];

                    while let Some(v) = stack.pop() {
                        match v {
                            CircuitParseElement::Circ(c) => {elements.insert(0, c);}
                            CircuitParseElement::Series => {return Err(());}
                            CircuitParseElement::Parallel => {break;}
                        }
                    }

                    stack.push(CircuitParseElement::Circ(Circuit::Parallel(elements)));
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

impl std::fmt::Display for Circuit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Circuit::Element(Element::Resistor) => write!(f, "R"),
            Circuit::Element(Element::Capacitor) => write!(f, "C"),
            Circuit::Element(Element::Inductor) => write!(f, "L"),
            Circuit::Element(Element::Warburg) => write!(f, "W"),
            Circuit::Element(Element::CPE) => write!(f, "Q"),
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

impl Circuit {
    pub fn paramlist(&self) -> Vec<ParameterBase>{
        match self {
            Circuit::Element(Element::Resistor) => vec![RESISTANCE],
            Circuit::Element(Element::Capacitor) => vec![CAPACITY],
            Circuit::Element(Element::Inductor) => vec![INDUCTANCE],
            Circuit::Element(Element::Warburg) => vec![WARBURG_A],
            Circuit::Element(Element::CPE) => vec![CPE_Q, CPE_N],
            Circuit::Series(elems) => elems.iter().map(|x| x.paramlist()).fold(vec![], |mut a, b| {a.extend(b); a}),
            Circuit::Parallel(elems) => elems.iter().map(|x| x.paramlist()).fold(vec![], |mut a, b| {a.extend(b); a}),
        }
    }

    fn _impedance(&self, omega: f64, params: &[f64]) -> Cplx{
        match self {
            Circuit::Element(Element::Resistor) => Cplx::new(params[0], 0.0),

            Circuit::Element(Element::Capacitor) => 1.0 / (I * omega * params[0]),

            Circuit::Element(Element::Inductor) => I * omega * params[0],

            Circuit::Element(Element::Warburg) => (1.0 - I) * params[0] / omega.sqrt(),

            Circuit::Element(Element::CPE) => {
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
                    let cend = cval + c.paramlist().len();
                    let slc = &params[cval..cend];
                    let ipd = c.impedance(omega, slc);
                    imped += ipd;
                    cval = cend;
                }
                return imped;
            }

            Circuit::Parallel(elems) => {
                let mut cval = 0;
                let mut admit = Cplx::new(0.0, 0.0);
                for c in elems.iter() {
                    let cend = cval + c.paramlist().len();
                    let slc = &params[cval..cend];
                    let ipd = c.impedance(omega, slc);
                    admit += 1.0/ipd;
                    cval = cend;
                }
                return 1.0/admit;
            }
        }
    }

    pub fn _d_impedance(&self, omega: f64, params: &[f64], dparam: usize) -> Cplx {
        if dparam >= self.paramlist().len() {
            //panic!("Gradient requested for {} parameter of {}", dparam, self)
            panic!()
        }
        match self {
            Circuit::Element(Element::Resistor) => Cplx::new(1.0, 0.0),

            Circuit::Element(Element::Capacitor) => I / (omega * params[0].powi(2)),

            Circuit::Element(Element::Inductor) => I * omega,

            Circuit::Element(Element::Warburg) => ((1.0 - I)*params[0])/(omega.sqrt()),

            Circuit::Element(Element::CPE) => {
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
                    let cend = cval + c.paramlist().len();
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
                    let cend = cval + c.paramlist().len();
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

    pub fn paint(&self, ctx: &cairo::Context, blocksize: f64, pos: (f64,f64)) {
        match self {
            Circuit::Element(Element::Resistor) => {
                ctx.move_to(pos.0, pos.1 + blocksize/2.);
                ctx.line_to(pos.0 + blocksize/4., pos.1 + blocksize/2.);

                ctx.rectangle(pos.0 + blocksize/4., pos.1 + blocksize/4., blocksize*3./2., blocksize/2.);

                ctx.move_to(pos.0 + blocksize*7./4., pos.1 + blocksize/2.);
                ctx.line_to(pos.0 + blocksize*2., pos.1 + blocksize/2.);

                sign_element(ctx, &"R", pos, blocksize);
            }
            
            Circuit::Element(Element::Capacitor) => {
                ctx.move_to(pos.0, pos.1 + blocksize/2.);
                ctx.line_to(pos.0 + blocksize*3./4., pos.1 + blocksize/2.);

                ctx.move_to(pos.0 + blocksize*3./4., pos.1);
                ctx.line_to(pos.0 + blocksize*3./4., pos.1 + blocksize);
                ctx.move_to(pos.0 + blocksize*5./4., pos.1);
                ctx.line_to(pos.0 + blocksize*5./4., pos.1 + blocksize);

                ctx.move_to(pos.0 + blocksize*5./4., pos.1 + blocksize/2.);
                ctx.line_to(pos.0 + blocksize*2., pos.1 + blocksize/2.);

                sign_element(ctx, &"C", pos, blocksize);
            },

            Circuit::Element(Element::Inductor) => {
                ctx.move_to(pos.0, pos.1 + blocksize/2.);
                ctx.line_to(pos.0 + blocksize/4., pos.1 + blocksize/2.);

                ctx.curve_to(pos.0 + blocksize*5./12., pos.1,
                            pos.0 + blocksize*7./12., pos.1,
                            pos.0 + blocksize*9./12., pos.1 + blocksize/2.);
                ctx.curve_to(pos.0 + blocksize*11./12., pos.1,
                            pos.0 + blocksize*13./12., pos.1,
                            pos.0 + blocksize*15./12., pos.1 + blocksize/2.);
                ctx.curve_to(pos.0 + blocksize*17./12., pos.1,
                            pos.0 + blocksize*19./12., pos.1,
                            pos.0 + blocksize*21./12., pos.1 + blocksize/2.);

                ctx.line_to(pos.0 + blocksize*2., pos.1 + blocksize/2.);

                sign_element(ctx, &"L", pos, blocksize);
            }

            Circuit::Element(Element::Warburg) => {
                ctx.move_to(pos.0, pos.1 + blocksize/2.);
                ctx.line_to(pos.0 + blocksize*3./4., pos.1 + blocksize/2.);

                ctx.move_to(pos.0 + blocksize*3./4., pos.1);
                ctx.line_to(pos.0 + blocksize*3./4., pos.1 + blocksize);
                ctx.move_to(pos.0 + blocksize*6./4., pos.1);
                ctx.line_to(pos.0 + blocksize*5./4., pos.1 + blocksize/2.);
                ctx.line_to(pos.0 + blocksize*6./4., pos.1 + blocksize);

                ctx.move_to(pos.0 + blocksize*5./4., pos.1 + blocksize/2.);
                ctx.line_to(pos.0 + blocksize*2., pos.1 + blocksize/2.);

                sign_element(ctx, &"W", pos, blocksize);
            }

            Circuit::Element(Element::CPE) => {
                ctx.move_to(pos.0, pos.1 + blocksize/2.);
                ctx.line_to(pos.0 + blocksize*3./4., pos.1 + blocksize/2.);

                ctx.move_to(pos.0 + blocksize*4./4., pos.1);
                ctx.line_to(pos.0 + blocksize*3./4., pos.1 + blocksize/2.);
                ctx.line_to(pos.0 + blocksize*4./4., pos.1 + blocksize);

                ctx.move_to(pos.0 + blocksize*6./4., pos.1);
                ctx.line_to(pos.0 + blocksize*5./4., pos.1 + blocksize/2.);
                ctx.line_to(pos.0 + blocksize*6./4., pos.1 + blocksize);

                ctx.move_to(pos.0 + blocksize*5./4., pos.1 + blocksize/2.);
                ctx.line_to(pos.0 + blocksize*2., pos.1 + blocksize/2.);

                sign_element(ctx, &"Q", pos, blocksize);
            }

            Circuit::Series(elems) => {
                let (mut x, y) = pos;

                elems[0].paint( ctx, blocksize, (x, y) );
                x += elems[0].painted_size().0 as f64 * blocksize;

                for c in elems[1..].iter() {
                    c.paint( ctx, blocksize, (x, y) );
                    x += c.painted_size().0 as f64 * blocksize;
                }
            }

            Circuit::Parallel(elems) => {
                let (_, mut y) = pos;
                let xsize = self.painted_size().0;

                ctx.move_to(pos.0, pos.1 + blocksize/2.);
                ctx.line_to(pos.0 + blocksize, pos.1 + blocksize/2.);
                ctx.move_to(pos.0 + (xsize-1) as f64 * blocksize, pos.1 + blocksize/2.);
                ctx.line_to(pos.0 + (xsize-1) as f64 * blocksize + blocksize, pos.1 + blocksize/2.);

                for c in elems.iter() {
                    let psize = c.painted_size().0;
                    let drawend = pos.0 + (xsize as f64)*blocksize;
                    let elemblock = (xsize-2-psize)/2 + 1;
                    let elemstart = (elemblock as f64)*blocksize + pos.0;
                    let elemend = elemstart + (psize as f64)*blocksize;
                    c.paint( ctx, blocksize, (elemstart, y) );

                    ctx.move_to(pos.0 + blocksize/2., y+blocksize/2.);
                    ctx.line_to(elemstart, y+blocksize/2.);
                    ctx.move_to(elemend, y+blocksize/2.);
                    ctx.line_to(pos.0 + (xsize as f64)*blocksize - blocksize/2., y+blocksize/2.);

                    ctx.move_to(pos.0 + blocksize/2., pos.1+blocksize/2.);
                    ctx.line_to(pos.0 + blocksize/2., y+blocksize/2.);

                    ctx.move_to(drawend - blocksize/2., pos.1+blocksize/2.);
                    ctx.line_to(drawend - blocksize/2., y+blocksize/2.);
                    y += c.painted_size().1 as f64 * blocksize;
                }
            }
        }
    }

    pub fn impedance(&self, omega: f64, params: &[f64]) -> Cplx {
        assert!(params.len() == self.paramlist().len());
        return self._impedance(omega, params);
    }

    pub fn replace(&mut self, coord: (u16, u16), element: Circuit) -> Option<Circuit> {
        match self {
            Circuit::Element(_) => Some(element),

            Circuit::Series(elems) => {
                assert!(elems.len() > 1);
                if let Some((i,start_x)) = series_index_by_block(&elems, coord) {
                    let el = &mut elems[i];
                    if let Some(rp) = el.replace((coord.0 - start_x, coord.1), element) {
                        *el = rp;
                    }
                }
                None
            }

            Circuit::Parallel(elems) => {
                assert!(elems.len() > 1);
                let ib = parallel_index_by_block(&elems, coord);

                match ib {
                    ParallelBlockPicked::This => Some(element),
                    ParallelBlockPicked::None => None,
                    ParallelBlockPicked::Child(i, elemblock, start_y) => {
                        let el = &mut elems[i];
                        if let Some(rp) = el.replace((coord.0 - elemblock, coord.1 - start_y), element) {
                            *el = rp;
                        };
                        None
                    }
                }
            }
        }
    }

    pub fn add_series(&mut self, coord: (u16, u16), element: Circuit) -> Option<Circuit>{
        match self {
            Circuit::Element(_) => Some(element),

            Circuit::Series(elems) => {
                assert!(elems.len() > 1);
                if let Some((i, start_x)) = series_index_by_block(&elems, coord) {
                    if let Some(elem) = elems[i].add_series((coord.0-start_x, coord.1), element) {
                        elems.insert(i+1, elem);
                    }
                }
                None
            }

            Circuit::Parallel(elems) => {
                assert!(elems.len() > 1);
                let ib = parallel_index_by_block(&elems, coord);

                match ib {
                    ParallelBlockPicked::Child(i, elemblock, start_y) => {
                        if let Some(elem) = elems[i].add_series((coord.0-elemblock, coord.1-start_y), element) {
                            let prev = elems.remove(i);
                            elems.insert(i, Circuit::Series(vec![prev, elem]));
                        }
                        None
                    }
                    _ => None
                }
            }
        }
    }

    pub fn add_parallel(&mut self, coord: (u16, u16), element: Circuit) -> Option<Circuit>{
        match self {
            Circuit::Element(_) => Some(element),

            Circuit::Series(elems) => {
                assert!(elems.len() > 1);
                if let Some((i, startx)) = series_index_by_block(&elems, coord){
                    if let Some(elem) = elems[i].add_parallel((coord.0-startx, coord.1), element) {
                        let prev = elems.remove(i);
                        elems.insert(i, Circuit::Parallel(vec![prev, elem]));
                    }
                    None
                }
                else {
                    Some(element)
                }
            }

            Circuit::Parallel(elems) => {
                assert!(elems.len() > 1);

                let ib = parallel_index_by_block(&elems, coord);
                
                match ib {
                    ParallelBlockPicked::Child(i, elemblock, start_y) => {
                        if let Some(elem) = elems[i].add_parallel((coord.0-elemblock, coord.1-start_y), element) {
                            elems.insert(i+1, elem);
                        }
                        None
                    }
                    ParallelBlockPicked::This => {
                        None
                    }
                    ParallelBlockPicked::None => {
                        None
                    }
                }
            }
        }
    }


    pub fn remove(&mut self, coord: (u16, u16)) -> RemoveAction {
        match self {
            Circuit::Element(_) => RemoveAction::Remove,

            Circuit::Series(elems) => {
                assert!(elems.len() > 1);
                
                if let Some((i,start_x)) = series_index_by_block(&elems, coord) {
                    let el = &mut elems[i];
                    let remove_sub = el.remove((coord.0 - start_x, coord.1));
                    match remove_sub {
                        RemoveAction::DoNothing => {return RemoveAction::DoNothing;}
                        RemoveAction::ChangeTo(newelem) => {*el = newelem; return RemoveAction::DoNothing;}
                        RemoveAction::Remove => {elems.remove(i);}
                    }
                }

                if elems.len() == 1 {
                    RemoveAction::ChangeTo(elems.pop().unwrap())
                }
                else {
                    RemoveAction::DoNothing
                }
            }

            Circuit::Parallel(elems) => {
                assert!(elems.len() > 1);
                let ib = parallel_index_by_block(&elems, coord);

                match ib {
                    ParallelBlockPicked::Child(i, elemblock, start_y) => {
                        let remove_sub = elems[i].remove((coord.0-elemblock, coord.1-start_y));
                        match remove_sub {
                            RemoveAction::DoNothing => {return RemoveAction::DoNothing;},
                            RemoveAction::ChangeTo(newelem) => {elems[i] = newelem; return RemoveAction::DoNothing;}
                            RemoveAction::Remove => {elems.remove(i);}
                        }

                        if elems.len() == 1 
                            { RemoveAction::ChangeTo(elems.pop().unwrap()) }
                        else
                            { RemoveAction::DoNothing }
                    }
                    ParallelBlockPicked::This => {
                        RemoveAction::Remove
                    }
                    ParallelBlockPicked::None => {
                        RemoveAction::DoNothing
                    }
                }
            }
        }
    }
}


// TODO Positive tests only
#[cfg(test)]
mod test {
    use super::*;

    fn approx_cplx(x: Cplx, y: Cplx, dev: f64) -> bool {
        let diff = y-x;
        let d2 = diff * diff.conj();
        return d2.re < dev*dev;
    }
    const APPROX_VAL : f64 = 1e-12;


    #[test]
    fn test_elements() {
        
        assert!(approx_cplx(Circuit::Element(Element::Resistor).impedance(1.0, &[20.0]), Cplx::new(20.0, 0.0), APPROX_VAL));
        assert!(approx_cplx(Circuit::Element(Element::Resistor).impedance(1.0, &[200.0]), Cplx::new(200.0, 0.0), APPROX_VAL));
        assert!(approx_cplx(Circuit::Element(Element::Resistor).impedance(1.0, &[2000.0]), Cplx::new(2000.0, 0.0), APPROX_VAL));

        assert!(approx_cplx(Circuit::Element(Element::Capacitor).impedance(1.0, &[(20.0)]), Cplx::new(0.0, -1.0/20.0), APPROX_VAL));
        assert!(approx_cplx(Circuit::Element(Element::Capacitor).impedance(1.0, &[(200.0)]), Cplx::new(0.0, -1.0/200.0), APPROX_VAL));
        assert!(approx_cplx(Circuit::Element(Element::Capacitor).impedance(10.0, &[(20.0)]), Cplx::new(0.0, -1.0/200.0), APPROX_VAL));
        assert!(approx_cplx(Circuit::Element(Element::Capacitor).impedance(10.0, &[(200.0)]), Cplx::new(0.0, -1.0/2000.0), APPROX_VAL));

        assert!(approx_cplx(Circuit::Element(Element::Inductor).impedance(1.0, &[(20.0)]), Cplx::new(0.0, 20.0), APPROX_VAL));
        assert!(approx_cplx(Circuit::Element(Element::Inductor).impedance(1.0, &[(200.0)]), Cplx::new(0.0, 200.0), APPROX_VAL));
        assert!(approx_cplx(Circuit::Element(Element::Inductor).impedance(10.0, &[(20.0)]), Cplx::new(0.0, 200.0), APPROX_VAL));
        assert!(approx_cplx(Circuit::Element(Element::Inductor).impedance(10.0, &[(200.0)]), Cplx::new(0.0, 2000.0), APPROX_VAL));

        assert!(approx_cplx(Circuit::Element(Element::Warburg).impedance(1.0, &[(20.0)]), Cplx::new(20.0, -20.0), APPROX_VAL));
        assert!(approx_cplx(Circuit::Element(Element::Warburg).impedance(1.0, &[(200.0)]), Cplx::new(200.0, -200.0), APPROX_VAL));
        assert!(approx_cplx(Circuit::Element(Element::Warburg).impedance(100.0, &[(20.0)]), Cplx::new(2.0, -2.0), APPROX_VAL));
        assert!(approx_cplx(Circuit::Element(Element::Warburg).impedance(100.0, &[(200.0)]), Cplx::new(20.0, -20.0), APPROX_VAL));

        assert!(approx_cplx(Circuit::Element(Element::CPE).impedance(1.0, &[1.0/20.0, 0.0]), Cplx::new(20.0, 0.0), APPROX_VAL));
        assert!(approx_cplx(Circuit::Element(Element::CPE).impedance(10.0, &[1.0/20.0, 0.0]), Cplx::new(20.0, 0.0), APPROX_VAL));
        assert!(approx_cplx(Circuit::Element(Element::CPE).impedance(1.0, &[20.0, 1.0]), Cplx::new(0.0, -1.0/20.0), APPROX_VAL));
        assert!(approx_cplx(Circuit::Element(Element::CPE).impedance(10.0, &[20.0, 1.0]), Cplx::new(0.0, -1.0/200.0), APPROX_VAL));
    }

    #[test]
    fn test_resistance() {
        let circ1 = Circuit::Series(vec![
            Circuit::Element(Element::Resistor),
            Circuit::Element(Element::Resistor),
        ]);

        let circ2 = Circuit::Parallel(vec![
            Circuit::Element(Element::Resistor),
            Circuit::Element(Element::Resistor),
        ]);

        let params = [
            (40.0),
            (40.0),
        ];
        
        println!("{:.3}", circ1.impedance(1.0, &params).norm_sqr().sqrt());
        println!("{:.3}", circ2.impedance(1.0, &params).norm_sqr().sqrt());

        assert!(approx_cplx(circ1.impedance(1.0, &params), Cplx::new(80.0, 0.0), APPROX_VAL));
        assert!(approx_cplx(circ2.impedance(1.0, &params), Cplx::new(20.0, 0.0), APPROX_VAL));
        assert!(approx_cplx(circ1.impedance(10.0, &params), Cplx::new(80.0, 0.0), APPROX_VAL));
        assert!(approx_cplx(circ2.impedance(10.0, &params), Cplx::new(20.0, 0.0), APPROX_VAL));
    }

    #[test]
    fn test_c() {
        let circ1 = Circuit::Series(vec![
            Circuit::Element(Element::Resistor),
            Circuit::Element(Element::Capacitor),
        ]);

        let circ2 = Circuit::Parallel(vec![
            Circuit::Element(Element::Resistor),
            Circuit::Element(Element::Capacitor),
        ]);

        let params = [
            (40.0),
            (1.0),
        ];

        println!("{:.3}", circ1.impedance(1.0, &params));
        println!("{:.3}", circ2.impedance(1.0, &params));
        println!("{:.3}", circ1.impedance(10.0, &params));
        println!("{:.3}", circ2.impedance(10.0, &params));

        assert!(approx_cplx(circ1.impedance(1.0, &params), Cplx::new(40.0, -1.0), APPROX_VAL));
        assert!(approx_cplx(circ1.impedance(10.0, &params), Cplx::new(40.0, -0.1), APPROX_VAL));

        assert!(approx_cplx(circ2.impedance(1.0, &params), 1.0/Cplx::new(1.0/40.0, 1.0), APPROX_VAL));
        assert!(approx_cplx(circ2.impedance(10.0, &params), 1.0/Cplx::new(1.0/40.0, 10.0), APPROX_VAL));
    }
}
