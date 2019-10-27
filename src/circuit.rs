
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


/// An unaided circuit having its own
/// parameter set and frequency response
pub trait Circuit {
    /// A circuit-specific routine to calculate the impedance
    fn _impedance(&self, omega: f64, params: &[f64]) -> Cplx;

    /// Get a list of all circuit parameters. The parameter values are to provide
    /// when calculating the impedance.
    fn paramlist(&self) -> Vec<ParameterBase>;

    /// A size of painted circuit in <blocks>
    fn painted_size(&self) -> (u16, u16) {(2,2)}

    /// Draw a circuit(subcircuit) on a widget
    fn paint(&self, ctx: &cairo::Context, blocksize: f64, pos: (f64,f64));

    /// Calculate the impedance value
    /// * `omega` is the angular frequency
    /// * `params` is the slice of circuit parameters in the order given by `paramlist` function
    fn impedance(&self, omega: f64, params: &[f64]) -> Cplx {
        assert!(params.len() == self.paramlist().len());
        return self._impedance(omega, params);
    }

    fn replace(&mut self, coord: (u16, u16), element: Box<dyn Circuit>) -> Option<Box<dyn Circuit>>{
        Some(element)
    }

    fn add_series(&mut self, coord: (u16, u16), element: Box<dyn Circuit>) -> Option<Box<dyn Circuit>>{
        Some(element)
    }

    fn add_parallel(&mut self, coord: (u16, u16), element: Box<dyn Circuit>) -> Option<Box<dyn Circuit>>{
        Some(element)
    }

    fn remove(&mut self, coord: (u16, u16)) -> Option<()>{
        Some(())
    }
}

// Basic circuit elements

pub struct Resistor {}
pub struct Capacitor {}
pub struct Inductor {}
pub struct Warburg {}
pub struct CPE {}

/// Paint a sign under the element
fn sign_element(ctx: &cairo::Context, text: &str, pos: (f64, f64), blocksize: f64) {
    ctx.move_to(pos.0 + blocksize*3./4., pos.1 + blocksize*3./2.);
    ctx.set_font_size(blocksize*3./4.);
    ctx.show_text(text);
}

impl Circuit for Resistor {
    fn paramlist(&self) -> Vec<ParameterBase> {vec![RESISTANCE]}
    fn _impedance(&self, _omega: f64, params: &[f64]) -> Cplx {
        let r = params[0];
        Cplx::new(r, 0.0)
    }

    fn paint(&self, ctx: &cairo::Context, blocksize: f64, pos: (f64,f64)) {
        ctx.move_to(pos.0, pos.1 + blocksize/2.);
        ctx.line_to(pos.0 + blocksize/4., pos.1 + blocksize/2.);

        ctx.rectangle(pos.0 + blocksize/4., pos.1 + blocksize/4., blocksize*3./2., blocksize/2.);

        ctx.move_to(pos.0 + blocksize*7./4., pos.1 + blocksize/2.);
        ctx.line_to(pos.0 + blocksize*2., pos.1 + blocksize/2.);

        sign_element(ctx, &"R", pos, blocksize);
    }
}
impl Circuit for Capacitor {
    fn paramlist(&self) -> Vec<ParameterBase> {vec![CAPACITY]}
    fn _impedance(&self, omega: f64, params: &[f64]) -> Cplx {
        let c = params[0];
        1.0 / (Cplx::new(0.0, 1.0) * omega * c)
    }

    fn paint(&self, ctx: &cairo::Context, blocksize: f64, pos: (f64,f64)) {
        ctx.move_to(pos.0, pos.1 + blocksize/2.);
        ctx.line_to(pos.0 + blocksize*3./4., pos.1 + blocksize/2.);

        ctx.move_to(pos.0 + blocksize*3./4., pos.1);
        ctx.line_to(pos.0 + blocksize*3./4., pos.1 + blocksize);
        ctx.move_to(pos.0 + blocksize*5./4., pos.1);
        ctx.line_to(pos.0 + blocksize*5./4., pos.1 + blocksize);

        ctx.move_to(pos.0 + blocksize*5./4., pos.1 + blocksize/2.);
        ctx.line_to(pos.0 + blocksize*2., pos.1 + blocksize/2.);

        sign_element(ctx, &"C", pos, blocksize);
    }
}
impl Circuit for Inductor {
    fn paramlist(&self) -> Vec<ParameterBase> {vec![INDUCTANCE]}
    fn _impedance(&self, omega: f64, params: &[f64]) -> Cplx {
        let l = params[0];
        Cplx::new(0.0, 1.0) * omega * l
    }

    fn paint(&self, ctx: &cairo::Context, blocksize: f64, pos: (f64,f64)) {
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
}
impl Circuit for Warburg {
    fn paramlist(&self) -> Vec<ParameterBase> {vec![WARBURG_A]}
    fn _impedance(&self, omega: f64, params: &[f64]) -> Cplx {
        let aw = params[0];
        let sqrtom = omega.sqrt();
        aw/sqrtom + aw/(Cplx::new(0.0, 1.0)*sqrtom)
    }

    fn paint(&self, ctx: &cairo::Context, blocksize: f64, pos: (f64,f64)) {
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
}
impl Circuit for CPE {
    fn paramlist(&self) -> Vec<ParameterBase> {vec![CPE_Q, CPE_N]}
    fn _impedance(&self, omega: f64, params: &[f64]) -> Cplx {
        let q = params[0];
        let n = params[1];
        let numer = (-I*PI/2.0*n).exp();
        let denom = q * omega.powf(n);
        numer/denom
    }

    fn paint(&self, ctx: &cairo::Context, blocksize: f64, pos: (f64,f64)) {
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
}


/// The base for parallel and series circuits
pub struct ComplexCirc {
    c : Vec<Box<dyn Circuit>>
}
impl ComplexCirc {
    pub fn new(circ: Vec<Box<dyn Circuit>>) -> ComplexCirc {
        ComplexCirc{c: circ}
    }

    fn complex_paramlist(&self) -> Vec<ParameterBase> {
        self.c
            .iter()
            .map(|x| x.paramlist())
            .fold(vec![], |mut a, b| {a.extend(b); a})
    }
}

/// `Series`, `Parallel`:
/// 
/// The complex circuits
/// 
/// ## Example:
/// ```
/// let randles = Series{ elems: ComplexCirc::new(vec![
///     Box::new(Parallel {elems: ComplexCirc::new(vec![
///         Box::new(Series{ elems: ComplexCirc::new(vec![
///            Box::new(Resistor{}),
///            Box::new(Warburg{}),
///        ]}),
///        Box::new(Capacitor{}),
///     ])}),
///     Box::new(Resistor{}),
/// ])};
/// 
/// let randles_parameters = randles.paramlist();
/// ```
pub struct Series { pub elems : ComplexCirc } 
pub struct Parallel { pub elems : ComplexCirc }

impl Series {
    fn _index_by_block(&self, block: (u16, u16)) -> Option<(usize, u16)> {
        let mut start_x = 0_u16;
        for (i, el) in &mut self.elems.c.iter().enumerate() {
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
}

impl Circuit for Series {
    fn paramlist(&self) -> Vec<ParameterBase> {
        self.elems.complex_paramlist()
    }
    fn _impedance(&self, omega: f64, params: &[f64]) -> Cplx {
        let mut cval = 0;
        let mut imped = Cplx::new(0.0, 0.0);
        for c in &self.elems.c {
            let cend = cval + c.paramlist().len();
            let slc = &params[cval..cend];
            let ipd = c.impedance(omega, slc);
            imped += ipd;
            cval = cend;
        }
        return imped;
    }

    fn painted_size(&self) -> (u16, u16) {
        let s = self.elems.c.iter().map(|x| x.painted_size()).fold((0,0), |a, b| (a.0+b.0, std::cmp::max(a.1,b.1)));

        (s.0, s.1)
    }

    fn paint(&self, ctx: &cairo::Context, blocksize: f64, pos: (f64,f64)) {
        let (mut x, y) = pos;

        self.elems.c[0].paint( ctx, blocksize, (x, y) );
        x += self.elems.c[0].painted_size().0 as f64 * blocksize;
        
        for c in &self.elems.c[1..] {
            c.paint( ctx, blocksize, (x, y) );
            x += c.painted_size().0 as f64 * blocksize;
        }
    }

    fn replace(&mut self, coord: (u16, u16), element: Box<dyn Circuit>) -> Option<Box<dyn Circuit>> {
        if self.elems.c.len() == 1 {
            Some(element)
        }
        else {
            if let Some((i,start_x)) = self._index_by_block(coord) {
                let el = &mut self.elems.c[i];
                if let Some(rp) = el.replace((coord.0 - start_x, coord.1), element) {
                    *el = rp;
                }
            }
            None
        }
    }

    fn add_series(&mut self, coord: (u16, u16), element: Box<dyn Circuit>) -> Option<Box<dyn Circuit>>{
        if let Some((i, start_x)) = self._index_by_block(coord) {
            if let Some(elem) = self.elems.c[i].add_series((coord.0-start_x, coord.1), element) {
                self.elems.c.insert(i+1, elem);
            }
        }
        None
    }
    
    fn add_parallel(&mut self, coord: (u16, u16), element: Box<dyn Circuit>) -> Option<Box<dyn Circuit>>{
        if let Some((i, startx)) = self._index_by_block(coord){
            if let Some(elem) = self.elems.c[i].add_parallel((coord.0-startx, coord.1), element) {
                let prev = self.elems.c.remove(i);
                self.elems.c.insert(i, Box::new(Parallel{
                    elems: ComplexCirc {
                        c: vec![prev, elem]
                    }
                }));
            }
            None
        }
        else {
            Some(element)
        }
    }

    fn remove(&mut self, coord: (u16, u16)) -> Option<()>{
        if self.elems.c.len() == 1 {
            Some(())
        }
        else {
            if let Some((i,start_x)) = self._index_by_block(coord) {
                let el = &mut self.elems.c[i];
                if let Some(_) = el.remove((coord.0 - start_x, coord.1)) {
                    self.elems.c.remove(i);
                }
            }
            None
        }
    }
}

enum ParallelBlockPicked {
    Child(usize, u16, u16),
    This,
    None
}

impl Parallel {
    fn _index_by_block(&self, block: (u16, u16)) -> ParallelBlockPicked {
            let wsize = self.painted_size();
            if block.1 < wsize.1 && (block.0 == 0 || block.0 == wsize.0-1) {
                return ParallelBlockPicked::This;
            }

            let xsize = self.painted_size().0;

            let mut start_coord_y = 0_u16;
            for (i,el) in &mut self.elems.c.iter().enumerate() {
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
}

impl Circuit for Parallel {
    fn paramlist(&self) -> Vec<ParameterBase> {
        self.elems.complex_paramlist()
    }
    fn _impedance(&self, omega: f64, params: &[f64]) -> Cplx {
        let mut cval = 0;
        let mut admit = Cplx::new(0.0, 0.0);
        for c in &self.elems.c {
            let cend = cval + c.paramlist().len();
            let slc = &params[cval..cend];
            let ipd = c.impedance(omega, slc);
            admit += 1.0/ipd;
            cval = cend;
        }
        return 1.0/admit;
    }

    /// A size of painted circuit in <blocks>
    fn painted_size(&self) -> (u16, u16) {
        let s = self.elems.c.iter().map(|x| x.painted_size()).fold((0,0), |a, b| (std::cmp::max(a.0,b.0), a.1+b.1));

        (s.0 + 2, s.1)
    }

    fn paint(&self, ctx: &cairo::Context, blocksize: f64, pos: (f64,f64)) {
        let (_, mut y) = pos;
        let xsize = self.painted_size().0;

        ctx.move_to(pos.0, pos.1 + blocksize/2.);
        ctx.line_to(pos.0 + blocksize, pos.1 + blocksize/2.);
        ctx.move_to(pos.0 + (xsize-1) as f64 * blocksize, pos.1 + blocksize/2.);
        ctx.line_to(pos.0 + (xsize-1) as f64 * blocksize + blocksize, pos.1 + blocksize/2.);

        for c in &self.elems.c {
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

    fn replace(&mut self, coord: (u16, u16), element: Box<dyn Circuit>) -> Option<Box<dyn Circuit>> {
        if self.elems.c.len() == 1 {
            Some(element)
        }
        else {
            let ib = self._index_by_block(coord);

            match ib {
                ParallelBlockPicked::This => Some(element),
                ParallelBlockPicked::None => None,
                ParallelBlockPicked::Child(i, elemblock, start_y) => {
                    let el = &mut self.elems.c[i];
                    if let Some(rp) = el.replace((coord.0 - elemblock, coord.1 - start_y), element) {
                        *el = rp;
                    };
                    None
                }
            }
        }
    }

    fn add_series(&mut self, coord: (u16, u16), element: Box<dyn Circuit>) -> Option<Box<dyn Circuit>>{
        let ib = self._index_by_block(coord);
        
        match ib {
            ParallelBlockPicked::Child(i, elemblock, start_y) => {
                if let Some(elem) = self.elems.c[i].add_series((coord.0-elemblock, coord.1-start_y), element) {
                    let prev = self.elems.c.remove(i);
                    self.elems.c.insert(i, Box::new(Series{
                        elems: ComplexCirc {
                            c: vec![prev, elem]
                        }
                    }));
                }
                None
            }
            ParallelBlockPicked::This => {
                Some(element)
            }
            ParallelBlockPicked::None => {
                None
            }
        }
    }

    fn add_parallel(&mut self, coord: (u16, u16), element: Box<dyn Circuit>) -> Option<Box<dyn Circuit>>{
        let ib = self._index_by_block(coord);
        
        match ib {
            ParallelBlockPicked::Child(i, elemblock, start_y) => {
                if let Some(elem) = self.elems.c[i].add_parallel((coord.0-elemblock, coord.1-start_y), element) {
                    self.elems.c.insert(i+1, elem);
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

    fn remove(&mut self, coord: (u16, u16)) -> Option<()>{
        if self.elems.c.len() == 1 {
            Some(())
        }
        else {
            let ib = self._index_by_block(coord);

            match ib {
                ParallelBlockPicked::Child(i, elemblock, start_y) => {
                    if let Some(_) = self.elems.c[i].remove((coord.0-elemblock, coord.1-start_y)) {
                        self.elems.c.remove(i);
                    }
                    None
                }
                ParallelBlockPicked::This => {
                    Some(())
                }
                ParallelBlockPicked::None => {
                    None
                }
            }
        }
    }
}


// ---------- Unit tests ----------

// TODO Positive only tests
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
        
        assert!(approx_cplx(Resistor{}.impedance(1.0, &[20.0]), Cplx::new(20.0, 0.0), APPROX_VAL));
        assert!(approx_cplx(Resistor{}.impedance(1.0, &[200.0]), Cplx::new(200.0, 0.0), APPROX_VAL));
        assert!(approx_cplx(Resistor{}.impedance(1.0, &[2000.0]), Cplx::new(2000.0, 0.0), APPROX_VAL));

        assert!(approx_cplx(Capacitor{}.impedance(1.0, &[(20.0)]), Cplx::new(0.0, -1.0/20.0), APPROX_VAL));
        assert!(approx_cplx(Capacitor{}.impedance(1.0, &[(200.0)]), Cplx::new(0.0, -1.0/200.0), APPROX_VAL));
        assert!(approx_cplx(Capacitor{}.impedance(10.0, &[(20.0)]), Cplx::new(0.0, -1.0/200.0), APPROX_VAL));
        assert!(approx_cplx(Capacitor{}.impedance(10.0, &[(200.0)]), Cplx::new(0.0, -1.0/2000.0), APPROX_VAL));

        assert!(approx_cplx(Inductor{}.impedance(1.0, &[(20.0)]), Cplx::new(0.0, 20.0), APPROX_VAL));
        assert!(approx_cplx(Inductor{}.impedance(1.0, &[(200.0)]), Cplx::new(0.0, 200.0), APPROX_VAL));
        assert!(approx_cplx(Inductor{}.impedance(10.0, &[(20.0)]), Cplx::new(0.0, 200.0), APPROX_VAL));
        assert!(approx_cplx(Inductor{}.impedance(10.0, &[(200.0)]), Cplx::new(0.0, 2000.0), APPROX_VAL));

        assert!(approx_cplx(Warburg{}.impedance(1.0, &[(20.0)]), Cplx::new(20.0, -20.0), APPROX_VAL));
        assert!(approx_cplx(Warburg{}.impedance(1.0, &[(200.0)]), Cplx::new(200.0, -200.0), APPROX_VAL));
        assert!(approx_cplx(Warburg{}.impedance(100.0, &[(20.0)]), Cplx::new(2.0, -2.0), APPROX_VAL));
        assert!(approx_cplx(Warburg{}.impedance(100.0, &[(200.0)]), Cplx::new(20.0, -20.0), APPROX_VAL));

        assert!(approx_cplx(CPE{}.impedance(1.0, &[1.0/20.0, 0.0]), Cplx::new(20.0, 0.0), APPROX_VAL));
        assert!(approx_cplx(CPE{}.impedance(10.0, &[1.0/20.0, 0.0]), Cplx::new(20.0, 0.0), APPROX_VAL));
        assert!(approx_cplx(CPE{}.impedance(1.0, &[20.0, 1.0]), Cplx::new(0.0, -1.0/20.0), APPROX_VAL));
        assert!(approx_cplx(CPE{}.impedance(10.0, &[20.0, 1.0]), Cplx::new(0.0, -1.0/200.0), APPROX_VAL));
    }

    #[test]
    fn test_resistance() {
        let circ1 = Series{elems: ComplexCirc::new(vec![
            Box::new(Resistor{}),
            Box::new(Resistor{}),
        ])};

        let circ2 = Parallel{elems: ComplexCirc::new(vec![
            Box::new(Resistor{}),
            Box::new(Resistor{}),
        ])};

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
        let circ1 = Series{elems: ComplexCirc::new(vec![
            Box::new(Resistor{}),
            Box::new(Capacitor{}),
        ])};

        let circ2 = Parallel{elems: ComplexCirc::new(vec![
            Box::new(Resistor{}),
            Box::new(Capacitor{}),
        ])};

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
