
extern crate num;

pub type Cplx = num::complex::Complex<f64>;

use std::f64::consts::PI;
const I: Cplx = Cplx{ re: 0.0, im: 1.0 };

/// A helper structure representing a parameter type of circuit elements
#[derive(Clone)]
pub struct ParameterBase {
    letter : char,
    limits : (f64, f64),
}


pub const RESISTANCE: ParameterBase = ParameterBase {letter: 'R', limits: (0.0, std::f64::INFINITY)};
pub const CAPACITY:   ParameterBase = ParameterBase {letter: 'C', limits: (0.0, std::f64::INFINITY)};
pub const WARBURG_A:  ParameterBase = ParameterBase {letter: 'A', limits: (0.0, std::f64::INFINITY)};
pub const INDUCTANCE: ParameterBase = ParameterBase {letter: 'L', limits: (0.0, std::f64::INFINITY)};
pub const CPE_Q:      ParameterBase = ParameterBase {letter: 'Q', limits: (0.0, std::f64::INFINITY)};
pub const CPE_N:      ParameterBase = ParameterBase {letter: 'n', limits: (0.0, 1.0)};


/// An unaided circuit having its own
/// parameter set and frequency response

pub trait Circuit {
    /// A circuit-specific routine to calculate the impedance
    fn _impedance(&self, omega: f64, params: &[f64]) -> Cplx;

    /// Get a list of all circuit parameters. The parameter values are to provide
    /// when calculating the impedance.
    fn paramlist(&self) -> &[ParameterBase];

    /// Calculate the impedance value
    /// * `omega` is the angular frequency
    /// * `params` is the slice of circuit parameters in the order given by `paramlist` function
    fn impedance(&self, omega: f64, params: &[f64]) -> Cplx {
        assert!(params.len() == self.paramlist().len());
        return self._impedance(omega, params);
    }
}


// Basic circuit elements

pub struct Resistor {}
pub struct Capacitor {}
pub struct Inductor {}
pub struct Warburg {}
pub struct CPE {}

impl Circuit for Resistor {
    fn paramlist(&self) -> &[ParameterBase] {return &[RESISTANCE];}
    fn _impedance(&self, _omega: f64, params: &[f64]) -> Cplx {
        let r = params[0];
        Cplx::new(r, 0.0)
    }
}
impl Circuit for Capacitor {
    fn paramlist(&self) -> &[ParameterBase] {return &[CAPACITY];}
    fn _impedance(&self, omega: f64, params: &[f64]) -> Cplx {
        let c = params[0];
        1.0 / (Cplx::new(0.0, 1.0) * omega * c)
    }
}
impl Circuit for Inductor {
    fn paramlist(&self) -> &[ParameterBase] {return &[INDUCTANCE];}
    fn _impedance(&self, omega: f64, params: &[f64]) -> Cplx {
        let l = params[0];
        Cplx::new(0.0, 1.0) * omega * l
    }
}
impl Circuit for Warburg {
    fn paramlist(&self) -> &[ParameterBase] {return &[WARBURG_A];}
    fn _impedance(&self, omega: f64, params: &[f64]) -> Cplx {
        let aw = params[0];
        let sqrtom = omega.sqrt();
        aw/sqrtom + aw/(Cplx::new(0.0, 1.0)*sqrtom)
    }
}
impl Circuit for CPE {
    fn paramlist(&self) -> &[ParameterBase] {return &[CPE_Q, CPE_N];}
    fn _impedance(&self, omega: f64, params: &[f64]) -> Cplx {
        let q = params[0];
        let n = params[1];
        let numer = (-I*PI/2.0*n).exp();
        let denom = q * omega.powf(n);
        numer/denom
    }
}


/// The base for parallel and series circuits
pub struct ComplexCirc {
    c : Vec<Box<dyn Circuit>>,
    params : Vec<ParameterBase>
}
impl ComplexCirc {
    pub fn new(circ: Vec<Box<dyn Circuit>>) -> ComplexCirc {
        let mut pt: Vec<ParameterBase> = vec![];
        for c in &circ {
            for p in c.paramlist() {
                pt.push(p.clone());
            }
        }
        ComplexCirc{c: circ, params: pt}
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


impl Circuit for Series {
    fn paramlist(&self) -> &[ParameterBase] {&self.elems.params}
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
}

impl Circuit for Parallel {
    fn paramlist(&self) -> &[ParameterBase] {&self.elems.params}
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
