use crate::{Circuit, ParameterEditability};
use float_pretty_print::PrettyPrintFloat;

pub type Cplx = num::complex::Complex<f64>;

// A single point (frequency-impedance) on impedance spectrum
// Impedance spectra are represented as Vec<DataPiece>

#[derive(Debug, Copy, Clone)]
pub struct DataPiece {
    pub freq: f64,
    pub imp: Cplx,
}


#[derive(Debug, Clone)]
pub struct Model {
    pub circuit: Circuit,
    pub name: String,
    pub parameters: Vec<ParameterEditability>,
    pub lock: bool,
}

// ParameterDesc and StringParameterDesc
// Circuit parameter descriptors: their values and
// min/max bounds used for fitting

#[derive(Debug, Clone)]
pub struct ParameterDesc {
    pub vals : Vec<f64>,
    pub bounds : Vec<(f64, f64)>
}

// String representation of `ParameterDesc` for GUI
#[derive(Debug, Clone)]
pub struct StringParameterDesc {
    pub vals : Vec<String>,
    pub bounds : Vec<(String, String)>
}

impl From<ParameterDesc> for StringParameterDesc {
    fn from(v: ParameterDesc) -> Self {
        Self {
            vals: v.vals.into_iter().map(|x| format!("{}", PrettyPrintFloat(x))).collect(),
            bounds: v.bounds.into_iter().map(|(a,b)| (format!("{}", PrettyPrintFloat(a)), format!("{}", PrettyPrintFloat(b)))).collect(), 
        }
    }
}

impl TryFrom<&StringParameterDesc> for ParameterDesc {
    type Error = ();
    fn try_from(v: &StringParameterDesc) -> Result<Self, ()> {
        Ok(
            Self {
                vals: v.vals.iter().map(|x| x.parse::<f64>().ok()).collect::<Option<Vec<f64>>>().ok_or(())?,
                bounds: v.bounds.iter().map(|(a,b)| {
                    let (a,b) = (a.parse::<f64>().ok(), b.parse::<f64>().ok());
                    if let (Some(a), Some(b)) = (a, b) {
                        return Some((a,b))
                    }
                    None
                }).collect::<Option<Vec<(f64, f64)>>>().ok_or(())?, 
            }
        )
    }
}


pub fn try_paramdesc_into_numbers(x: &[Vec<StringParameterDesc>]) -> Option<Vec<Vec<ParameterDesc>>> {
    x.iter().map(|ds| {
        ds.iter().map(|r|->Option<ParameterDesc> {
            r.try_into().ok()
        }).collect::< Option<Vec<_>> >()
    }).collect::< Option<Vec<_>> >()
}


impl ParameterDesc {
    pub fn insert(&mut self, index: usize, (val, bounds): (f64, (f64, f64))) {
        self.vals.insert(index, val);
        self.bounds.insert(index, bounds);
    }
    pub fn remove(&mut self, index: usize) -> (f64, (f64, f64)) {
        (self.vals.remove(index), self.bounds.remove(index))
    }
}

impl StringParameterDesc {
    pub fn insert(&mut self, index: usize, (val, bounds): (f64, (f64, f64))) {
        self.vals.insert(index, format!("{}", PrettyPrintFloat(val)));
        self.bounds.insert(index, (format!("{}", PrettyPrintFloat(bounds.0)), format!("{}", PrettyPrintFloat(bounds.1))));
    }
    pub fn remove(&mut self, index: usize) -> (String, (String, String)) {
        (self.vals.remove(index), self.bounds.remove(index))
    }
}
