

use crate::circuit::*;

pub struct ParameterDesc {
    pub vals : Vec<f64>,
    pub bounds : Vec<(f64, f64)>
}
impl ParameterDesc{
    pub fn new(paramlist: &[ParameterBase]) -> Self {
        let vals = paramlist.iter().map(|x| x.default).collect::<Vec<_>>();;
        let bounds = paramlist.iter().map(|x| x.limits).collect::<Vec<_>>();;

        ParameterDesc{vals, bounds}
    }
}

/// A model description consists of
/// * The circuit description and metadata
/// * Its current parameters and bounds
pub struct Model {
    pub circ : Box<dyn Circuit>,
    pub params : ParameterDesc
}


#[derive(Debug, Copy, Clone)]
pub struct DataPiece {
    pub omega: f64,
    pub imp: Cplx,
}

/// A simple 2D vector
#[derive(PartialEq, Debug, Copy, Clone)]
pub struct V2 {
    pub x: f64,
    pub y: f64,
}
impl std::ops::Add for V2 {
    type Output = Self;
    fn add(self, other: Self) -> Self {  Self { x: self.x + other.x, y: self.y + other.y }  }
}
impl std::ops::Sub for V2 {
    type Output = Self;
    fn sub(self, other: Self) -> Self {  Self { x: self.x - other.x, y: self.y + other.y }  }
}

/// A simple rectangle
#[derive(PartialEq, Debug, Copy, Clone)]
pub struct Bounds {
    pub min: V2,
    pub max: V2,
}

/// Iterator for a `Model` providing `DataPiece` output for each of the given points (omega)
pub struct ModelIter<'model> {
    pub model : &'model Model,
    pub points : &'model mut dyn Iterator<Item=f64>,
}

impl Iterator for ModelIter<'_> {
    type Item = DataPiece;

    fn next(&mut self) -> Option<DataPiece> {
        match self.points.next() {
            Some(omega) => {
                let imp = self.model.circ.impedance(omega, &self.model.params.vals);
                Some(DataPiece{omega, imp})
            }
            None => None,
        }
    }
}


pub fn geomspace(first: f64, last: f64, count: usize) -> impl Iterator<Item=f64>
{
    let (lf, ll) = (first.ln(), last.ln());
    let delta = (ll - lf) / ((count-1) as f64);
    return (0..count).map(move |i| (lf + (i as f64) * delta).exp());
}
