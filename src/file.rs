use crate::circuit::{Cplx, Circuit};
use crate::imped_math;
extern crate csv;


pub fn load_csv_freq_re_im(filename: &str) -> Result<Vec<imped_math::DataPiece>, Box<dyn std::error::Error>> {
    let file = std::fs::File::open(filename)?;
    let mut rdr = csv::Reader::from_reader(file);

    let mut out : Vec<imped_math::DataPiece> = vec![];

    for result in rdr.records() {
        let record = result?;

        let dpoint = imped_math::DataPiece {
            omega: record[0].parse::<f64>()? * 2.0 * std::f64::consts::PI,
            imp: Cplx{
                re: record[1].parse::<f64>()?, 
                im: record[2].parse::<f64>()?}};

        out.push(dpoint);
    }

    Ok(out)
}

pub fn save_model(model: &imped_math::Model, filename: &str) -> Result<(), ()> {
    if let Ok(mut file) = std::fs::File::create(filename) {
        use std::io::Write;
        writeln!(&mut file, "{}", model.circ).unwrap();
        for p in 0..model.params.vals.len() {
            writeln!(&mut file, "{} ({}, {})", model.params.vals[p], model.params.bounds[p].0, model.params.bounds[p].1).unwrap();
        }

        Ok(())
    }
    else {
        Err(())
    }
}



fn split_params(mut s: String) -> Result<(f64,f64,f64), ()>
{
    s.retain(|c| c != ' ');
    let s = s
        .replace('(', " ")
        .replace(',', " ")
        .replace(')', " ");
    let val : &str = &s;
    let line : Vec<&str> = val.split_whitespace().collect();
    if line.len() != 3 {return Err(())}
    let vals : Vec<_> = line.iter().map(|x| x.parse::<f64>()).collect();

    let val = match vals[0] {Ok(f) => f, Err(_) => {return Err(())}};
    let min = match vals[1] {Ok(f) => f, Err(_) => {return Err(())}};
    let max = match vals[2] {Ok(f) => f, Err(_) => {return Err(())}};
    
    Ok((val,min,max))
}

pub fn load_model(filename: &str, model: &mut imped_math::Model) -> Result<(), ()> {
    if let Ok(file) = std::fs::File::open(filename) {
        use std::io::BufRead;

        let buf = std::io::BufReader::new(&file);
        let mut lines = buf.lines();
        if let Some(Ok(firstline)) = lines.next() {
            use std::str::FromStr;
            if let Ok(circ) = Circuit::from_str(&firstline) {
                model.circ = circ;
            }
            else {return Err(());}
        }
        else {return Err(());}

        model.params.vals.clear();
        model.params.bounds.clear();

        for line in lines {
            if let Ok(par) = line {
                let (val,min,max) = split_params(par)?;
                model.params.vals.push(val);
                model.params.bounds.push((min,max));
            }
        }

        Ok(())
    }
    else {
        Err(())
    }

}

