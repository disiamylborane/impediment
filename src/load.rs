use crate::circuit::Cplx;
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
