use crate::ImportWindowMode;

extern crate csv;

fn rec_to_datapiece(
    rec: &csv::StringRecord,
    params: ImportWindowMode
) -> Option<crate::DataPiece> {
    let freq_d = rec.get(params.freq_col)?.to_string().replace(',', ".").parse::<f64>().ok()?;
    let rp1 = rec.get(params.re_z_col)?.to_string().replace(',', ".").parse::<f64>().ok()?;
    let rp2 = rec.get(params.im_z_col)?.to_string().replace(',', ".").parse::<f64>().ok()?;

    let freq = match params.freq_type {
        crate::FreqOpenParam::Hz => {freq_d}
        crate::FreqOpenParam::Khz => {freq_d*1000.}
        crate::FreqOpenParam::Rads => {freq_d/std::f64::consts::TAU}
        crate::FreqOpenParam::Krads => {freq_d*1000./std::f64::consts::TAU}
    };
    let imp = match params.imp_type {
        crate::ImpOpenParam::PlusOhm => {crate::Cplx::new(rp1, rp2)}
        crate::ImpOpenParam::MinusOhm => {crate::Cplx::new(rp1, -rp2)}
        crate::ImpOpenParam::PlusKohm => {crate::Cplx::new(rp1*1000., rp2*1000.)}
        crate::ImpOpenParam::MinusKohm => {crate::Cplx::new(rp1*1000., -rp2*1000.)}
    };

    Some(crate::DataPiece{freq, imp})
}

pub fn csv_to_impediment(
    text: &str,
    params: ImportWindowMode,
) -> Option<Vec<crate::DataPiece>> {
    csv_to_impediment_delim(text, params, b';')
        .or_else(|| csv_to_impediment_delim(text, params, b','))
}


pub fn csv_to_impediment_delim(
            text: &str,
            params: ImportWindowMode,
            delim: u8
) -> Option<Vec<crate::DataPiece>> {
    let mut rdr = csv::ReaderBuilder::new().delimiter(delim).from_reader(text.as_bytes());

    let mut out = Vec::with_capacity(32);

    let mut rec_iter = rdr.records();
    let first_rec = rec_iter.next()?.ok()?;

    if let Some(fdrec) = rec_to_datapiece(&first_rec, params) {
        out.push(fdrec);
    }

    for rec in rec_iter {
        let rec = rec.ok()?;
        out.push(rec_to_datapiece(&rec, params)?);
    }

    Some(out)
}
