use crate::circuit::{Cplx, Circuit};
use crate::imped_math;
extern crate csv;

use gtk::prelude::*;

struct FileLoadingError;
impl std::fmt::Display for FileLoadingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "No file loaded")
    }
}
impl std::fmt::Debug for FileLoadingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "No file loaded")
    }
}
impl std::error::Error for FileLoadingError{}

pub fn load_csv(filename: &str, builder: &gtk::Builder, mainwindow: &gtk::Window) 
                    -> Result<Vec<imped_math::DataPiece>, Box<dyn std::error::Error>> {
    use std::io::Seek;

    let file = std::fs::File::open(filename)?;

    let dialog: gtk::Dialog = builder.get_object("dialog_load_csv").unwrap();

    #[derive(Debug)]
    enum FreqType {FreqHz, FreqKHz, AngularHz, AngularKHz};
    #[derive(Debug)]
    enum ImpedanceType {RealImagOhm, RealImagKOhm};

    impl std::fmt::Display for FreqType {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            use FreqType::*;
            write!(f, "{}", match self {
                FreqHz => "Hz", FreqKHz => "kHz", AngularHz => "rad/s", AngularKHz => "krad/s"
            })
        }
    }
    impl std::fmt::Display for ImpedanceType {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            use ImpedanceType::*;
            write!(f, "{}", match self {
                RealImagOhm => "Ohm", RealImagKOhm => "kOhm"
            })
        }
    }

    #[derive(Debug)]
    struct CSVParams {
        column_x: u32,
        column_y_prim: u32,
        column_y_sec: u32,
        skip_start: u32,
        freq_type: FreqType,
        imp_type: ImpedanceType,

        response_do_open: bool,

        file: std::fs::File,
    };

    let e_freq_col = builder.get_object::<gtk::SpinButton>("e_freq_col").unwrap();
    let e_cplx_primary_col = builder.get_object::<gtk::SpinButton>("e_cplx_primary_col").unwrap();
    let e_cplx_secondary_col = builder.get_object::<gtk::SpinButton>("e_cplx_secondary_col").unwrap();
    let e_startline = builder.get_object::<gtk::SpinButton>("e_spstartline").unwrap();

    let cb_freq_select = builder.get_object::<gtk::ComboBoxText>("cb_freq_select").unwrap();
    let cb_complex_select = builder.get_object::<gtk::ComboBoxText>("cb_complex_select").unwrap();

    let t_preview = builder.get_object::<gtk::TextView>("t_preview").unwrap();

    let freqtype_get = |cb: &gtk::ComboBoxText| match cb.get_active() {
        Some(0) => FreqType::FreqHz,
        Some(1) => FreqType::FreqKHz,
        Some(2) => FreqType::AngularHz,
        Some(3) => FreqType::AngularKHz,
        _ => FreqType::FreqHz
    };

    let imptype_get = |cb: &gtk::ComboBoxText| match cb.get_active() {
        Some(0) => ImpedanceType::RealImagOhm,
        Some(1) => ImpedanceType::RealImagKOhm,
        _ => ImpedanceType::RealImagOhm
    };

    let csv_params = std::sync::Arc::new(std::cell::RefCell::new(CSVParams{
        column_x: e_freq_col.get_value() as u32,
        column_y_prim: e_cplx_primary_col.get_value() as u32,
        column_y_sec: e_cplx_secondary_col.get_value() as u32,
        skip_start: e_startline.get_value() as u32,
        freq_type: freqtype_get(&cb_freq_select),
        imp_type: imptype_get(&cb_complex_select),

        response_do_open: false,

        file,
    }));


    let display_preview = {
        let t_preview_clone = t_preview.clone();
        move |params: &mut CSVParams| -> Result<(), Box<dyn std::error::Error>> {
            let override_color = || {t_preview_clone.override_color(gtk::StateFlags::NORMAL, Some(&gdk::RGBA::red())); Ok(())};

            params.file.seek(std::io::SeekFrom::Start(0)).unwrap();
            let mut reader = csv::Reader::from_reader(&params.file);

            let recs = reader.records().skip(params.skip_start as _);
            let mut s_out = String::new();
            for result in recs.take(5) {
                if let Ok(rec) = result {

                    let fval = match rec.get(params.column_x as _) {
                        Some(c)=>c, None=>{return override_color()}
                    }.parse::<f64>();
                    let ival1 = match rec.get(params.column_y_prim as _) {
                        Some(c)=>c, None=>{return override_color()}
                    }.parse::<f64>();
                    let ival2 = match rec.get(params.column_y_sec as _) {
                        Some(c)=>c, None=>{return override_color()}
                    }.parse::<f64>();

                    if let (Ok(fval),Ok(ival1),Ok(ival2)) = (fval,ival1,ival2) {
                        s_out += &format!("{} {}: {} + {}i {}\n", fval, params.freq_type, ival1, ival2, params.imp_type);
                    }
                    else {
                        return override_color();
                    }
                }
            }

            t_preview_clone.override_color(gtk::StateFlags::NORMAL, None);
            t_preview_clone.get_buffer().unwrap().set_text(&s_out);

            Ok(())
        }
    };

    let display_preview = std::sync::Arc::new(display_preview);

    if let Ok(_) = display_preview(&mut csv_params.borrow_mut())
        {}

    {
        let csv_params_clone = csv_params.clone();
        let display_preview_clone = display_preview.clone();
        cb_freq_select.connect_changed(move |cb| {
            if let Ok(mut params) = csv_params_clone.try_borrow_mut() {
                params.freq_type = freqtype_get(cb);

                display_preview_clone(&mut params).unwrap();
            };
        });
    }
    {
        let csv_params_clone = csv_params.clone();
        let display_preview_clone = display_preview.clone();
        cb_complex_select.connect_changed(move |cb| {
            if let Ok(mut params) = csv_params_clone.try_borrow_mut() {
                params.imp_type = imptype_get(cb);

                display_preview_clone(&mut params).unwrap();
            };
        });
    }

    {
        let csv_params_clone = csv_params.clone();
        let display_preview_clone = display_preview.clone();
        e_freq_col.connect_changed(move |efcol| {
            if let Ok(mut params) = csv_params_clone.try_borrow_mut() {
                params.column_x = efcol.get_value() as u32;
                display_preview_clone(&mut params).unwrap();
            };
        });
    }
    {
        let csv_params_clone = csv_params.clone();
        let display_preview_clone = display_preview.clone();
        e_cplx_primary_col.connect_changed(move |efcol| {
            if let Ok(mut params) = csv_params_clone.try_borrow_mut() {
                params.column_y_prim = efcol.get_value() as u32;
                display_preview_clone(&mut params).unwrap();
            };
        });
    }
    {
        let csv_params_clone = csv_params.clone();
        let display_preview_clone = display_preview.clone();
        e_cplx_secondary_col.connect_changed(move |efcol| {
            if let Ok(mut params) = csv_params_clone.try_borrow_mut() {
                params.column_y_sec = efcol.get_value() as u32;
                display_preview_clone(&mut params).unwrap();
            };
        });
    }
    {
        let csv_params_clone = csv_params.clone();
        let display_preview_clone = display_preview.clone();
        e_startline.connect_changed(move |efsl| {
            if let Ok(mut params) = csv_params_clone.try_borrow_mut() {
                params.skip_start = efsl.get_value() as u32;
                display_preview_clone(&mut params).unwrap();
            };
        });
    }

    {
        let csv_params_clone = csv_params.clone();
        dialog.connect_response(move |dia, _resp| {
            if _resp == gtk::ResponseType::Ok {
                csv_params_clone.borrow_mut().response_do_open = true;
            }
            else {
                csv_params_clone.borrow_mut().response_do_open = false;
            }
            dia.hide();
        });
    }

    dialog.set_transient_for(Some(mainwindow));
    let resp = dialog.run();

    if resp == gtk::ResponseType::Ok {

        let mut params = csv_params.borrow_mut();

        let mut out : Vec<imped_math::DataPiece> = vec![];

        let skip_start = params.skip_start as _;
        let idx_freq = params.column_x as _;
        let idx_cplx_1 = params.column_y_prim as _;
        let idx_cplx_2 = params.column_y_sec as _;

        params.file.seek(std::io::SeekFrom::Start(0)).unwrap();
        let mut reader = csv::Reader::from_reader(&params.file);
        let rec_iter = reader.records();
        let rec_iter = rec_iter.skip(skip_start);
        for result in rec_iter {
            let record = result?;

            let ue_none = |x| match x{Some(x)=>Ok(x), None=>Err(FileLoadingError)};

            let freq_val = ue_none(record.get(idx_freq))?.parse::<f64>()?;
            let re_val = ue_none(record.get(idx_cplx_1))?.parse::<f64>()?;
            let im_val = ue_none(record.get(idx_cplx_2))?.parse::<f64>()?;

            let omega = freq_val * match params.freq_type {
                FreqType::FreqHz => 2. * std::f64::consts::PI,
                FreqType::FreqKHz => 2000. * std::f64::consts::PI,
                FreqType::AngularHz => 1.,
                FreqType::AngularKHz => 1000.,
            };
            let re = re_val * match params.imp_type {
                ImpedanceType::RealImagOhm => 1.,
                ImpedanceType::RealImagKOhm => 1000.,
            };
            let im = im_val * match params.imp_type {
                ImpedanceType::RealImagOhm => 1.,
                ImpedanceType::RealImagKOhm => 1000.,
            };

            let dpoint = imped_math::DataPiece { omega, imp: Cplx{ re, im } };

            out.push(dpoint);
        }

        Ok(out)
    }
    else {
        Err(Box::new(FileLoadingError))
    }
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
/*
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

*/
