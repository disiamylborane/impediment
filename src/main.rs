#![warn(clippy::nursery)]

use std::{fmt::Display, str::FromStr};

use circuit::Circuit;
use data::DataPiece;
use eframe::{egui::{self, Color32, Vec2, vec2}, epi};

use crate::circuit::Element;

mod circuit;
mod data;
mod file;

pub type Cplx = num::complex::Complex<f64>;

#[derive(PartialEq, Debug)]
pub enum FitMethod { BOBYQA, TNC, SLSQP, LBfgsB }

#[derive(PartialEq, Debug, Clone, Copy)]
pub enum ElectricComponent { R, C, W, L, Q }

#[derive(PartialEq, Debug, Clone, Copy)]
pub enum ComponentInteraction { Change, Series, Parallel, Delete }

#[derive(Debug, Clone)]
pub struct ParameterDesc {
    pub vals : Vec<f64>,
    pub bounds : Vec<(f64, f64)>
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

#[derive(Debug, Clone)]
pub enum ParameterEditability { Plural, Single, Immutable }

pub struct Model {
    circuit: Circuit,
    name: String,
    parameters: Vec<ParameterEditability>,
}

#[derive(Clone, Copy, PartialEq)]
pub enum FreqOpenParam {Hz, Khz, Rads, Krads}
impl Display for FreqOpenParam {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use FreqOpenParam::*;
        write!(f, "{}", match self {Hz => "Hz", Khz => "kHz", Rads => "rad/s", Krads => "krad/s",})
    }
}
#[derive(Clone, Copy, PartialEq)]
pub enum ImpOpenParam {PlusOhm, MinusOhm, PlusKohm, MinusKohm}
impl Display for ImpOpenParam {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use ImpOpenParam::*;
        write!(f, "{}", match self {PlusOhm => "re + im, Ω", MinusOhm => "re - im, Ω", PlusKohm => "re + im, kΩ", MinusKohm => "re - im, kΩ",})
    }
}

pub struct TemplateApp {
    pub fit_method: FitMethod,
    pub component: ElectricComponent,
    pub interaction: ComponentInteraction,

    pub models: Vec<Model>,
    pub datasets: Vec<(Vec<DataPiece>, String)>,
    pub params: Vec< Vec< ParameterDesc > >,

    pub current_circ: usize,
    pub current_dataset: usize,

    pub editor: String,

    pub opening_data: Vec<(String, String)>,
    pub opening_mode: (FreqOpenParam, ImpOpenParam, usize, usize, usize, usize),

    pub copied_paramlist: Option<(usize, usize)>,
}

type AppData = (Vec<Model>, Vec<(Vec<DataPiece>, String)>, Vec< Vec< ParameterDesc >>);

impl TemplateApp {
    fn save_to_string(models: &[Model], datasets: &[(Vec<DataPiece>, String)], params: &[Vec< ParameterDesc >]) -> Result<String, std::fmt::Error> {
        let mut f = String::new();
        use std::fmt::Write;
        for m in models.iter() {
            writeln!(f, "Model {}", m.name)?;
            writeln!(f, "{}", m.circuit)?;
        }
        writeln!(f)?;
        for ds in datasets.iter() {
            writeln!(f, "Dataset {}", ds.1)?;
            writeln!(f, "{}", dataset_to_string(&ds.0))?;
        }
        writeln!(f)?;
        for (ip, ps_circ) in params.iter().enumerate() {
            writeln!(f, "Paramlist for Circuit {}:", ip)?;
            for ps in ps_circ {
                write!(f, "{:?} with bounds [", ps.vals)?;
                for b in &ps.bounds {
                    write!(f, "({:?}..{:?}), ", b.0, b.1)?;
                }
                writeln!(f, "]")?;
            }
        }

        Ok(f)
    }

    fn load_from_string(string: &str) -> Option<AppData> {
        let h_lines = string.split('\n').filter(|&x| !x.trim().is_empty()).collect::<Vec<_>>();
        let mut lines = &h_lines as &[&str];

        let mut models = vec![];

        loop {
            let linestart = lines.get(0)?;
            let marker = "Model ";
            if let Some(name) = linestart.strip_prefix(marker) {
                let circuit = Circuit::from_str( lines.get(1)? ).ok()?;
                let parameters = vec![ParameterEditability::Plural; circuit.paramlen()];
                models.push(Model{circuit, name: name.to_string(), parameters });
                lines = &lines[2..];
            }
            else {
                break;
            }
        }

        let mut datasets = vec![];
        loop {
            let linestart = lines.get(0)?;
            let marker = "Dataset ";
            let mut data = vec![];
            if let Some(name) = linestart.strip_prefix(marker) {
                lines = &lines[1..];
                while let Some(dp) = lines.get(0).and_then(|x| datapiece_from_string(x))  {
                    data.push(dp);
                    lines = &lines[1..];
                }
                datasets.push((data, name.to_string()));
            }
            else {
                break;
            }
        }

        let mut paramsets = vec![];
        let mut curr_circ = 0;
        while let Some(linestart) = lines.get(0) {
            let marker = "Paramlist for Circuit ";
            let mut pset = vec![];
            if let Some(circ) = linestart.strip_prefix(marker).and_then(|x| x.strip_suffix(':')).and_then(|x|x.parse::<usize>().ok()) {
                if curr_circ != circ {return None;}
                for _ in 0..datasets.len() {
                    lines = &lines[1..];
                    let (val_str,bounds_str) = lines.get(0)?.split_once(" with bounds ")?;
                    let vals = val_str.trim()
                                     .strip_prefix('[')
                                     .and_then(|x| x.strip_suffix(']'))?
                                     .split(',')
                                     .filter(|&x| !x.trim().is_empty())
                                     .map(|x| x.trim().parse::<f64>().ok())
                                     .collect::<Option<Vec<f64>>>()?;
                    let bounds = bounds_str.trim()
                                           .strip_prefix('[')
                                           .and_then(|x| x.strip_suffix(']'))?
                                           .split(',')
                                           .filter(|&x| !x.trim().is_empty())
                                           .map(|x| {
                                               let (smin, smax) = x.trim().strip_prefix('(')?.strip_suffix(')')?.split_once("..")?;
                                               let min = smin.parse::<f64>().ok()?;
                                               let max = smax.parse::<f64>().ok()?;
                                               Some((min,max))
                                           })
                                           .collect::<Option<Vec<(f64,f64)>>>()?;
                    if vals.len() != models.get(circ)?.circuit.paramlen() {
                        return None
                    }
                    if bounds.len() != models.get(circ)?.circuit.paramlen() {
                        return None
                    }
                    pset.push(ParameterDesc{vals, bounds});
                }
                paramsets.push(pset);
            }
            else {
                break;
            }
            lines = &lines[1..];
            curr_circ += 1;
        }

        if paramsets.len() != models.len() {
            return None
        }

        Some((models, datasets, paramsets))
    }
}

fn dataset_to_string(data: &[DataPiece]) -> String {
    let mut out = String::with_capacity(64);

    for d in data {
        if d.imp.im >= 0. {
            out += &format!("{}: {} + {}i\n", d.freq, d.imp.re, d.imp.im);
        } else {
            out += &format!("{}: {} - {}i\n", d.freq, d.imp.re, -d.imp.im);
        }
    }

    out
}

fn datapiece_from_string(l: &str) -> Option<DataPiece> {
    let cdataline = l.trim();
    let colon = cdataline.find(':')?;

    let freq: f64 = cdataline[0..colon].trim().parse().ok()?;
    let cimpline = &cdataline[(colon+1)..];

    let sep = cimpline.find('+').or_else(|| cimpline.find('-'))?;
    let re: f64 = cimpline[0..sep].trim().parse().ok()?;

    let im_str = cimpline[(sep+1)..].trim();
    if !im_str.ends_with('i') {return None;}

    let im: f64 = im_str[..(im_str.len()-1)].parse().ok()?;
    let im = if cimpline.chars().nth(sep)?=='+' {im} else {-im};
    Some(DataPiece{ freq, imp: Cplx::new(re, im) })
}

fn dataset_from_string(data: &str) -> Option<Vec<DataPiece>> {
    data.split('\n')
        .filter(|&x| !x.trim().is_empty())
        .map(datapiece_from_string)
        .collect()
}

/// Get the coordinates of drawing block given the coordinates of a point in a `gtk::DrawingArea`
fn block_by_coords(circuit: &Circuit, widsize: Vec2, clickpos: Vec2, blocksize: f32) 
    -> Option<(u16, u16)>
{
    // The circuit is located at the center of the DrawingArea

    // Get the circuit size
    let (i_sx, i_sy) = circuit.painted_size();
    let (sx,sy) = (i_sx as f32 * blocksize, i_sy as f32 * blocksize);

    // Recalc (cursor vs canvas) => (cursor vs circuit)
    let (xcirc, ycirc) = (clickpos.x - (widsize.x-sx)/2., clickpos.y-(widsize.y-sy)/2.);
    if xcirc < 0. || ycirc < 0. {return None;}

    let (x,y) = (xcirc / blocksize, ycirc / blocksize);

    Some((x as u16, y as u16))
}

impl Default for TemplateApp {
    fn default() -> Self {
        let mut out = Self {
            fit_method: FitMethod::BOBYQA,
            component: ElectricComponent::R,
            interaction: ComponentInteraction::Change,
            models: vec![
                Model{
                    circuit: Circuit::Series(vec![Circuit::Element(Element::Resistor), Circuit::Element(Element::Capacitor)]), 
                    name: "model1".to_owned(), 
                    parameters: vec![ParameterEditability::Plural, ParameterEditability::Plural]},
                Model{
                    circuit: Circuit::Parallel(vec![Circuit::Element(Element::Resistor), Circuit::Element(Element::Capacitor)]),
                    name: "model2".to_owned(),
                    parameters: vec![ParameterEditability::Plural, ParameterEditability::Plural]
                },
            ],
            datasets: vec![
                (vec![
                    data::DataPiece {freq: 100000., imp: Cplx::new(2., 2.)},
                    data::DataPiece {freq: 80000., imp: Cplx::new(3., 3.)},
                    data::DataPiece {freq: 60000., imp: Cplx::new(3., 4.)},
                ], "data1".to_owned()),
                (vec![], "data2".to_owned()),
            ],
            params: vec![
                vec![
                    ParameterDesc { vals : vec![100.,0.01], bounds : vec![(1., 10000.),(1e-5, 1e1)] },
                    ParameterDesc { vals : vec![100.,0.01], bounds : vec![(1., 10000.),(1e-5, 1e1)] },
                ],
                vec![
                    ParameterDesc { vals : vec![100.,0.01], bounds : vec![(1., 10000.),(1e-5, 1e1)] },
                    ParameterDesc { vals : vec![100.,0.01], bounds : vec![(1., 10000.),(1e-5, 1e1)] },
                ],
            ],
            current_circ: 0,
            current_dataset: 0,
            editor: String::with_capacity(64),
            opening_data: vec![],
            opening_mode: (FreqOpenParam::Hz, ImpOpenParam::PlusOhm, 0, 1, 2, 0),
            copied_paramlist: None
        };
        out.editor = dataset_to_string(&out.datasets[out.current_dataset].0);
        out
    }
}

impl epi::App for TemplateApp {
    fn name(&self) -> &str {
        "egui template"
    }

    /// Called each time the UI needs repainting, which may be many times per second.
    /// Put your widgets into a `SidePanel`, `TopPanel`, `CentralPanel`, `Window` or `Area`.
    fn update(&mut self, ctx: &egui::CtxRef, _frame: &mut epi::Frame<'_>) {
        let Self {
            fit_method, 
            component, 
            interaction,
            models,
            current_circ,
            current_dataset,
            datasets,
            params,
            editor,
            opening_data,
            opening_mode,
            copied_paramlist,
        ..} = self;

        if !opening_data.is_empty() {
            egui::Window::new("Data").show(ctx, |ui|
                ui.vertical(|ui| {
                    ui.vertical(|ui| {
                        ui.label("Columns");
                        ui.separator();
                        ui.horizontal(|ui| {
                            ui.label("Freq");
                            if ui.small_button("<").clicked() && opening_mode.2 > 0 {opening_mode.2 -= 1}
                            ui.label(&opening_mode.2.to_string());
                            if ui.small_button(">").clicked() {opening_mode.2 += 1}
                        });
                        ui.horizontal(|ui| {
                            ui.selectable_value(&mut opening_mode.0, FreqOpenParam::Hz, &format!("{}", FreqOpenParam::Hz));
                            ui.selectable_value(&mut opening_mode.0, FreqOpenParam::Khz, &format!("{}", FreqOpenParam::Khz));
                            ui.selectable_value(&mut opening_mode.0, FreqOpenParam::Rads, &format!("{}", FreqOpenParam::Rads));
                            ui.selectable_value(&mut opening_mode.0, FreqOpenParam::Krads, &format!("{}", FreqOpenParam::Krads));
                        });
                        ui.horizontal(|ui| {
                            ui.label("Re Z");
                            if ui.small_button("<").clicked() && opening_mode.3 > 0 {opening_mode.3 -= 1}
                            ui.label(&opening_mode.3.to_string());
                            if ui.small_button(">").clicked() {opening_mode.3 += 1}
                        });
                        ui.horizontal(|ui| {
                            ui.label("Im Z");
                            if ui.small_button("<").clicked() && opening_mode.4 > 0 {opening_mode.4 -= 1}
                            ui.label(&opening_mode.4.to_string());
                            if ui.small_button(">").clicked() {opening_mode.4 += 1}
                        });
                        ui.vertical(|ui| {
                            ui.selectable_value(&mut opening_mode.1, ImpOpenParam::PlusOhm, &format!("{}", ImpOpenParam::PlusOhm));
                            ui.selectable_value(&mut opening_mode.1, ImpOpenParam::MinusOhm, &format!("{}", ImpOpenParam::MinusOhm));
                            ui.selectable_value(&mut opening_mode.1, ImpOpenParam::PlusKohm, &format!("{}", ImpOpenParam::PlusKohm));
                            ui.selectable_value(&mut opening_mode.1, ImpOpenParam::MinusKohm, &format!("{}", ImpOpenParam::MinusKohm));
                        });
                    });
                    ui.vertical(|ui| {
                        let new_dset: Option<Vec<(Vec<DataPiece>, String)>>= opening_data.iter().map(|od| {
                            Some((file::csv_to_impediment(&od.0, *opening_mode)?, od.1.clone()))
                        }).collect();

                        if let Some(new_dset) = new_dset {
                          if ui.button("Load").clicked() {
                              let mut newds: Vec<(Vec<DataPiece>, String)> = new_dset.into_iter().collect();
                              for paramset in params.iter_mut() {
                                  for _ in 0..newds.len() {
                                    paramset.push(paramset[0].clone());
                                  }
                              }
                              datasets.append(&mut newds);
                              *opening_data = vec![];
                              return;
                          }
                          if ui.button("Cancel").clicked() {
                              *opening_data = vec![];
                              return;
                          };
                          ui.horizontal(|ui| {
                              if ui.small_button("<").clicked() && opening_mode.5 > 0 {
                                  opening_mode.5 -= 1;
                              }
                              ui.label(&new_dset[opening_mode.5].1);
                              if ui.small_button(">").clicked() && opening_mode.5+1 < opening_data.len() {
                                  opening_mode.5 += 1;
                              }
                          });
                          
                          let dstring = dataset_to_string(&new_dset[opening_mode.5].0);

                          egui::ScrollArea::auto_sized().show(ui, |ui| {
                              ui.label(&dstring);
                          });
                        }
                        else if ui.button("Cancel").clicked(){
                            *opening_data = vec![];
                        }
                    });
                })
            );
        }

        egui::SidePanel::left("models").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui.button("Load").clicked() {
                    if let Ok(Some(filename)) = native_dialog::FileDialog::new().show_open_single_file() {
                        if let Ok(file) = std::fs::File::open(filename.clone()) {
                            use std::io::BufRead;
                            let buf = std::io::BufReader::new(&file);
                            if let Some(Ok(line)) = buf.lines().next() {
                                if let Ok(circ) = Circuit::from_str(&line) {
                                    params.push(
                                        vec![circ.generate_new_params(); datasets.len()]
                                    );
                                    models.push(Model {
                                        name: filename.to_string_lossy().as_ref().to_owned(),
                                        parameters: vec![ParameterEditability::Plural; circ.paramlen()],
                                        circuit: circ,
                                    });
                                }
                            }
                        }
                    }
                }

                if ui.button("Save").clicked() {
                    if let Ok(Some(filename)) = native_dialog::FileDialog::new().show_save_single_file() {
                        if let Ok(mut file) = std::fs::File::create(filename) {
                            if let Some(mdl) = models.get(*current_circ) {
                                use std::io::Write;
                                writeln!(&mut file, "{}", mdl.circuit).unwrap();
                            }
                        }
                    }
                }

                if ui.button("+").clicked() {
                    let newmodel = Model {
                        circuit: Circuit::Element(Element::Resistor),
                        name: "new model".to_string(),
                        parameters: vec![ParameterEditability::Plural],
                    };
                    params.push(
                        vec![ParameterDesc{
                            vals: vec![10.],
                            bounds: vec![(0.1, 100000.0)]
                        }; datasets.len()]
                    );
                    models.push(newmodel);
                };
                if ui.button("-").clicked() && models.len() > 1 {
                    models.remove(*current_circ);
                    params.remove(*current_circ);
                    if *current_circ != 0 {*current_circ -= 1;}
                };
            });

            ui.separator();

            ui.horizontal(|ui| {
                ui.selectable_value(component, ElectricComponent::R, "R");
                ui.selectable_value(component, ElectricComponent::C, "C");
                ui.selectable_value(component, ElectricComponent::L, "L");
                ui.selectable_value(component, ElectricComponent::W, "W");
                ui.selectable_value(component, ElectricComponent::Q, "Q");

                ui.selectable_value(interaction, ComponentInteraction::Change, ":");
                ui.selectable_value(interaction, ComponentInteraction::Series, "--");
                ui.selectable_value(interaction, ComponentInteraction::Parallel, "=");
                ui.selectable_value(interaction, ComponentInteraction::Delete, "x");
            });

            let blocksize = 10.;
            let size = egui::vec2(200.0, 100.0);
            let (response, painter) = ui.allocate_painter(size, egui::Sense::click());
            painter.rect_filled(response.rect, 0., Color32::from_rgb(80, 80, 80));
            if let Some(c) = models.get_mut(*current_circ) {
                let widsize = size;
                let size = c.circuit.painted_size();
                let size = egui::vec2(size.0 as _, size.1 as _)*blocksize;
                c.circuit.paint(response.rect.min + (widsize-size)/2., blocksize, &painter, 0);

                if response.clicked() {
                    if let Some(pos) = response.interact_pointer_pos() {
                        let user_element = match component {
                            ElectricComponent::R => (Element::Resistor),
                            ElectricComponent::C => (Element::Capacitor),
                            ElectricComponent::W => (Element::Warburg),
                            ElectricComponent::L => (Element::Inductor),
                            ElectricComponent::Q => (Element::Cpe),
                        };

                        let canvas_pos = pos - response.rect.min;
                        let block = block_by_coords(&c.circuit, widsize, canvas_pos, blocksize);
                        if let Some(block) = block {
                            match interaction {
                                ComponentInteraction::Change => {
                                    c.circuit.replace(block, user_element, params[*current_circ].iter_mut(), &mut c.parameters)
                                },
                                ComponentInteraction::Series => {
                                    c.circuit._add_series(block, user_element, params[*current_circ].iter_mut(), 0, &mut c.parameters)
                                },
                                ComponentInteraction::Parallel => {
                                    c.circuit._add_parallel(block, user_element, params[*current_circ].iter_mut(), 0, &mut c.parameters)
                                }
                                ComponentInteraction::Delete => {
                                    c.circuit._remove(block, params[*current_circ].iter_mut(), 0, &mut c.parameters)
                                }
                            }
                        }
                    }
                }
            }

            ui.separator();

            egui::ScrollArea::auto_sized().show(ui, |ui| {
                for (i, c) in models.iter_mut().enumerate() {
                    let tedit = egui::TextEdit::singleline(&mut c.name);
                    let tedit = if *current_circ==i {tedit.text_color(Color32::RED)} else {tedit};
                    if ui.add(tedit).clicked() {
                        *current_circ=i;
                    };
                }
            });
        });
        egui::SidePanel::right("rdata").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui.button("Load").clicked() {
                    if let Ok(files) = native_dialog::FileDialog::new()
                    .show_open_multiple_file() {
                        let vcd = files.into_iter().map(|ref f| {
                            use std::io::BufRead;
                            if let Ok(file) = std::fs::File::open(f) {
                                let buf = std::io::BufReader::new(file);
                                let lines = buf.lines().fold(String::with_capacity(256), |mut x,y| {x.extend(y); x.push('\n'); x});
                                let fname = f.file_name().map_or(String::new(), |x| x.to_string_lossy().to_string());
                                Ok((lines, fname))
                            } else {Err(f.to_string_lossy().to_string())}
                        }).collect::<Result<Vec<(String, String)>, String>>();
                        match vcd {
                            Ok(v) => {
                              *opening_data = v;
                            }
                            Err(e) => println!("Error: {}", e),
                        }
                    }
                }

                // ui.button("Save");

                if ui.button("+").clicked() {
                    datasets.push((vec![], "new dataset".into()));
                    for (p,m) in params.iter_mut().zip(models.iter_mut()) {
                        p.push(m.circuit.generate_new_params())
                    }
                }
                if ui.button("-").clicked() && datasets.len() > 1 {
                    datasets.remove(*current_dataset);
                    for p in params.iter_mut() {
                        p.remove(*current_dataset);
                    }
                    if *current_dataset > 0 {*current_dataset -= 1;}
                }
            });

            ui.separator();

            egui::ScrollArea::auto_sized().show(ui, |ui| {
                for (i, d) in datasets.iter_mut().enumerate() {
                    let tedit = egui::TextEdit::singleline(&mut d.1);
                    let tedit = if *current_dataset==i {tedit.text_color(Color32::RED)} else {tedit};
                    if ui.add(tedit).clicked() {
                        *current_dataset=i;
                        *editor = dataset_to_string(&d.0);
                    };
                }
            });
        });

        egui::SidePanel::left("datalist").show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                if ui.button("Save project").clicked() {
                    if let Ok(Some(filename)) = native_dialog::FileDialog::new().show_save_single_file() {
                        if let Ok(mut file) = std::fs::File::create(filename) {
                            if let Ok(string) = Self::save_to_string(models, datasets, params) {
                                use std::io::Write;
                                writeln!(&mut file, "{}", string).unwrap();
                            } else {
                                println!("Error writing to the file");
                            }
                        }
                    }
                }

                if ui.button("Load project").clicked() {
                    if let Ok(Some(filename)) = native_dialog::FileDialog::new().show_open_single_file() {
                        if let Ok(sfile) = std::fs::read_to_string(filename) {
                            if let Some((m, d, p)) = Self::load_from_string(&sfile) {
                                *models = m;
                                *datasets = d;
                                *params = p;
                                *current_circ = 0;
                                *current_dataset = 0;
                                *opening_data = vec![];
                                *copied_paramlist = None;
                            }
                        }
                    }                        
                }
            });

            ui.separator();

            ui.horizontal_wrapped(|ui| {
                if ui.button("Fit").clicked() {
                    let paramlist = &mut params[*current_circ][*current_dataset];
                    let mut opt = nlopt::Nlopt::new(
                        match fit_method {
                            FitMethod::BOBYQA => nlopt::Algorithm::Bobyqa,
                            FitMethod::TNC => nlopt::Algorithm::TNewton,
                            FitMethod::SLSQP => nlopt::Algorithm::Slsqp,
                            FitMethod::LBfgsB => nlopt::Algorithm::Lbfgs,
                        }, 
                        paramlist.vals.len(), 
                        |params: &[f64], mut gradient_out: Option<&mut [f64]>, _: &mut ()| {
                            let circ = &models[*current_circ].circuit;
                            if let Some(gout) = &mut gradient_out {
                                for i in 0..gout.len() {
                                    gout[i] = 0.0;
                                }
                            };
                            let exps = &datasets[*current_dataset].0;
                            
                            let mut loss = 0.0_f64;
                            for point in exps {
                                let model_imp = circ.impedance(std::f64::consts::TAU*point.freq, params);
                                let diff = model_imp - point.imp;
                                loss += diff.norm_sqr() / point.imp.norm_sqr();

                                if let Some(gout) = &mut gradient_out {
                                    for i in 0..gout.len() {
                                        let dmdx = circ._d_impedance(std::f64::consts::TAU*point.freq, params, i);
                                        // (a*)b + a(b*)  = [(a*)b] + [(a*)b]* = 2*re[(a*)b]
                                        let ml = (point.imp-model_imp)*dmdx.conj().re * 2.0;
                                        gout[i] += -1.0 / point.imp.norm_sqr() * ml.re;
                                    }
                                };
                            }
                        
                            loss
                        },
                        nlopt::Target::Minimize,
                        (),
                    );

                    opt.set_lower_bounds(&paramlist.bounds.iter().map(|x| x.0).collect::<Vec<f64>>()).unwrap();
                    opt.set_upper_bounds(&paramlist.bounds.iter().map(|x| x.1).collect::<Vec<f64>>()).unwrap();
                    opt.set_maxeval((-1_i32) as u32).unwrap();
                    opt.set_maxtime(10.0).unwrap();

                    opt.set_xtol_rel(1e-10).unwrap();

                    let optresult = opt.optimize(&mut paramlist.vals);
                    println!("{:?}", optresult);
                }

                egui::ComboBox::from_id_source("Select_method")
                    .selected_text(&format!("{:?}", fit_method))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(fit_method, FitMethod::BOBYQA, "Bobyqa");
                        ui.selectable_value(fit_method, FitMethod::LBfgsB, "L-Bfgs-B");
                        ui.selectable_value(fit_method, FitMethod::TNC, "TNC");
                        ui.selectable_value(fit_method, FitMethod::SLSQP, "SLSQP");
                    }
                );
            });

            ui.horizontal_wrapped(|ui| {
                if ui.button("Copy").clicked() {
                    *copied_paramlist = Some((*current_circ, *current_dataset));
                }
                if ui.button("Paste").clicked() {
                    if let Some(cp) = copied_paramlist {
                        if cp.0 == *current_circ {
                            if let Some(dpl) = params.get(cp.0).and_then(|x| x.get(cp.1)) {
                                params[*current_circ][*current_dataset] = dpl.to_owned();
                            }
                        }
                    }
                }
            });

            egui::ScrollArea::auto_sized().show(ui, |ui| {
                
                ui.vertical(|ui| {
                    use egui::Widget;

                    let mut empty_ed = vec![];

                    let param_names = models.get_mut(*current_circ).map_or(vec![], |x| x.circuit.param_names());
                    let paramlist = &mut params[*current_circ][*current_dataset];
                    let param_ed = models.get_mut(*current_circ).map_or(&mut empty_ed, |x| &mut x.parameters);

                    for (((name, val), (min, max)), ed) in 
                            param_names.iter()
                            .zip(&mut paramlist.vals)
                            .zip(&mut paramlist.bounds)
                            .zip(param_ed)
                    {
                        ui.horizontal(|ui| {
                            if ui.selectable_label(true, match ed {
                                ParameterEditability::Plural => "✔",
                                ParameterEditability::Single => "1",
                                ParameterEditability::Immutable => "x",
                            }).clicked() {
                                *ed = match ed {
                                    ParameterEditability::Plural => ParameterEditability::Single,
                                    ParameterEditability::Single => ParameterEditability::Immutable,
                                    ParameterEditability::Immutable => ParameterEditability::Plural,
                                };
                            }

                            ui.label(name);

                            let mut smin = min.to_string();
                            let mut smax = max.to_string();
                            let mut sval = val.to_string();

                            if egui::TextEdit::singleline(&mut smin).desired_width(50.).ui(ui).changed() {
                                if let Ok(new) = smin.parse::<f64>() { *min = new; }
                            }
                            if egui::TextEdit::singleline(&mut smax).desired_width(50.).ui(ui).changed() {
                                if let Ok(new) = smax.parse::<f64>() { *max = new; }
                            }
                            if egui::TextEdit::singleline(&mut sval).desired_width(80.).ui(ui).changed() {
                                if let Ok(new) = sval.parse::<f64>() { *val = new; }
                            }

                            if ui.small_button("<").clicked() {
                                if ctx.input().modifiers.alt  { *val /= 1.01 }
                                else if ctx.input().modifiers.command { *val /= 2.0 }
                                else { *val /= 1.1 }
                            }
                            if ui.small_button(">").clicked() {
                                if ctx.input().modifiers.alt  { *val *= 1.01 }
                                else if ctx.input().modifiers.command { *val *= 2.0 }
                                else { *val *= 1.1 }
                            }
                        });
                    }
                });
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical_centered_justified(|ui| {
                use eframe::egui::Widget;

                let dataset = &mut match datasets.get_mut(*current_dataset) {
                    Some(s) => s,
                    None => return,
                }.0;

                let plt = egui::plot::Plot::new("plot1")
                    .points(egui::plot::Points::new(
                        egui::plot::Values::from_values( dataset.iter().map(|d| egui::plot::Value::new(d.imp.re, -d.imp.im)).collect::<Vec<_>>() )
                        )
                        .shape(egui::plot::MarkerShape::Circle)
                        .radius(4.)
                    );
                
                let plt = if let Some(m) = models.get(*current_circ) {
                    let values = 
                    dataset.iter().map(|d| {
                        let imp = m.circuit.impedance(std::f64::consts::TAU*d.freq, &params[*current_circ][*current_dataset].vals);
                        egui::plot::Value::new(
                            imp.re, -imp.im
                        )
                    }).collect();
                    plt.line(egui::plot::Line::new(egui::plot::Values::from_values(values)).stroke((1.0, Color32::WHITE)))
                } else {plt};

                ui.add(plt.view_aspect(1.0));

                egui::ScrollArea::auto_sized().show(ui, |ui| {
                    let edit = egui::TextEdit::multiline(editor)
                            .desired_rows(20)
                            .ui(ui);
                    if edit.lost_focus() {
                        if let Some(v) = dataset_from_string(editor) {
                            *dataset = v;
                        } else {println!("Wrong data")}
                    }
                });
            });
        });
    }

    fn warm_up_enabled(&self) -> bool {
        false
    }

    fn on_exit(&mut self) {}

    fn auto_save_interval(&self) -> std::time::Duration {
        std::time::Duration::from_secs(30)
    }

    fn max_size_points(&self) -> egui::Vec2 {
        egui::Vec2::new(1024.0, 2048.0)
    }

    fn clear_color(&self) -> egui::Rgba {
        egui::Color32::from_rgba_unmultiplied(12, 12, 12, 180).into()
    }
}

fn main() {
    let app = TemplateApp::default();
    let native_options = eframe::NativeOptions{initial_window_size: Some(vec2(1100., 600.)), ..Default::default()};
    eframe::run_native(Box::new(app), native_options);
}

