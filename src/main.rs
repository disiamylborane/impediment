#![warn(clippy::nursery)]
#![warn(clippy::pedantic)]
#![allow(clippy::blocks_in_if_conditions)]

use float_pretty_print::PrettyPrintFloat;

use std::{fmt::Display, str::FromStr};

use circuit::Circuit;
use eframe::{egui::{self, Color32, Vec2, vec2, plot::Value}, epi};


#[derive(Debug, Copy, Clone)]
pub struct DataPiece {
    pub freq: f64,
    pub imp: crate::Cplx,
}

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


fn try_into_numbers(x: &[Vec<StringParameterDesc>]) -> Option<Vec<Vec<ParameterDesc>>> {
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

#[derive(Debug, Clone)]
pub enum ParameterEditability { Plural, Single, Immutable }

#[derive(Debug, Clone)]
pub struct Model {
    circuit: Circuit,
    name: String,
    parameters: Vec<ParameterEditability>,
    lock: bool,
}

#[derive(Clone, Copy, PartialEq)]
pub enum FreqOpenParam {Hz, Khz, Rads, Krads}
impl Display for FreqOpenParam {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use FreqOpenParam::{Hz, Khz, Krads, Rads};
        write!(f, "{}", match self {Hz => "Hz", Khz => "kHz", Rads => "rad/s", Krads => "krad/s",})
    }
}
#[derive(Clone, Copy, PartialEq)]
pub enum ImpOpenParam {PlusOhm, MinusOhm, PlusKohm, MinusKohm}
impl Display for ImpOpenParam {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use ImpOpenParam::{MinusKohm, MinusOhm, PlusKohm, PlusOhm};
        write!(f, "{}", match self {PlusOhm => "re + im, Ω", MinusOhm => "re - im, Ω", PlusKohm => "re + im, kΩ", MinusKohm => "re - im, kΩ",})
    }
}

#[derive(Clone, Copy, PartialEq)]
pub enum PlotType {Nyquist, BodePhase, BodeAmp}

pub struct TemplateApp {
    pub fit_method: FitMethod,
    pub component: ElectricComponent,
    pub interaction: ComponentInteraction,

    pub models: Vec<Model>,
    pub datasets: Vec<(Vec<DataPiece>, String)>,
    pub str_params: Vec< Vec< StringParameterDesc > >,

    pub current_circ: usize,
    pub current_dataset: usize,

    pub editor: String,

    pub opening_data: Vec<(String, String)>,
    pub opening_mode: (FreqOpenParam, ImpOpenParam, usize, usize, usize, usize),

    pub copied_paramlist: Option<(usize, usize)>,

    pub plot_type: PlotType,

    pub text_status: String,
}

type AppData = (Vec<Model>, Vec<(Vec<DataPiece>, String)>, Vec< Vec< ParameterDesc >>);

impl TemplateApp {
    fn save_to_string(models: &[Model], datasets: &[(Vec<DataPiece>, String)], params: &[Vec< ParameterDesc >]) -> Result<String, std::fmt::Error> {
        use std::fmt::Write;

        let mut f = String::new();
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
                models.push(Model{circuit, name: name.to_string(), parameters, lock: true });
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
                                               let (str_min, str_max) = x.trim().strip_prefix('(')?.strip_suffix(')')?.split_once("..")?;
                                               let min = str_min.parse::<f64>().ok()?;
                                               let max = str_max.parse::<f64>().ok()?;
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

/// Get the coordinates of drawing block given the coordinates of a point
fn block_by_coords(circuit: &Circuit, widsize: Vec2, clickpos: Vec2, blocksize: f32) 
    -> Option<(u16, u16)>
{
    // The circuit is located at the center of the canvas

    // Get the circuit size
    let (idx_s_x, idx_s_y) = circuit.painted_size();
    let (sx,sy) = (f32::from(idx_s_x) * blocksize, f32::from(idx_s_y) * blocksize);

    // Recalc (cursor vs canvas) => (cursor vs circuit)
    let (x_circ, y_circ) = (clickpos.x - (widsize.x-sx)/2., clickpos.y-(widsize.y-sy)/2.);
    if x_circ < 0. || y_circ < 0. {return None;}

    let (x,y) = (x_circ / blocksize, y_circ / blocksize);

    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    Some((x as u16, y as u16))
}

impl Default for TemplateApp {
    fn default() -> Self {
        let circ = Circuit::Series(vec![
            Circuit::Parallel(vec![
                Circuit::Element(Element::Resistor), 
                Circuit::Element(Element::Capacitor),
            ]),
            Circuit::Element(Element::Resistor),
        ]);
        let params = ParameterDesc { vals : vec![100.,5e-6,100.], bounds : vec![(1., 10000.),(1e-6, 1e-1),(1., 10000.0)] };

        let data = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5].iter().map(|&freq|
            DataPiece {freq, imp: circ.impedance(std::f64::consts::TAU*freq, &params.vals)}
        ).collect();

        let mut out = Self {
            fit_method: FitMethod::BOBYQA,
            component: ElectricComponent::R,
            interaction: ComponentInteraction::Change,
            models: vec![
                Model{
                    circuit: circ, 
                    name: "model1".to_owned(), 
                    parameters: vec![ParameterEditability::Plural, ParameterEditability::Plural, ParameterEditability::Plural],
                    lock: false},
            ],
            datasets: vec![
                (data, "data1".to_owned()),
            ],
            str_params: vec![
                vec![
                    StringParameterDesc {
                        vals : vec!["110.0".into(), "4.50e-6".into(), "102.0".into()],
                        bounds : vec![("1.0".into(), "10000".into()),("1e-6".into(), "0.1".into()),("1.0".into(), "10000".into())]
                    },
                ],
            ],
            current_circ: 0,
            current_dataset: 0,
            editor: String::with_capacity(64),
            opening_data: vec![],
            opening_mode: (FreqOpenParam::Hz, ImpOpenParam::PlusOhm, 0, 1, 2, 0),
            copied_paramlist: None,
            plot_type: PlotType::Nyquist,
            text_status: String::new(),
        };
        out.editor = dataset_to_string(&out.datasets[out.current_dataset].0);
        out
    }
}

pub fn geomspace(first: f64, last: f64, count: u32) -> impl Iterator<Item=f64>
{
    let (lf, ll) = (first.ln(), last.ln());
    let delta = (ll - lf) / f64::from(count-1);
    (0..count).map(move |i| (f64::from(i).mul_add(delta, lf)).exp())
}


fn display_import_window(
        ctx: &egui::Context, 
        opening_mode: &mut (FreqOpenParam, ImpOpenParam, usize, usize, usize, usize),
        opening_data: &mut Vec<(String, String)>,
        datasets: &mut Vec<(Vec<DataPiece>, String)>,
        str_params: &mut [Vec< StringParameterDesc >],
) {
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
                ui.horizontal(|ui| {
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
                    let new_dset = ui.horizontal(|ui| {
                        if ui.button("Load").clicked() {
                            let mut newds: Vec<(Vec<DataPiece>, String)> = new_dset.into_iter().collect();
                            for paramset in str_params.iter_mut() {
                                for _ in 0..newds.len() {
                                  paramset.push(paramset[0].clone());
                                }
                            }
                            datasets.append(&mut newds);
                            *opening_data = vec![];
                            opening_mode.5 = 0;  // FIXME: revamp the logics
                            return None;
                        }
                        if ui.button("Cancel").clicked() {
                            *opening_data = vec![];
                            opening_mode.5 = 0;  // FIXME: revamp the logics
                            return None;
                        };
                        Some(new_dset)
                    }).inner;
                    if let Some(new_dset) = new_dset {
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

                        egui::ScrollArea::vertical().show(ui, |ui| {
                            ui.label(&dstring);
                        });
                    }
                }
                else if ui.button("Cancel").clicked(){
                    *opening_data = vec![];
                }
            });
        })
    );
}


impl epi::App for TemplateApp {
    fn name(&self) -> &str {
        "Impediment"
    }

    #[allow(clippy::too_many_lines)]
    fn update(&mut self, ctx: &egui::Context, _frame: &epi::Frame) {
        ctx.set_visuals(egui::Visuals::dark());

        let Self {
            fit_method, 
            component, 
            interaction,
            models,
            current_circ,
            current_dataset,
            datasets,
            str_params,
            editor,
            opening_data,
            opening_mode,
            copied_paramlist,
            plot_type,
            text_status,
        ..} = self;

        if !opening_data.is_empty() {
            display_import_window(ctx, opening_mode, opening_data, datasets, str_params);
        }

        egui::SidePanel::left("models").show(ctx, |ui| {
            ui.vertical_centered_justified( |ui| {
                ui.horizontal(|ui| {
                    if ui.button("+").on_hover_ui(|ui| {ui.label("Add a new circuit");}).clicked() {
                        let newmodel = Model {
                            circuit: Circuit::Element(Element::Resistor),
                            name: "new model".to_string(),
                            parameters: vec![ParameterEditability::Plural],
                            lock: false,
                        };
                        str_params.push(
                            vec![StringParameterDesc{
                                vals: vec!["10.".into()],
                                bounds: vec![("0.1".into(), "100000".into())]
                            }; datasets.len()]
                        );
                        models.push(newmodel);
                    };
                    if ui.button("D").on_hover_ui(|ui| {ui.label("Duplicate circuit");}).clicked() {
                        str_params.push(str_params[*current_circ].clone());
                        models.push(models[*current_circ].clone());
                    };
                    if ui.button("-").on_hover_ui(|ui| {ui.label("Remove current circuit");}).clicked() && models.len() > 1 {
                        models.remove(*current_circ);
                        str_params.remove(*current_circ);
                        if *current_circ != 0 {*current_circ -= 1;}
                    };
                    if ui.button("Up").clicked() && *current_circ > 0 {
                        (models as &mut[_]).swap(*current_circ, *current_circ - 1);
                        (str_params as &mut[_]).swap(*current_circ, *current_circ - 1);
                        *current_circ -= 1;
                    }
                    if ui.button("Down").clicked() && *current_circ < models.len() - 1 {
                        (models as &mut[_]).swap(*current_circ, *current_circ + 1);
                        (str_params as &mut[_]).swap(*current_circ, *current_circ + 1);
                        *current_circ += 1;
                    }
                });
    
                ui.separator();
    
                ui.horizontal(|ui| {
                    ui.selectable_value(component, ElectricComponent::R, "R").on_hover_ui(|ui| {ui.label("Resistor");});
                    ui.selectable_value(component, ElectricComponent::C, "C").on_hover_ui(|ui| {ui.label("Capacitor");});
                    ui.selectable_value(component, ElectricComponent::L, "L").on_hover_ui(|ui| {ui.label("Inductor");});
                    ui.selectable_value(component, ElectricComponent::W, "W").on_hover_ui(|ui| {ui.label("Warburg");});
                    ui.selectable_value(component, ElectricComponent::Q, "Q").on_hover_ui(|ui| {ui.label("Constant phase");});
    
                    ui.separator();
    
                    ui.selectable_value(interaction, ComponentInteraction::Change, ":").on_hover_ui(|ui| {ui.label("Replace");});
                    ui.selectable_value(interaction, ComponentInteraction::Series, "--").on_hover_ui(|ui| {ui.label("Add series");});
                    ui.selectable_value(interaction, ComponentInteraction::Parallel, "=").on_hover_ui(|ui| {ui.label("Add parallel");});
                    ui.selectable_value(interaction, ComponentInteraction::Delete, "x").on_hover_ui(|ui| {ui.label("Remove clicked");});
                });
    
                let blocksize = 10.;
                let size = egui::vec2(200.0, 100.0);
                let (response, painter) = ui.allocate_painter(size, egui::Sense::click());
                painter.rect_filled(response.rect, 0., Color32::from_rgb(80, 80, 80));
                if let Some(c) = models.get_mut(*current_circ) {
                    let widsize = size;
                    let size = c.circuit.painted_size();
                    let size = egui::vec2(size.0.into(), size.1.into())*blocksize;
                    c.circuit.paint(response.rect.min + (widsize-size)/2., blocksize, &painter, 0, if c.lock {Color32::WHITE} else {Color32::LIGHT_RED});
    
                    if response.clicked() && !c.lock {
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
                                        c.circuit.replace(block, user_element, str_params[*current_circ].iter_mut(), &mut c.parameters);
                                    },
                                    ComponentInteraction::Series => {
                                        c.circuit._add_series(block, user_element, str_params[*current_circ].iter_mut(), 0, &mut c.parameters);
                                    },
                                    ComponentInteraction::Parallel => {
                                        c.circuit._add_parallel(block, user_element, str_params[*current_circ].iter_mut(), 0, &mut c.parameters);
                                    }
                                    ComponentInteraction::Delete => {
                                        c.circuit._remove(block, str_params[*current_circ].iter_mut(), 0, &mut c.parameters);
                                    }
                                }
                            }
                        }
                    }
                }
    
                if models[*current_circ].lock {
                    if ui.small_button("Unlock").clicked() {
                        models[*current_circ].lock = false;
                    }
                } else {
                    ui.add_enabled(false, egui::Button::new("Unlock").small());
                }
    
                ui.separator();
    
                egui::ScrollArea::vertical().show(ui, |ui| {
                    for (i, c) in models.iter_mut().enumerate() {
                        let tedit = egui::TextEdit::singleline(&mut c.name);
                        let tedit = if *current_circ==i {tedit.text_color(Color32::RED)} else {tedit};
                        if ui.add(tedit).clicked() {
                            *current_circ=i;
                        };
                    }
                });
            });
        });
        egui::SidePanel::right("rdata").show(ctx, |ui| {
            ui.vertical_centered( |ui| {
                ui.horizontal(|ui| {
                    if ui.button("Import").on_hover_ui(|ui| {ui.label("Load a dataset");}).clicked() {
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
    
                    if ui.button("+").on_hover_ui(|ui| {ui.label("Add a new dataset");}).clicked() {
                        datasets.push((vec![], "new dataset".into()));
                        for (p, m) in str_params.iter_mut().zip(models.iter_mut()) {
                            p.push(m.circuit.generate_new_params().into());
                        }
                    }
                    if ui.button("D").on_hover_ui(|ui| {ui.label("Duplicate dataset");}).clicked() {
                        datasets.push(datasets[*current_dataset].clone());
                        for p in str_params.iter_mut() {
                            let dc = p[*current_dataset].clone();
                            p.push(dc);
                        }
                    };
                    if ui.button("-").on_hover_ui(|ui| {ui.label("Remove current dataset");}).clicked() && datasets.len() > 1 {
                        datasets.remove(*current_dataset);
                        for p in str_params.iter_mut() {
                            p.remove(*current_dataset);
                        }
                        if *current_dataset > 0 {*current_dataset -= 1;}
                    }
                    if ui.button("Up").clicked() && *current_dataset > 0 {
                        (datasets as &mut[_]).swap(*current_dataset, *current_dataset - 1);
                        for sp in str_params.iter_mut() {
                            sp.swap(*current_dataset, *current_dataset - 1);
                        }
                        *current_dataset -= 1;
                    }
                    if ui.button("Down").clicked() && *current_dataset < datasets.len() - 1 {
                        (datasets as &mut[_]).swap(*current_dataset, *current_dataset + 1);
                        for sp in str_params.iter_mut() {
                            sp.swap(*current_dataset, *current_dataset + 1);
                        }
                        *current_dataset += 1;
                    }
                });
    
                ui.separator();
    
                egui::ScrollArea::vertical().show(ui, |ui| {
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
        });

        egui::SidePanel::left("datalist").show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                if ui.button("Save project as").clicked() {
                    try_into_numbers(str_params).map_or_else(|| {
                        *text_status = "Can't save due to errors in parameters".into();
                    }, |nparams| if let Ok(Some(filename)) = native_dialog::FileDialog::new().show_save_single_file() {
                            if let Ok(mut file) = std::fs::File::create(filename) {
                                if let Ok(string) = Self::save_to_string(models, datasets, &nparams) {
                                    use std::io::Write;
                                    writeln!(&mut file, "{}", string).unwrap();
                                } else {
                                    println!("Error writing to the file");
                                }
                            }
                        });
                }

                if ui.button("Load project").clicked() {
                    if let Ok(Some(filename)) = native_dialog::FileDialog::new().show_open_single_file() {
                        if let Ok(sfile) = std::fs::read_to_string(filename) {
                            if let Some((m, d, p)) = Self::load_from_string(&sfile) {
                                *models = m;
                                *datasets = d;
                                *str_params = p.into_iter().map(|ds| ds.into_iter().map(Into::into).collect()).collect();
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

            if ui.horizontal_wrapped(|ui| {
                let mut lock_now = false;

                if ui.button("Fit").clicked() {
                    if let Ok(mut paramlist) = ParameterDesc::try_from(&str_params[*current_circ][*current_dataset]) {
                        //let paramlist = &mut params[*current_circ][*current_dataset];
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
                        opt.set_maxeval(u32::MAX).unwrap();
                        opt.set_maxtime(10.0).unwrap();
    
                        opt.set_xtol_rel(1e-10).unwrap();
    
                        let optresult = opt.optimize(&mut paramlist.vals);
    
                        *text_status = match optresult {
                            Ok((okstate, _)) => {
                                use nlopt::SuccessState::{FtolReached, MaxEvalReached, MaxTimeReached, StopValReached, Success, XtolReached};
                                let res_str = match okstate {
                                    Success | FtolReached | StopValReached | XtolReached => {"Fitting finished".into()}
                                    MaxEvalReached => {"Max evaluations reached".into()}
                                    MaxTimeReached => {"Max time reached".into()}
                                };

                                str_params[*current_circ][*current_dataset] = paramlist.into();

                                res_str
                            }
                            Err((failstate, _)) => {
                                use nlopt::FailState::{Failure, ForcedStop, InvalidArgs, OutOfMemory, RoundoffLimited};
                                match failstate {
                                    Failure => {"Fitting failure".into()}
                                    InvalidArgs => {"The parameter values exceed the bounds".into()}
                                    OutOfMemory => {"Memory error".into()}
                                    RoundoffLimited => {"Roundoff limited".into()}
                                    ForcedStop => {"Forced stop of fitting".into()}
                                }
                            }
                        };
                        println!("{:?}", optresult);
    
                        lock_now = true;
                    }
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

                lock_now
            }).inner {
                models[*current_circ].lock = true;
            }

            if ui.horizontal_wrapped(|ui| {
                let mut lock_now = false;

                if ui.button("Copy").on_hover_ui(|ui| {ui.label("Copy the parameters from the current circuit");}).clicked() {
                    *copied_paramlist = Some((*current_circ, *current_dataset));
                }
                if ui.button("Paste").on_hover_ui(|ui| {ui.label("Replace the parameters with copied ones");}).clicked() {
                    if let Some(cp) = copied_paramlist {
                        if cp.0 == *current_circ {
                            if let Some(dpl) = str_params.get(cp.0).and_then(|x| x.get(cp.1)) {
                                str_params[*current_circ][*current_dataset] = dpl.clone();
                            }
                        }
                    }
                    lock_now = true;
                }
                lock_now
            }).inner {
                models[*current_circ].lock = true;
            }

            if egui::ScrollArea::vertical().show(ui, |ui| {
                ui.vertical(|ui|->bool {
                    use egui::Widget;

                    let mut lock_now = false;

                    let mut empty_ed = vec![];

                    let param_names = models.get_mut(*current_circ).map_or(vec![], |x| x.circuit.param_names());
                    let paramlist = &mut str_params[*current_circ][*current_dataset];
                    let param_ed = models.get_mut(*current_circ).map_or(&mut empty_ed, |x| &mut x.parameters);

                    let wi = ui.available_width() - 155.0;  // Magic numbers

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
                            })
                            .on_hover_ui(|ui| {ui.label("Feature not yet implemented");})
                            .clicked() {
                                *ed = match ed {
                                    ParameterEditability::Plural => ParameterEditability::Single,
                                    ParameterEditability::Single => ParameterEditability::Immutable,
                                    ParameterEditability::Immutable => ParameterEditability::Plural,
                                };
                            }

                            ui.label(name);

                            let mut x_txt = |s: &mut String, div, hint| {
                                let t_clr = |s:&str| if s.parse::<f64>().is_ok() {None} else {Some(Color32::RED)};
                                let mclr = t_clr(s);
                                if egui::TextEdit::singleline(s)
                                    .text_color_opt(mclr)
                                    .desired_width(wi/div)
                                    .ui(ui)
                                    .on_hover_ui(|ui| {ui.label(hint);})
                                    .changed()
                                {
                                    lock_now = true;
                                }
                            };

                            x_txt(min, 4., "Min");
                            x_txt(max, 4., "Max");
                            x_txt(val, 2., "Value");

                            if ui.small_button("<").on_hover_ui(|ui| {ui.label("Decrease (Ctrl=fast, Shift=slow)");}).clicked() {
                                if let Ok(mut nval) = val.parse::<f64>() {
                                    if ctx.input().modifiers.shift  { nval /= 1.01 }
                                    else if ctx.input().modifiers.command { nval /= 2.0 }
                                    else { nval /= 1.1 }
                                    *val = format!("{}", PrettyPrintFloat(nval));
                                }
                                text_status.clear();
                                lock_now = true;
                            }
                            if ui.small_button(">").on_hover_ui(|ui| {ui.label("Increase (Ctrl=fast, Shift=slow)");}).clicked() {
                                if let Ok(mut nval) = val.parse::<f64>() {
                                    if ctx.input().modifiers.shift  { nval *= 1.01 }
                                    else if ctx.input().modifiers.command { nval *= 2.0 }
                                    else { nval *= 1.1 }
                                    *val = format!("{}", PrettyPrintFloat(nval));
                                }
                                text_status.clear();
                                lock_now = true;
                            }
                        });
                    }

                    if !text_status.is_empty() {
                        ui.separator();
                        ui.label(&*text_status);
                    }

                    lock_now
                }).inner
            }).inner {
                models[*current_circ].lock = true;
            }
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical_centered_justified(|ui| {
                use eframe::egui::Widget;

                let dataset = &mut match datasets.get_mut(*current_dataset) {
                    Some(s) => s,
                    None => return,
                }.0;

                ui.horizontal(|ui|
                    if ui.small_button("Nyquist").clicked() {
                        *plot_type = PlotType::Nyquist;
                    }
                    else if ui.small_button("Bode Amplitude").clicked() {
                        *plot_type = PlotType::BodeAmp;
                    }
                    else if ui.small_button("Bode Phase").clicked() {
                        *plot_type = PlotType::BodePhase;
                    }
                );

                let eplot_type = *plot_type;

                let points_dataset = egui::plot::Points::new(
                        egui::plot::Values::from_values(
                            dataset.iter().map(|d| 
                                match eplot_type {
                                    PlotType::Nyquist => egui::plot::Value::new(d.imp.re, -d.imp.im),
                                    PlotType::BodePhase => egui::plot::Value::new(d.freq.log10(), -d.imp.arg().to_degrees()),
                                    PlotType::BodeAmp => egui::plot::Value::new(d.freq.log10(), d.imp.norm()),
                                }
                            ).collect::<Vec<_>>() 
                        )
                    )
                    .shape(egui::plot::MarkerShape::Circle)
                    .radius(4.);
                
                let dlen = dataset.len();

                let rmin;
                let rmax;

                match dlen {
                    0 => {
                        rmin = 1.0;
                        rmax = 1000.0;
                    }
                    1 => {
                        rmin = dataset[0].freq/2.0;
                        rmax = dataset[0].freq*2.0;
                    }
                    _ => {
                        rmin = dataset[0].freq;
                        rmax = dataset[dlen-1].freq;
                    }
                }

                let plt = egui::plot::Plot::new("plot1")
                    .width(ui.available_width())
                    .height(ui.available_height()/2.0)
                    .label_formatter(move |_s, &Value{x, y}| -> String {
                        match eplot_type {
                            PlotType::Nyquist => format!("{} Ohm\n{} Ohm", x, y),
                            PlotType::BodeAmp => format!("{} Hz:\n{} Ohm", 10.0_f64.powf(x), y),
                            PlotType::BodePhase => format!("{} Hz:\n{}°", 10.0_f64.powf(x), y),
                        }
                    })
                    .x_axis_formatter(move |v,_e| {
                        match eplot_type {
                            PlotType::Nyquist => v.to_string(),
                            PlotType::BodePhase|PlotType::BodeAmp => (10.0_f64).powf(v).to_string(),
                        }
                    });

                plt.show(ui, |plot_ui| {
                    plot_ui.points(points_dataset);

                    if let Ok(cparams) = ParameterDesc::try_from(&str_params[*current_circ][*current_dataset]) {
                        if let Some(m) = models.get(*current_circ) {
                            let line_values: Vec<egui::widgets::plot::Value> = geomspace(rmin, rmax, 1000).map(
                                |f| {
                                let imp = m.circuit.impedance(std::f64::consts::TAU*f, &cparams.vals);
                                match plot_type {
                                    PlotType::Nyquist => egui::plot::Value::new(imp.re, -imp.im),
                                    PlotType::BodePhase => egui::plot::Value::new(f.log10(), -imp.arg().to_degrees()),
                                    PlotType::BodeAmp => egui::plot::Value::new(f.log10(), imp.norm()),
                                }
                            }).collect();
                            plot_ui.line(egui::plot::Line::new(egui::plot::Values::from_values(line_values)).stroke((0.5, Color32::WHITE)));
    
                            let data_values = 
                            dataset.iter().map(|d| {
                                let imp = m.circuit.impedance(std::f64::consts::TAU*d.freq, &cparams.vals);
                                match plot_type {
                                    PlotType::Nyquist => egui::plot::Value::new(imp.re, -imp.im),
                                    PlotType::BodePhase => egui::plot::Value::new(d.freq.log10(), -imp.arg().to_degrees()),
                                    PlotType::BodeAmp => egui::plot::Value::new(d.freq.log10(), imp.norm()),
                                }
                            }).collect();
                            plot_ui.points(egui::plot::Points::new(egui::plot::Values::from_values(data_values)).color(Color32::WHITE).shape(egui::plot::MarkerShape::Circle).radius(2.));
                        }
                    }

                });

                ui.label(match plot_type{
                    PlotType::Nyquist => "-Im Z vs Re Z",
                    PlotType::BodeAmp => "|Z| vs freq",
                    PlotType::BodePhase => "arg Z vs freq",
                });

                egui::ScrollArea::vertical().show(ui, |ui| {
                    let edit = egui::TextEdit::multiline(editor)
                            .desired_rows(20)
                            .ui(ui);
                    if edit.lost_focus() {
                        dataset_from_string(editor).map_or_else(|| println!("Wrong data"), |v| {
                            *dataset = v;
                        });
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
    let native_options = eframe::NativeOptions{initial_window_size: Some(vec2(1100., 600.)), ..eframe::NativeOptions::default()};
    eframe::run_native(Box::new(app), native_options);
}

