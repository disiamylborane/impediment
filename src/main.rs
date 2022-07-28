#![warn(clippy::nursery)]
#![warn(clippy::pedantic)]
//#![allow(clippy::blocks_in_if_conditions)]


mod circuit;
use std::vec;

use circuit::{Circuit, Element};
use eframe::{egui::{self, Color32, vec2, Widget}, emath::Vec2};


#[derive(Debug, Clone, Copy)]
pub enum ParameterEditability { Plural, Single, Immutable }

pub type Cplx = num::complex::Complex<f64>;


#[derive(Debug, Clone)]
pub enum ParameterType {
    Individual(String),
    GroupValue(String),
    GroupLinear{name: String, const_idx: usize},
}

impl ParameterType {
    fn name(&self) -> &str {
        match self {
            ParameterType::Individual(x) => x,
            ParameterType::GroupValue(x) => x,
            ParameterType::GroupLinear { name, const_idx:_ } => name,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    pub circuit: Circuit,
    pub param_types: Vec<ParameterType>,
    pub name: String,
    pub lock: bool,
}


#[derive(Debug, Clone)]
pub enum VarIndex {
    Ind(usize),
    Group(usize),
}


impl Model {
    #[must_use]
    pub fn impedance(&self, omega: f64, ind_vars: &[f64], grp_vars: &[f64], consts: &[f64]) -> Cplx {
        let mut params = Vec::<f64>::with_capacity(self.circuit.paramlen());
        
        let mut curr_indvar = 0;
        let mut curr_grpvar = 0;
        for ptype in &self.param_types {
            match ptype {
                ParameterType::Individual(_) => {
                    params.push(ind_vars[curr_indvar]);
                    curr_indvar += 1;
                },
                ParameterType::GroupValue(_) => {
                    params.push(grp_vars[curr_grpvar]);
                    curr_grpvar += 1;
                },
                &ParameterType::GroupLinear { name:_, const_idx } => {
                    let cst = consts[const_idx];
                    let a = grp_vars[curr_grpvar];
                    let b = grp_vars[curr_grpvar+1];
                    params.push(b.mul_add(cst, a));
                    curr_grpvar += 2;
                },
            }
        }

        self.circuit.impedance(omega, &params)
    }


    #[must_use]
    pub fn start_var_index(&self, param_idx: usize) -> VarIndex {
        let mut curr_indvar = 0;
        let mut curr_grpvar = 0;
        for ptype in &self.param_types[0..param_idx] {
            match ptype {
                ParameterType::Individual(_) => {
                    curr_indvar += 1;
                },
                ParameterType::GroupValue(_) => {
                    curr_grpvar += 1;
                },
                &ParameterType::GroupLinear { name:_, const_idx } => {
                    curr_grpvar += 2;
                },
            }
        }

        match self.param_types[param_idx] {
            ParameterType::Individual(_) => VarIndex::Ind(curr_indvar),
            ParameterType::GroupValue(_) => VarIndex::Group(curr_grpvar),
            ParameterType::GroupLinear { name:_, const_idx:_ } => VarIndex::Group(curr_grpvar),
        }
    }


/*
    #[allow(clippy::pedantic)]
    pub fn replace(&mut self, coord: (u16, u16), element: Element, param_idx: usize, ind_vars: &mut Vec<Paramlist>, grp_vars: &mut Vec<Paramlist>) {
        let circ = &mut self.circuit;
        match circ {
            Circuit::Element(e) => {
                /*let var_idx = self.start_var_index()
                for i in 0..e.param_count() {
                    ...
                }*/
                
                *circ = Circuit::Element(element);
            }
            Circuit::Series(s) => {
                
            }
            Circuit::Parallel(p) => {
                
            }
        }

    }
*/

    /*pub fn append_series(&mut self, coord: (u16, u16), element: Element, param_idx: usize) {
        let circ = &mut self.circuit;
        match circ {
            Circuit::Element(e) => {
                let new = Circuit::Series(vec![Circuit::Element(*e), Circuit::Element(element)]);
                let paramlen = circ.paramlen();

                //self.param_types.insert(index, element);
                //element.insert_param(param_idx+paramlen, paramlist, editability);
            }
            Circuit::Series(s) => {
                
            }
            Circuit::Parallel(p) => {
                
            }
        }
    }*/


}


fn small_style(ui: &egui::Ui) -> egui::text::TextFormat {
    ui.style().as_ref().text_styles.get(&egui::TextStyle::Small)
        .map_or_else(
            || egui::text::TextFormat{italics: true, ..Default::default()}, 
            |font_id| egui::text::TextFormat{font_id: font_id.clone(), ..Default::default()})
}

impl Model {
    fn get_individual_vars(&self, ui: &egui::Ui) -> Vec<egui::Label> {
        let mut out = vec![];
        for (pt, &letter) in self.param_types.iter().zip(self.circuit.param_letters().iter()) {
            if let ParameterType::Individual(pts) = pt {
                let mut job = egui::text::LayoutJob::default();
                job.append(letter, 0.0, egui::text::TextFormat{..Default::default()});

                job.append(pts, 0.0, small_style(ui));

                out.push(egui::Label::new(job));
            }
        }
        out
    }

    fn get_group_vars(&self, ui: &egui::Ui) -> Vec<egui::Label> {
        let mut out = vec![];
        for (pt, &letter) in self.param_types.iter().zip(self.circuit.param_letters().iter()) {
            match pt {
                ParameterType::Individual(_) => {}
                ParameterType::GroupValue(gs) => {
                    let mut job = egui::text::LayoutJob::default();
                    job.append(letter, 0.0, egui::text::TextFormat{..Default::default()});
                    job.append(gs, 0.0, small_style(ui));
                    out.push(egui::Label::new(job));
                }
                ParameterType::GroupLinear { name, const_idx } => {
                    let mut job1 = egui::text::LayoutJob::default();
                    job1.append(letter, 0.0, egui::text::TextFormat{..Default::default()});
                    job1.append(name, 0.0, small_style(ui));
                    job1.append("free", 0.0, egui::text::TextFormat{..Default::default()});
                    out.push(egui::Label::new(job1));

                    let mut job2 = egui::text::LayoutJob::default();
                    job2.append(letter, 0.0, egui::text::TextFormat{..Default::default()});
                    job2.append(name, 0.0, small_style(ui));
                    job2.append(&format!("({})", const_idx), 0.0, egui::text::TextFormat{..Default::default()});
                    out.push(egui::Label::new(job2));
                }
            }
        }
        out
    }
}

// TODO: Ineffective memory usage
#[derive(Debug, Copy, Clone)]
pub struct DataPiece {
    pub freq: f64,
    pub imp: crate::Cplx,
    pub enabled: bool,
}


#[derive(Debug, Clone)]
pub struct ModelVariable {
    pub val : f64,
    pub bounds : (f64, f64),
    pub enabled: bool,
}


#[derive(Debug, Clone)]
pub struct Paramlist {
    pub vals : Vec<ModelVariable>,
}


#[derive(Debug, Clone)]
pub struct Dataset {
    pub points: Vec<DataPiece>,
    pub name: String,
    pub individual_vars: Vec<Paramlist>,

    pub consts: Vec<f64>,
}


#[derive(Debug, Clone)]
pub struct DatasetStack {
    pub datasets: Vec<Dataset>,
    pub group_vars: Vec<Paramlist>,
}


//#[derive(PartialEq, Eq, Debug, Clone, Copy)]
//pub enum ElectricComponent { R, C, W, L, Q }

#[derive(PartialEq, Eq, Debug, Clone, Copy)]
pub enum ComponentInteraction { Replace, Series, Parallel, Delete }


#[derive(PartialEq, Eq, Debug)]
pub enum FitMethod { Bobyqa, Tnc, Slsqp, LBfgsB }


pub struct ImpedimentApp {
    pub models: Vec<Model>,
    pub datagrid: Vec<DatasetStack>,
    pub data_consts: Vec<String>,

    pub current_dataset: Option<(usize, usize)>,
    pub current_circ: Option<usize>,

    // circuit editor variables
    pub ce_element: Element,
    pub ce_interaction: ComponentInteraction,

    // fitter variables
    pub fit_method: FitMethod,

    // dataset editor variables
    pub curr_value: String,
}


fn dflt_dataset(name: &str, potential: f64) -> Dataset {
    Dataset {
        points: vec![
            DataPiece{ freq: 1000.0, imp: Cplx::new(5.1, -3.), enabled: true },
            DataPiece{ freq: 100.0, imp: Cplx::new(5.2, -4.), enabled: true },
            DataPiece{ freq: 10.0, imp: Cplx::new(5.3, -5.), enabled: true },
            DataPiece{ freq: 1.0, imp: Cplx::new(5.4, -6.), enabled: true },
        ],
        name: name.to_string(), 
        individual_vars: vec![
            Paramlist{vals: vec![ModelVariable{ val: 1.0, bounds: (0.1, 100.0), enabled: true }]},
            Paramlist{vals: vec![]},
        ],
        consts: vec![1.0, potential],
    }
}
fn dflt_grvars() -> Vec<Paramlist> {
    vec![Paramlist{vals:vec![]}, Paramlist{vals:vec![ModelVariable{ val: 0.001, bounds: (0.000_001, 1.0), enabled: true }]}]
}


#[derive(Clone, Copy)]
#[repr(u8)]
enum UniqueID6bit {
    Dataset = 0b10_1010,
    Const = 0b10_1111,
    IndividualVar = 0b00_0010,
    GroupVar = 0b00_0011,
}


impl Default for ImpedimentApp {
    fn default() -> Self {
        Self{
            models: vec![
                Model { circuit: Circuit::Element(Element::Resistor), param_types: vec![(ParameterType::Individual("ct".to_string()))], name: "Default".to_string(), lock: false },
                Model { circuit: Circuit::Element(Element::Capacitor), param_types: vec![(ParameterType::GroupValue("dl".to_string()))], name: "C-Default".to_string(), lock: false },
            ],
            datagrid: vec![
                DatasetStack {
                    datasets: vec![
                        dflt_dataset("dA1", 1.0), dflt_dataset("dA2", 2.0), dflt_dataset("dA3", 3.0),
                    ],
                    group_vars: dflt_grvars(),
                },
                DatasetStack {
                    datasets: vec![dflt_dataset("dB1", 1.0), dflt_dataset("dB2", 2.0), dflt_dataset("dB3", 3.0)],
                    group_vars: dflt_grvars(),
                },
                DatasetStack {
                    datasets: vec![dflt_dataset("dD1", 1.0), dflt_dataset("dD2", 2.0), dflt_dataset("dD3", 3.0), dflt_dataset("dD4", 4.0)],
                    group_vars: dflt_grvars(),
                },
            ],
            data_consts: vec!["mass".to_string(), "potential".to_string()],

            current_dataset: None,
            current_circ: None,

            ce_element: Element::Resistor,
            ce_interaction: ComponentInteraction::Replace,

            fit_method: FitMethod::Bobyqa,

            curr_value: String::new()
        }
    }
}


macro_rules! hinted_btn {
    ($ui:expr, $txt:literal, $hint:literal) => {
        $ui.button($txt).on_hover_ui(|ui| {ui.label($hint);}).clicked()
    };
}


fn insert_variable_editor(ctx: &egui::Context, ui: &mut egui::Ui, curr_value: &mut String, (i_param, param): (usize, &mut ModelVariable), lbl: egui::Label, hash_6bit: u8) {
    ui.horizontal(|ui| {
        if ui.selectable_label(param.enabled, "·").clicked() {
            param.enabled = !param.enabled;
        }

        ui.add(lbl);

        let wi = ui.available_width() - 100.0;  // Magic numbers
        /*let mut x_txt = |s: &mut String, div, hint| {
            let t_clr = |s:&str| if s.parse::<f64>().is_ok() {None} else {Some(Color32::RED)};
            let mclr = t_clr(s);

            use eframe::egui::Widget;
            egui::TextEdit::singleline(s)
                .text_color_opt(mclr)
                .desired_width(wi/div)
                .ui(ui)
                .on_hover_ui(|ui| {ui.label(hint);});
        };*/

        let focus = ctx.memory().focus();

        let min_id = egui::Id::new(i_param << 8 | 0 << 6 | (hash_6bit as usize));
        let max_id = egui::Id::new(i_param << 8 | 1 << 6 | (hash_6bit as usize));
        let val_id = egui::Id::new(i_param << 8 | 2 << 6 | (hash_6bit as usize));
        value_editor(ui, focus, min_id, curr_value, wi/4.0, &mut param.bounds.0);
        value_editor(ui, focus, max_id, curr_value, wi/4.0, &mut param.bounds.1);
        value_editor(ui, focus, val_id, curr_value, wi/2.0, &mut param.val);
        //x_txt(&mut param.bounds.0, 4., "Min");
        //x_txt(&mut param.bounds.1, 4., "Max");
        //x_txt(&mut param.val, 2., "Value");

        ui.small_button("<").on_hover_ui(|ui| {ui.label("Decrease (Ctrl=fast, Shift=slow)");});
        ui.small_button(">").on_hover_ui(|ui| {ui.label("Increase (Ctrl=fast, Shift=slow)");});   
    });
}


fn value_editor(ui: &mut egui::Ui, focus: Option<egui::Id>, id: egui::Id, str_buffer: &mut String, desired_width: f32, val: &mut f64, ) {
    if focus == Some(id) {
        let text_color = if str_buffer.parse::<f64>().is_ok() {None} else {Some(egui::Color32::RED)};
        let resp = egui::TextEdit::singleline(str_buffer).desired_width(desired_width).text_color_opt(text_color).id(id).ui(ui);
        if resp.lost_focus() {
            str_buffer.clear();
        }
        if let Ok(x) = str_buffer.parse() {
            *val = x;
        }
    }
    else {
        let mut strval = val.to_string();
        let resp = egui::TextEdit::singleline(&mut strval).desired_width(desired_width).id(id).ui(ui);
        if resp.gained_focus() {
            *str_buffer = strval;
        }
    }
}


fn impedance_datapoint_editor(ctx: &egui::Context, ui: &mut egui::Ui, curr_value: &mut String, (point_idx, p): (usize, &mut DataPiece)) {
    ui.horizontal(|ui| {
        if ui.selectable_label(p.enabled, "·").clicked() {
            p.enabled = !p.enabled;
        }

        let desired_width = (ui.available_width() - 30.0)/3.0;  // Magic numbers

        let focus = ctx.memory().focus();

        for (idx, val) in [&mut p.freq, &mut p.imp.re, &mut p.imp.im].into_iter().enumerate() {
            let id = egui::Id::new(point_idx << 8 | idx << 6 | UniqueID6bit::Dataset as usize);
            value_editor(ui, focus, id, curr_value, desired_width, val);
        }
    });
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



impl eframe::App for ImpedimentApp {
    #[allow(clippy::too_many_lines)]
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        ctx.set_visuals(egui::Visuals::dark());

        let Self {
            models,
            datagrid,
            current_dataset,
            current_circ,
            ce_element,
            ce_interaction,
            fit_method,
            curr_value,
            data_consts,
        } = self;


        egui::SidePanel::left("lp").show(ctx, |ui| {
            ui.vertical_centered_justified( |ui| {
                ui.horizontal(|ui| {
                    if hinted_btn!(ui, "+", "Add a new circuit") {
                        let newmodel = Model {
                            circuit: Circuit::Element(Element::Resistor),
                            name: "new model".to_string(),
                            param_types: vec![ParameterType::Individual("0".to_string())],
                            lock: false,
                        };

                        // without iter_mut it moves here (or may be < &mut *datagrid >)
                        for dk in datagrid.iter_mut() {
                            for ds in &mut dk.datasets {
                                ds.individual_vars.push(Paramlist{vals: vec![ModelVariable{val: 10.0, bounds: (1.0, 10000.0), enabled: true}]});
                            }
                        }

                        models.push(newmodel);

                    }
                    if hinted_btn!(ui, "D", "Duplicate circuit") {
                        if let Some(cc) = *current_circ {
                            if let Some(mdl) = models.get(cc) {
                                let newmodel = mdl.clone();
    
                                for dk in datagrid.iter_mut() {
                                    dk.group_vars.push(dk.group_vars[cc].clone());
                                    for ds in &mut dk.datasets {
                                        ds.individual_vars.push(ds.individual_vars[cc].clone());
                                    }
                                }
    
                                models.push(newmodel);
                            }
                        }
                    }
                    if hinted_btn!(ui, "-", "Remove current circuit") {
                        if let Some(cc) = *current_circ {
                            if models.get(cc).is_some() {
                                for dk in datagrid.iter_mut() {
                                    dk.group_vars.remove(cc);
                                    for ds in &mut dk.datasets {
                                        ds.individual_vars.remove(cc);
                                    }
                                }
                                models.remove(cc);
                                *current_circ = None;
                            }
                        }
                    }
                });


                ui.horizontal(|ui| {
                    macro_rules! hinted_component {
                        ($val:ident, $txt:literal, $hint:literal) => {
                            ui.selectable_value(ce_element, Element::$val, $txt).on_hover_ui(|ui| {ui.label($hint);});
                        };
                    }
                    macro_rules! hinted_interact {
                        ($val:ident, $txt:literal, $hint:literal) => {
                            ui.selectable_value(ce_interaction, ComponentInteraction::$val, $txt).on_hover_ui(|ui| {ui.label($hint);});
                        };
                    }

                    hinted_component!{Resistor, "R", "Resistor"}
                    hinted_component!{Capacitor, "C", "Capacitor"}
                    hinted_component!{Inductor, "L", "Inductor"}
                    hinted_component!{Warburg, "W", "Warburg"}
                    hinted_component!{Cpe, "Q", "Constant phase"}

                    ui.separator();

                    hinted_interact!{Replace, ":", "Replace"}
                    hinted_interact!{Series, "--", "Add series"}
                    hinted_interact!{Parallel, "=", "Add parallel"}
                    hinted_interact!{Delete, "x", "Remove clicked"}
                });

                let blocksize = 10.;
                let widsize = egui::vec2(200.0, 100.0);
                let (response, painter) = ui.allocate_painter(widsize, egui::Sense::click());
                painter.rect_filled(response.rect, 0., Color32::from_rgb(80, 80, 80));
                if let Some(mdl) = current_circ.and_then(|cc| models.get_mut(cc))  {
                    let widsize = egui::vec2(ui.available_width(), 100.0);
                    let size = mdl.circuit.painted_size();
                    let size = egui::vec2(size.0.into(), size.1.into())*blocksize;
                    mdl.circuit.paint(response.rect.min + (widsize-size)/2., blocksize, &painter, 0, if mdl.lock {Color32::WHITE} else {Color32::LIGHT_RED}, &mdl.param_types);

                    if response.clicked() && !mdl.lock {
                        if let Some(pos) = response.interact_pointer_pos() {
                            let user_element = *ce_element;
                            let canvas_pos = pos - response.rect.min;
                            if let Some(block) = block_by_coords(&mdl.circuit, widsize, canvas_pos, blocksize) {
                                #[allow(clippy::match_same_arms)]
                                match ce_interaction {
                                    ComponentInteraction::Replace => {
                                        if let Some((element, params)) = mdl.circuit.pick_block(block) {
                                            // Replace circuit element
                                            *mdl.circuit.get_element_mut(element).unwrap() = user_element;
                                            // Replace parameter definitions
                                            todo!();
                                            // Replace parameters
                                            todo!();
                                        }
                                        //mdl.circuit.replace(block, user_element, str_params[*current_circ].iter_mut(), &mut c.parameters);
                                    },
                                    ComponentInteraction::Series => {
                                        todo!();
                                        //mdl.circuit._add_series(block, user_element, str_params[*current_circ].iter_mut(), 0, &mut c.parameters);
                                    },
                                    ComponentInteraction::Parallel => {
                                        todo!();
                                        //mdl.circuit._add_parallel(block, user_element, str_params[*current_circ].iter_mut(), 0, &mut c.parameters);
                                    }
                                    ComponentInteraction::Delete => {
                                        todo!();
                                        //mdl.circuit._remove(block, str_params[*current_circ].iter_mut(), 0, &mut c.parameters);
                                    }
                                }
                            }
                        }
                    }
                }

                if let Some(cc) = current_circ {
                    if ui.small_button("Unlock").clicked() {
                        models[*cc].lock = false;
                    }
                } else {
                    ui.add_enabled(false, egui::Button::new("Unlock").small());
                }

                ui.push_id("ScrollCirc", |ui| {
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        for (i, c) in models.iter_mut().enumerate() {
                            let tedit = egui::TextEdit::singleline(&mut c.name);
                            let tedit = if *current_circ==Some(i) {tedit.text_color(Color32::RED)} else {tedit};
                            if ui.add(tedit).clicked() {
                                *current_circ=Some(i);
                            };
                        }
                    });
                });
            });


            ui.separator();


            ui.vertical( |ui| {
                ui.horizontal(|ui| {
                    if hinted_btn!(ui, "Import", "Import a csv") {
                    }
                    if hinted_btn!(ui, "+", "Create a new dataset") {
                    }
                    if hinted_btn!(ui, "D", "Duplicate dataset") {
                    }
                    if hinted_btn!(ui, "-", "Remove current dataset") {
                    }
                });


                ui.push_id("ScrollData", |ui| {
                    egui::ScrollArea::both().show(ui, |ui| {

                        egui::Grid::new("datagrid").min_col_width(60.0).show(ui, |ui| {
                            if let Some(vlen) = datagrid.iter().map(|d| d.datasets.len()).max() {
                                for i in 0..vlen {
                                    for (dkidx, dstack) in datagrid.iter_mut().enumerate() {
                                        if let Some(ds) = dstack.datasets.get_mut(i) {

                                            let tedit = egui::TextEdit::singleline(&mut ds.name);
                                            let tedit = if *current_dataset==Some((dkidx,i)) {tedit.text_color(Color32::RED)} else {tedit};
                                            let tresp = ui.add(tedit);
                                            if tresp.clicked() {
                                                *current_dataset = Some((dkidx, i));
                                            }
                                        }
                                        else {
                                            ui.label("");
                                        }
                                    }

                                    ui.end_row();
                                }
                            }
                        });
                    });
                });
            });
        });


        egui::SidePanel::right("rp").width_range(200.0..=1000.0).show(ctx, |ui| {
            ui.vertical(|ui|{
                if let Some(ds) = current_dataset.map(|(k,s)| &mut datagrid[k].datasets[s]) {

                    let plt = egui::plot::Plot::new("plot1")
                        .width(ui.available_width())
                        .height(ui.available_height()/2.0);

                    let vecvalues = ds.points.iter().map(|d| 
                        egui::plot::Value::new(d.imp.re, -d.imp.im)
                    ).collect::<Vec<_>>();

                    let points_dataset = egui::plot::Points::new(
                        egui::plot::Values::from_values(vecvalues)
                    )
                    .shape(egui::plot::MarkerShape::Circle)
                    .radius(4.);

                    plt.show(ui, |plot_ui| {
                        plot_ui.points(points_dataset);
                    });


                    for point in ds.points.iter_mut().enumerate() {
                        impedance_datapoint_editor(ctx, ui, curr_value, point);
                    }
                }
            })
        });


        egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical(|ui|{

                if let (Some(imd), Some((idk, ids))) = (*current_circ, *current_dataset) {
                    let model = &mut models[imd];
                    let dk = &mut datagrid[idk];

                    {
                        let ds = &mut dk.datasets[ids];

                        ui.label("Dataset constants");
                        for (i, (name, val)) in data_consts.iter().zip(&mut ds.consts).enumerate() {
                            let focus = ctx.memory().focus();
                            let awi = ui.available_width();
                            ui.horizontal(|ui|{

                                let namelabel = egui::Label::new(name).ui(ui);
                                let lblwidth = namelabel.rect.width();

                                let id = egui::Id::new(i << 6 | UniqueID6bit::Const as usize);
                                value_editor(ui, focus, id, curr_value, awi-50.0-lblwidth, val);
                            });
                        }

                        ui.separator();

                        ui.label("Individual parameters");
                        let ivars = model.get_individual_vars(ui);
                        for (iip, lbl) in ivars.into_iter().enumerate() {
                            insert_variable_editor(ctx, ui, curr_value, (iip, &mut ds.individual_vars[imd].vals[iip]), lbl, UniqueID6bit::IndividualVar as u8);
                        }
                    }

                    ui.separator();

                    ui.label("Group parameters");
                    let gvars = model.get_group_vars(ui);
                    for (igp, lbl) in gvars.into_iter().enumerate() {
                        insert_variable_editor(ctx, ui, curr_value, (igp, &mut dk.group_vars[imd].vals[igp]), lbl, UniqueID6bit::GroupVar as u8);
                    }

                    ui.separator();
                }

            });
        });

    }
}


fn main() {
    let app = ImpedimentApp::default();
    let native_options = eframe::NativeOptions{initial_window_size: Some(vec2(900., 500.)), ..eframe::NativeOptions::default()};
    eframe::run_native( "Impediment", native_options, Box::new(|_| Box::new(app)));
}

