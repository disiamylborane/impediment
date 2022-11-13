use std::{f64::consts::PI, fmt::Display};

use circuit::Element;
use eframe::{egui::{self, Ui, Widget}, epaint::Color32};
use egui::vec2;


pub mod project;
pub mod circuit;
pub use project::Cplx;
use project::{ProjectData, DataPoint, UndoBuffer, Action, ParameterDescriptor, ModelVariable};

pub mod file;

#[derive(PartialEq, Eq, Debug, Clone, Copy, Default)]
pub enum ComponentInteraction { #[default] Replace, Series, Parallel, Delete }

#[derive(PartialEq, Eq, Debug, Clone, Copy, Default)]
pub enum PlotType { #[default] Nyquist, BodeAmp, BodePhase, NyquistAdmittance }


pub struct Ephemerals {
    pub current_spectrum: Option<(usize, usize)>,
    pub current_circ: Option<usize>,

    pub ce_element: circuit::Element,
    pub ce_interaction: ComponentInteraction,

    pub plot_type: PlotType,

    pub blocksize: f32,
}
impl Default for Ephemerals{
    fn default() -> Self {
        Self { 
            current_spectrum: Default::default(), 
            current_circ: Default::default(), 
            ce_element: Default::default(), 
            ce_interaction: Default::default(), 
            plot_type: Default::default(), 
            blocksize: 15.0 
        }
    }
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

pub struct ImportData {
    content: String,
    name: String,
}

pub struct ImportModule {
    data: Vec<ImportData>,

    freq_type: FreqOpenParam,
    imp_type: ImpOpenParam,
    freq_col: usize,
    imp1_col: usize,
    imp2_col: usize,
    skip_head: usize,

    curr_spectrum: usize,
    specgroup: usize,
}

#[derive(Default)]
pub struct ImpedimentApp {
    pub prjdata: project::ProjectData,
    pub editor_variables: Ephemerals,
    pub import_data: Option<ImportModule>,
    pub edit_buffer: String,
    pub action_buffer: UndoBuffer,
}


fn hinted_btn(ui: &mut egui::Ui, txt: &str, hint: &str ) -> bool {
    ui.button(txt).on_hover_ui(|ui| {ui.label(hint);}).clicked()
}



/// Get the coordinates of drawing block given the coordinates of a point
fn block_by_coords(circuit: &circuit::Circuit, widsize: eframe::epaint::Vec2, clickpos: eframe::epaint::Vec2, blocksize: f32) 
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


fn render_circuit_editor(ui: &mut Ui, iapp: &mut ImpedimentApp)->Result<(), Action> {

    ui.horizontal(|ui| {
        fn hinted_selector_e(ui: &mut Ui, iapp: &mut ImpedimentApp, val: Element, text: &str, hint: &str) {
            ui.selectable_value(&mut iapp.editor_variables.ce_element, val, text).on_hover_ui(|ui| {ui.label(hint);});
        }
        fn hinted_selector_i(ui: &mut Ui, iapp: &mut ImpedimentApp, val: ComponentInteraction, text: &str, hint: &str) {
            ui.selectable_value(&mut iapp.editor_variables.ce_interaction, val, text).on_hover_ui(|ui| {ui.label(hint);});
        }

        hinted_selector_e(ui, iapp, Element::Resistor, "R", "Resistor");
        hinted_selector_e(ui, iapp, Element::Capacitor, "C", "Capacitor");
        hinted_selector_e(ui, iapp, Element::Inductor, "L", "Inductor");
        hinted_selector_e(ui, iapp, Element::Warburg, "W", "Warburg");
        hinted_selector_e(ui, iapp, Element::Cpe, "Q", "Constant phase");
        
        ui.separator();

        hinted_selector_i(ui, iapp, ComponentInteraction::Replace, ":", "Replace");
        hinted_selector_i(ui, iapp, ComponentInteraction::Series, "--", "Add series");
        hinted_selector_i(ui, iapp, ComponentInteraction::Parallel, "=", "Add parallel");
        hinted_selector_i(ui, iapp, ComponentInteraction::Delete, "x", "Remove clicked");

        Ok(())
    }).inner?;


    let blocksize = iapp.editor_variables.blocksize;
    let widsize = egui::vec2(200.0, 100.0);
    let (response, painter) = ui.allocate_painter(widsize, egui::Sense::click());
    if let Some((imdl, mdl)) = iapp.editor_variables.current_circ.and_then(|cc| Some((cc, iapp.prjdata.models.get_mut(cc)?)))  {
        let widsize = egui::vec2(ui.available_width(), 100.0);
        let size = mdl.circuit.painted_size();
        let size = egui::vec2(size.0.into(), size.1.into())*blocksize;
        mdl.circuit.paint(response.rect.min + (widsize-size)/2., blocksize, &painter, 0, if mdl.lock {Color32::WHITE} else {Color32::LIGHT_RED}, &mdl.component_names);

        if response.clicked() && !mdl.lock {
            if let Some(pos) = response.interact_pointer_pos() {
                let canvas_pos = pos - response.rect.min;
                if let Some(block) = block_by_coords(&mdl.circuit, widsize, canvas_pos, blocksize) {
                    match iapp.editor_variables.ce_interaction {
                        ComponentInteraction::Replace => {
                            let new_element = iapp.editor_variables.ce_element;
                            if let Some(chg) = mdl.circuit.replace_element(block, new_element) {
                                let old_self = iapp.prjdata.clone();
                                let mdl = iapp.prjdata.models.get_mut(imdl).unwrap();

                                let (ivars, gvars) = mdl.var_ranges(chg.clone());

                                for sgrp in &mut iapp.prjdata.dataset {
                                    sgrp.group_vars[imdl].drain(gvars.clone());
                                    for spc in &mut sgrp.spectra {
                                        spc.ind_params[imdl].drain(ivars.clone());
                                    }
                                }

                                mdl.params.drain(chg.clone());

                                let newparams = new_element.gen_individual_params();

                                for (n, nparam) in newparams.into_iter().enumerate() {
                                    mdl.params.insert(chg.start + n, ParameterDescriptor::Individual);

                                    for sgrp in &mut iapp.prjdata.dataset {
                                        for spc in &mut sgrp.spectra {
                                            spc.ind_params[imdl].insert(chg.start + n, nparam.clone());
                                        }
                                    }
                                }

                                iapp.prjdata.check_for_consistency().unwrap();
                                return Err(Action::Unknown(Box::new(old_self)));
                            }
                        }
                        ComponentInteraction::Series => {
                            let new_element = iapp.editor_variables.ce_element;
                            if let Some((chg_cm, chg_param)) = mdl.circuit.add_series_element(block, new_element) {
                                let old_self = iapp.prjdata.clone();
                                let mdl = iapp.prjdata.models.get_mut(imdl).unwrap();

                                let newparams = new_element.gen_individual_params();

                                let npcount = newparams.len();

                                mdl.component_names.insert(chg_cm, format!("{chg_cm}"));
                                for ip in chg_param..(chg_param+npcount) {
                                    mdl.params.insert(ip, ParameterDescriptor::Individual);
                                }

                                let (ivars, gvars) = mdl.var_ranges(chg_param..(chg_param+npcount));
                                assert!(gvars.is_empty());
                                assert!(ivars.len() == npcount);

                                for (nparam, iidx) in newparams.into_iter().zip(ivars) {
                                    for sgrp in &mut iapp.prjdata.dataset {
                                        for spc in &mut sgrp.spectra {
                                            spc.ind_params[imdl].insert(iidx, nparam.clone());
                                        }
                                    }
                                }

                                iapp.prjdata.check_for_consistency().unwrap();
                                return Err(Action::Unknown(Box::new(old_self)));
                            }
                        }
                        ComponentInteraction::Parallel => {
                            let new_element = iapp.editor_variables.ce_element;
                            if let Some((chg_cm, chg_param)) = mdl.circuit.add_parallel_element(block, new_element) {
                                let old_self = iapp.prjdata.clone();
                                let mdl = iapp.prjdata.models.get_mut(imdl).unwrap();

                                let newparams = new_element.gen_individual_params();

                                let npcount = newparams.len();

                                mdl.component_names.insert(chg_cm, format!("{chg_cm}"));
                                for ip in chg_param..(chg_param+npcount) {
                                    mdl.params.insert(ip, ParameterDescriptor::Individual);
                                }

                                let (ivars, gvars) = mdl.var_ranges(chg_param..(chg_param+npcount));
                                assert!(gvars.is_empty());
                                assert!(ivars.len() == npcount);

                                for (nparam, iidx) in newparams.into_iter().zip(ivars) {
                                    for sgrp in &mut iapp.prjdata.dataset {
                                        for spc in &mut sgrp.spectra {
                                            spc.ind_params[imdl].insert(iidx, nparam.clone());
                                        }
                                    }
                                }

                                iapp.prjdata.check_for_consistency().unwrap();
                                return Err(Action::Unknown(Box::new(old_self)));
                            }
                        }
                        ComponentInteraction::Delete => {
                            if let Some(chg) = mdl.circuit.delete_element(block) {
                                let old_self = iapp.prjdata.clone();
                                let mdl = iapp.prjdata.models.get_mut(imdl).unwrap();

                                let (ivars, gvars) = mdl.var_ranges(chg.clone());

                                for sgrp in &mut iapp.prjdata.dataset {
                                    sgrp.group_vars[imdl].drain(gvars.clone());
                                    for spc in &mut sgrp.spectra {
                                        spc.ind_params[imdl].drain(ivars.clone());
                                    }
                                }

                                mdl.params.drain(chg);

                                iapp.prjdata.check_for_consistency().unwrap();
                                return Err(Action::Unknown(Box::new(old_self)));
                            }
                        }
                    }
                }
            }
        }

        iapp.editor_variables.blocksize += ui.input().scroll_delta.y/50.0;

        if let Some(cc) = iapp.editor_variables.current_circ {
            if ui.small_button("Unlock").clicked() {
                iapp.prjdata.models[cc].lock = false;
            }
        } else {
            ui.add_enabled(false, egui::Button::new("Unlock").small());
        }
    }

    Ok(())
}


fn render_circuit_box(ui: &mut Ui, iapp: &mut ImpedimentApp)->Result<(), Action> {
    ui.horizontal(|ui| {
        if hinted_btn(ui, "+", "Add a new circuit") {
            let new_circ = iapp.prjdata.add_new_circuit_action();
            return Err(new_circ.act(&mut iapp.prjdata))
        }
        hinted_btn(ui, "D", "Duplicate circuit");
        if hinted_btn(ui, "-", "Remove current circuit") {
            if let Some(idx) = iapp.editor_variables.current_circ {
                let action = Action::RemoveCircuit { idx };
                iapp.editor_variables.current_circ = None;
                return Err(action.act(&mut iapp.prjdata))
            }
        }
        Ok(())
    }).inner?;

    for (i, model) in iapp.prjdata.models.iter_mut().enumerate() {
        let old_name = model.name.clone();

        let tedit = egui::TextEdit::singleline(&mut model.name);
        let tedit = if iapp.editor_variables.current_circ==Some(i) {tedit.text_color(Color32::RED)} else {tedit};

        let tres = ui.add(tedit);

        if tres.clicked() {
            iapp.editor_variables.current_circ = Some(i);
        };

        if tres.changed() {
            return Err(Action::EditCircuitName { idx: i, name: old_name });
        }
    }

    Ok(())
}



fn render_dataset_box(ui: &mut Ui, iapp: &mut ImpedimentApp)->Result<(), Action> {
    ui.push_id("ScrollData", |ui| {
        ui.horizontal(|ui| {
            if ui.button("Add Group").clicked() {

                let d_old = iapp.prjdata.clone();
                let gr_vars = iapp.prjdata.models
                    .iter_mut()
                    .map(|mdl| vec![project::ModelVariable::new_unknown(); mdl.group_vars_count()])
                    .collect();
                
                iapp.prjdata.dataset.push(project::SpectrumGroup{ spectra: vec![], group_vars: gr_vars });
                return Err(Action::Unknown(Box::new(d_old)));
            }
            if ui.button("Delete Group").clicked() {
                let d_old = iapp.prjdata.clone();
                if let Some((sg,_)) = iapp.editor_variables.current_spectrum {
                    iapp.prjdata.dataset.remove(sg);
                    iapp.editor_variables.current_spectrum = None;
                }
                return Err(Action::Unknown(Box::new(d_old)));
            }
            Ok(())
        }).inner?;

        egui::ScrollArea::both().show(ui, |ui|->Result<(), Action> {
            egui::Grid::new("datagrid").min_col_width(60.0).show(ui, |ui| {
                if let Some(vlen) = iapp.prjdata.dataset.iter().map(|sgroup| sgroup.spectra.len()).max() {
                    for i in 0..(vlen+1) {
                        for (dkidx, sgroup) in iapp.prjdata.dataset.iter_mut().enumerate() {
                            if let Some(ds) = sgroup.spectra.get_mut(i) {
                                let old_name = ds.name.clone();

                                let tedit = egui::TextEdit::singleline(&mut ds.name);
                                let tedit = if iapp.editor_variables.current_spectrum==Some((dkidx,i)) {tedit.text_color(Color32::RED)} else {tedit};

                                let mut duplicate = false;
                                let mut delete = false;
                                let tres = ui.add(tedit).context_menu(|ui| {
                                    if ui.button("Duplicate").clicked() {
                                        duplicate = true;
                                        ui.close_menu();
                                    }
                                    if ui.button("Delete").clicked() {
                                        delete = true;
                                        ui.close_menu();
                                    }
                                });
                                if duplicate {
                                    let d_old = iapp.prjdata.clone();
                                    let n_clone = iapp.prjdata.dataset[dkidx].spectra[i].clone();
                                    iapp.prjdata.dataset[dkidx].spectra.push(n_clone);
                                    return Err(Action::Unknown(Box::new(d_old)));
                                }
                                if delete {
                                    let d_old = iapp.prjdata.clone();
                                    iapp.prjdata.dataset[dkidx].spectra.remove(i);
                                    iapp.editor_variables.current_spectrum = None;
                                    return Err(Action::Unknown(Box::new(d_old)));
                                }

                                if tres.clicked() {
                                    iapp.editor_variables.current_spectrum = Some((dkidx, i));
                                }

                                if tres.changed() {
                                    return Err(Action::EditSpectrumName { idx: (dkidx, i), name: old_name });
                                }
                            }
                            else if i == sgroup.spectra.len() {
                                let mut add = false;
                                let mut import = false;

                                ui.horizontal(|ui|{
                                    if ui.button("+").clicked() {
                                        add = true;
                                    }
                                    if ui.button("Import").clicked() {
                                        import = true;
                                    }
                                });

                                if add {
                                    let d_old = iapp.prjdata.clone();
                                    let constants = vec![0.0; iapp.prjdata.constants.len()];
                                    let ind_params = iapp.prjdata.models
                                        .iter_mut()
                                        .map(|mdl| vec![project::ModelVariable::new_unknown(); mdl.individual_vars_count()])
                                        .collect();
                                    iapp.prjdata.dataset[dkidx].spectra.push(project::Spectrum{ points: vec![], name: "New spec".to_string(), ind_params, constants });
                                    return Err(Action::Unknown(Box::new(d_old)));
                                }

                                if import {
                                    if let Ok(files) = native_dialog::FileDialog::new().show_open_multiple_file() {
                                        if !files.is_empty() {
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
                                                    if !v.is_empty() {
                                                        iapp.import_data = Some(ImportModule { 
                                                            data: v.into_iter().map(|(content, name)| ImportData { content, name }).collect(),
                                                            freq_type: FreqOpenParam::Hz,
                                                            imp_type: ImpOpenParam::MinusOhm,
                                                            freq_col: 0,
                                                            imp1_col: 1,
                                                            imp2_col: 2,
                                                            skip_head: 0,

                                                            curr_spectrum: 0,
                                                            specgroup: dkidx,
                                                        });
                                                    }
                                                }
                                                Err(e) => println!("Error: {}", e),
                                            }
                                        }
                                    }
                                }
                            }
                            else {
                                ui.label("");
                            }
                        }

                        ui.end_row();
                    }
                }
                Ok(())
            }).inner?;
            Ok(())
        }).inner?;
        Ok(())
    }).inner?;

    Ok(())
}



fn hinted_selector_ptype(ui: &mut Ui, iapp: &mut ImpedimentApp, val: PlotType, text: &str, hint: &str) {
    ui.selectable_value(&mut iapp.editor_variables.plot_type, val, text).on_hover_ui(|ui| {ui.label(hint);});
}

fn render_plot(ui: &mut Ui, iapp: &mut ImpedimentApp)->Result<(), Action> {
    ui.horizontal(|ui|{
        hinted_selector_ptype(ui, iapp, PlotType::Nyquist, "Nyquist", "Draw impedance godograph with inverted Im Z axis");
        hinted_selector_ptype(ui, iapp, PlotType::BodeAmp, "Bode Amp", "Draw amplitude-log frequency plot");
        hinted_selector_ptype(ui, iapp, PlotType::BodePhase, "Bode Phase", "Draw negative phase-lg frequency plot");
        hinted_selector_ptype(ui, iapp, PlotType::NyquistAdmittance, "Admittance", "Draw admittance godograph with inverted Im Z axis");
    });

    let plt = egui::plot::Plot::new("plot1")
        .width(ui.available_width())
        .height(ui.available_height()/2.0);
    
    let mut pltshow = vec![];
    let mut pltlines = vec![];
    let freqrange = 0.01..10000.0;
    let extractor = match iapp.editor_variables.plot_type {
        PlotType::Nyquist => |datapoint: DataPoint| egui::plot::Value::new(datapoint.imp.re, -datapoint.imp.im),
        PlotType::BodePhase => |d: DataPoint| egui::plot::Value::new(d.freq.log10(), -d.imp.arg().to_degrees()),
        PlotType::BodeAmp => |d: DataPoint| egui::plot::Value::new(d.freq.log10(), d.imp.norm()),
        PlotType::NyquistAdmittance => |d: DataPoint| {let adm = 1.0/d.imp; egui::plot::Value::new(adm.re, adm.im)},
    };

    if let Some((k,s)) = iapp.editor_variables.current_spectrum {
        let spectrum = &iapp.prjdata.dataset[k].spectra[s];
        let actives = spectrum.points.iter().filter(|&datapoint| datapoint.enabled).cloned().map(extractor);
        let passives = spectrum.points.iter().filter(|&datapoint| !datapoint.enabled).cloned().map(extractor);

        let active_points = egui::plot::Points::new(
            egui::plot::Values::from_values_iter(actives)
        )
        .shape(egui::plot::MarkerShape::Circle)
        .radius(4.);

        let passive_points = egui::plot::Points::new(
            egui::plot::Values::from_values_iter(passives)
        )
        .shape(egui::plot::MarkerShape::Circle)
        .color(Color32::WHITE)
        .radius(4.);

        pltshow.push(active_points);
        pltshow.push(passive_points);

        if let Some(cc) = iapp.editor_variables.current_circ {
            let mdl = &iapp.prjdata.models[cc];
            let freqs = geomspace(freqrange.start, freqrange.end, 500);

            let grps = &iapp.prjdata.dataset[k].group_vars[cc];
            let inds = &iapp.prjdata.dataset[k].spectra[s].ind_params[cc];

            let params = mdl.build_params(inds, grps);

            let points = freqs.map(|freq| DataPoint{freq, imp: mdl.circuit.impedance(2.*PI*freq, &params), enabled:true}).map(extractor);

            let line = egui::plot::Line::new(egui::plot::Values::from_values_iter(points)).stroke((0.5, Color32::YELLOW));
            pltlines.push(line);

            let predicted = spectrum.points.iter().map(|pt| DataPoint { freq: pt.freq, imp: mdl.circuit.impedance(2.*PI*pt.freq, &params), enabled: true }).map(extractor);
            
            let predicted_points = egui::plot::Points::new(
                egui::plot::Values::from_values_iter(predicted)
            )
            .shape(egui::plot::MarkerShape::Circle)
            .color(Color32::YELLOW)
            .radius(2.);
            pltshow.push(predicted_points);
        }
    }

    plt.show(ui, |plot_ui| {
        for pts in pltshow {
            plot_ui.points(pts);
        }
        for pts in pltlines {
            plot_ui.line(pts);
        }
    });

    Ok(())
}

pub fn geomspace(first: f64, last: f64, count: u32) -> impl Iterator<Item=f64>
{
    let (lf, ll) = (first.ln(), last.ln());
    let delta = (ll - lf) / f64::from(count-1);
    (0..count).map(move |i| (f64::from(i).mul_add(delta, lf)).exp())
}

fn render_data_editor(ctx: &egui::Context, ui: &mut Ui, iapp: &mut ImpedimentApp)->Result<(), Action> {
    let focus = ctx.memory().focus();

    if let Some((k,s)) = iapp.editor_variables.current_spectrum {
        let spectrum = &mut iapp.prjdata.dataset[k].spectra[s];
        let mut delete_it = None;

        for (i_point, point) in spectrum.points.iter_mut().enumerate() {
            ui.horizontal(|ui| {
                if ui.selectable_label(point.enabled, "·").context_menu(|ui|{
                    if ui.button("Delete").clicked() {
                        delete_it = Some(i_point);
                        ui.close_menu();
                    }    
                }).clicked() {
                    point.enabled = !point.enabled;
                }

                if delete_it.is_some() {
                    return Ok(());
                }

                let desired_width = (ui.available_width() - 30.0)/3.0;  // Magic numbers

                let freq_id = egui::Id::new(i_point << 8 | 0b_1010_0000);
                if let Err(old_val) = value_editor(ui, freq_id, focus==Some(freq_id), &mut iapp.edit_buffer, desired_width, &mut point.freq) {
                    return Err(Action::EditDataPoint { idx: (k,s,i_point), part: project::DataPointVal::Freq, value: old_val });
                }

                let re_id = egui::Id::new(i_point << 8 | 0b_1010_0001);
                if let Err(old_val) = value_editor(ui, re_id, focus==Some(re_id), &mut iapp.edit_buffer, desired_width, &mut point.imp.re) {
                    return Err(Action::EditDataPoint { idx: (k,s,i_point), part: project::DataPointVal::Re, value: old_val });
                }

                let im_id = egui::Id::new(i_point << 8 | 0b_1010_0010);
                if let Err(old_val) = value_editor(ui, im_id, focus==Some(im_id), &mut iapp.edit_buffer, desired_width, &mut point.imp.im) {
                    return Err(Action::EditDataPoint { idx: (k,s,i_point), part: project::DataPointVal::Im, value: old_val });
                }

                Ok(())
            }).inner?;

            if let Some(di) = delete_it {
                let d = Action::Unknown(Box::new(iapp.prjdata.clone()));
                iapp.prjdata.dataset[k].spectra[s].points.remove(di);
                return Err(d);
            }
        }

        ui.horizontal(|ui| {
            if ui.button("+").clicked() {
                let d = Action::Unknown(Box::new(iapp.prjdata.clone()));
                let dp_default = DataPoint { freq: 1.0, imp: Cplx{re:0.0,im:0.0}, enabled: true };
                let pts = &mut iapp.prjdata.dataset[k].spectra[s].points;
                pts.push(pts.last().cloned().unwrap_or(dp_default));
                return Err(d);
            }

            if ui.button("Sort").clicked() {
                let d = Action::Unknown(Box::new(iapp.prjdata.clone()));
                let pts = &mut iapp.prjdata.dataset[k].spectra[s].points;
                pts.sort_unstable_by(|a,b| a.freq.partial_cmp(&b.freq).unwrap_or(std::cmp::Ordering::Equal) );
                return Err(d);
            }

            Ok(())
        }).inner?;
    }
    Ok(())
}



fn render_consts(ctx: &egui::Context, ui: &mut Ui, iapp: &mut ImpedimentApp)->Result<(), Action> {
    let focus = ctx.memory().focus();

    if let Some((grp, spc)) = iapp.editor_variables.current_spectrum {
        let consts = &mut iapp.prjdata.dataset[grp].spectra[spc].constants;
        let names = &mut iapp.prjdata.constants;

        let mut delete = None;
        for (i_cst, (name, val)) in names.iter_mut().zip(consts.iter_mut()).enumerate() {
            ui.horizontal(|ui| {
                let id = egui::Id::new(i_cst << 8 | 0b_1110_1111);

                let dlabel = egui::Label::new(name as &String as &str);
                let elabel = editable_label(focus, id, ui, name, dlabel, ctx);

                elabel.context_menu(|ui| {
                    if ui.button("Delete").clicked() {
                        delete = Some(i_cst);
                        ui.close_menu();
                    }
                });

                let awi = ui.available_width();

                let cst_id = egui::Id::new(i_cst << 8 | 0b_1111_0000);
                if let Err(old_val) = value_editor(ui, cst_id, focus==Some(cst_id), &mut iapp.edit_buffer, awi-50.0, val) {
                    return Err(Action::EditConst { spec: (grp, spc), cst: i_cst, value: old_val });
                }

                Ok(())
            }).inner?;
        }

        if let Some(del) = delete {
            let oldd = iapp.prjdata.clone();
            iapp.prjdata.constants.remove(del);
            for sgrp in &mut iapp.prjdata.dataset {
                for sp in &mut sgrp.spectra {
                    sp.constants.remove(del);
                }
            }
            return Err(Action::Unknown(Box::new(oldd)));
        }

        if ui.button("Add").clicked() {
            let oldd = iapp.prjdata.clone();
            iapp.prjdata.constants.push("Constant".to_string());
            for sgrp in &mut iapp.prjdata.dataset {
                for sp in &mut sgrp.spectra {
                    sp.constants.push(0.0);
                }
            }
            return Err(Action::Unknown(Box::new(oldd)));
        }
    }
    Ok(())
}


fn editable_label(
    focus: Option<egui::Id>,
    id: egui::Id,
    ui: &mut Ui,
    name: &mut String,
    label: egui::Label,
    ctx: &egui::Context
)->egui::Response {
    if focus == Some(id) {
        let lay = ui.painter().layout(name.clone(), egui::FontId::default(), Color32::WHITE, 0.0).as_ref().rect.size().y / 2.0;

        let tedit = egui::TextEdit::singleline(name).desired_width(lay+10.0).id(id);
            
        ui.add(tedit)
    } else {
        let namelabel = label.sense(egui::Sense::click()).ui(ui);
        if namelabel.clicked() {
            ctx.memory().request_focus(id);
        }
        namelabel
    }
}


fn render_ind_params(ctx: &egui::Context, ui: &mut Ui, iapp: &mut ImpedimentApp)->Result<(), Action> {
    let focus = ctx.memory().focus();

    if let (Some(mdl), Some((grp, spc))) = (iapp.editor_variables.current_circ, iapp.editor_variables.current_spectrum) {
        let vals = &mut iapp.prjdata.dataset[grp].spectra[spc].ind_params[mdl];
        let mut names = iapp.prjdata.models[mdl].get_individual_vars();
        let mut ivals = vals.iter_mut().enumerate();

        let mut make_group: Option<(usize, ModelVariable)> = None;

        while let (Some(name), Some((i_ind, val))) = (names.next(ui), ivals.next()) {
            ui.horizontal(|ui| {
                let selabel = ui.selectable_label(val.enabled, "·");
                if selabel.clicked() {
                    val.enabled = !val.enabled;
                }

                selabel.context_menu(|ui|{
                    if ui.button("Make group").clicked() {
                        make_group = Some((name.1, val.clone()));
                        ui.close_menu();
                    }
                });

                let lab_id = egui::Id::new(i_ind << 8 | 0b_0011_1111);

                editable_label(focus, lab_id, ui, name.0, name.2, ctx);

                let wi = ui.available_width() - 100.0;  // Magic numbers

                let min_id = egui::Id::new(i_ind << 8 | 0b_0011_0000);
                if let Err(old_val) = value_editor(ui, min_id, focus==Some(min_id), &mut iapp.edit_buffer, wi/4.0, &mut val.bounds.0) {
                    return Err(Action::EditIndividualVar { mdl, spec: (grp, spc), var: i_ind, part: project::DataVarPart::Min, value: old_val });
                }

                let max_id = egui::Id::new(i_ind << 8 | 0b_0011_0001);
                if let Err(old_val) = value_editor(ui, max_id, focus==Some(max_id), &mut iapp.edit_buffer, wi/4.0, &mut val.bounds.1) {
                    return Err(Action::EditIndividualVar { mdl, spec: (grp, spc), var: i_ind, part: project::DataVarPart::Max, value: old_val });
                }

                let val_id = egui::Id::new(i_ind << 8 | 0b_0011_0011);
                if let Err(old_val) = value_editor(ui, val_id, focus==Some(val_id), &mut iapp.edit_buffer, wi/2.0, &mut val.val) {
                    return Err(Action::EditIndividualVar { mdl, spec: (grp, spc), var: i_ind, part: project::DataVarPart::Val, value: old_val });
                }

                if ui
                    .small_button("<")
                    .on_hover_ui(|ui| {ui.label("Decrease (Ctrl=fast, Shift=slow)");})
                    .clicked() 
                {
                    if ctx.input().modifiers.shift  { val.val /= 1.01 }
                    else if ctx.input().modifiers.command { val.val /= 2.0 }
                    else { val.val /= 1.1 }
                }
                if ui
                    .small_button(">")
                    .on_hover_ui(|ui| {ui.label("Increase (Ctrl=fast, Shift=slow)");})
                    .clicked() 
                {
                    if ctx.input().modifiers.shift  { val.val *= 1.01 }
                    else if ctx.input().modifiers.command { val.val *= 2.0 }
                    else { val.val *= 1.1 }
                }

                Ok(())
            }).inner?;
        }


        if let Some((param, mvar)) = make_group {
            let oldd = Box::new(iapp.prjdata.clone());
            let ipar = iapp.prjdata.models[mdl].var_ranges(param..(param+1)).0.start;
            iapp.prjdata.models[mdl].params[param] = ParameterDescriptor::Group(project::GroupParameterType::Value);
            let gpar = iapp.prjdata.models[mdl].var_ranges(param..(param+1)).1.start;

            for sg in &mut iapp.prjdata.dataset {
                sg.group_vars[mdl].insert(gpar, mvar.clone());
                for sp in &mut sg.spectra {
                    sp.ind_params[mdl].remove(ipar);
                }
            }
            return Err(Action::Unknown(oldd));
        }
    }

    Ok(())
}


fn render_grp_vars(ctx: &egui::Context, ui: &mut Ui, iapp: &mut ImpedimentApp)->Result<(), Action> {
    let focus = ctx.memory().focus();

    if let (Some(mdl), Some((grp, _))) = (iapp.editor_variables.current_circ, iapp.editor_variables.current_spectrum) {
        let vals = &mut iapp.prjdata.dataset[grp].group_vars[mdl];
        let mut names = iapp.prjdata.models[mdl].get_group_vars();
        let mut ivals = vals.iter_mut().enumerate();

        while let (Some(name), Some((i_grvar, val))) = (names.next(ui, &iapp.prjdata.constants), ivals.next()) {
            ui.horizontal(|ui| {
                if ui.selectable_label(val.enabled, "·").clicked() {
                    val.enabled = !val.enabled;
                }

                let lab_id = egui::Id::new(i_grvar << 8 | 0b_1001_1111);

                editable_label(focus, lab_id, ui, name.0, name.1, ctx);

                let wi = ui.available_width() - 100.0;  // Magic numbers

                let min_id = egui::Id::new(i_grvar << 8 | 0b_1001_0000);
                if let Err(old_val) = value_editor(ui, min_id, focus==Some(min_id), &mut iapp.edit_buffer, wi/4.0, &mut val.bounds.0) {
                    return Err(Action::EditGroupVar { mdl, spec: grp, var: i_grvar, part: project::DataVarPart::Min, value: old_val });
                }

                let max_id = egui::Id::new(i_grvar << 8 | 0b_1001_0001);
                if let Err(old_val) = value_editor(ui, max_id, focus==Some(max_id), &mut iapp.edit_buffer, wi/4.0, &mut val.bounds.1) {
                    return Err(Action::EditGroupVar { mdl, spec: grp, var: i_grvar, part: project::DataVarPart::Max, value: old_val });
                }

                let val_id = egui::Id::new(i_grvar << 8 | 0b_1001_0011);
                if let Err(old_val) = value_editor(ui, val_id, focus==Some(val_id), &mut iapp.edit_buffer, wi/2.0, &mut val.val) {
                    return Err(Action::EditGroupVar { mdl, spec: grp, var: i_grvar, part: project::DataVarPart::Val, value: old_val });
                }

                if ui
                    .small_button("<")
                    .on_hover_ui(|ui| {ui.label("Decrease (Ctrl=fast, Shift=slow)");})
                    .clicked() 
                {
                    if ctx.input().modifiers.shift  { val.val /= 1.01 }
                    else if ctx.input().modifiers.command { val.val /= 2.0 }
                    else { val.val /= 1.1 }
                }
                if ui
                    .small_button(">")
                    .on_hover_ui(|ui| {ui.label("Increase (Ctrl=fast, Shift=slow)");})
                    .clicked() 
                {
                    if ctx.input().modifiers.shift  { val.val *= 1.01 }
                    else if ctx.input().modifiers.command { val.val *= 2.0 }
                    else { val.val *= 1.1 }
                }

                Ok(())
            }).inner?;
        }
    }

    Ok(())
}


fn value_editor(ui: &mut egui::Ui, id: egui::Id, focused_now: bool, str_buffer: &mut String, desired_width: f32, val: &mut f64) -> Result<(), f64> {
    if focused_now {
        let text_color = if str_buffer.parse::<f64>().is_ok() {None} else {Some(egui::Color32::RED)};
        let resp = egui::TextEdit::singleline(str_buffer).desired_width(desired_width).text_color_opt(text_color).id(id).ui(ui);
        if !resp.lost_focus() && resp.changed() {
            if let Ok(x) = str_buffer.parse() {
                let old_val = *val;
                *val = x;
                return Err(old_val);
            }
        }
    } else {
        let mut strval = val.to_string();
        let resp = egui::TextEdit::singleline(&mut strval).desired_width(desired_width).id(id).ui(ui);

        if resp.gained_focus() {
            *str_buffer = strval;
        }
    }

    Ok(())
}


fn display_import_window(ctx: &eframe::egui::Context, iapp: &mut ImpedimentApp) -> Result<(), Action> {
    let mut close = false;
    let mut ret = Ok(());

    if let Some(impdata) = &mut iapp.import_data {
        egui::Window::new("Data").show(ctx, |ui| {
            ui.vertical(|ui| {
                
                ui.label("Columns");
                ui.separator();

                ui.horizontal(|ui| {
                    ui.label("Freq");
                    if ui.small_button("<").clicked() && impdata.freq_col > 0 {impdata.freq_col -= 1}
                    ui.label(&impdata.freq_col.to_string());
                    if ui.small_button(">").clicked() {impdata.freq_col += 1}
                });
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut impdata.freq_type, FreqOpenParam::Hz, &format!("{}", FreqOpenParam::Hz));
                    ui.selectable_value(&mut impdata.freq_type, FreqOpenParam::Khz, &format!("{}", FreqOpenParam::Khz));
                    ui.selectable_value(&mut impdata.freq_type, FreqOpenParam::Rads, &format!("{}", FreqOpenParam::Rads));
                    ui.selectable_value(&mut impdata.freq_type, FreqOpenParam::Krads, &format!("{}", FreqOpenParam::Krads));
                });
                ui.horizontal(|ui| {
                    ui.label("Re Z");
                    if ui.small_button("<").clicked() && impdata.imp1_col > 0 {impdata.imp1_col -= 1}
                    ui.label(&impdata.imp1_col.to_string());
                    if ui.small_button(">").clicked() {impdata.imp1_col += 1}

                    ui.separator();
                    ui.label("Im Z");
                    if ui.small_button("<").clicked() && impdata.imp2_col > 0 {impdata.imp2_col -= 1}
                    ui.label(&impdata.imp2_col.to_string());
                    if ui.small_button(">").clicked() {impdata.imp2_col += 1}

                });
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut impdata.imp_type, ImpOpenParam::PlusOhm, &format!("{}", ImpOpenParam::PlusOhm));
                    ui.selectable_value(&mut impdata.imp_type, ImpOpenParam::MinusOhm, &format!("{}", ImpOpenParam::MinusOhm));
                    ui.selectable_value(&mut impdata.imp_type, ImpOpenParam::PlusKohm, &format!("{}", ImpOpenParam::PlusKohm));
                    ui.selectable_value(&mut impdata.imp_type, ImpOpenParam::MinusKohm, &format!("{}", ImpOpenParam::MinusKohm));
                });


                ui.separator();


                let new_dset: Option<Vec<(Vec<DataPoint>, String)>> = impdata.data.iter().map(|od| {
                    Some((file::csv_to_impediment(&od.content, (impdata.freq_type, impdata.imp_type, impdata.freq_col, impdata.imp1_col, impdata.imp2_col, impdata.skip_head))?, od.name.clone()))
                }).collect();

                if let Some(opening_data) = new_dset {
                    ui.horizontal(|ui| {
                        if ui.small_button("<").clicked() && impdata.curr_spectrum > 0 {
                            impdata.curr_spectrum -= 1;
                        }
                        ui.label(&opening_data[impdata.curr_spectrum].1);
                        if ui.small_button(">").clicked() && impdata.curr_spectrum+1 < opening_data.len() {
                            impdata.curr_spectrum += 1;
                        }
                    });

                    ui.horizontal(|ui| {
                        let plt = egui::plot::Plot::new("plot_import")
                            .width(ui.available_width()/2.0)
                            .height(100.0);

                        let extractor = |datapoint: DataPoint| egui::plot::Value::new(datapoint.imp.re, -datapoint.imp.im);

                        let dpoints = opening_data[impdata.curr_spectrum].0.iter().cloned().map(extractor);

                        let text = opening_data[impdata.curr_spectrum].0.iter().cloned().map(|x| {
                            format!("{:.4}: ({:.4})\n", x.freq, x.imp)
                        }).collect::<String>();

                        let dpoints = egui::plot::Points::new(
                            egui::plot::Values::from_values_iter(dpoints)
                        );
                        plt.show(ui, |plot_ui| {
                            plot_ui.points(dpoints);
                        });

                        egui::ScrollArea::vertical()
                            .max_width(ui.available_width()-3.0)
                            .max_height(ui.available_height()-3.0)
                            .show(ui, |ui|
                        {
                            egui::TextEdit::multiline(&mut {text})
                                .desired_width(ui.available_width())
                                .ui(ui);
                        });

                    });

                    ui.horizontal(|ui| {
                        if ui.button("Load").clicked() {
                            //...

                            let d_old = iapp.prjdata.clone();
                            let constants = vec![0.0; iapp.prjdata.constants.len()];
                            let ind_params: Vec<Vec<ModelVariable>> = iapp.prjdata.models
                                .iter_mut()
                                .map(|mdl| vec![project::ModelVariable::new_unknown(); mdl.individual_vars_count()])
                                .collect();
                            
                            for (spec, sname) in opening_data {
                                iapp.prjdata.dataset[impdata.specgroup].spectra.push(project::Spectrum{ 
                                    points: spec,
                                    name: sname, 
                                    ind_params: ind_params.clone(),
                                    constants: constants.clone(),
                                });
                            }

                            close = true;
                            ret = Err(Action::Unknown(Box::new(d_old)));
                        }
                        if ui.button("Cancel").clicked() {
                            close = true;
                        }
                    });
                } else if ui.button("Cancel").clicked() {
                    close = true;
                }
            });
        });
    }

    if close {iapp.import_data = None;}
    ret
}



fn fit_individuals(ctx: &eframe::egui::Context, iapp: &mut ImpedimentApp) -> Result<(), Action> {
    let (Some(imodel), Some((icg, icc))) = (iapp.editor_variables.current_circ, iapp.editor_variables.current_spectrum) else {return Ok(())};

    

    Ok(())
}


impl eframe::App for ImpedimentApp {
    fn update(&mut self, ctx: &eframe::egui::Context, _frame: &mut eframe::Frame) {
        ctx.set_visuals(egui::Visuals::dark());


        let mut p_action = ||->Result<(), Action> {
            display_import_window(ctx, self)?;

            egui::SidePanel::left("lp").show(ctx, |ui|->Result<(), Action> {
                ui.vertical_centered_justified(|ui| render_circuit_editor(ui, self)).inner?;
                
                ui.separator();

                ui.vertical_centered_justified(|ui| render_circuit_box(ui, self)).inner?;

                ui.separator();

                ui.vertical(|ui| render_dataset_box(ui, self)).inner?;
                Ok(())
            }).inner?;


            egui::SidePanel::right("rp").width_range(200.0..=1000.0).show(ctx, |ui| {

                ui.vertical(|ui| {
                    render_plot(ui, self)?;
                    render_data_editor(ctx, ui, self)
                }).inner

            }).inner?;


            egui::CentralPanel::default().show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label("Dataset constants");
                    Ok(())
                }).inner?;
                render_consts(ctx, ui, self)?;

                ui.separator();

                ui.horizontal(|ui| {
                    ui.label("Individual parameters");

                    if ui.button("Fit").clicked() {

                    }
                    if ui.button("Fit log").clicked() {

                    }

                    Ok(())
                }).inner?;

                render_ind_params(ctx, ui, self)?;

                ui.separator();

                ui.horizontal(|ui| {
                    ui.label("Group parameters");

                    if ui.button("Fit").clicked() {

                    }
                    if ui.button("Fit log").clicked() {

                    }
                    if ui.button("Fit all").clicked() {

                    }
                    if ui.button("Fit all log").clicked() {

                    }

                    Ok(())
                }).inner?;
                render_grp_vars(ctx, ui, self)
            }).inner?;

            Ok(())
        };

        if let Err(undo) = p_action() {
            self.prjdata.check_for_consistency().unwrap();
            // let undo = action.act(&mut self.prjdata);
            self.action_buffer.add_undo(undo);
            println!("{:?}", self.action_buffer);
        }
    }
}

fn main() {
    let app = ImpedimentApp{prjdata: ProjectData::sample(), ..Default::default()};
    let native_options = eframe::NativeOptions{initial_window_size: Some(vec2(900., 500.)), ..eframe::NativeOptions::default()};
    eframe::run_native( "Impediment", native_options, Box::new(|_| Box::new(app)));
}
