use std::f64::consts::PI;

use circuit::Element;
use eframe::{egui::{self, Ui, Widget}, epaint::Color32};
use egui::vec2;


pub mod project;
pub mod circuit;
pub use project::Cplx;
use project::{ProjectData, DataPoint, UndoBuffer, Action, ParameterDescriptor};


#[derive(PartialEq, Eq, Debug, Clone, Copy, Default)]
pub enum ComponentInteraction { #[default] Replace, Series, Parallel, Delete }

#[derive(PartialEq, Eq, Debug, Clone, Copy, Default)]
pub enum PlotType { #[default] Nyquist, BodeAmp, BodePhase, NyquistAdmittance }


#[derive(Default)]
pub struct Ephemerals {
    pub current_spectrum: Option<(usize, usize)>,
    pub current_circ: Option<usize>,

    pub ce_element: circuit::Element,
    pub ce_interaction: ComponentInteraction,

    pub plot_type: PlotType,
}


#[derive(Default)]
pub struct ImpedimentApp {
    pub prjdata: project::ProjectData,
    pub editor_variables: Ephemerals,
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


    let blocksize = 10.;
    let widsize = egui::vec2(200.0, 100.0);
    let (response, painter) = ui.allocate_painter(widsize, egui::Sense::click());
    if let Some((imdl, mdl)) = iapp.editor_variables.current_circ.and_then(|cc| Some((cc, iapp.prjdata.models.get_mut(cc)?)))  {
        let widsize = egui::vec2(ui.available_width(), 100.0);
        let size = mdl.circuit.painted_size();
        let size = egui::vec2(size.0.into(), size.1.into())*blocksize;
        mdl.circuit.paint(response.rect.min + (widsize-size)/2., blocksize, &painter, 0, if mdl.lock {Color32::WHITE} else {Color32::LIGHT_RED}, &mdl.params);

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
                                    mdl.params.insert(chg.start + n, ParameterDescriptor::Individual(format!("{}", chg.start + n)));

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
                            if let Some(chg) = mdl.circuit.add_series_element(block, new_element) {
                                let old_self = iapp.prjdata.clone();
                                let mdl = iapp.prjdata.models.get_mut(imdl).unwrap();

                                let newparams = new_element.gen_individual_params();

                                let npcount = newparams.len();

                                for ip in chg..(chg+npcount) {
                                    mdl.params.insert(ip, ParameterDescriptor::Individual(format!("{}", ip)));
                                }

                                let (ivars, gvars) = mdl.var_ranges(chg..(chg+npcount));
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
                            if let Some(chg) = mdl.circuit.add_parallel_element(block, new_element) {
                                let old_self = iapp.prjdata.clone();
                                let mdl = iapp.prjdata.models.get_mut(imdl).unwrap();

                                let newparams = new_element.gen_individual_params();

                                let npcount = newparams.len();

                                for ip in chg..(chg+npcount) {
                                    mdl.params.insert(ip, ParameterDescriptor::Individual(format!("{}", ip)));
                                }

                                let (ivars, gvars) = mdl.var_ranges(chg..(chg+npcount));
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
    ui.horizontal(|ui| {
        if hinted_btn(ui, "Import", "Import a csv") {
        }
        if hinted_btn(ui, "+", "Create a new dataset") {
        }
        if hinted_btn(ui, "D", "Duplicate dataset") {
        }
        if hinted_btn(ui, "-", "Remove current dataset") {
        }
        Ok(())
    }).inner?;

    ui.push_id("ScrollData", |ui| {
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
                                if ui.button("+").clicked() {
                                    let d_old = iapp.prjdata.clone();
                                    let constants = vec![0.0; iapp.prjdata.constants.len()];
                                    let ind_params = iapp.prjdata.models
                                        .iter()
                                        .map(|mdl| vec![project::ModelVariable::new_unknown(); mdl.get_individual_vars(ui).len()])
                                        .collect();
                                    iapp.prjdata.dataset[dkidx].spectra.push(project::Spectrum{ points: vec![], name: "New spec".to_string(), ind_params, constants });
                                    return Err(Action::Unknown(Box::new(d_old)));
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



fn render_plot(ui: &mut Ui, iapp: &mut ImpedimentApp)->Result<(), Action> {
    let plt = egui::plot::Plot::new("plot1")
        .width(ui.available_width())
        .height(ui.available_height()/2.0);
    
    let mut pltshow = vec![];
    let mut pltlines = vec![];
    let freqrange = 0.01..10000.0;
    let extractor = |datapoint: DataPoint| egui::plot::Value::new(datapoint.imp.re, -datapoint.imp.im);

    if let Some((k,s)) = iapp.editor_variables.current_spectrum {
        let spectrum = &mut iapp.prjdata.dataset[k].spectra[s];
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

            let line = egui::plot::Line::new(egui::plot::Values::from_values_iter(points)).stroke((0.5, Color32::WHITE));
            pltlines.push(line);
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
        let names = &iapp.prjdata.constants;

        for (i_cst, (name, val)) in names.iter().zip(consts.iter_mut()).enumerate() {
            let awi = ui.available_width();
            ui.horizontal(|ui|{
                let namelabel = egui::Label::new(name).ui(ui);
                let lblwidth = namelabel.rect.width();

                let cst_id = egui::Id::new(i_cst << 8 | 0b_1111_0000);
                if let Err(old_val) = value_editor(ui, cst_id, focus==Some(cst_id), &mut iapp.edit_buffer, awi-50.0-lblwidth, val) {
                    return Err(Action::EditConst { spec: (grp, spc), cst: i_cst, value: old_val });
                }

                Ok(())
            }).inner?;
        }
    }
    Ok(())
}


fn render_ind_params(ctx: &egui::Context, ui: &mut Ui, iapp: &mut ImpedimentApp)->Result<(), Action> {
    let focus = ctx.memory().focus();

    if let (Some(mdl), Some((grp, spc))) = (iapp.editor_variables.current_circ, iapp.editor_variables.current_spectrum) {
        let vals = &mut iapp.prjdata.dataset[grp].spectra[spc].ind_params[mdl];
        let names = iapp.prjdata.models[mdl].get_individual_vars(ui);

        for (i_ind, (name, val)) in names.into_iter().zip(vals.iter_mut()).enumerate() {
            ui.horizontal(|ui| {
                if ui.selectable_label(val.enabled, "·").clicked() {
                    val.enabled = !val.enabled;
                }
                ui.add(name);

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

                ui.small_button("<").on_hover_ui(|ui| {ui.label("Decrease (Ctrl=fast, Shift=slow)");});
                ui.small_button(">").on_hover_ui(|ui| {ui.label("Increase (Ctrl=fast, Shift=slow)");});

                Ok(())
            }).inner?;
        }
    }

    Ok(())
}


fn render_grp_vars(ctx: &egui::Context, ui: &mut Ui, iapp: &mut ImpedimentApp)->Result<(), Action> {
    let focus = ctx.memory().focus();

    if let (Some(mdl), Some((grp, _))) = (iapp.editor_variables.current_circ, iapp.editor_variables.current_spectrum) {
        let vals = &mut iapp.prjdata.dataset[grp].group_vars[mdl];
        let names = iapp.prjdata.models[mdl].get_group_vars(ui, &iapp.prjdata.constants);

        for (i_grvar, (name, val)) in names.into_iter().zip(vals.iter_mut()).enumerate() {
            ui.horizontal(|ui| {
                if ui.selectable_label(val.enabled, "·").clicked() {
                    val.enabled = !val.enabled;
                }

                ui.add(name);

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

                ui.small_button("<").on_hover_ui(|ui| {ui.label("Decrease (Ctrl=fast, Shift=slow)");});
                ui.small_button(">").on_hover_ui(|ui| {ui.label("Increase (Ctrl=fast, Shift=slow)");});
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



impl eframe::App for ImpedimentApp {
    fn update(&mut self, ctx: &eframe::egui::Context, _frame: &mut eframe::Frame) {
        ctx.set_visuals(egui::Visuals::dark());

        let mut p_action = ||->Result<(), Action> {
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
                ui.label("Dataset constants");
                render_consts(ctx, ui, self)?;

                ui.separator();

                ui.label("Individual parameters");
                render_ind_params(ctx, ui, self)?;

                ui.separator();

                ui.label("Group parameters");
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
