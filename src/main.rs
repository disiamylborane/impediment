#![allow(dead_code)]

extern crate nlopt;

extern crate cairo;
extern crate gio;
extern crate gtk;
extern crate gdk;

use gio::prelude::*;
use gtk::prelude::*;
use gdk::RGBA;

mod circuit;
use circuit::*;

mod plotting;

mod imped_math;
use imped_math::*;

mod file;

#[derive(Debug,Clone)]
struct ParameterDescHistory(Vec<ParameterDesc>, usize);

impl ParameterDescHistory {
    fn append(&mut self, pd: ParameterDesc) {
        self.0.truncate(self.1 + 1);
        self.0.push(pd);
        self.1 += 1;
    }

    fn back(&mut self) {
        if self.1 > 0 {
            self.1 -= 1;
        }
    }

    fn forward(&mut self) {
        if self.1 < self.0.len() {
            self.1 += 1;
        }
    }

    fn current<'a>(&'a mut self) -> &'a mut ParameterDesc {
        &mut self.0[self.1]
    }
}

#[derive(Debug)]
struct ImpedimentData {
    pub circs : Vec<(Circuit, String, bool)>,  // Always at least 1
    pub datasets : Vec<(Dataset, String)>,  // Always at least 1

    // For each in 2D structure [Circuit][Data]:
    pub params: Vec< Vec<ParameterDescHistory> >,

    pub current_circuit: usize,
    pub current_dataset: usize,
}

impl ImpedimentData {
    fn current_datastruct<'a>(&'a mut self) -> (&'a mut Circuit, &'a mut ParameterDesc, &'a mut Dataset) {
        let circ = &mut self.circs[self.current_circuit].0;
        let ph = &mut self.params[self.current_circuit][self.current_dataset];
        let ds = &mut self.datasets[self.current_dataset].0;

        (circ, ph.current(), ds)
    }

    fn curr_circuit_editable(&self) -> bool {
        self.circs[self.current_circuit].2
    }

    fn set_curr_circuit_editable(&mut self, ed: bool) {
        self.circs[self.current_circuit].2 = ed;
    }

    fn current_circuit(&mut self) -> &mut Circuit {
        &mut self.circs[self.current_circuit].0
    }

    fn current_dataset(&mut self) -> &mut Dataset {
        &mut self.datasets[self.current_dataset].0
    }

    fn add_dataset(&mut self, dataset: Dataset, name: String) {
        self.datasets.push((dataset, name));

        for (v,c) in self.params.iter_mut() .zip (self.circs.iter()) {
            v.push(ParameterDescHistory(vec![ParameterDesc::new(&c.0.paramlist())], 0));
        }

        self.current_dataset = self.datasets.len() - 1;
    }

    fn current_paramhistory(&mut self) -> &mut ParameterDescHistory {
        &mut self.params[self.current_circuit][self.current_dataset]
    }

    fn current_paramlist(&mut self) -> &mut ParameterDesc {
        self.current_paramhistory().current()
    }

    fn undo(&mut self){
        self.current_paramhistory().back()
    }
    fn redo(&mut self){
        self.current_paramhistory().forward()
    }
}

pub type Dataset = Vec<DataPiece>;


#[allow(non_upper_case_globals)]
const g_blocksize : f64 = 15.0;


/// Paint the g_model on a given DrawingArea
fn draw_main_circuit<'a>(widget: &gtk::DrawingArea, context: &cairo::Context){
    let ida = unsafe{G_DATA.as_mut().unwrap()};

    context.set_source_rgb(1.0, 1.0, 1.0);
    context.paint();

    if ida.curr_circuit_editable()
        {context.set_source_rgb(0.0, 0.7, 0.4);}
    else
        {context.set_source_rgb(0.0, 0.0, 0.0);}

    let circuit = ida.current_circuit();

    let size = circuit.painted_size();

    let winalloc = widget.get_allocation();
    let widsize = (winalloc.width, winalloc.height);

    if true {
        let blocksize: f64 = g_blocksize;
                circuit.paint(&context, blocksize, (
            (widsize.0 as f64-size.0 as f64 * blocksize)/2., 
            (widsize.1 as f64-size.1 as f64 * blocksize)/2.)
        );
    }

    context.stroke();
}

fn remake_circuit_combobox(ida:  &ImpedimentData, cb: &gtk::ComboBoxText) {
    cb.remove_all();

    for v in ida.circs.iter() {
        cb.append_text(&v.1);
    }
    cb.set_active(Some(ida.current_circuit as u32));
}

fn remake_dataset_combobox(ida: &ImpedimentData, cb: &gtk::ComboBoxText) {
    cb.remove_all();

    for v in ida.datasets.iter() {
        cb.append_text(&v.1);
    }
    cb.set_active(Some(ida.current_dataset as u32));
}

/// Get a numeric value from the gtk::Entry
fn entryval<T: std::str::FromStr>(entry: &gtk::Entry) -> Result<T, T::Err> {
    entry.get_text().unwrap().as_str().parse::<T>()
}

/// The [on change] event for a `gtk::Entry`
/// editing a f64 parameter
fn edit_value<T: std::str::FromStr>(val: &mut T, entry: &gtk::Entry, window: &gtk::Window) {
    match entryval::<T>(&entry) {
        Ok(b_val) => {
            *val = b_val;
            entry.override_color(gtk::StateFlags::NORMAL, None);
        }
        Err(_) => {
            entry.override_color(gtk::StateFlags::NORMAL, Some(&RGBA::red()));
        }
    }
    window.queue_draw();
}

fn remake_circuit_param_lists(ida: &mut ImpedimentData)
{
    let circ = ida.current_circuit();
    let list = circ.paramlist();

    for clist in ida.params[ida.current_circuit].iter_mut() {
        *clist = ParameterDescHistory(vec![ParameterDesc::new(&list)], 0);
    }
}

/// Fill the parameter list with a current model's data
fn redraw_param_list(parambox: &gtk::Box, window: &gtk::Window) {
    // Remove all the previous editboxes
    for w in parambox.get_children() {
        parambox.remove(&w);
    }

    let params = ida().current_paramlist();
    {
        // Create the name label, the bounds editboxes and the value editboxes for each element
        for i in 0..params.vals.len() {
            let single_param = gtk::Box::new(gtk::Orientation::Horizontal, 5);

            let lbl = gtk::Label::new(Some(&i.to_string()));
            let ebounds_min = gtk::Entry::new();
            let ebounds_max = gtk::Entry::new();
            let evalue = gtk::Entry::new();

            ebounds_min.set_text(&params.bounds[i].0.to_string());
            ebounds_max.set_text(&params.bounds[i].1.to_string());
            evalue.set_text(&params.vals[i].to_string());

            ebounds_min.set_width_chars(6);
            ebounds_max.set_width_chars(6);

            single_param.pack_start(&lbl, /*expand*/false, /*fill*/false, /*padding*/0);
            single_param.pack_start(&ebounds_min, /*expand*/true, /*fill*/true, /*padding*/0);
            single_param.pack_start(&ebounds_max, /*expand*/true, /*fill*/true, /*padding*/0);
            single_param.pack_start(&evalue, /*expand*/true, /*fill*/true, /*padding*/5);

            let w1 = window.clone();
            let w2 = window.clone();
            let w3 = window.clone();

            ebounds_min.connect_changed(move |emin| {
                ida().set_curr_circuit_editable(false);
                edit_value( &mut ida().current_paramlist().bounds[i].0, emin, &w1 )
            });
            ebounds_max.connect_changed(move |emax| {
                ida().set_curr_circuit_editable(false);
                edit_value( &mut ida().current_paramlist().bounds[i].1, emax, &w2 )
            });
            evalue.connect_changed(move |ev| {
                ida().set_curr_circuit_editable(false);
                edit_value( &mut ida().current_paramlist().vals[i], ev, &w3 )
            });

            parambox.pack_start(&single_param, /*expand*/false, /*fill*/false, /*padding*/0);
        }

    }

    // Redraw the window
    parambox.show_all();
}

/// Get the coordinates of drawing block given the coordinates of a point in a `gtk::DrawingArea`
fn block_by_coords(circuit: &Circuit, wid: &gtk::DrawingArea, event: &gdk::EventButton) 
    -> Option<(u16, u16)>
{
    // The circuit is located at the center of the DrawingArea

    // Get the canvas size
    let winalloc = wid.get_allocation();
    let (wx, wy) = (winalloc.width as f64, winalloc.height as f64);

    // Get the circuit size
    let (i_sx, i_sy) = circuit.painted_size();
    let (sx,sy) = (i_sx as f64 * g_blocksize, i_sy as f64 * g_blocksize);

    // Get the cursor pos
    let (xpos, ypos) = event.get_position();

    // Recalc (cursor vs canvas) => (cursor vs circuit)
    let (xcirc, ycirc) = (xpos - (wx-sx)/2., ypos-(wy-sy)/2.);
    if xcirc < 0. || ycirc < 0. {return None;}

    let (x,y) = (xcirc / g_blocksize, ycirc / g_blocksize);

    Some((x as u16, y as u16))
}

/// Build a function that returns the circuit element chosen by the user
/// 
/// [Resistor, Inductor or CPE?]
fn build_element_selector(builder: &gtk::Builder) -> impl Fn() -> Circuit {
    let rb_element_r: gtk::RadioButton = builder.get_object("rb_element_r").unwrap();
    let rb_element_c: gtk::RadioButton = builder.get_object("rb_element_c").unwrap();
    let rb_element_w: gtk::RadioButton = builder.get_object("rb_element_w").unwrap();
    let rb_element_l: gtk::RadioButton = builder.get_object("rb_element_l").unwrap();
    let rb_element_q: gtk::RadioButton = builder.get_object("rb_element_q").unwrap();

    let create_user_element = move || -> Circuit {
        if rb_element_r.get_active() {
            Circuit::Element(Element::Resistor)
        }
        else if rb_element_c.get_active() {
            Circuit::Element(Element::Capacitor)
        }
        else if rb_element_w.get_active() {
            Circuit::Element(Element::Warburg)
        }
        else if rb_element_l.get_active() {
            Circuit::Element(Element::Inductor)
        }
        else if rb_element_q.get_active() {
            Circuit::Element(Element::CPE)
        }
        else {
            panic!();
        }
    };

    return create_user_element;
}

/// Build a function that performs the circuit editing operation chosen by the user
/// 
/// [Replace, Remove, Add series or Add parallel?]
fn build_circuit_editor(builder: &gtk::Builder) -> impl Fn(&mut Circuit, u16, u16) -> () {
    let rb_edit_replace: gtk::RadioButton = builder.get_object("rb_edit_replace").unwrap();
    let rb_edit_series: gtk::RadioButton = builder.get_object("rb_edit_series").unwrap();
    let rb_edit_parallel: gtk::RadioButton = builder.get_object("rb_edit_parallel").unwrap();
    let rb_edit_remove: gtk::RadioButton = builder.get_object("rb_edit_remove").unwrap();

    let create_user_element = build_element_selector(&builder);
    
    let perform_user_edit = move |circ: &mut Circuit, x, y| {
        if rb_edit_replace.get_active() {
            if let Some(el) = circ.replace((x, y), create_user_element()) {
                *circ = el;
            };
        }
        else if rb_edit_series.get_active() {
            if let Some(el) = circ.add_series((x, y), create_user_element()) {
                *circ = Circuit::Series(vec![circ.clone(), el]);
            }
        }
        else if rb_edit_parallel.get_active() {
            if let Some(el) = circ.add_parallel((x, y), create_user_element()) {
                *circ = Circuit::Parallel(vec![circ.clone(), el]);
            }
        }
        else if rb_edit_remove.get_active() {
            match circ.remove((x, y)) {
                RemoveAction::DoNothing => {}
                RemoveAction::ChangeTo(el) => {*circ = el;}
                RemoveAction::Remove => {}
            }
        }
        else {
            panic!();
        }
    };

    return perform_user_edit;
}

/// Plot the experimental and theoretical data for a single DrawingArea
fn draw_graph<T>(ida: &mut ImpedimentData, widget: &gtk::DrawingArea, context: &cairo::Context) -> Inhibit
    where T: plotting::DataExtractor
{
    let sz = widget.get_allocation();

    let (circ, params, dataset) = ida.current_datastruct();

    //let dataset = match dataset{Some(x)=>Some(&x[..]), None=>None};

    let dataset = if dataset.len() == 0 {
        None
    }
    else {
        Some(dataset.as_slice())
    };

    context.set_source_rgb(1.0, 1.0, 1.0);
    context.paint();
    plotting::plot_model::<T>(
        Model{circ, params},
        context,
        V2{x: sz.width as f64, y: sz.height as f64},
        None,
        dataset
    );

    Inhibit(false)
}

/// Loss function for circuit parameter fitting
/// 
/// Compute the difference between the experimental data
/// and model output, given a set of `params` to the model
fn loss(circ: &Circuit, data: &Dataset, params: &[f64], gradient_out: Option<&mut [f64]>) -> f64
{
    let mut gradient_out = gradient_out;
    //let circ = ida.current_circuit();
    let exps = data;

    if let Some(gout) = &mut gradient_out {
        for i in 0..gout.len() {
            gout[i] = 0.0;
        }
    };

    let mut loss = 0.0_f64;
    for point in exps {
        let model_imp = circ.impedance(point.omega, &params);
        let diff = model_imp - point.imp;
        loss += diff.norm_sqr() / point.imp.norm_sqr();

        if let Some(gout) = &mut gradient_out {
            for i in 0..gout.len() {
                let dmdx = circ._d_impedance(point.omega, &params, i);
                // (a*)b + a(b*)  = [(a*)b] + [(a*)b]* = 2*re[(a*)b]
                let ml = (point.imp-model_imp)*dmdx.conj().re * 2.0;
                gout[i] += -1.0 / point.imp.norm_sqr() * ml.re;
            }
        };
    }

    loss
}


fn user_filename<'a>(do_save: bool, title: &str, parent_window: &gtk::Window) -> Option<String>
{
    let dialog = gtk::FileChooserDialog::new(
        Some(title), 
        Some(parent_window), 
        if do_save {gtk::FileChooserAction::Save} else {gtk::FileChooserAction::Open} 
    );

    dialog.add_button(if do_save {"Save"} else {"Open"}, gtk::ResponseType::Ok.into());
    dialog.add_button("Cancel", gtk::ResponseType::Cancel.into());
    let runres = dialog.run();
    let filename = dialog.get_filename();
    dialog.destroy();
    if runres == gtk::ResponseType::Ok {
        if let Some(filename) = filename {
            if let Some(filename) = filename.to_str() {
                return Some(filename.to_string());
            }
        }
    }

    None
}

static mut G_DATA: Option<ImpedimentData> = None;

fn ida()->&'static mut ImpedimentData {unsafe{G_DATA.as_mut().unwrap()}}

fn main() {
    unsafe {
        let circuit = Circuit::Series(vec![
                Circuit::Parallel(vec![
                    Circuit::Series(vec![
                        Circuit::Element(Element::Resistor),
                        Circuit::Element(Element::Warburg),
                    ]),
                    Circuit::Element(Element::Capacitor),
                ]),
                Circuit::Element(Element::Resistor),
            ]);
        let desc = ParameterDesc::new(&circuit.paramlist());
        
        G_DATA = Some(ImpedimentData {
            circs : vec![(circuit, "Model1".to_string(), true)],

            datasets : vec![(vec![], "Empty".to_string())],

            params: vec![vec![ParameterDescHistory(vec![desc], 0)]],

            current_circuit: 0,
            current_dataset: 0,
        });
    }

    let app = gtk::Application::new(Some("app.impediment"), Default::default()).expect("GTK failed");

    app.connect_activate(|app| {

        let builder = gtk::Builder::new_from_string(include_str!("impedui.glade"));

        let main_window: gtk::Window = builder.get_object("main_window").unwrap();
        main_window.set_application(Some(app));

        let cb_datasets = builder.get_object::<gtk::ComboBoxText>("cb_datasets").unwrap();
        let cb_circuits = builder.get_object::<gtk::ComboBoxText>("cb_circuits").unwrap();

        remake_circuit_combobox(ida(), &cb_circuits);
        remake_dataset_combobox(ida(), &cb_datasets);

        let perform_user_edit = build_circuit_editor(&builder);

        let graph: gtk::DrawingArea = builder.get_object("graphCircuit").unwrap();
        graph.connect_draw(|widget, context| {
            draw_main_circuit(&widget, &context);
            Inhibit(false)
        });

        graph.add_events(gdk::EventMask::BUTTON_PRESS_MASK);

        let cpbox : gtk::Box = builder.get_object("boxParams").unwrap();
        redraw_param_list(&cpbox, &main_window);

        let main_window_bybutton = main_window.clone();
        let cpbox_bpress = cpbox.clone();
        graph.connect_button_press_event(move |wid, event: &gdk::EventButton| {
            if ida().curr_circuit_editable() {
                let block = block_by_coords(ida().current_circuit(), &wid, &event);

                if let Some((x,y)) = block {
                    perform_user_edit(ida().current_circuit(), x,y);
                    remake_circuit_param_lists(ida());
                    redraw_param_list(&cpbox_bpress, &main_window_bybutton);
                    main_window_bybutton.queue_draw();
                }
            }

            Inhibit(false)
        });

        builder.get_object::<gtk::DrawingArea>("graph1")
                .unwrap()
                .connect_draw(|w,c|draw_graph::<plotting::BodePhaseExtractor>(ida(),w,c));

        builder.get_object::<gtk::DrawingArea>("graph2")
                .unwrap()
                .connect_draw(|w,c|draw_graph::<plotting::BodeAmpExtractor>(ida(),w,c));

        builder.get_object::<gtk::DrawingArea>("graphR")
                .unwrap()
                .connect_draw(|w,c|draw_graph::<plotting::NyquistExtractor>(ida(),w,c));

        let main_window_byfit = main_window.clone();

        let cb_method = builder.get_object::<gtk::ComboBox>("cbMethod").unwrap();

        cb_method.set_active(Some(0));

        /*let main_window_save = main_window.clone();
        builder.get_object::<gtk::Button>("b_save_model")
            .unwrap()
            .connect_clicked(move |_btn| {
                match user_filename(true, "Save model", &main_window_save) {
                    Some(filename) => {
                        file::save_model(unsafe{g_model.as_ref().unwrap()}, &filename).unwrap();
                    }

                    None => {}
                }
            });
        */

        let main_window_open = main_window.clone();
        let builder_loadfile_ref = builder.clone();
        let cb_datasets_loadfile = cb_datasets.clone();
        builder.get_object::<gtk::Button>("b_open_data")
            .unwrap()
            .connect_clicked(move |_btn| {
                match user_filename(false, "Open data file", &main_window_open) {
                    Some(filename) => {
                        match file::load_csv(&filename, &builder_loadfile_ref, &main_window_open) {
                            Ok(data) => { ida().add_dataset(data, filename); }
                            Err(err) => { println!("{}", err);}
                        }
                        remake_dataset_combobox(ida(), &cb_datasets_loadfile);
                    }
                    None => {}                    
                }
            });

        let main_window_b_addcirc = main_window.clone();
        let cb_circuits_addcirc = cb_circuits.clone();
        builder.get_object::<gtk::Button>("b_add_model")
            .unwrap()
            .connect_clicked(move |_btn| {
                let ida = ida();

                let circ = Circuit::Element(Element::Resistor);
                let vparams = ParameterDescHistory(vec![ParameterDesc::new(&circ.paramlist())], 0);

                ida.circs.push((circ, "Circuit".to_string(), true));
                ida.params.push(vec![vparams; ida.datasets.len()]);
                ida.current_circuit = ida.circs.len() - 1;

                remake_circuit_combobox(&ida, &cb_circuits_addcirc);
                main_window_b_addcirc.queue_draw();
            });
        
        let main_window_cb_datasets = main_window.clone();
        let cpbox_cb_datasets = cpbox.clone();
        cb_datasets.connect_changed(move |cb|{
            match cb.get_active() {
                Some(v) => {ida().current_dataset = v as usize}
                None => {}
            }
            redraw_param_list(&cpbox_cb_datasets, &main_window_cb_datasets);
            main_window_cb_datasets.queue_draw();
        });

        let main_window_cb_circuits = main_window.clone();
        let cpbox_cb_circuits = cpbox.clone();
        cb_circuits.connect_changed(move |cb|{
            match cb.get_active() {
                Some(v) => {ida().current_circuit = v as usize}
                None => {}
            }
            redraw_param_list(&cpbox_cb_circuits, &main_window_cb_circuits);
            main_window_cb_circuits.queue_draw();
        });

        /*let cpbox_openmodel = cpbox.clone();
        let main_window_openmodel = main_window.clone();
        builder.get_object::<gtk::Button>("b_open_model")
            .unwrap()
            .connect_clicked(move |_btn| {
                match user_filename(false, "Open model", &main_window_openmodel) {
                    Some(filename) => {
                        match file::load_model(&filename, unsafe{g_model.as_mut().unwrap()}) {
                            Ok(_) => { }
                            Err(_) => { println!("Cannot load model from {}", filename);}
                        }
                    }
                    None => {}                    
                }

                recreate_param_list(&cpbox_openmodel, &main_window_openmodel, false);
                main_window_openmodel.queue_draw();
            });*/

        builder.get_object::<gtk::Button>("bFit")
            .unwrap()
            .connect_clicked(move |_btn| {
                let algo = match cb_method.get_active() {
                    Some(algoindex)  => {
                        match algoindex{
                            0 => Some(nlopt::Algorithm::Bobyqa),
                            1 => Some(nlopt::Algorithm::TNewton),
                            2 => Some(nlopt::Algorithm::Slsqp),
                            3 => Some(nlopt::Algorithm::Lbfgs),
                            _ => None,
                        }
                    }
                    None => None
                };

                ida().set_curr_circuit_editable(false);

                let (circuit, paramlist,dataset) = ida().current_datastruct();
                {
                    if let Some(algo) = algo {
                        let mut opt = nlopt::Nlopt::new(
                            algo,
                            paramlist.vals.len(), 
                            |x: &[f64], grad: Option<&mut [f64]>, _: &mut ()| loss(circuit, dataset, x, grad), 
                            nlopt::Target::Minimize, 
                            ());

                        opt.set_lower_bounds(&paramlist.bounds.iter().map(|x| x.0).collect::<Vec<f64>>()).unwrap();
                        opt.set_upper_bounds(&paramlist.bounds.iter().map(|x| x.1).collect::<Vec<f64>>()).unwrap();
                        opt.set_maxeval((-1_i32) as u32).unwrap();
                        opt.set_maxtime(10.0).unwrap();

                        opt.set_xtol_rel(1e-10).unwrap();

                        let optresult = opt.optimize(&mut paramlist.vals);
                        println!("{:?}", optresult);
                        println!("{:?}", loss(circuit, dataset, &paramlist.vals, None));

                        redraw_param_list(&cpbox, &main_window_byfit);
                        main_window_byfit.queue_draw();
                    }
                    else {
                        println!("Optimization algorithm failed to be chosen")
                    }
                }
            });

        main_window.show_all();
    });

    app.run(&["Impediment".to_string()]);
}
