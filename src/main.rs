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

#[allow(non_upper_case_globals)]
static mut g_model : Option<Model> = None;

#[allow(non_upper_case_globals)]
static mut g_experimental : Option<Vec<DataPiece>> = None;

#[allow(non_upper_case_globals)]
const g_blocksize : f64 = 15.0;


/// Paint the g_model on a given DrawingArea
fn draw_main_circuit(widget: &gtk::DrawingArea, context: &cairo::Context){
    context.set_source_rgb(1.0, 1.0, 1.0);
    context.paint();

    match unsafe{&g_model} {
        Some(model) => {
            context.set_source_rgb(0.0, 0.7, 0.4);

            let size = model.circ.painted_size();

            let winalloc = widget.get_allocation();
            let widsize = (winalloc.width, winalloc.height);

            if true {
                let blocksize: f64 = g_blocksize;
                model.circ.paint(&context, blocksize, (
                    (widsize.0 as f64-size.0 as f64 * blocksize)/2., 
                    (widsize.1 as f64-size.1 as f64 * blocksize)/2.)
                );
            }

            context.stroke();
        }
        _ => {panic!();}
    }

    context.stroke();
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

/// Fill the parameter list with a current model's data
fn recreate_param_list(parambox: &gtk::Box, window: &gtk::Window, renew_params: bool) {
    unsafe {
        // Remove all the previous editboxes
        for w in parambox.get_children() {
            parambox.remove(&w);
        }

        let _model = g_model.as_mut().unwrap();
        if renew_params {
            _model.params = ParameterDesc::new(&(_model.circ).paramlist());
        }

        // Create the name label, the bounds editboxes and the value editboxes for each element
        for i in 0.._model.params.vals.len() {
            let single_param = gtk::Box::new(gtk::Orientation::Horizontal, 5);

            let lbl = gtk::Label::new(Some(&i.to_string()));
            let ebounds_min = gtk::Entry::new();
            let ebounds_max = gtk::Entry::new();
            let evalue = gtk::Entry::new();

            ebounds_min.set_text(&_model.params.bounds[i].0.to_string());
            ebounds_max.set_text(&_model.params.bounds[i].1.to_string());
            evalue.set_text(&_model.params.vals[i].to_string());

            ebounds_min.set_width_chars(6);
            ebounds_max.set_width_chars(6);

            single_param.pack_start(&lbl, /*expand*/false, /*fill*/false, /*padding*/0);
            single_param.pack_start(&ebounds_min, /*expand*/true, /*fill*/true, /*padding*/0);
            single_param.pack_start(&ebounds_max, /*expand*/true, /*fill*/true, /*padding*/0);
            single_param.pack_start(&evalue, /*expand*/true, /*fill*/true, /*padding*/5);

            let w1 = window.clone();
            let w2 = window.clone();
            let w3 = window.clone();
            ebounds_min.connect_changed(move |emin| {edit_value(&mut g_model.as_mut().unwrap().params.bounds[i].0, emin, &w1 )});
            ebounds_max.connect_changed(move |emax| {edit_value(&mut g_model.as_mut().unwrap().params.bounds[i].1, emax, &w2 )});
            evalue.connect_changed(move |ev| {edit_value(&mut g_model.as_mut().unwrap().params.vals[i], ev, &w3)});

            parambox.pack_start(&single_param, /*expand*/false, /*fill*/false, /*padding*/0);
        }

        // Redraw the window
        parambox.show_all();
    }
}

/// Get the coordinates of drawing block given the coordinates of a point in a `gtk::DrawingArea`
fn block_by_coords(model: &Model, wid: &gtk::DrawingArea, event: &gdk::EventButton) 
    -> Option<(u16, u16)>
{
    // The circuit is located at the center of the DrawingArea

    // Get the canvas size
    let winalloc = wid.get_allocation();
    let (wx, wy) = (winalloc.width as f64, winalloc.height as f64);

    // Get the circuit size
    let (i_sx, i_sy) = model.circ.painted_size();
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
fn build_circuit_editor(builder: &gtk::Builder) -> impl Fn(&mut Model, u16, u16) -> () {
    let rb_edit_replace: gtk::RadioButton = builder.get_object("rb_edit_replace").unwrap();
    let rb_edit_series: gtk::RadioButton = builder.get_object("rb_edit_series").unwrap();
    let rb_edit_parallel: gtk::RadioButton = builder.get_object("rb_edit_parallel").unwrap();
    let rb_edit_remove: gtk::RadioButton = builder.get_object("rb_edit_remove").unwrap();

    let create_user_element = build_element_selector(&builder);
    
    let perform_user_edit = move |model: &mut Model, x, y| {
        if rb_edit_replace.get_active() {
            if let Some(el) = model.circ.replace((x, y), create_user_element()) {
                model.circ = el;
            };
        }
        else if rb_edit_series.get_active() {
            if let Some(el) = model.circ.add_series((x, y), create_user_element()) {
                model.circ = Circuit::Series(vec![model.circ.clone(), el]);
            }
        }
        else if rb_edit_parallel.get_active() {
            if let Some(el) = model.circ.add_parallel((x, y), create_user_element()) {
                model.circ = Circuit::Parallel(vec![model.circ.clone(), el]);
            }
        }
        else if rb_edit_remove.get_active() {
            match model.circ.remove((x, y)) {
                RemoveAction::DoNothing => {}
                RemoveAction::ChangeTo(el) => {model.circ = el;}
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
fn draw_graph<T>(widget: &gtk::DrawingArea, context: &cairo::Context) -> Inhibit
    where T: plotting::DataExtractor
{
    let sz = widget.get_allocation();

    context.set_source_rgb(1.0, 1.0, 1.0);
    context.paint();
    unsafe{
        plotting::plot_model::<T>(
            g_model.as_ref().unwrap(),
            context,
            V2{x: sz.width as f64, y: sz.height as f64}, 
            None,
            match &g_experimental {
                Some(ex) => {Some(&ex)} 
                None => None
            }
        );
    }
    Inhibit(false)
}

/// Loss function for circuit parameter fitting
/// 
/// Compute the difference between the experimental data
/// and model output, given a set of `params` to the model
fn loss(params: &[f64], gradient_out: Option<&mut [f64]>) -> f64
{
    let mut gradient_out = gradient_out;
    let circ = unsafe{&g_model.as_ref().unwrap().circ};
    let _e = unsafe{ g_experimental.as_ref() };

    if let Some(gout) = &mut gradient_out {
        for i in 0..gout.len() {
            gout[i] = 0.0;
        }
    };

    if let Some(exps) = _e { 
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
                    //let ml = (point.imp-model_imp)*dmdx.conj() + (point.imp.conj()-model_imp.conj())*dmdx;
                    gout[i] += -1.0 / point.imp.norm_sqr() * ml.re;
                }
            };
        }

        loss
    }
    else {0.0}
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


fn main() {
    let circ = Circuit::Series(vec![
        Circuit::Parallel(vec![
            Circuit::Series(vec![
                Circuit::Element(Element::Resistor),
                Circuit::Element(Element::Warburg),
            ]),
            Circuit::Element(Element::Capacitor),
        ]),
        Circuit::Element(Element::Resistor),
    ]);

    let params = ParameterDesc::new(&circ.paramlist());

    unsafe {g_model = Some(Model{circ, params});}

    let app = gtk::Application::new(Some("app.impediment"), Default::default()).expect("GTK failed");

    app.connect_activate(|app| {
        let builder = gtk::Builder::new_from_string(include_str!("impedui.glade"));

        let main_window: gtk::Window = builder.get_object("main_window").unwrap();
        main_window.set_application(Some(app));

        let perform_user_edit = build_circuit_editor(&builder);

        let graph: gtk::DrawingArea = builder.get_object("graphCircuit").unwrap();
        graph.connect_draw(|widget, context| {
            draw_main_circuit(&widget, &context);
            Inhibit(false)
        });

        graph.add_events(gdk::EventMask::BUTTON_PRESS_MASK);

        let cpbox : gtk::Box = builder.get_object("boxParams").unwrap();
        recreate_param_list(&cpbox, &main_window, true);

        let main_window_bybutton = main_window.clone();
        let cpbox_bpress = cpbox.clone();
        graph.connect_button_press_event(move |wid, event: &gdk::EventButton| {
            let model = unsafe{g_model.as_mut().unwrap()};

            if let Some((x,y)) = block_by_coords(&model, &wid, &event) {
                perform_user_edit(model, x,y);
                recreate_param_list(&cpbox_bpress, &main_window_bybutton, true);
                main_window_bybutton.queue_draw();
            }

            Inhibit(false)
        });

        builder.get_object::<gtk::DrawingArea>("graph1")
                .unwrap()
                .connect_draw(draw_graph::<plotting::BodePhaseExtractor>);

        builder.get_object::<gtk::DrawingArea>("graph2")
                .unwrap()
                .connect_draw(draw_graph::<plotting::BodeAmpExtractor>);

        builder.get_object::<gtk::DrawingArea>("graphR")
                .unwrap()
                .connect_draw(draw_graph::<plotting::NiquistExtractor>);

        let main_window_byfit = main_window.clone();

        let cb_method = builder.get_object::<gtk::ComboBox>("cbMethod").unwrap();

        cb_method.set_active(Some(0));

        let main_window_save = main_window.clone();
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

        let main_window_open = main_window.clone();
        builder.get_object::<gtk::Button>("b_open_data")
            .unwrap()
            .connect_clicked(move |_btn| {
                match user_filename(false, "Open data file", &main_window_open) {
                    Some(filename) => {
                        match file::load_csv_freq_re_im(&filename) {
                            Ok(data) => { unsafe{g_experimental = Some(data);} }
                            Err(err) => { println!("{}", err);}
                        }
                    }
                    None => {}                    
                }
            });

        let cpbox_openmodel = cpbox.clone();
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
            });

        builder.get_object::<gtk::Button>("bFit")
            .unwrap()
            .connect_clicked(move |_btn| {
                let model = unsafe{g_model.as_mut().unwrap()};

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


                if let Some(algo) = algo {
                    let mut opt = nlopt::Nlopt::new(
                        algo,
                        model.params.vals.len(), 
                        |x: &[f64], grad: Option<&mut [f64]>, _: &mut ()| loss(x, grad), 
                        nlopt::Target::Minimize, 
                        ());

                    opt.set_lower_bounds(&model.params.bounds.iter().map(|x| x.0).collect::<Vec<f64>>()).unwrap();
                    opt.set_upper_bounds(&model.params.bounds.iter().map(|x| x.1).collect::<Vec<f64>>()).unwrap();
                    opt.set_maxeval((-1_i32) as u32).unwrap();
                    opt.set_maxtime(10.0).unwrap();

                    opt.set_xtol_rel(1e-10).unwrap();

                    let optresult = opt.optimize(&mut model.params.vals);
                    println!("{:?}", optresult);
                    println!("{:?}", loss(&model.params.vals, None));

                    recreate_param_list(&cpbox, &main_window_byfit, false);
                    main_window_byfit.queue_draw();
                }
                else {
                    println!("Optimization algorithm failed to be chosen")
                }
            });

        main_window.show_all();
    });

    app.run(&["Impediment".to_string()]);
}
