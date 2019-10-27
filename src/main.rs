// An example: 
// The series RC circuit has an impedance of 100-3i Ohm at angular frequency 100/s
// Use the BOBYQA algorithm from Nlopt to fit the circuit parameters

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

#[derive(Debug)]
struct ParameterDesc {
    vals : Vec<f64>,
    bounds : Vec<(f64, f64)>
}

#[derive(Debug)]
struct Model {
    circ : Box<dyn Circuit>,
    params : ParameterDesc
}



#[allow(non_upper_case_globals)]
static mut g_model : Option<Model> = None;

#[allow(non_upper_case_globals)]
const g_blocksize : f64 = 20.0;



impl ParameterDesc{
    fn new(paramlist: &[ParameterBase]) -> Self {
        let vals = paramlist.iter().map(|x| x.default).collect::<Vec<_>>();;
        let bounds = paramlist.iter().map(|x| x.limits).collect::<Vec<_>>();;

        ParameterDesc{vals, bounds}
    }
}

fn draw(widget: &gtk::DrawingArea, context: &cairo::Context){
    unsafe {
        context.set_source_rgb(1.0, 1.0, 1.0);
        context.paint();

        match &g_model {
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

                for i in 0..model.params.vals.len() {
                    context.move_to(g_blocksize, 70.+(i as f64)*g_blocksize);
                    context.show_text(&model.params.vals[i].to_string());
                }

                context.stroke();
            }
            _ => {panic!();}
        }
    }

    context.stroke();
}

fn entryval<T: std::str::FromStr>(entry: &gtk::Entry) -> Result<T, T::Err> {
    entry.get_text().unwrap().as_str().parse::<T>()
}

fn remake_param_list(parambox: &gtk::Box, graph: &gtk::DrawingArea) {
    unsafe {
        for w in parambox.get_children() {
            parambox.remove(&w);
        }

        let _model = g_model.as_mut().unwrap();
        println!("..creating boxes..{:?}", _model.circ);
        _model.params = ParameterDesc::new(&(_model.circ).paramlist());
        let params = &mut _model.params;

        println!("Parameter len: {}", params.vals.len());

        for i in 0..params.vals.len() {
            let single_param = gtk::Box::new(gtk::Orientation::Horizontal, 5);

            println!("Creating components ...iteration {}", i);
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
            single_param.pack_start(&evalue, /*expand*/true, /*fill*/true, /*padding*/0);

            let g = graph.clone();
            ebounds_min.connect_changed(move |emin| {
                match entryval::<f64>(&emin) {
                    Ok(b_min) => {
                        g_model.as_mut().unwrap().params.bounds[i].0 = b_min;
                        emin.override_color(gtk::StateFlags::NORMAL, None);
                    }
                    Err(_) => {
                        emin.override_color(gtk::StateFlags::NORMAL, Some(&RGBA::red()));
                    }
                }
            });
            ebounds_max.connect_changed(move |emax| {
                match entryval::<f64>(&emax) {
                    Ok(b_max) => {
                        g_model.as_mut().unwrap().params.bounds[i].1 = b_max;
                        emax.override_color(gtk::StateFlags::NORMAL, None);
                    }
                    Err(_) => {
                        emax.override_color(gtk::StateFlags::NORMAL, Some(&RGBA::red()));
                    }
                }
            });
            evalue.connect_changed(move |ev| {
                match entryval::<f64>(&ev) {
                    Ok(val) => {
                        g_model.as_mut().unwrap().params.vals[i] = val;
                        g.queue_draw();
                        ev.override_color(gtk::StateFlags::NORMAL, None);
                    }
                    Err(_) => {
                        ev.override_color(gtk::StateFlags::NORMAL, Some(&RGBA::red()));
                    }
                }
            });

            parambox.pack_start(&single_param, /*expand*/false, /*fill*/false, /*padding*/0);
        }

        parambox.show_all();
    }
}

fn block_by_coords(model: &Model, wid: &gtk::DrawingArea, event: &gdk::EventButton) 
    -> Option<(u16, u16)>
{
    let winalloc = wid.get_allocation();
    let (wx, wy) = (winalloc.width as f64, winalloc.height as f64);
    let (xpos, ypos) = event.get_position();
    let (i_sx, i_sy) = model.circ.painted_size();

    let (sx,sy) = (i_sx as f64 * g_blocksize, i_sy as f64 * g_blocksize);

    let (xcirc, ycirc) = (xpos - (wx-sx)/2., ypos-(wy-sy)/2.);
    if xcirc < 0. || ycirc < 0. {return None;}

    let (x,y) = (xcirc / g_blocksize, ycirc / g_blocksize);

    Some((x as u16, y as u16))
}

fn main() {
    let circ : Box<dyn Circuit> =  
        Box::new(Series{elems: ComplexCirc::new(vec![
            Box::new(Inductor{}),
            Box::new(Parallel{elems: ComplexCirc::new(vec![
                Box::new(CPE{}),
                Box::new(Series{elems: ComplexCirc::new(vec![
                    Box::new(Resistor{}),
                    Box::new(Warburg{}),
                ])}),
            ])}),
        ])});

    let params = ParameterDesc::new(&circ.paramlist());

    unsafe {
        
        g_model = Some(Model{circ, params});

        let app = gtk::Application::new(Some("app.impediment"), Default::default()).expect("GTK failed");

        app.connect_activate(|app| {
            let builder = gtk::Builder::new_from_string(include_str!("impedui.glade"));

            let main_window: gtk::Window = builder.get_object("main_window").unwrap();
            main_window.set_application(Some(app));

            let rb_element_r: gtk::RadioButton = builder.get_object("rb_element_r").unwrap();
            let rb_element_c: gtk::RadioButton = builder.get_object("rb_element_c").unwrap();
            let rb_element_w: gtk::RadioButton = builder.get_object("rb_element_w").unwrap();
            let rb_element_l: gtk::RadioButton = builder.get_object("rb_element_l").unwrap();
            let rb_element_q: gtk::RadioButton = builder.get_object("rb_element_q").unwrap();
            let create_user_element = move || -> Box<dyn Circuit> {
                if rb_element_r.get_active() {
                    Box::new(Resistor{})
                }
                else if rb_element_c.get_active() {
                    Box::new(Capacitor{})
                }
                else if rb_element_w.get_active() {
                    Box::new(Warburg{})
                }
                else if rb_element_l.get_active() {
                    Box::new(Inductor{})
                }
                else if rb_element_q.get_active() {
                    Box::new(CPE{})
                }
                else {
                    panic!();
                }
            };


            let rb_edit_replace: gtk::RadioButton = builder.get_object("rb_edit_replace").unwrap();
            let rb_edit_series: gtk::RadioButton = builder.get_object("rb_edit_series").unwrap();
            let rb_edit_parallel: gtk::RadioButton = builder.get_object("rb_edit_parallel").unwrap();
            let perform_user_edit = move |model: &mut Model, x, y, newelem| {
                if rb_edit_replace.get_active() {
                    model.circ.replace((x, y), newelem);
                }
                else if rb_edit_series.get_active() {
                    model.circ.add_series((x, y), newelem);
                }
                else if rb_edit_parallel.get_active() {
                    // TODO: implement
                    //model.circ.add_parallel((x, y), newelem);
                }
                else {
                    panic!();
                }
            };


            let graph: gtk::DrawingArea = builder.get_object("graphCircuit").unwrap();
            graph.connect_draw(|widget, context| {
                draw(&widget, &context);
                Inhibit(false)
            });

            graph.add_events(gdk::EventMask::BUTTON_PRESS_MASK);

            let cpbox : gtk::Box = builder.get_object("boxParams").unwrap();

            let cp = cpbox.clone();
            graph.connect_button_press_event(move |wid, event: &gdk::EventButton| {
                println!("Press event at {:?}", event.get_position());

                let model = g_model.as_mut().unwrap();

                if let Some((x,y)) = block_by_coords(&model, &wid, &event) {
                    let newelem: Box<dyn Circuit> = create_user_element();

                    println!("....at cell {:?}", (x, y));

                    //model.circ.replace((x, y), newelem);
                    perform_user_edit(model, x,y, newelem);
                    println!("....{:?}", model.circ);
                    println!("....{:?}", g_model.as_ref().unwrap().circ);

                    remake_param_list(&cp, &wid);
                    wid.queue_draw();
                }

                Inhibit(false)
            });

            remake_param_list(&cpbox, &graph);
            main_window.show_all();
        });

        app.run(&["Impediment".to_string()]);
    }
}
