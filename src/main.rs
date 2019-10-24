// An example: 
// The series RC circuit has an impedance of 100-3i Ohm at angular frequency 100/s
// Use the BOBYQA algorithm from Nlopt to fit the circuit parameters

extern crate nlopt;

extern crate cairo;
extern crate gio;
extern crate gtk;

use gio::prelude::*;
use gtk::prelude::*;

mod circuit;
use circuit::*;

struct ParameterDesc {
    vals : Vec<f64>,
    bounds : Vec<(f64, f64)>
}

struct Model {
    circ : Box<dyn Circuit>,
    params : ParameterDesc
}

static mut g_model : Option<Model> = None;

impl ParameterDesc{
    fn new(paramlist: &[ParameterBase]) -> Self {
        let vals = paramlist.iter().map(|x| x.default).collect::<Vec<_>>();;
        let bounds = paramlist.iter().map(|x| x.limits).collect::<Vec<_>>();;

        ParameterDesc{vals, bounds}
    }
}

fn draw(context: &cairo::Context){
    unsafe {
        match &g_model {
            Some(model) => {
                context.set_source_rgb(0.0, 0.7, 0.4);
                
                if true {
                    model.circ.paint(&context, 20., (50., 50.));
                } else {

                    Parallel{elems: ComplexCirc::new(vec![
                        Box::new(Resistor{}),
                        Box::new(Capacitor{}),
                        Box::new(CPE{}),
                        Box::new(Resistor{}),
                    ])}.paint(&context, 18., (50., 50.));
                }

                context.stroke();
            }
            _ => {panic!();}
        }
    }
}

fn main() {
    let mut circ : Box<dyn Circuit> =  
        Box::new(Series{elems: ComplexCirc::new(vec![
            Box::new(Resistor{}),
            Box::new(Parallel{elems: ComplexCirc::new(vec![
                Box::new(CPE{}),
                Box::new(Series{elems: ComplexCirc::new(vec![
                    Box::new(Resistor{}),
                    Box::new(Warburg{}),
                ])}),
            ])}),
        ])});

    let mut params = ParameterDesc::new(circ.paramlist());

    unsafe {
        
    g_model = Some(Model{circ, params});

    let app = gtk::Application::new(Some("app.impediment"), Default::default()).expect("GTK failed");

    app.connect_activate(|app| {
        let builder = gtk::Builder::new_from_string(include_str!("impedui.glade"));

        let main_window: gtk::Window = builder.get_object("main_window").unwrap();
        main_window.set_application(Some(app));

        let graph: gtk::DrawingArea = builder.get_object("graphCircuit").unwrap();
        graph.connect_draw(|_widget, context| {
            context.set_source_rgb(1.0, 1.0, 1.0);
            context.paint();

            draw(&context);

            context.stroke();
            Inhibit(false)
        });

        let parambox: gtk::Box = builder.get_object("boxParams").unwrap();
        /*match &g_model {
            Some(model) => {

            }

            _ => panic!()
        }*/
        let _model = g_model.as_ref().unwrap();
        let params = ParameterDesc::new((_model.circ).paramlist());

        for i in 0..params.vals.len() {
            let single_param = gtk::Box::new(gtk::Orientation::Horizontal, 5);
            
            let lbl = gtk::Label::new(Some(&i.to_string()));
            let ebounds = gtk::Entry::new();
            let evalue = gtk::Entry::new();
            ebounds.set_text(&format!("{}, {}", params.bounds[i].0, params.bounds[i].1));
            evalue.set_text(&params.vals[i].to_string());

            single_param.pack_start(&lbl, /*expand*/false, /*fill*/false, /*padding*/0);
            single_param.pack_start(&ebounds, /*expand*/true, /*fill*/true, /*padding*/0);
            single_param.pack_start(&evalue, /*expand*/true, /*fill*/true, /*padding*/0);

            parambox.pack_start(&single_param, /*expand*/false, /*fill*/false, /*padding*/0);
        }

        main_window.show_all();
    });

    app.run(&[String::new()]);
    }

}
