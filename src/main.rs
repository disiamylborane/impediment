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

fn draw(g: &cairo::Context){

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

    unsafe {g_model = Some(Model{circ, params});}

    let app = gtk::Application::new(Some("app.impediment"), Default::default()).expect("GTK failed");

    app.connect_activate(|app| {
        let builder = gtk::Builder::new_from_string(include_str!("impedui.glade"));

        let main_window: gtk::Window = builder.get_object("main_window").unwrap();
        main_window.set_application(Some(app));

        let graph: gtk::DrawingArea = builder.get_object("graphCircuit").unwrap();
        graph.connect_draw(|_widget, context| {

            context.set_source_rgb(1.0, 1.0, 1.0);
            context.paint();

            context.set_source_rgb(0.0, 0.0, 0.0);

            context.move_to(50.0, 50.0);
            context.line_to(55.0, 50.0);

            context.move_to(55.0, 45.0);
            context.line_to(55.0, 55.0);
            context.line_to(75.0, 55.0);
            context.line_to(75.0, 45.0);
            context.line_to(55.0, 45.0);

            context.move_to(75.0, 50.0);
            context.line_to(90.0, 50.0);

            context.move_to(90.0, 45.0);
            context.line_to(90.0, 55.0);
            context.move_to(95.0, 45.0);
            context.line_to(95.0, 55.0);
            context.move_to(95.0, 50.0);
            context.line_to(110.0, 50.0);

            unsafe {
                match &g_model {
                    Some(model) => {
                        let imp = model.circ.impedance(100.0, &model.params.vals);
                        context.set_source_rgb(0.0, 0.0, 0.0);
                        context.set_font_size(14.0);

                        context.move_to(95.0, 60.0);
                        context.show_text(& (imp.re as f32).to_string());

                        context.move_to(95.0, 80.0);
                        context.show_text(& (imp.im as f32).to_string());
                    }
                    _ => {panic!();}
                }
            }


            context.stroke();
            Inhibit(false)
        });
        
        main_window.show_all();
    });

    app.run(&[String::new()]);
}
