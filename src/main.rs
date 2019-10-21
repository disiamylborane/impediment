// An example: 
// The series RC circuit has an impedance of 100-3i Ohm at angular frequency 100/s
// Use the BOBYQA algorithm from Nlopt to fit the circuit parameters

extern crate nlopt;

extern crate gio;
extern crate gtk;

use gio::prelude::*;
use gtk::prelude::*;

fn main() {
    let app =
        gtk::Application::new(Some("app.impediment"), Default::default()).expect("GTK failed");

    app.connect_activate(|app| {
        let builder = gtk::Builder::new_from_string(include_str!("impedui.glade"));

        let main_window: gtk::Window = builder.get_object("main_window").unwrap();
        main_window.set_application(Some(app));

        let graph: gtk::DrawingArea = builder.get_object("graphCircuit").unwrap();
        graph.connect_draw(move |_widget, context| {
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

            context.stroke();
            Inhibit(false)
        });
        
        main_window.show_all();
    });

    app.run(&[String::new()]);
}
