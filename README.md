# impediment

Impediment is an interactive tool for impedance spectra modelling and fitting

Currently supports:

* An equivalent circuit builder
* CSV file import
* Model parameter fitting using [nlopt](https://crates.io/crates/nlopt)
* Manual parameter fitting

# Usage

See [interface guide](doc/manual/Usage.md).

# Building

The software is written in [Rust](http://github.com/rust-lang/rust/), it requires [rust toolchain](https://rustup.rs/) to be built.

The software depends on `egui` crate. See egui [requirements](https://github.com/emilk/egui_template/#testing-locally).

Run `cargo build --release` to build the application.
