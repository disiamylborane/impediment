# Basic structure

The application works with _Circuits_ and _Datasets_ separately. Circuit is 
represented by a number of passive elements connected in series-parallel. 
Dataset is a number of points "frequency-impedance" to be (normally) fitted to.
 The circuit provides a few parameters to be adjusted. The dataset may be empty.

An application instance serves multiple circuits and datasets.
All the combinations of circuit and dataset are described by independent 
parameter sets:

|           |    Circuit 1    |   Circuit 2     |
|----------:|:---------------:|:---------------:|
| Dataset 1 | Parameter set 1 | Parameter set 2 |
| Dataset 2 | Parameter set 3 | Parameter set 4 |
| Dataset 3 | Parameter set 5 | Parameter set 6 |

# GUI

![Application GUI Image](interface.png)

## Circuit editor

The circuits are created editable (displayed green). Once the fitting is started
or parameter is set manually, the circuit is no more editable (displayed black).

Navigate through the circuits by _circuit selector box_. To change the name of
the current circuit, type it inside _dataset selector_ and press Enter. Press
Escape while changing the name to reset it.

Click "+" button to add a circuit.

Click "-" button to delete the current circuit.

[Open and save buttons are WIP].

To edit the circuit element:

* Click the _action_ button to choose either replace element [:], remove element
 [x], or add the element in series[--]/parallel[=].
* Choose the appropriate _element_, if needed: [R] Resistor, [C] Capacitor, [W] Warburg element, [L] Inductor, [Q] Constant phase.
* Click the element on _circuit graph_.

## Dataset editor

Navigate through the datasets by _dataset selector_. To change the name of the current dataset, type a new name inside _dataset selector_ and press Enter.

Click "Load" button to create a new dataset from CSV file.

Click "-" button to delete the current dataset.

[Save and + buttons are WIP].

## Parameter editor

The parameters are listed in _parameter box_.

[The parameter names are WIP.]

All the parameters contain bound min and max values (used by fitting routines) and the parameter value itself.

When the parameter values are edited, the plots are immediately updated.

Click "Fit" button to perform a fitting routine with the selected method.

## Plots [Deeply WIP]

Three plot types are currently available:

* Bode phase-log frequency plot
* Bode amplitude-log frequency plot
* Nyquist Im Z-Re Z plot

# CSV loading dialog

![CSV dialog GUI Image](csv_dialog.png)

# Customizations

When the CSV file is loading, the following may be customized:

* Start line &mdash; the amount of header lines to be skipped
* The columns of frequency, Real(Impedance) and Imag(Impedance) data
* Frequency and impedance units.

The preview window shows several data rows. When a file or configuration is unparseable, the preview is displayed red.
