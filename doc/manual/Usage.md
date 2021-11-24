# Basic structure

The impediment's project consists of a list of _Circuits_ and a list of
_Datasets_.Circuit is represented by a number of passive elements connected
in series-parallel. Dataset is a number of points "frequency-impedance" to
be (normally) fitted to. The circuit provides a few parameters to be adjusted
according to the electrical components. Each pair of circuit-dataset has its
own set of parameters.

|           |    Circuit 1    |   Circuit 2     |
|----------:|:---------------:|:---------------:|
| Dataset 1 | Parameter set 1 | Parameter set 2 |
| Dataset 2 | Parameter set 3 | Parameter set 4 |
| Dataset 3 | Parameter set 5 | Parameter set 6 |

# GUI

![Application GUI Image](uinterface.png)

## Circuit and dataset selection

Click a textbox in circuit editor to select a circuit. It highlights with red.

Click a textbox in dataset editor to select a circuit. It also highlights with red.

The circuits and datasets may be renamed in-place.

## Circuit editor

Click "+" button to add a new circuit.

Click "-" button to delete the current circuit.

Press Load and Save button to load a new circuit from or save the current circuit to file.
Only a single circuit with no parameter values is saved.

To edit the circuit elements:

* Click the _action_ button to choose either replace element [:], remove element
 [x], or add the element in series[--]/parallel[=].
* Choose the appropriate _element_, if needed: [R] Resistor, [C] Capacitor,
[W] Warburg element, [L] Inductor, [Q] Constant phase.
* Click the element on _circuit graph_.

## Dataset editor

Click "+" button to add a new empty dataset.

Click "-" button to delete the current dataset.

The dataset may be edited in a _data editor_. The values are stored
in the format of {Frequency, Hz}: {Re Z} + {Im Z}i.

Click "Load" button to create a new dataset from CSV file.

The data graph shows the Niquist plot of the data and the fitting curve.
Drag the circuit graph to move it, Ctrl+scroll to scale it, double click to reset.

## Parameter editor

The parameters are listed in _parameter box_.

All the parameters contain min and max bounds (used by fitting routines) and
the parameter value itself.

When the parameter values are edited, the plots are immediately updated. One
can use "<" and ">" buttons to increase or decrease the parameter value
logatithmically. Alt-click for slower, Ctrl-click for faster edit.

Click "Fit" button to perform a fitting routine with the selected method.

The complete parameter set may be copied and pasted ("Copy"/"Paste" buttons) for
the same circuit.

# CSV loading

The CSV loading procedure allows to load nultiple files with the same data layout
at once. After the files are selected, customize the CSV column numbers for
frequency, real and imaginary impedance values. The data preview is available.

![Application GUI Image](csv.png)
