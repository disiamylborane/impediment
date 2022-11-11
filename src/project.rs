use std::collections::VecDeque;
use std::ops::Range;

use eframe::egui;
use num::Complex;

use crate::circuit::Circuit;
use crate::circuit::Element;

pub type Float = f64;

pub type Cplx = num::complex::Complex<Float>;


// TODO: Ineffective memory usage
#[derive(Debug, Copy, Clone)]
pub struct DataPoint {
    pub freq: Float,
    pub imp: Cplx,
    pub enabled: bool,
}

// A circuit component's impedance is calculated from
// a set of __parameters__ with some physical meaning.
// 
// Each __parameter__ is calculated from one or several 
// __variables__, which are user-edited or fitted 
#[derive(Debug, Clone)]
pub struct ModelVariable {
    // The current value of variable
    pub val : Float,
    // Lower and upper bounds for fitting
    pub bounds : (Float, Float),
    // The variable is enabled to vary in fitting process
    pub enabled: bool,
}


#[derive(Debug, Clone)]
pub struct Spectrum {
    pub points: Vec<DataPoint>,
    pub name: String,

    // The individual variable is strictly the value of parameter
    // Always has the shape of ind_params[model count][individual params in the model count]
    pub ind_params: Vec<Vec<ModelVariable>>,

    // The spectrum-assotiated constants (like potential, current, temperature) and so on
    // Always has a size of ProjectData.constants.len()
    pub constants: Vec<Float>,
}


#[derive(Debug, Clone)]
pub struct SpectrumGroup {
    pub spectra: Vec<Spectrum>,

    // Always has the shape of group_vars[model count][(group param * varcount()) in the model ]
    pub group_vars: Vec<Vec<ModelVariable>>,
}


#[derive(Debug, Clone, Copy)]
pub enum GroupParameterType {
    Value,
    Linear(usize),
}

impl GroupParameterType {
    /// How many variables does this group parameter occupy
    pub fn varcount(self) -> usize {
        match self {
            GroupParameterType::Value => 1,
            GroupParameterType::Linear(_) => 2,
        }
    }
}


#[derive(Debug, Clone)]
pub enum ParameterDescriptor {
    Individual(String),
    Group(String, GroupParameterType),
}
impl ParameterDescriptor{
    pub fn name(&self) -> &str {
        match self {
            ParameterDescriptor::Individual(nm) | ParameterDescriptor::Group(nm, _) => nm,
        }
    }
}


#[derive(Debug, Clone)]
pub struct Model {
    pub circuit: Circuit,
    pub params: Vec<ParameterDescriptor>,
    pub name: String,
    pub lock: bool,
}


impl Model {
    fn small_style(ui: &egui::Ui) -> egui::text::TextFormat {
        ui.style().as_ref().text_styles.get(&egui::TextStyle::Small)
            .map_or_else(
                || egui::text::TextFormat{italics: true, ..Default::default()}, 
                |font_id| egui::text::TextFormat{font_id: font_id.clone(), ..Default::default()})
    }


    pub fn var_ranges(&self, param_range: Range<usize>) -> (Range<usize>, Range<usize>) {
        let mut ind_var = 0;
        let mut grp_var = 0;

        for param in 0..(param_range.start) {
            match self.params[param] {
                ParameterDescriptor::Individual(_) => ind_var += 1,
                ParameterDescriptor::Group(_, GroupParameterType::Value) => grp_var += 1,
                ParameterDescriptor::Group(_, GroupParameterType::Linear(_)) => grp_var += 2,
            }
        }

        let mut ind_end = ind_var;
        let mut grp_end = grp_var;

        for param in param_range {
            match self.params[param] {
                ParameterDescriptor::Individual(_) => ind_end += 1,
                ParameterDescriptor::Group(_, GroupParameterType::Value) => grp_end += 1,
                ParameterDescriptor::Group(_, GroupParameterType::Linear(_)) => grp_end += 2,
            }
        }

        (ind_var..ind_end, grp_var..grp_end)
    }
    

    pub fn get_individual_vars(&self, ui: &egui::Ui) -> Vec<egui::Label> {
        let mut out = vec![];
        for (pt, &letter) in self.params.iter().zip(self.circuit.param_letters().iter()) {
            if let ParameterDescriptor::Individual(pts) = pt {
                let mut job = egui::text::LayoutJob::default();
                job.append(letter, 0.0, egui::text::TextFormat{..Default::default()});
                job.append(pts, 0.0, Self::small_style(ui));
                out.push(egui::Label::new(job));
            }
        }
        out
    }

    pub fn get_group_vars(&self, ui: &egui::Ui, consts: &[String]) -> Vec<egui::Label> {
        let mut out = vec![];
        for (pt, &letter) in self.params.iter().zip(self.circuit.param_letters().iter()) {
            if let ParameterDescriptor::Group(pts, gtype) = pt {

                match gtype {
                    GroupParameterType::Value => {
                        let mut job = egui::text::LayoutJob::default();
                        job.append(letter, 0.0, egui::text::TextFormat{..Default::default()});
                        job.append(pts, 0.0, Self::small_style(ui));
                        out.push(egui::Label::new(job));
                    }

                    &GroupParameterType::Linear(cst) => {
                        let mut job = egui::text::LayoutJob::default();
                        job.append(letter, 0.0, egui::text::TextFormat{..Default::default()});
                        job.append(pts, 0.0, Self::small_style(ui));
                        job.append(": +", 0.0, egui::text::TextFormat{..Default::default()});
                        out.push(egui::Label::new(job));

                        let mut job = egui::text::LayoutJob::default();
                        job.append(letter, 0.0, egui::text::TextFormat{..Default::default()});
                        job.append(pts, 0.0, Self::small_style(ui));
                        job.append(&format!(": {}*", consts[cst]), 0.0, egui::text::TextFormat{..Default::default()});
                        out.push(egui::Label::new(job));
                    }
                }
            }
        }
        out
    }
    
}



#[derive(Default, Debug, Clone)]
pub struct ProjectData {
    pub models: Vec<Model>,
    pub dataset: Vec<SpectrumGroup>,
    pub constants: Vec<String>,
}

#[derive(Debug)]
pub enum ConsistencyError {
    Unknown,
    ModelCnt{spectrum_group: usize},
    ConstLen{spectrum_group: usize, spectrum: usize},
    IndividualLen{spectrum_group: usize, spectrum: usize},

    ParamCount{model: usize},
    GlobalVars{model: usize, spectrum_group: usize},
    IndividualParam{model: usize, spectrum_group: usize, spectrum: usize},
}

impl ProjectData {
    /// Check if project data are well-formed, e.g. 
    /// all the array sizes match
    pub fn check_for_consistency(&self) -> Result<(), ConsistencyError> {
        macro_rules! constrain {
            ($e1:expr, $e2:expr => $error:expr) => {
                if ($e1 != $e2) {return Err($error)}
            };
        }

        use ParameterDescriptor::{Group, Individual};
        use ConsistencyError::*;

        let consts_cnt = self.constants.len();
        let model_cnt = self.models.len();

        for (i_sgroup, sgroup) in self.dataset.iter().enumerate() {
            constrain!(sgroup.group_vars.len(), model_cnt => ModelCnt{spectrum_group: i_sgroup});

            for (i_spectrum, spectrum) in sgroup.spectra.iter().enumerate() {
                constrain!(spectrum.constants.len(), consts_cnt => ConstLen{spectrum_group: i_sgroup, spectrum: i_spectrum});
                constrain!(spectrum.ind_params.len(), model_cnt => IndividualLen{spectrum_group: i_sgroup, spectrum: i_spectrum});
            }
        }

        for (imodel, model) in self.models.iter().enumerate() {
            constrain!(model.circuit.param_count(), model.params.len() => ParamCount{model: imodel});

            let ind_len: usize = model.params.iter().map(|x| match x {Individual(_) => 1, Group(_, _) => 0}).sum();
            let gvar_count: usize = model.params.iter().map(|x| match x {Individual(_) => 0, Group(_, tp) => tp.varcount()}).sum();

            for (i_sgroup, sgroup) in self.dataset.iter().enumerate() {
                let gvars = &sgroup.group_vars[imodel];
                
                constrain!(gvars.len(), gvar_count => GlobalVars{model: imodel, spectrum_group: i_sgroup});
                
                for (i_spectrum, spectrum) in sgroup.spectra.iter().enumerate() {
                    constrain!(spectrum.ind_params[imodel].len(), ind_len => IndividualParam{model: imodel, spectrum_group: i_sgroup, spectrum: i_spectrum});
                }
            }
        }

        Ok(())
    }
}


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataPointVal {
    Freq, Re, Im
}


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataVarPart {
    Val, Min, Max
}


/// An user-defined action on project data. Each Action has an inverse
/// action (undo) to be done on the new ProjectData.
#[derive(Debug)]
pub enum Action {
    Unknown(Box<ProjectData>),

    EditCircuitName{idx: usize, name: String},
    EditSpectrumName{idx: (usize,usize), name: String},

    AddCircuit{idx: usize, mdl: Model, inds: Vec<Vec<Vec<ModelVariable>>>, grps: Vec<Vec<ModelVariable>>},
    RemoveCircuit{idx: usize},

    EditDataPoint{idx: (usize,usize,usize), part: DataPointVal, value: f64},
    EditIndividualVar{mdl: usize, spec: (usize,usize), var: usize, part: DataVarPart, value: f64},
    EditGroupVar{mdl: usize, spec: usize, var: usize, part: DataVarPart, value: f64},
    EditConst{spec: (usize,usize), cst: usize, value: f64},

}


impl Action {
    pub fn act(self, on: &mut ProjectData) -> Action {
        match self {
            Action::Unknown(mut prdata) => {
                std::mem::swap(prdata.as_mut(), on);
                Action::Unknown(prdata)
            }
            Action::EditCircuitName { idx, mut name } => {
                std::mem::swap(&mut name, &mut on.models[idx].name);
                Action::EditCircuitName { idx, name }
            }
            Action::EditSpectrumName { idx, mut name } => {
                std::mem::swap(&mut name, &mut on.dataset[idx.0].spectra[idx.1].name);
                Action::EditSpectrumName { idx, name }
            }
            Action::EditDataPoint { idx, part, mut value } => {
                let dp = &mut on.dataset[idx.0].spectra[idx.1].points[idx.2];
                let data_val = match part {
                    DataPointVal::Freq => &mut dp.freq,
                    DataPointVal::Re => &mut dp.imp.re,
                    DataPointVal::Im => &mut dp.imp.im,
                };
                std::mem::swap(data_val, &mut value);
                Action::EditDataPoint { idx, part, value }
            }

            Action::AddCircuit{idx, mdl, inds, grps} => {
                on.models.insert(idx, mdl);
                for ((sgroup, gvars), ivarstack) in on.dataset.iter_mut().zip(grps).zip(inds) {
                    sgroup.group_vars.insert(idx, gvars);
                    for (spec, iparams) in sgroup.spectra.iter_mut().zip(ivarstack) {
                        spec.ind_params.insert(idx, iparams);
                    }
                }
                Action::RemoveCircuit { idx }
            }
            Action::RemoveCircuit { idx } => {
                let mdl = on.models.remove(idx);
                let (grps, inds) = on.dataset.iter_mut().map(|sg| {
                    let gvars = sg.group_vars.remove(idx);
                    let iparams = sg.spectra.iter_mut().map(|sp| {sp.ind_params.remove(idx)}).collect::<Vec<_>>();
                    (gvars, iparams)
                }).unzip::<_, _, Vec<_>, Vec<_>>();

                Action::AddCircuit{idx, mdl, inds, grps}
            }
            Action::EditIndividualVar { mdl, spec, var, part, mut value } => {
                let dv = &mut on.dataset[spec.0].spectra[spec.1].ind_params[mdl][var];
                let data_val = match part {
                    DataVarPart::Min => &mut dv.bounds.0,
                    DataVarPart::Max => &mut dv.bounds.1,
                    DataVarPart::Val => &mut dv.val,
                };
                std::mem::swap(data_val, &mut value);
                Action::EditIndividualVar { mdl, spec, var, part, value }
            }
            Action::EditGroupVar { mdl, spec, var, part, mut value } => {
                let dv = &mut on.dataset[spec].group_vars[mdl][var];
                let data_val = match part {
                    DataVarPart::Min => &mut dv.bounds.0,
                    DataVarPart::Max => &mut dv.bounds.1,
                    DataVarPart::Val => &mut dv.val,
                };
                std::mem::swap(data_val, &mut value);
                Action::EditGroupVar { mdl, spec, var, part, value }
            }
            Action::EditConst { spec, cst, mut value } => {
                let dv = &mut on.dataset[spec.0].spectra[spec.1].constants[cst];
                std::mem::swap(dv, &mut value);
                Action::EditConst { spec, cst, value }
            }
        }
    }

    pub fn combine_undos(&self, prev: &Action) -> Option<Action> {
        match (self, prev) {
            //(Action::__Unknown, Action::__Unknown) => return Some(Action::__Unknown),
            (Action::EditCircuitName { idx, name:_ }, Action::EditCircuitName { idx: idx2, name }) => {
                if idx == idx2 {
                    return Some(Action::EditCircuitName { idx: *idx2, name: name.clone() });
                }
            },
            (Action::EditSpectrumName { idx, name:_ }, Action::EditSpectrumName { idx: idx2, name }) => {
                if idx == idx2 {
                    return Some(Action::EditSpectrumName { idx: *idx2, name: name.clone() });
                }
            },
            (&Action::EditDataPoint { idx, part, value:_ }, &Action::EditDataPoint { idx: idx2, part: part2, value }) => {
                if idx == idx2 && part == part2 {
                    return Some(Action::EditDataPoint { idx, part, value });
                }
            },
            (&Action::EditIndividualVar { mdl, spec, var, part, value:_ }, &Action::EditIndividualVar { mdl: mdl2, spec: spec2, var: var2, part: part2, value }) => {
                if mdl == mdl2 && spec == spec2 && var == var2 && part == part2 {
                    return Some(Action::EditIndividualVar { mdl: mdl2, spec: spec2, var: var2, part: part2, value });
                }
            },
            (&Action::EditGroupVar { mdl, spec, var, part, value:_ }, &Action::EditGroupVar { mdl: mdl2, spec: spec2, var: var2, part: part2, value }) => {
                if mdl == mdl2 && spec == spec2 && var == var2 && part == part2 {
                    return Some(Action::EditGroupVar { mdl: mdl2, spec: spec2, var: var2, part: part2, value });
                }
            },
            (&Action::EditConst { spec, cst, value:_ }, &Action::EditConst { spec: spec2, cst: cst2, value }) => {
                if spec == spec2 && cst == cst2 {
                    return Some(Action::EditConst { spec: spec2, cst: cst2, value });
                }
            },
            _ => {},
        }

        None
    }
}


impl ProjectData {
    pub fn sample() -> Self {
        let models = vec![
            Model{ 
                circuit: Circuit::Element(Element::Resistor), 
                params: vec![ParameterDescriptor::Individual("ct".to_string())],
                name: "Model R".to_string(),
                lock: false,
            },
            Model{ 
                circuit: Circuit::Element(Element::Capacitor), 
                params: vec![ParameterDescriptor::Group("dl".to_string(), GroupParameterType::Value)],
                name: "Model C".to_string(),
                lock: false,
            },
        ];
        let dataset = vec![SpectrumGroup{ spectra: vec![
            Spectrum{ 
                points: vec![
                    DataPoint{ freq: 10000.0, imp: Complex{ re: 10.0, im: -10.0 }, enabled: true },
                    DataPoint{ freq: 1000.0, imp: Complex{ re: 15.0, im: -15.0 }, enabled: false },
                    DataPoint{ freq: 100.0, imp: Complex{ re: 20.0, im: -15.0 }, enabled: true },
                    DataPoint{ freq: 10.0, imp: Complex{ re: 30.0, im: -10.0 }, enabled: true },
                ],
                name: "Spectrum A".to_string(),
                ind_params: vec![
                    vec![ModelVariable{ val: 100.0, bounds: (1.0, 1000.0), enabled: true }],
                    vec![],
                ],
                constants: vec![14.0],
            }
        ], group_vars: vec![
            vec![],
            vec![ModelVariable{ val: 0.01, bounds: (0.0001, 0.1), enabled: true }],
        ] }];
        let constants = vec!["Temperature".to_string()];

        let out = Self { models, dataset, constants };
        out.check_for_consistency().unwrap();
        out
    }

    pub fn add_new_constant(&mut self, name: String, value: f64) {
        self.constants.push(name);

        for sgroup in &mut self.dataset {
            for spectrum in &mut sgroup.spectra {
                spectrum.constants.push(value);
            }
        }
    }

    pub fn edit_constant_value(&mut self, const_idx: usize, spectrum: (usize, usize), value: f64) -> f64 {
        let mval = &mut self.dataset[spectrum.0].spectra[spectrum.1].constants[const_idx];
        let old_val = *mval;
        *mval = value;
        old_val
    }

    pub fn remove_constant(&mut self, const_idx: usize) -> (String, Vec<Vec<f64>>) {
        let values: Vec<Vec<f64>> = self.dataset.iter_mut().map(|sgroup| {
            let c_old: Vec<f64> = sgroup.spectra.iter_mut().map(|spec| {
                spec.constants.remove(const_idx)
            }).collect();
            c_old
        }).collect();

        let cname = self.constants.remove(const_idx);

        (cname, values)
    }

    pub fn insert_constant(&mut self, const_idx: usize, (cname, values): (String, Vec<Vec<f64>>)) {
        self.constants.insert(const_idx, cname);
        
        for (sgroup, svalues) in self.dataset.iter_mut().zip(values.into_iter()) {
            for (spec, svalue) in sgroup.spectra.iter_mut().zip(svalues.into_iter()) {
                spec.constants.insert(const_idx, svalue);
            }
        }
    }

    pub fn add_new_circuit_action(&self) -> Action {
        let idx = self.models.len();
        let mdl = Model{
            circuit: Circuit::Element(Element::Resistor),
            params: vec![ParameterDescriptor::Individual("0".to_string())],
            name: "Circuit".to_string(),
            lock: false,
        };

        let inds = self.dataset.iter().map(|sg| {
            vec![vec![ModelVariable{ val: 100.0, bounds: (1.0, 1000.0), enabled: true }]; sg.spectra.len()]
        }).collect();

        let grps = vec![vec![]; self.dataset.len()];

        Action::AddCircuit { idx, mdl, inds, grps}
    }
}


#[test]
fn x() {
    let mut d = ProjectData::sample();
    println!("{:#?}", d);

    d.add_new_constant("value".to_string(), 0.0);
    d.check_for_consistency().unwrap();
    println!("{:#?}", d);

    d.edit_constant_value(1, (0, 0), 99.0);

    let rd = d.remove_constant(0);
    d.check_for_consistency().unwrap();
    println!("{:#?}", d);

    d.insert_constant(0, rd);
    d.check_for_consistency().unwrap();
    println!("{:#?}", d);
}


#[derive(Debug)]
pub struct UndoBuffer {
    maxsize: usize,
    undos: VecDeque<Action>,
    redos: VecDeque<Action>,
}


impl Default for UndoBuffer {
    fn default() -> Self {
        Self { maxsize: 16, undos: VecDeque::with_capacity(17), redos: VecDeque::with_capacity(17) }
    }
}

impl UndoBuffer {
    pub fn add_undo(&mut self, undo: Action) {
        self.redos.clear();

        if let Some((last, combo)) = self.undos.back_mut().and_then(|last| {
            let nw = undo.combine_undos(last)?;
            Some((last, nw))
        }) {
            *last = combo;
        }
        else {
            self.undos.push_back(undo);

            if self.undos.len() > self.maxsize {
                self.undos.pop_front();
            }
        }
    }
}


