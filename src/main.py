import numpy as np  # Make sure numpy is imported if not already at the top
import matplotlib.pyplot as plt  # Ensure this is imported
from matplotlib.figure import Figure as MatplotlibFigure
import warnings

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

from qiskit_experiments.library import StandardRB
from qiskit_experiments.framework import ExperimentData

# --- Your Warning Filters (keep these at the top) ---
warnings.filterwarnings(
    "ignore",
    message="Providing `coupling_map` and/or `basis_gates` along with `backend` is not recommended.*",
    category=UserWarning,
    module="qiskit.compiler.transpiler"
)
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=".*Leaving `dataframe` unset or setting it to `False` for `ExperimentData.analysis_results` is deprecated.*"
)

# --- Global constants from original script (can be overridden by function args) ---
N_QUBITS_DEFAULT = 2  # nQ_orig
BASIS_GATES_DEFAULT = ['u1', 'u2', 'u3', 'cx']
SHOTS_DEFAULT = 200
rb_pattern_orig = [[0, 1]]


# ==============================================================================
# REUSABLE FUNCTION TO RUN AN RB EXPERIMENT AND GET RESULTS
# ==============================================================================
def run_rb_experiment(
        experiment_name: str,
        physical_qubits_tuple: tuple,
        clifford_lengths: list,
        num_seeds: int,
        p1q_noise: float,
        p2q_noise: float,
        basis_gates: list = BASIS_GATES_DEFAULT,
        shots: int = SHOTS_DEFAULT,
        save_plot: bool = True,
        plot_filename_prefix: str = "rb_plot"
):
    """
    Runs a StandardRB experiment with specified parameters and saves/analyzes results.
    """
    print(f"\n========== RUNNING EXPERIMENT: {experiment_name} ==========")
    print(f"Parameters: Qubits={physical_qubits_tuple}, Lengths={clifford_lengths}, Seeds={num_seeds}")
    print(f"Noise: p1Q={p1q_noise}, p2Q={p2q_noise}")
    print(f"Basis Gates: {basis_gates}, Shots: {shots}")

    # 1. Setup Noise Model
    current_noise_model = NoiseModel(basis_gates=basis_gates)  # Important for noise model
    if p1q_noise > 0:
        current_noise_model.add_all_qubit_quantum_error(depolarizing_error(p1q_noise, 1), ['u2'])
        current_noise_model.add_all_qubit_quantum_error(depolarizing_error(2 * p1q_noise, 1), ['u3'])
    if p2q_noise > 0:
        current_noise_model.add_all_qubit_quantum_error(depolarizing_error(p2q_noise, 2), ['cx'])
    # Note: u1 is assumed error-free in this model

    # 2. Setup Backend
    current_backend = AerSimulator(noise_model=current_noise_model, basis_gates=basis_gates)

    # 3. Create and Configure RB Experiment
    rb_exp_instance = StandardRB(
        physical_qubits=physical_qubits_tuple,
        lengths=clifford_lengths,
        num_samples=num_seeds,
        seed=42  # Keep seed for comparability across runs where other params change
    )
    rb_exp_instance.set_transpile_options(basis_gates=basis_gates, optimization_level=0)

    # 4. Run Experiment
    print("Submitting experiment...")
    exp_data_instance: ExperimentData = rb_exp_instance.run(backend=current_backend, shots=shots)
    print("Blocking for results...")
    exp_data_instance.block_for_results()
    print("Experiment and analysis finished.")

    # 5. Extract Key Results
    epc_res = exp_data_instance.analysis_results("EPC", dataframe=False)
    alpha_res = exp_data_instance.analysis_results("alpha", dataframe=False)

    epc_val, epc_err, alpha_val, alpha_err = None, None, None, None
    if epc_res:
        epc_val = epc_res.value.nominal_value
        epc_err = epc_res.value.std_dev
        print(f"  Error Per Clifford (EPC): {epc_val:.4e} \u00B1 {epc_err:.4e}")
    if alpha_res:
        alpha_val = alpha_res.value.nominal_value
        alpha_err = alpha_res.value.std_dev
        print(f"  Fit parameter alpha: {alpha_val:.4f} \u00B1 {alpha_err:.4f}")

    # 6. Process and Save Plot
    if save_plot:
        figure_data_cont = exp_data_instance.figure(0)
        output_fname = f"{plot_filename_prefix}_{experiment_name.replace(' ', '_').lower()}.png"
        try:
            if figure_data_cont and hasattr(figure_data_cont, 'figure'):
                actual_fig: MatplotlibFigure = figure_data_cont.figure
                if isinstance(actual_fig, MatplotlibFigure) and actual_fig.get_axes():
                    ax = actual_fig.get_axes()[0]
                    ax.set_title(f'{len(physical_qubits_tuple)} Qubit RB - {experiment_name}', fontsize=16)
                    ax.set_ylabel("Ground State Population", fontsize=14)

                    found_errorbar = False
                    if ax.containers:
                        for i, container in enumerate(ax.containers):
                            if hasattr(container, 'lines') and len(container.lines) == 3:
                                print(f"Found ErrorbarContainer at ax.containers[{i}]. Restyling to red 'x'.")
                                data_line, caplines, barlines_list = container.lines

                                data_line.set_color('red')
                                data_line.set_marker('x')
                                data_line.set_markersize(7)
                                data_line.set_markerfacecolor('red')
                                data_line.set_markeredgecolor('red')
                                data_line.set_linestyle('')

                                for cap in caplines:
                                    cap.set_color('red')
                                    cap.set_linewidth(1.5)
                                for bar_collection in barlines_list:
                                    bar_collection.set_color('red')
                                    bar_collection.set_linewidth(1.5)
                                found_errorbar = True
                                break

                    if alpha_val is not None and epc_val is not None:  # Add text box
                        alpha_s = f"{alpha_val:.3f}"
                        alpha_e = f"{alpha_err:.1e}"
                        epc_s = f"{epc_val:.3e}"
                        epc_e = f"{epc_err:.1e}"
                        txt = f"alpha: {alpha_s}({alpha_e})\nEPC: {epc_s}({epc_e})"  # Newline for better fit
                        ax.text(0.97, 0.97, txt, transform=ax.transAxes, fontsize=8,
                                verticalalignment='top', horizontalalignment='right',
                                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='black'))

                    actual_fig.savefig(output_fname, dpi=150)
                    print(f"  Plot saved as '{output_fname}'")
                else:
                    print(f"  Could not save plot: Figure object issue.")
            else:
                print(f"  Could not save plot: No figure data.")
        except Exception as e_plot:
            print(f"  Error during plot processing/saving: {e_plot}")

    print(f"========== FINISHED EXPERIMENT: {experiment_name} ==========")
    return {"name": experiment_name, "epc": epc_val, "epc_err": epc_err, "alpha": alpha_val, "alpha_err": alpha_err}


# ==============================================================================
# MAIN SCRIPT EXECUTION - Running different experiments
# ==============================================================================

baseline_lengths = [1, 10, 20, 50, 75, 100, 125, 150, 175, 200]
baseline_nseeds = 5
baseline_p1Q = 0.002 # Your original value
baseline_p2Q = 0.01  # Your original value
baseline_physical_qubits = tuple(rb_pattern_orig[0] if 'rb_pattern_orig' in locals() else [[0,1]][0])


all_results = [] # To store results if you want a summary table later

# Experiment 0: Baseline (Re-run for a clean comparison point if you like)
print("\n--- Experiment 0: Baseline ---")
results_baseline = run_rb_experiment(
    experiment_name="Baseline",
    physical_qubits_tuple=baseline_physical_qubits,
    clifford_lengths=baseline_lengths,
    num_seeds=baseline_nseeds,
    p1q_noise=baseline_p1Q,
    p2q_noise=baseline_p2Q
)
all_results.append(results_baseline)

# --- Experiment 1: Varying Single-Qubit Noise (p1Q) ---
# 1a. Increase p1Q
print("\n--- Experiment 1a: Increased Single-Qubit Noise ---")
increased_p1Q = baseline_p1Q * 2.5 # Example: 0.002 -> 0.005
results_inc_p1q = run_rb_experiment(
    experiment_name="Increased_p1Q",
    physical_qubits_tuple=baseline_physical_qubits,
    clifford_lengths=baseline_lengths,
    num_seeds=baseline_nseeds,
    p1q_noise=increased_p1Q, # Changed parameter
    p2q_noise=baseline_p2Q   # Keep p2Q at baseline
)
all_results.append(results_inc_p1q)

# 1b. Decrease p1Q (or set to near zero)
print("\n--- Experiment 1b: Decreased Single-Qubit Noise ---")
decreased_p1Q = baseline_p1Q / 4 # Example: 0.002 -> 0.0005
# decreased_p1Q = 0.00001 # Near zero single-qubit error
results_dec_p1q = run_rb_experiment(
    experiment_name="Decreased_p1Q",
    physical_qubits_tuple=baseline_physical_qubits,
    clifford_lengths=baseline_lengths,
    num_seeds=baseline_nseeds,
    p1q_noise=decreased_p1Q, # Changed parameter
    p2q_noise=baseline_p2Q   # Keep p2Q at baseline
)
all_results.append(results_dec_p1q)


# --- Experiment 2: Varying Two-Qubit Noise (p2Q) ---
# 2a. Increase p2Q
print("\n--- Experiment 2a: Increased Two-Qubit Noise ---")
increased_p2Q = baseline_p2Q * 2.5 # Example: 0.01 -> 0.025
results_inc_p2q = run_rb_experiment(
    experiment_name="Increased_p2Q",
    physical_qubits_tuple=baseline_physical_qubits,
    clifford_lengths=baseline_lengths,
    num_seeds=baseline_nseeds,
    p1q_noise=baseline_p1Q,   # Keep p1Q at baseline
    p2q_noise=increased_p2Q  # Changed parameter
)
all_results.append(results_inc_p2q)

# 2b. Decrease p2Q
print("\n--- Experiment 2b: Decreased Two-Qubit Noise ---")
decreased_p2Q = baseline_p2Q / 4 # Example: 0.01 -> 0.0025
results_dec_p2q = run_rb_experiment(
    experiment_name="Decreased_p2Q",
    physical_qubits_tuple=baseline_physical_qubits,
    clifford_lengths=baseline_lengths,
    num_seeds=baseline_nseeds,
    p1q_noise=baseline_p1Q,   # Keep p1Q at baseline
    p2q_noise=decreased_p2Q  # Changed parameter
)
all_results.append(results_dec_p2q)

# --- Experiment 3: Varying Clifford Lengths ---

# 3a. Shorter and Fewer Clifford Lengths
print("\n--- Experiment 3a: Shorter and Fewer Clifford Lengths ---")
shorter_fewer_lengths = [1, 5, 10, 20, 30, 40, 50] # Fewer points, max length 50
results_shorter_lengths = run_rb_experiment(
    experiment_name="Shorter_Fewer_Lengths",
    physical_qubits_tuple=baseline_physical_qubits,
    clifford_lengths=shorter_fewer_lengths, # Changed parameter
    num_seeds=baseline_nseeds,
    p1q_noise=baseline_p1Q,
    p2q_noise=baseline_p2Q
)
all_results.append(results_shorter_lengths)

# 3b. More Dense Clifford Lengths (covering similar or extended range)
print("\n--- Experiment 3b: More Dense Clifford Lengths ---")
longer_dense_lengths = np.linspace(1, 200, 15, dtype=int).tolist() # ~15 points from 1 to 200
if 1 not in longer_dense_lengths and longer_dense_lengths[0] > 1:
    longer_dense_lengths = [1] + longer_dense_lengths
longer_dense_lengths = sorted(list(set(longer_dense_lengths)))
print(f"Using lengths for 3b: {longer_dense_lengths}")

results_longer_dense_lengths = run_rb_experiment(
    experiment_name="Longer_Dense_Lengths",
    physical_qubits_tuple=baseline_physical_qubits,
    clifford_lengths=longer_dense_lengths,
    num_seeds=baseline_nseeds,
    p1q_noise=baseline_p1Q,
    p2q_noise=baseline_p2Q
)
all_results.append(results_longer_dense_lengths)

print("\n\n--- SUMMARY OF RESULTS ---")
for res in all_results:
    print(f"Experiment: {res['name']:<20} | EPC: {res['epc']:.4e} \u00B1 {res['epc_err']:.1e} | Alpha: {res['alpha']:.4f} \u00B1 {res['alpha_err']:.1e}")
