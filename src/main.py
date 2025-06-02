# Import general libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure as MatplotlibFigure  # For isinstance check
import warnings

# Import Qiskit classes (Modern)
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

# Import from qiskit_experiments
from qiskit_experiments.library import StandardRB
from qiskit_experiments.framework import ExperimentData

# --- Suppress specific warnings ---
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

# --- Original Ignis Script Parameters ---
nQ_orig = 2
lengths_orig = [1, 10, 20, 50, 75, 100, 125, 150, 175, 200]
nseeds_orig = 5
rb_pattern_orig = [[0, 1]]
p1Q_orig = 0.002
p2Q_orig = 0.01
basis_gates_orig = ['u1', 'u2', 'u3', 'cx']
shots_orig = 200

# --- Setup for qiskit_experiments ---
physical_qubits = tuple(rb_pattern_orig[0])

noise_model = NoiseModel()
noise_model.add_all_qubit_quantum_error(depolarizing_error(p1Q_orig, 1), ['u2'])
noise_model.add_all_qubit_quantum_error(depolarizing_error(2 * p1Q_orig, 1), ['u3'])
noise_model.add_all_qubit_quantum_error(depolarizing_error(p2Q_orig, 2), ['cx'])

backend = AerSimulator(noise_model=noise_model, basis_gates=basis_gates_orig)

rb_exp = StandardRB(
    physical_qubits=physical_qubits,
    lengths=lengths_orig,
    num_samples=nseeds_orig,
)
rb_exp.set_transpile_options(basis_gates=basis_gates_orig, optimization_level=0)

print(f"Setting up StandardRB Experiment for {physical_qubits} qubits.")

print(f"\nRunning StandardRB experiment...")
exp_data: ExperimentData = rb_exp.run(backend=backend, shots=shots_orig)
print("Experiment execution submitted. Blocking for results...")
exp_data.block_for_results()
print("Experiment and analysis finished.")

# Extract Key Results
epc_result = exp_data.analysis_results("EPC", dataframe=False)
alpha_result = exp_data.analysis_results("alpha", dataframe=False)
figure_data_container = exp_data.figure(0)

epc_value, epc_stderr, alpha_value, alpha_stderr = None, None, None, None

if epc_result:
    epc_value = epc_result.value.nominal_value
    epc_stderr = epc_result.value.std_dev
    print(f"\nError Per Clifford (EPC): {epc_value:.4e} \u00B1 {epc_stderr:.4e}")
else:
    print("\nEPC result not found.")

if alpha_result:
    alpha_value = alpha_result.value.nominal_value
    alpha_stderr = alpha_result.value.std_dev
    print(f"Fit parameter alpha: {alpha_value:.4f} \u00B1 {alpha_stderr:.4f}")
else:
    print("Alpha result not found.")

print("\nCustomizing and saving RB plot as PNG...")
output_filename = "customized_rb_plot.png"
try:
    if figure_data_container and hasattr(figure_data_container, 'figure'):
        actual_mpl_figure: MatplotlibFigure = figure_data_container.figure

        if isinstance(actual_mpl_figure, MatplotlibFigure):
            print("Matplotlib Figure object obtained for customization.")

            if not actual_mpl_figure.get_axes():
                print("Figure has no axes. Cannot customize or save.")
            else:
                ax = actual_mpl_figure.get_axes()[0]

                # 1. Set Title
                ax.set_title(f'{nQ_orig} Qubit RB', fontsize=18)

                # 2. Set Y-axis Label
                ax.set_ylabel("Ground State Population", fontsize=14)
                print("Y-axis label set.")

                # 3. Restyle the averaged data points and error bars to red 'x'
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
                if not found_errorbar:
                    print("Could not robustly identify the error bar container for averaged data to restyle to red.")

                # 4. Add Alpha/EPC Text Box
                if alpha_value is not None and epc_value is not None:
                    alpha_val_str = f"{alpha_value:.3f}"
                    alpha_err_str = f"{alpha_stderr:.1e}"
                    epc_val_str = f"{epc_value:.3e}"
                    epc_err_str = f"{epc_stderr:.1e}"
                    text_str = f"alpha: {alpha_val_str}({alpha_err_str}) EPC: {epc_val_str}({epc_err_str})"

                    ax.text(0.97, 0.97, text_str,
                            transform=ax.transAxes,
                            fontsize=9,
                            verticalalignment='top',
                            horizontalalignment='right',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='black'))
                    print(f"Added text box: {text_str}")
                else:
                    print("Alpha or EPC values not available for text box.")

                # 5. Save the customized figure
                actual_mpl_figure.savefig(output_filename, dpi=150)  # dpi can be adjusted
                print(f"Customized figure saved as '{output_filename}'")
        else:
            print(
                f"Figure content is of unexpected type: {type(actual_mpl_figure)}. Cannot save as PNG from this type.")
    else:
        print("No FigureData object returned or it lacks a 'figure' attribute. Cannot save plot.")

except Exception as e:
    print(f"Error during plot customization or saving: {e}")
    import traceback

    traceback.print_exc()

print("\nScript finished.")