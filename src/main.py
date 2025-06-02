# Import general libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure as MatplotlibFigure # For isinstance check
import warnings # Import the warnings module

# Import Qiskit classes (Modern)
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
# from qiskit import transpile # Not strictly needed for this script anymore

# Import from qiskit_experiments
from qiskit_experiments.library import StandardRB
from qiskit_experiments.framework import ExperimentData

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

noise_model = NoiseModel(basis_gates=basis_gates_orig)
noise_model.add_all_qubit_quantum_error(depolarizing_error(p1Q_orig, 1), ['u2'])
noise_model.add_all_qubit_quantum_error(depolarizing_error(2 * p1Q_orig, 1), ['u3'])
noise_model.add_all_qubit_quantum_error(depolarizing_error(p2Q_orig, 2), ['cx'])

backend = AerSimulator(noise_model=noise_model, basis_gates=basis_gates_orig)

# 4. Create the StandardRB Experiment
rb_exp = StandardRB(
    physical_qubits=physical_qubits,
    lengths=lengths_orig,
    num_samples=nseeds_orig,
    seed=42,
)
# Set transpile_options AFTER creating the instance
rb_exp.set_transpile_options(basis_gates=basis_gates_orig, optimization_level=0)

print(f"Setting up StandardRB Experiment for {physical_qubits} qubits.")
print(f"  Lengths: {lengths_orig}")
print(f"  Num Samples (seeds): {nseeds_orig}")
print(f"  Transpiling to basis: {basis_gates_orig}")

# 5. Run the Experiment
print(f"\nRunning StandardRB experiment...")

# Initialize these outside the warning block for broader scope
epc_result = None
alpha_result = None
figure_data_container = None

# --- Suppress the specific DeprecationWarning locally ---
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        message=".*Leaving `dataframe` unset or setting it to `False` for `ExperimentData.analysis_results` is deprecated.*"
    )

    exp_data: ExperimentData = rb_exp.run(backend=backend, shots=shots_orig)
    print("Experiment execution submitted. Blocking for results...")
    exp_data.block_for_results()
    print("Experiment and analysis finished.")

    # 6. Extract and Print Key Results
    epc_result = exp_data.analysis_results("EPC", dataframe=False)
    alpha_result = exp_data.analysis_results("alpha", dataframe=False)

    # 7. Get Figure Data
    print("\nProcessing RB plot data...")
    figure_data_container = exp_data.figure(0)

# --- End of warning suppression block ---


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

# Display/Process the Figure
print("\nDisplaying RB plot...")
try:
    if figure_data_container and hasattr(figure_data_container, 'figure'):
        actual_mpl_figure = figure_data_container.figure

        if isinstance(actual_mpl_figure, MatplotlibFigure):
            print("Matplotlib Figure object obtained.")
            if actual_mpl_figure.get_axes():
                ax = actual_mpl_figure.get_axes()[0]
                ax.set_title(f'{nQ_orig} Qubit RB', fontsize=18)
                print(f"Plot title set to: '{nQ_orig} Qubit RB'")

            # Save it first as a definite check
            save_filename = "rb_plot_attempt1.png"
            actual_mpl_figure.savefig(save_filename)
            print(f"Figure saved as {save_filename} (verify this file)")

            print("Attempting to display plot...")
            try:
                print("Trying canvas.draw_idle() and canvas.start_event_loop()...")
                if plt.get_backend().lower() not in ['tkagg', 'qt5agg', 'macosx']: # Add other known interactive backends
                     current_backend = plt.get_backend()
                     print(f"Current backend is {current_backend}. Forcing TkAgg for this attempt.")
                     plt.switch_backend('TkAgg') # Or 'Qt5Agg'

                canvas = actual_mpl_figure.canvas
                if canvas:
                    canvas.draw_idle()
                    plt.show()
                else:
                    print("Figure has no canvas. Falling back to plt.show().")
                    plt.show()

            except Exception as e_canvas:
                print(f"Canvas draw/event loop attempt failed: {e_canvas}")
                print("Falling back to standard plt.show().")
                plt.show() # Standard way

        else:
            print(f"Figure content is of unexpected type: {type(actual_mpl_figure)}")
    else:
        print("No FigureData object returned or it lacks a 'figure' attribute.")

except Exception as e:
    print(f"Error during plotting: {e}")
    import traceback
    traceback.print_exc()

print("\nScript finished.")