# Import general libraries (needed for functions)
import warnings
import matplotlib.pyplot as plt
from matplotlib.figure import Figure as MatplotlibFigure # For isinstance check

# Import Qiskit classes
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_experiments.library import StandardRB
from qiskit_experiments.framework import ExperimentData


# --- Parameters ---
lengths = [1, 10, 20, 50, 75, 100, 125, 150, 175, 200]
num_samples = 5
physical_qubits = (0, 1)
shots_per_circuit = 200

# --- Noise Model Setup ---
noise_model = NoiseModel()
p1Q = 0.002
p2Q = 0.01
noise_model.add_all_qubit_quantum_error(depolarizing_error(p1Q, 1), ['sx', 'x'])
noise_model.add_all_qubit_quantum_error(depolarizing_error(p1Q, 1), ['u2'])
noise_model.add_all_qubit_quantum_error(depolarizing_error(2 * p1Q, 1), ['u3'])
noise_model.add_all_qubit_quantum_error(depolarizing_error(p2Q, 2), ['cx'])

# --- Backend Setup ---
backend = AerSimulator(noise_model=noise_model)

print("Setting up StandardRB Experiment...")
rb_exp = StandardRB(
    physical_qubits=physical_qubits,
    lengths=lengths,
    num_samples=num_samples,
    seed=42
)

print(f"Running StandardRB on qubits {physical_qubits} with {num_samples} seeds and lengths up to {max(lengths)}.")

epc_result_data_obj = None

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        message=".*Leaving `dataframe` unset or setting it to `False` for `ExperimentData.analysis_results` is deprecated.*"
    )

    exp_data: ExperimentData = rb_exp.run(backend=backend, shots=shots_per_circuit)
    exp_data.block_for_results()

    print("Experiment finished. Analyzing results...")
    # ... (your analysis results printing if you want it here) ...

    # --- PLOTTING BLOCK BASED ON SOURCE CODE INSIGHTS ---
    print("\nProcessing and attempting to save/display RB plot...")
    try:
        # exp_data.figure(0) returns a FigureData object (let's call it figure_container)
        figure_container = exp_data.figure(0)

        if figure_container and hasattr(figure_container, 'figure'):
            # figure_container.figure is the actual figure data (MatplotlibFigure, str, or bytes)
            raw_figure_content = figure_container.figure
            figure_name = figure_container.name # e.g., "StandardRB_Q0_Q1_figure.svg"

            if isinstance(raw_figure_content, MatplotlibFigure):
                print(f"Obtained Matplotlib Figure object: {figure_name}")
                # Save it as PNG
                save_filename_png = "rb_plot_from_mpl_fig.png"
                raw_figure_content.savefig(save_filename_png)
                print(f"Figure saved as {save_filename_png}")

                # Attempt to show it using plt.show()
                # To make plt.show() aware of this figure, we can make it "current"
                # This might not always be necessary if it's the only figure.
                # plt.figure(raw_figure_content.number) # This makes it the current figure for pyplot
                print("Attempting to display Matplotlib Figure with plt.show()...")
                plt.show() # This should now display the active Matplotlib figures

            elif isinstance(raw_figure_content, (str, bytes)):
                print(f"Obtained SVG data (string or bytes) for figure: {figure_name}")
                save_filename_svg = figure_name if figure_name.endswith(".svg") else "rb_plot_from_svg_data.svg"
                mode = "w" if isinstance(raw_figure_content, str) else "wb"
                encoding = "utf-8" if isinstance(raw_figure_content, str) else None
                with open(save_filename_svg, mode, encoding=encoding) as f:
                    f.write(raw_figure_content)
                print(f"Figure saved as {save_filename_svg}")
                print("SVG data cannot be directly displayed with plt.show(). Open the .svg file manually.")

            else:
                print(f"Figure content is of an unexpected type: {type(raw_figure_content)}")
        else:
            print("No FigureData object returned by exp_data.figure(0) or it lacks a 'figure' attribute.")
            if figure_container:
                print(f"Object returned by exp_data.figure(0): {figure_container}")
                print(f"Type: {type(figure_container)}")

    except Exception as e:
        print(f"Error processing or displaying/saving figure: {e}")
        import traceback
        traceback.print_exc()
    # --- END OF PLOTTING BLOCK ---

    epc_result_data_obj = exp_data.analysis_results("EPC", dataframe=False)
    # ... (your EPC printing) ...

print("\nScript finished.")