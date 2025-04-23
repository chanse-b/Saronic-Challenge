# A main driver file to predict return the 2 state predictors. This file 
# is to be run after each of the respective python notebooks, and the models
# need to be saved. To run this file,  type 'streamlit python3 driver.py' in a terminal
# and follow the menu instructions
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import streamlit as st


AHRS = pd.read_csv('ahrs.csv')
CONTROL = pd.read_csv('vehicle_control.csv')
STATE_COLS = list(AHRS.columns.drop('ts'))
NEXT_AHRS_COLS = [col+"_next" for col in STATE_COLS]
CONTROL_COLS = list(CONTROL.columns.drop('ts'))
# print(STATE_COLS)
# print(CONTROL_COLS)

def format(x):
    if isinstance(x, pd.Series):
        x = x.to_frame().T
    else:
        x = x.reset_index(drop=True)
    if 'ts' in x.columns:
        x = x.drop(columns='ts')
    return x

def predictControl(ahrs_current: pd.DataFrame, ahrs_desired: pd.DataFrame):
    """
    Predicts the next set of controls given the current state and desired state
    """
    from util import MixedRandomForest #lazy import for consistency
    ahrs_current, ahrs_desired = format(ahrs_current), format(ahrs_desired)
    X_left, X_right = ahrs_current.values, ahrs_desired.values
    X = np.hstack([X_left, X_right]) #np.concatenate([X_left, X_right], axis=1)
    turn = MixedRandomForest('turn').predict(X)[0]
    trim = MixedRandomForest('trim').predict(X)[0]
    throttle = MixedRandomForest('throttle').predict(X)[0]
    gear = MixedRandomForest('gear').predict(X)[0]
    data = {
        "gear": int(gear),
        "throttle": throttle,
        "trim": int(trim),
        "turn": turn
    }
    df = pd.DataFrame([data])  # Note: wrap in list to get one row
    df = df.round(4)
    return df

# sample1 = AHRS.sample(n=1)
# sample2 = AHRS.sample(n=1)
# print(predictControl(sample1,sample2))
    
def predictNextAHRS(ahrs_current:pd.DataFrame, control_sequence:pd.DataFrame):
    """
    Predicts the next state given the current state and control
    """
    from util import MultiHeadModel # lazy import to avoid streamlit issue with torch
    ahrs_current, control_sequence = format(ahrs_current), format(control_sequence)

    model = MultiHeadModel(input_dim=len(ahrs_current.columns)+len(control_sequence.columns)-2+6)
    state_raw = ahrs_current[STATE_COLS].values[0]
    control_raw = control_sequence[CONTROL_COLS].values[0]
    X = np.hstack([state_raw, control_raw]).reshape(1, -1)
    vals = model.predict(X)
    df = pd.DataFrame(vals, columns=STATE_COLS)
    df = df.round(4)
    return df

# sample = AHRS.sample(n=1)
# control = CONTROL.sample(n=1)
# print(predictNextAHRS(sample,control))


def distance(state_a, state_b, tol):
    """
    Gets the distance between 2 vectors
    """
    a = np.array(state_a.values).flatten()
    b = np.array(state_b.values).flatten()
    distance = np.linalg.norm(a - b)
    print(distance)
    return distance < tol



def simulate_to_target(current_state, desired_state, tolerance=.1, output_dir="simulation_outputs", cycles=20):
    """
    Simulates boat motion toward a desired state using predicted controls.
    Saves results to CSV and plots control signals over time.
    """
    os.makedirs(output_dir, exist_ok=True)
    current_state, desired_state = format(current_state), format(desired_state)

    logs = []

    # Step -1: Initial state (no control applied yet)
    nan_controls = pd.DataFrame([[np.nan] * len(CONTROL_COLS)], columns=CONTROL_COLS)
    start = pd.concat([nan_controls, current_state], axis=1)
    start["step"] = -1
    logs.append(start)

    # Step 0: Desired state (also no control, just for reference)
    desired = pd.concat([nan_controls, desired_state], axis=1)
    desired["step"] = 0
    logs.append(desired)

    for i in tqdm(range(1, cycles + 1), desc="Simulating run"):
        control = predictControl(current_state, desired_state)
        control = format(control)
        next_state = predictNextAHRS(current_state,control)
        next_state = format(next_state)

        control.columns = CONTROL_COLS 
        log_row = pd.concat([control, next_state], axis=1)
        log_row["step"] = i
        logs.append(log_row)

        if distance(next_state, desired_state, tolerance):
            break

        current_state = next_state

    sim_df = pd.concat(logs, ignore_index=True)
    # Save to CSV
    csv_path = os.path.join(output_dir, "simulation_log.csv")
    sim_df.to_csv(csv_path, index=False)
    print(f"Saved simulation log to: {csv_path}")

    # Plotting
    plt.figure(figsize=(10, 6))
    for col in CONTROL_COLS:
        if col in sim_df:
            plt.plot(sim_df["step"], sim_df[col], label=col)
    plt.xlabel("Step")
    plt.ylabel("Control Value")
    plt.title("Control Signals Over Time")
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(output_dir, "control_signals_plot.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved control plot to: {plot_path}")
    return sim_df

# i = np.random.randint(0,len(AHRS))
# start = AHRS.iloc[i:i+1]
# end = AHRS.iloc[i+1:i+2]
# simulate_to_target(start, end)




def main():
    """
    Front end using streamlit. 3 options: simulate, predict AHRS, predict Control
    """
    if "initialized" not in st.session_state: # keep the stuff so its not reinitalized when copied
        currentIndex = np.random.randint(1, len(AHRS) - 1)
        st.session_state.state_sample = AHRS.iloc[currentIndex : currentIndex + 1, :].drop('ts', axis=1)
        st.session_state.nextState_sample = AHRS.iloc[currentIndex + 1 : currentIndex + 2, :].drop('ts', axis=1)
        st.session_state.control_sample = CONTROL.sample(n=1).drop('ts', axis=1)
        st.session_state.initialized = True  # Mark it as initialized

    state_sample = st.session_state.state_sample
    nextState_sample = st.session_state.nextState_sample
    control_sample = st.session_state.control_sample
    st.title("Saronic State Predictor Challenge")

    choice = st.selectbox("Select an option:", [
        "Simulate to random target",
        "Predict control between two states",
        "Predict next state given current state + control"
    ])

    st.markdown("### Input Format:")
    st.code(",".join(STATE_COLS), language="text")
    st.code(",".join(CONTROL_COLS), language="text")

    #options
    if choice == "Simulate to random target":
        st.markdown(
            "Uses both predictors to simulate a boat's journey from its current state to the desired state, "
            "based on predicted controls. At each step, the control predictor suggests how to move toward the goal, "
            "and the AHRS model estimates the resulting next state. "
            "This process repeats for a number of timesteps, or until the goal is reached."
        )
        st.markdown('Current')
        st.dataframe(state_sample)
        randomSample = AHRS.sample(n=1).drop('ts',axis=1)
        st.markdown("Target")
        st.dataframe(randomSample)
        numRuns = st.number_input("Number of steps", value=20)
        gap = st.number_input("Tolerance (Euclidian margin to target) for early stopping", value=10,step=1)
        if st.button("Simulate"):
            try:
                result = simulate_to_target(state_sample,randomSample,cycles=numRuns,tolerance=gap)
                result = pd.read_csv('simulation_outputs/simulation_log.csv')
                st.markdown("Note: Index 0 is the current, Index 1 is the desired")
                st.dataframe(result)
                st.image('simulation_outputs/control_signals_plot.png', caption="Control Signals Over Time", use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")
    elif choice == "Predict control between two states":
        st.markdown("#### Samples to get you started:")
        st.markdown("**Editable Start State Sample**")
        current_df = st.data_editor(state_sample.reset_index(drop=True), key="current_state_editor")

        st.markdown("**Editable Desired State Sample**")
        desired_df = st.data_editor(nextState_sample.reset_index(drop=True), key="desired_state_editor")

        # Copyable previews that sync
        st.markdown("**Current State**")
        st.code(",".join(map(str, current_df.iloc[0].tolist())))

        st.markdown("**Desired State**")
        st.code(",".join(map(str, desired_df.iloc[0].tolist())))

        # User-entered 
        st.markdown("**Enter your inputs (comma-separated):**")
        current_input = st.text_input("Current State:", "")
        desired_input = st.text_input("Desired State:", "")

        if st.button("Predict Control"):
            try:
                current_vals = [float(x) for x in current_input.strip().split(",")]
                desired_vals = [float(x) for x in desired_input.strip().split(",")]

                current = pd.DataFrame([current_vals], columns=STATE_COLS)
                desired = pd.DataFrame([desired_vals], columns=STATE_COLS)

                result = predictControl(current, desired)
                st.dataframe(result)
            except Exception as e:
                st.error(f"Error: {e}")

    elif choice == "Predict next state given current state + control":
        st.markdown("#### Samples to get you started:")
        st.markdown("**Editable Sample State**")
        state_df = st.data_editor(state_sample.reset_index(drop=True), key="state_editor")
        st.markdown("**Editable Sample Control**")
        control_df = st.data_editor(control_sample.reset_index(drop=True), key="control_editor")
        st.markdown("**Current State (copy-ready)**")
        st.code(",".join(map(str, state_df.iloc[0].tolist())))
        st.markdown("**Control Input (copy-ready)**")
        st.code(",".join(map(str, control_df.iloc[0].tolist())))
        state_input = st.text_input("Enter Current State (comma-separated):", "")
        control_input = st.text_area(
            "Enter Control Input (one per line, comma-separated):",
            height=150,
            placeholder="Example:\n1,0.5,0,-0.25\n0,0.3,1,0.1"
        )

        if st.button("Predict Next State"):
            try:
                # Parse state input
                state_vals = [float(x) for x in state_input.strip().split(",")]
                state = pd.DataFrame([state_vals], columns=STATE_COLS)

                # Parse control input (use the last line)
                lines = [line.strip() for line in control_input.strip().splitlines() if line.strip()]
                last_line = lines[-1]
                control_vals = [float(x) for x in last_line.split(",")]
                control = pd.DataFrame([control_vals], columns=CONTROL_COLS)

                result = predictNextAHRS(state, control)
                st.dataframe(result)

            except Exception as e:
                st.error(f"Error: {e}")

if __name__ == "__main__":
    main()    
