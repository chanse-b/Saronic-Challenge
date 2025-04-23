# OVERVIEW 

---

### requirements.txt
A convenience file to pip install the necessary packages for this challenge

### util.py
Utility file that contains the classes copied from both ipython notebook files. Mostly for convenience to run in driver.py

### driver.py
A user-facing python script that allows the user to input fresh commands to predict the next state or next control. Also has a simulation mode that uses both models to simultatnously predict the control and the next state. Run the file with the command 'streamlit run driver.py'

### Analyzer.ipynb
python notebook for a data exploration. Purpose is to guide my understanding of the data to choose the appropriate models for each of the predictors.

### predCONTROL.ipynb
This is the python notebook that builds the model to predict the controls given state. There are also metrics and performance data as well. Saves the models in 'Control Models'

### predAHRS.ipynb
This is the python notebook that builds the model to predict the next state given the controls and the current state. There are also metrics and performance data as well. Saves the models in 'AHRS Models'

### plots
Includes AHRS/Control plots for the predictors and how accurate they were. There are also plots for the importances of each variable when predicting gear,throttle,trim, and turn

### simulation_outputs
contains a csv file with the predicted state given the control. The first two entries are the start and desired, respectively


---

# INSTRUCTIONS

In a new terminal, run the following to create a new environment with the packages needed for this project. Then run the python notebooks

1. python3 -m venv state_predictor_env
2. source state_predictor_env/bin/activate
3. pip install --upgrade pip
4. pip install -r requirements.txt
5. run all predAHRS.ipynb
6. run all predControl.ipynb
6. streamlit run driver.py 

Notes: The inputs in the ipynb files are randomized so errors and performance statistics may vary accross runs.

---

# DESIGN DECISONS

My solution assumes that the vehicle dynamics are Markovian — meaning the next vehicle state depends only on the current state and the (most recently) applied control, rather than on the full sequence of previous states or control inputs. Likewise, for predicting a control sequence to reach a desired state, I assume that the requird controls are fully determined by the current state and target state, without needing access to prior history. 

FFrom a practical standpoint, attempting to predict full variable-length control sequences introduces a key challenge. These sequences naturally vary in length between different state transitions, leading to inconsistent input and output sizes, which are incompatible with standard feedforward models unless complex padding and truncation are introduced — all of which come with increased training time, memory use, and the need for more powerful hardware, such as a GPU. This would be difficult to support on limited resources, such as a local development environment or a standard MacBook. 

By inspection, I also was able to see the window of the control sequence between timestep t and t+1 were often consistant values for the entirety of the sequence (See Analyzer.ipynb). 

| i       | ts                                   | gear | throttle  | trim | turn     |
|--------:|--------------------------------------|:----:|----------:|:----:|---------:|
| 312222  | 2024-07-31 14:54:42.510909641        |  0   | 0.001221  |  0   | 0.002442 |
| 312223  | 2024-07-31 14:54:42.515483645        |  0   | 0.001221  |  0   | 0.002442 |
| 312224  | 2024-07-31 14:54:42.532317330        |  0   | 0.003663  |  0   | 0.002442 |
| 312225  | 2024-07-31 14:54:42.549252145        |  0   | 0.001221  |  0   | 0.002442 |
| 312226  | 2024-07-31 14:54:42.566114331        |  0   | 0.001221  |  0   | 0.002442 |
| 312227  | 2024-07-31 14:54:42.582947391        |  0   | 0.001221  |  0   | 0.002442 |
| 312228  | 2024-07-31 14:54:42.599776825        |  0   | 0.001221  |  0   | 0.002442 |

This indicates that control values tend to remain nearly static over short time intervals, which supports the Markovian assumption and simplifies the modeling problem, making it measurable and computationally feasible. I did, however, include a simulation option in `driver.py` that enables the sequential prediction of controls by using the state predictor model as a baseline for new sensor data.This assumption significantly simplifies the modeling problem, making it measurable, and computationally feasible.

With the assumption in place, we can now turn to model selection. For predicting control action given a current and desired state, I chose a random forest approach. Specifically, I trained four separate models: two regressors (throttle and turn) and two classifiers (trim and gear). This  decision was mostly grounded in intuition — if I were piloting a boat and wanted to 'increase' yaw, I’d instinctively 'turn right'. Such threshold-based decision-making, where discrete actions stem from directional changes in state, aligns well with the structurd decision boundaries random forests can learn. Their ensemble nature also helps in handling noisy or imprecise mappings between states and actions.

In contrast, predicting the next state given a current state and control input requires modeling more complex, continuous dynamics interwoven with discrete signals. While technically possible with a random forest, I beleived it would struggle to capture the nuanced interactions without significantly increasing model complexity. For this reason, I opted for a multi-layer perceptron architecture. Drawing from prior experience with noisy input spaces, I designed the network with four attention heads — each tuned to focus on a distinct dynamic: velocty, acceleration, angular rate, and orientation. This modular structure allowed for customization for each task. After experimentation, I unified the architecture for three of the four heads, while assigning a more complex structure to the acceleration head due to its initially weaker performance.

---

# REFLECTION AND FUTURE WORK

I chose to model a single-step control predictor. In a real-world application, this model would be likely in a closed-loop system, where it continuously receives the latest AHRS state, compares it to the goal state, and outputs the next control input. This loop naturally results in a full control sequence over time, while also allowing the system to adapt to disturbances (wind, current, etc.) in addition to delays, and feedback — just like a real autopilot for a aircraft. I attempted to model this in driver.py, in 'simulation' mode where I use both models to predict the control, and given the control and the predict the subsequent state. In realty, a 'autopilot' for a boat in this scenario would stream in live data while the controls would be continously adjusted based on the incoming states.

Given access to a GPU and additional telemetry data such as wind and current, I would consider switching to a different model for next-state prediction. A Long Short-Term Memory (LSTM) network could be a strong candidate, as it can incorporate temporal context by retaining a memory of past inputs. This could allow the model to better capture dynamic patterns over time.But beyond purely machine learning–based approaches, a physics-informed or hybrid model that combines ML with physical principles may offer even greater accuracy and interpretability. However, since I don’t have deep expertise in vehicle dynamics, I’d want to collaborate with domain engineers to develop such a model. That way, we avoid relying on oversimplified assumptions like neglecting drag, wind,current, or other real-world forces.Perhaphs even more data with conditions might yield even better results. If the current dataset was in choppy seas, how might the model perform in calm water? 

Overall, I am happy with the performance of both models. The plots indicate strong performance for most metrics, however, the acceleration components demonstrate considerable drop in performance. Possibly, acceleration is hard for most linear models to generalize, and perhaphs a physics-based hybrid model mentioned previously may pay dividends in the future. 