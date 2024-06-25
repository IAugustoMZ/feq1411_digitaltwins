# FEQ1411 - Digital Twins and Simulation

This is a web-based application for the course FEQ1411 - Digital Twins and Simulation. The dashboard is built using Plotly Dash and allows users to simulate and visualize various parameters related to digital twins.

## Features

- Interactive dashboard with real-time updates
- Simulation of feed temperature, feed pressure, anomaly probability, and average simulation error
- User interface components including buttons, switches, and graphs
- Responsive design with Bootstrap components

## Installation

### Prerequisites

- Python 3.7 or higher
- `pip` package manager

### Clone the Repository

```bash
git clone <repository-url>
cd <repository-directory>
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
python app.py
```

The application will be available at `http://127.0.0.1:8050/`.

## Project Structure

```
├── app.py               # Main application script
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## Usage

1. Open your web browser and navigate to `http://127.0.0.1:8050/`.
2. Use the sidebar to control the simulation parameters:
   - **Execute Button**: Start the simulation.
   - **Robust Switch**: Toggle the robustness mode.
3. The main content area displays real-time updates of the following parameters:
   - Feed Temperature (deg. C)
   - Feed Pressure (bar)
   - Anomaly Probability
   - Avg. Simulation Error (kJ/kg)
4. Interactive graphs provide a visual representation of the simulation data.

## Dependencies

- Dash
- Dash Bootstrap Components
- Joblib
- NumPy
- Pandas
- Plotly

## Acknowledgements

This project is developed as part of the FEQ1411 - Digital Twins and Simulation course.

## Screenshots of the solution

![image](https://github.com/IAugustoMZ/feq1411_digitaltwins/assets/42342168/71614254-5985-4598-8fed-e1fe65ee5904)

![image](https://github.com/IAugustoMZ/feq1411_digitaltwins/assets/42342168/40f3756a-38ec-4835-a72b-21e790064178)

![image](https://github.com/IAugustoMZ/feq1411_digitaltwins/assets/42342168/d882afcb-b9f2-4549-8be4-000d12dc0bf3)

