# GuardPulse: Stress Detection from Physiological Time-Series Data

Early stress and burnout detection for uniformed personnel using wearable sensor data, trained and validated on the WESAD dataset.

## Overview

Traditional stress monitoring relies on manual observation or self-report—both catch problems late. GuardPulse learns to recognize early stress and burnout signals from physiological time-series data:

- **Heart-rate variability (HRV)** and activity patterns as primary features
- **WESAD dataset** for honest benchmarking with labelled stress conditions
- **Leave-one-subject-out (LOSO) evaluation** to avoid overstating accuracy
- **Confidence bands over time** rather than single instantaneous readings
- **Sustained pattern detection** to avoid false positives from noisy spikes

## Key Features

### 1. Stress Detection Model
- Physiological feature extraction from chest and wrist sensor data
- scikit-learn and PyTorch implementations
- LOSO cross-validation for honest subject-generalization metrics
- Uncertainty quantification with confidence bands

### 2. Dashboard
- Chart.js visualization of stress trends with confidence bands
- Real-time escalation alerts based on sustained patterns (not single spikes)
- Integration-ready for wearable devices (MAX30102 pulse sensor on ESP32, ~$300)

### 3. Stretch Goal: Self-Report Integration
- Micro-survey integration to show how self-reports shift confidence bands
- Comparison of model-only vs. model + survey predictions

## Dataset

**WESAD** (Wearable Stress and Affect Detection)
- Multi-modal wearable sensors (chest and wrist)
- Labelled stress conditions from controlled experiments
- Standard benchmark for physiological stress detection
- Download: https://github.com/amd-a/wesad

## Tech Stack

- **ML/Data**: scikit-learn, PyTorch, pandas, numpy
- **Visualization**: Chart.js
- **Hardware Integration**: ESP32, MAX30102 pulse sensor
- **Python 3.8+**

## Project Structure

```
guardpulse-stress-detection/
├── README.md
├── requirements.txt
├── data/
│   └── wesad/              # WESAD dataset (not included, download separately)
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # WESAD data loading and preprocessing
│   ├── feature_extraction.py # HRV and activity feature engineering
│   ├── models/
│   │   ├── sklearn_model.py
│   │   └── pytorch_model.py
│   ├── evaluation.py        # LOSO cross-validation and metrics
│   └── confidence_bands.py  # Uncertainty quantification
├── dashboard/
│   ├── index.html
│   ├── app.js
│   └── chart-config.js     # Chart.js configuration
├── hardware/
│   ├── esp32_pulse_sensor/
│   │   └── main.ino        # MAX30102 integration sketch
│   └── sensor_config.py    # Hardware calibration
└── notebooks/
    └── exploratory_analysis.ipynb
```

## Getting Started

### 1. Clone and Install

```bash
git clone https://github.com/dhanusharajkumar236-cpu/guardpulse-stress-detection.git
cd guardpulse-stress-detection
pip install -r requirements.txt
```

### 2. Download WESAD Dataset

```bash
# Download from https://github.com/amd-a/wesad
# Extract to data/wesad/
ls data/wesad/  # Should contain S2, S3, ..., S17 directories
```

### 3. Run Feature Extraction Pipeline

```bash
python src/data_loader.py
python src/feature_extraction.py
```

### 4. Train and Evaluate (LOSO)

```bash
python src/evaluation.py
```

### 5. Launch Dashboard

```bash
cd dashboard
python -m http.server 8000
# Open http://localhost:8000 in browser
```

## Results

- **Model Accuracy** (LOSO evaluation): [TBD]
- **Confidence Band Coverage**: [TBD]
- **Sustained Pattern Detection Rate**: [TBD]

## Contributing

Contributions welcome! Please:
1. Create a feature branch: `git checkout -b feature/your-feature`
2. Commit with clear messages: `git commit -m "Add feature description"`
3. Push and open a pull request

## License

MIT License

## References

- WESAD: Schmidt, P., et al. (2018). Introducing WESAD—A Multimodal Dataset for Wearable Stress and Affect Detection.
- HRV Features: Task Force of the European Society of Cardiology (1996). Heart rate variability.
