# Road Sign Detection App

A real-time road sign detection application using YOLOv8 trained on the GTSRB (German Traffic Sign Recognition Benchmark) dataset.

## Features
- Real-time road sign detection using webcam
- Support for 43 different traffic sign classes
- Display of sign name, description, and detection confidence

## Requirements
- Python 3.8+
- OpenCV
- Streamlit
- Ultralytics YOLOv8
- Other dependencies listed in requirements.txt

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/traffic-sign-detector.git
cd traffic-sign-detector
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```


## Usage
Run the application using Streamlit:
```bash
streamlit run app.py
```

```

## Model Training
The model was trained on the GTSRB dataset using YOLOv8. The training data is not included in this repository due to size constraints.

