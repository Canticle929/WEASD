# WESAD Data Processing

This project processes the WESAD (Wearable Stress and Affect Detection) dataset, focusing on extracting and cleaning physiological signals for stress detection.

## Features

- Extracts ECG, EDA, and Resp signals from the WESAD dataset
- Retains only baseline (label 1) and stress (label 2) states
- Generates comprehensive statistics and visualizations
- Creates cleaned dataset ready for machine learning tasks

## Requirements

- Python 3.6+
- NumPy
- Pandas
- Matplotlib
- SciPy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/wesad-processing.git
cd wesad-processing
```

2. Install dependencies:
```bash
pip install numpy pandas matplotlib scipy
```

3. Place your WESAD dataset in the `WESAD/raw` directory.

## Usage

Run the data cleaning script:
```bash
python data_cleaning.py
```

The script will:
1. Process all subjects in the dataset
2. Extract ECG, EDA, and Resp signals
3. Filter data to retain only baseline and stress states
4. Generate statistics and visualizations
5. Save cleaned data in the `WESAD/cleaned_data` directory

## Output

The script generates:
- Cleaned signal data (ECG, EDA, Resp) for each subject
- Label information and sampling rates
- Signal visualizations
- Dataset statistics and summary reports
- Label distribution visualizations

## Directory Structure

```
wesad-processing/
├── data_cleaning.py      # Main processing script
├── README.md            # Project documentation
├── WESAD/
│   ├── raw/            # Original WESAD dataset
│   └── cleaned_data/   # Processed output data
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite the original WESAD dataset:

```
Schmidt, P., Reiss, A., Duerichen, R., Marberger, C., & Van Laerhoven, K. (2018). 
Introducing WESAD, a multimodal dataset for Wearable Stress and Affect Detection. 
In Proceedings of the 20th ACM International Conference on Multimodal Interaction. 