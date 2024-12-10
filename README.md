# Heart Disease Prediction System

A machine learning system that predicts the likelihood of heart disease based on patient data.

## Quick Start (2 Steps)

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Run Tests**
```bash
python test_predictions.py
```

## Project Files
```
heart_disease_prediction/
├── heart.csv                               # Dataset
├── create_scaler.py                        # Scaler creation script
├── train_model.py                          # Prediction model
├── test_predictions.py                     # Test suite
├── scaler.pkl                              # Pre-trained scaler
├── heart_disease_classification_model.pkl   # Pre-trained model
└── requirements.txt                        # Dependencies list
```

## Expected Output
The test suite will display colored output showing:
- Test case descriptions
- Input parameters for each test
- Prediction results with:
  - Heart Disease Risk (Yes/No)
  - Probability percentage
  - Risk Level (Low/Medium/High)
- Overall test summary

## Troubleshooting

If you encounter errors:
1. Verify Python 3.7+ is installed
2. Ensure all files are in the same directory
3. Check file permissions
4. Run `pip install -r requirements.txt` again

## Notes
- Results are color-coded:
  - Green: Low risk
  - Yellow: Medium risk
  - Red: High risk
- Keep all files in the same directory
- Do not modify the `.pkl` files
