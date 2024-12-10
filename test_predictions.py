from train_model import predict_heart_disease
from colorama import init, Fore, Style
import time

# Initialize colorama for colored output
init()

def print_header(text, char="=", color=Fore.BLUE):
    """Print a formatted header"""
    width = 70
    print(f"\n{color}{char * width}")
    print(f"{text.center(width)}")
    print(f"{char * width}{Style.RESET_ALL}")

def print_section(text, color=Fore.CYAN):
    """Print a section header"""
    print(f"\n{color}{text}{Style.RESET_ALL}")

def format_value(value):
    """Format values with appropriate coloring"""
    if isinstance(value, bool):
        return f"{Fore.GREEN}Yes{Style.RESET_ALL}" if value else f"{Fore.RED}No{Style.RESET_ALL}"
    if isinstance(value, (int, float)):
        return f"{Fore.YELLOW}{value}{Style.RESET_ALL}"
    return str(value)

def run_test_cases():
    """Run multiple test cases to verify the heart disease prediction model"""
    
    test_cases = [
        # Test Case 1: Normal case (Low Risk)
        {
            "Age": 45,
            "Sex": 0,  # Female
            "ChestPainType": 0,
            "RestingBP": 120,
            "Cholesterol": 200,
            "FastingBS": 0,
            "RestingECG": 0,
            "MaxHR": 160,
            "ExerciseAngina": 0,
            "Oldpeak": 0.0,
            "ST_Slope": 1,
            "NumMajorVessels": 0,
            "Thal": 2
        },
        
        # Test Case 2: High Risk Case
        {
            "Age": 65,
            "Sex": 1,  # Male
            "ChestPainType": 2,
            "RestingBP": 160,
            "Cholesterol": 300,
            "FastingBS": 1,
            "RestingECG": 2,
            "MaxHR": 120,
            "ExerciseAngina": 1,
            "Oldpeak": 2.5,
            "ST_Slope": 0,
            "NumMajorVessels": 3,
            "Thal": 3
        },
        
        # Test Case 3: Borderline Case
        {
            "Age": 55,
            "Sex": 1,
            "ChestPainType": 1,
            "RestingBP": 140,
            "Cholesterol": 250,
            "FastingBS": 0,
            "RestingECG": 1,
            "MaxHR": 145,
            "ExerciseAngina": 0,
            "Oldpeak": 1.5,
            "ST_Slope": 1,
            "NumMajorVessels": 1,
            "Thal": 2
        },
        
        # Test Case 4: Edge Case - Minimum Values
        {
            "Age": 29,
            "Sex": 0,
            "ChestPainType": 0,
            "RestingBP": 94,
            "Cholesterol": 126,
            "FastingBS": 0,
            "RestingECG": 0,
            "MaxHR": 71,
            "ExerciseAngina": 0,
            "Oldpeak": 0.0,
            "ST_Slope": 0,
            "NumMajorVessels": 0,
            "Thal": 0
        },
        
        # Test Case 5: Edge Case - Maximum Values
        {
            "Age": 77,
            "Sex": 1,
            "ChestPainType": 3,
            "RestingBP": 200,
            "Cholesterol": 564,
            "FastingBS": 1,
            "RestingECG": 2,
            "MaxHR": 202,
            "ExerciseAngina": 1,
            "Oldpeak": 6.2,
            "ST_Slope": 2,
            "NumMajorVessels": 4,
            "Thal": 3
        }
    ]
    
    print_header("Heart Disease Prediction Model - Test Suite", "=", Fore.MAGENTA)
    print(f"{Fore.WHITE}Starting test execution at: {time.strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}")
    
    total_tests = len(test_cases)
    successful_tests = 0
    
    for i, test_data in enumerate(test_cases, 1):
        print_header(f"Test Case {i}/{total_tests}", "-", Fore.BLUE)
        
        # Print test case description
        if i == 1:
            print(f"{Fore.CYAN}Description: Normal case (Low Risk){Style.RESET_ALL}")
        elif i == 2:
            print(f"{Fore.CYAN}Description: High Risk Case{Style.RESET_ALL}")
        elif i == 3:
            print(f"{Fore.CYAN}Description: Borderline Case{Style.RESET_ALL}")
        elif i == 4:
            print(f"{Fore.CYAN}Description: Edge Case - Minimum Values{Style.RESET_ALL}")
        elif i == 5:
            print(f"{Fore.CYAN}Description: Edge Case - Maximum Values{Style.RESET_ALL}")
        
        print_section("Input Parameters:")
        # Print input parameters in columns
        params = list(test_data.items())
        mid = len(params) // 2
        for (k1, v1), (k2, v2) in zip(params[:mid], params[mid:]):
            print(f"{k1:15}: {format_value(v1):10} {k2:15}: {format_value(v2)}")
        
        print_section("Prediction Results:")
        result = predict_heart_disease(test_data)
        
        if "error" in result:
            print(f"{Fore.RED}Error: {result['error']}{Style.RESET_ALL}")
        else:
            successful_tests += 1
            prediction = "Yes" if result['prediction'] == 1 else "No"
            probability = result['probability']
            
            # Color-code the prediction based on probability
            if probability > 0.7:
                pred_color = Fore.RED
            elif probability > 0.3:
                pred_color = Fore.YELLOW
            else:
                pred_color = Fore.GREEN
                
            print(f"Heart Disease Risk: {pred_color}{prediction}{Style.RESET_ALL}")
            print(f"Probability: {pred_color}{probability:.2%}{Style.RESET_ALL}")
            
            # Add risk level indicator
            risk_level = "High" if probability > 0.7 else "Medium" if probability > 0.3 else "Low"
            print(f"Risk Level: {pred_color}{risk_level}{Style.RESET_ALL}")
        
        time.sleep(0.5)  # Add small delay between tests for better readability
    
    # Print summary
    print_header("Test Summary", "=", Fore.MAGENTA)
    print(f"Total Tests Run: {total_tests}")
    print(f"Successful Tests: {Fore.GREEN}{successful_tests}{Style.RESET_ALL}")
    print(f"Failed Tests: {Fore.RED}{total_tests - successful_tests}{Style.RESET_ALL}")
    print(f"Success Rate: {Fore.YELLOW}{(successful_tests/total_tests)*100:.1f}%{Style.RESET_ALL}")
    print(f"\nTest execution completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    try:
        run_test_cases()
    except KeyboardInterrupt:
        print(f"\n{Fore.RED}Test execution interrupted by user.{Style.RESET_ALL}")
    except Exception as e:
        print(f"\n{Fore.RED}Error during test execution: {str(e)}{Style.RESET_ALL}")