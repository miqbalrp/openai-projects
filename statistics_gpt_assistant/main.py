from gpt_function import *
from util import *

print("Starting the program...")

print("STEP 1 : Declare and confirm the analysis objective...")
suggested_method, suggested_method_explanation = get_suggested_method()
method = check_method_availability(suggested_method)

print("STEP 2 : Upload Dataset...")
try:
    structure, example = get_dataset_structure(method=method)
    print(f"[ASSISTANT] Data structure : {structure}")
    print("[ASSISTANT] Please follow below example :")
    print(f"==================== \n{example}\n ====================")

except ValueError:
    print("[ASSISTANT] The method is unsupported")

print("[ASSISTANT] Select the file contain your dataset: ")
csv_file_path = select_file()
if csv_file_path:
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file_path)
    print("[ASSISTANT] Below is your dataset sample:")
    print(f"====================\n{df.head()}\n====================")

