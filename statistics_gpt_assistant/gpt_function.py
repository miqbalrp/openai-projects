import pandas as pd
import numpy as np
import os

from openai import OpenAI
import json
from tqdm import tqdm

test_list = ['t-test', 'mann-whitney', 'paired t-test', 'wilcoxon', 'anova', 'kruskal-wallis', 'chi-square', 'pearson', 'spearman', 'kendall', 'other']

def get_chat_completion(messages, tools=None, tool_choice=None):
    import os
    from dotenv import load_dotenv

    # Load environment variables from the .env file
    load_dotenv()
    openai_api_key = os.getenv('OPENAI_API_KEY')

    client = OpenAI(api_key=openai_api_key)

    return client.chat.completions.create(
        messages=messages,
        model="gpt-3.5-turbo",
        temperature=0.3,
        max_tokens=4096,
        top_p=0.8,

        tools=tools,
        tool_choice=tool_choice
    )

def get_suggested_method():
    system_content = """
    You are a statistics expert to help the user on giving the suggestion of what method that best to used best on the objective of the analysis.
    User will input the objective and the dataset, you will suggest the statistics method with out out in JSON with two fields: 'method' and 'explanation'.
    Don't make assumption on the method, ask for clarification if user input is ambiguous. 
    Make sure the suggested method is clear, for example if it's t-test, we should know is it an independent or pair-wise, one-tail or two tail, etc.
    If you need more clarification, please put "NEED_CLARIFICATION" as method.
    """
    
    # Initial objective
    user_objective = input("[USER] Input your analysis objective: ")
    
    user_content = f"""
    The user wants to perform a statistical analysis with the following objective: {user_objective}.
    Provide your suggestion.
    """

    messages = []
    messages.append({"role": "system", "content": system_content})
    messages.append({"role": "user", "content": user_content})

    chat_completion = get_chat_completion(messages)
    assistant_message = chat_completion.choices[0].message
    messages.append(assistant_message)

    suggested_method = json.loads(assistant_message.content)['method']
    suggested_method_explanation = json.loads(assistant_message.content)['explanation']

    print(f"[ASSISTANT] Suggested method: {suggested_method}")
    print(f"[ASSISTANT] Explanation: {suggested_method_explanation}")
    if suggested_method != 'NEED_CLARIFICATION':
        is_confirmed = input("[USER] Confirm the suggestion is correct [y/N]: ")

    # Loop to clarify the objective
    while (suggested_method == 'NEED_CLARIFICATION') or (is_confirmed=="N") :
        user_objective = input("[USER] Input your clarification: ")

        user_content = f""" 
        The user provide following clarification or additional context: {user_objective}.
        Revise your suggestion based on that clarification.
        """

        messages.append({"role": "user", "content": user_content})
        chat_completion = get_chat_completion(messages)
        assistant_message = chat_completion.choices[0].message
        messages.append(assistant_message)

        suggested_method = json.loads(assistant_message.content)['method']
        suggested_method_explanation = json.loads(assistant_message.content)['explanation']

        print(f"[ASSISTANT] Suggested method: {suggested_method}")
        print(f"[ASSISTANT] Explanation: {suggested_method_explanation}")

        if suggested_method != 'NEED_CLARIFICATION':
            is_confirmed = input("[USER] Confirm the suggestion is correct [y/N]: ")

    return suggested_method, suggested_method_explanation

def get_method_availability(method):
    if method in test_list[:-1]:
        print(f"[ASSISTANT] Method is available: {method}")
    else:
        raise ValueError("[ASSISTANT] Mehod is not available.")

def check_method_availability(suggested_method):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_method_availability",
                "description": "Use this function to check the availability of the method in the app.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "method": {
                            "type": "string",
                            "enum": test_list
                        }
                    },
                    "required": ["method"]
                }
            }
        }
    ]

    user_content = f""" 
    The user want to check if this method is available in the system: {suggested_method}.
    """

    messages = []
    messages.append({"role": "user", "content": user_content})
    chat_completion = get_chat_completion(messages, tools=tools, tool_choice='auto')
    assistant_message = chat_completion.choices[0].message
    tool_calls = assistant_message.tool_calls
    
    function = tool_calls[0].function
    if function.name == 'get_method_availability':
        method = json.loads(function.arguments).get('method')
        get_method_availability(method=method)

    return method
        
def get_dataset_structure(method):
    if method in ['t-test', 'mann-whitney']:
        structure = "Two columns, one for the group labels and one for the continuous variable."
        data = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B'],
            'score': [10, 12, 15, 18]
        })
    elif method in ['paired t-test', 'wilcoxon']:
        structure = "Two columns, both continuous variables for the paired samples."
        data = pd.DataFrame({
            'before': [20, 21, 19, 22],
            'after': [23, 22, 21, 24]
        })
    elif method in ['anova', 'kruskal-wallis']:
        structure = "One column for the group labels and one for the continuous variable."
        data = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B', 'C', 'C'],
            'score': [10, 12, 15, 18, 20, 22]
        })
    elif method == 'chi-square':
        structure = "Two columns, both categorical variables."
        data = pd.DataFrame({
            'category1': ['A', 'A', 'B', 'B'],
            'category2': ['X', 'Y', 'X', 'Y']
        })

    elif method in ['pearson', 'spearman', 'kendall']:
        structure = "Two columns, both continuous variables."
        data = pd.DataFrame({
            'variable1': [1, 2, 3, 4],
            'variable2': [10, 20, 30, 40]
        })

    else:
        raise ValueError("Unsupported method")
    
    return structure, data

def upload_dataset(method):
    try:
        structure, example = get_dataset_structure(method=method)
        print(f"[ASSISTANT] Data structure : {structure}")
        print("[ASSISTANT] Please follow below example :")
        print(f"==================== \n{example}\n ====================")

    except ValueError:
        print("[ASSISTANT] The method is unsupported")


    is_confirmed_dataset = "N"

    while is_confirmed_dataset == "N":
        print("[ASSISTANT] Select the file contain your dataset: ")
        csv_file_path = select_file()
        if csv_file_path:
            # Load the CSV file into a pandas DataFrame
            df = pd.read_csv(csv_file_path)
            print("[ASSISTANT] Below is your dataset sample:")
            print(f"====================\n{df.head()}\n====================")

            is_confirmed_dataset = input("[USER] Confirm the dataset is correct and follow the instruction [y/N]: ")
    
    return df