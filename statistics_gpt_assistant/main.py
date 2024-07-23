import pandas as pd
import numpy as np
import os

from openai import OpenAI
import json

from util import *
from stats_function import *

def chat_completion_request(messages, tools=None, tool_choice=None):
    import os
    from dotenv import load_dotenv

    # Load environment variables from the .env file
    load_dotenv()
    openai_api_key = os.getenv('OPENAI_API_KEY')

    client = OpenAI(api_key=openai_api_key)
    GPT_MODEL = "gpt-3.5-turbo-1106"

    return client.chat.completions.create(
        messages=messages,
        model=GPT_MODEL,
        temperature=0.3,
        max_tokens=4096,
        top_p=0.8,

        tools=tools,
        tool_choice=tool_choice
    )

def input_initial_objective():
    return input("[USER_INPUT] Input your analysis objective: ")

def confirm_additional_input():
    input_text = ''
    while input_text not in ['y', 'N']:
        input_text = input("[USER_INPUT] Do you want to add additional input? ")
        if input_text=='y': return True
        elif input_text=='N': return False

def input_additional_input():
    return input("[USER_INPUT] Add your additional input: ")

def get_suggested_method(method, explanation):
    text = f"The suggested method is {method}. {explanation}"
    print("[ASSISTANT] ", text)
    return {'method': method, 'explanation': explanation, 'text': text}

def check_suggested_method(assistant_message):
    if assistant_message.tool_calls:
        suggested_method_result = get_suggested_method(**json.loads(assistant_message.tool_calls[0].function.arguments))
        tool_message = {
            "role": "tool",
            "content": suggested_method_result['method'],
            "tool_call_id": assistant_message.tool_calls[0].id,
        }
        return suggested_method_result, tool_message
    else:
        text = assistant_message.content
        print("[ASSISTANT] ", text)
        return {'method': None, 'explanation': None, 'text': text}, None
    
def get_final_method(suggested_method_result):
    method = suggested_method_result['method']
    if method==None:
        print("[ASSISTANT] Sorry I cannot suggest the method and we cannot continue the process.")
        exit()
        return None
    else:
        print(f"The method that we will use: {method}")
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

def show_dataset_structure(assistant_message):
    if assistant_message.tool_calls:
        structure, example = get_dataset_structure(**json.loads(assistant_message.tool_calls[0].function.arguments))
        print(f"[ASSISTANT] Data structure : {structure}")
        print("[ASSISTANT] Please follow below example :")
        print(f"==================== \n{example}\n ====================")
    else:
        exit()

def upload_dataset():
    is_confirmed_dataset='N'

    while is_confirmed_dataset!='y':
        print("[ASSISTANT] Select the file contain your dataset: ")
        csv_file_path = select_file()
        if csv_file_path:
            # Load the CSV file into a pandas DataFrame
            df = pd.read_csv(csv_file_path)
            print("[ASSISTANT] Below is your dataset sample:")
            print(f"====================\n{df.sample(5)}\n====================")

            is_confirmed_dataset = input("[USER] Confirm the dataset is correct [y/N]: ")
    
    return df

if __name__ == "__main__":
    system_content = """
    You are a statistics expert to help the user on giving the suggestion of what method that best to used best on the objective of the analysis.
    User will input the objective and the dataset, you will suggest the statistics method.
    Don't make assumption on the method, ask for clarification if user input is ambiguous. 
    Make sure the suggested method is clear, for example if it's t-test, we should know is it an independent or pair-wise, one-tail or two tail, etc.
    """

    # Get the objective, initial input
    user_objective = input_initial_objective()

    messages = []
    messages.append({"role": "system", "content": system_content})
    messages.append({"role": "user", "content": user_objective})

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_suggested_method",
                "description": "Determine the best statistics method based on objective from the user and additional context.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "method": {
                            "type": "string",
                            "description": "The suggested method."
                        },
                        "explanation": {
                            "type": "string",
                            "description": "The explanation why the suggested method is suitable with the analysis objective."
                        }
                        },
                    "required": ["method", "explanation"]
                }
            }
        },
    ]

    chat_completion = chat_completion_request(messages, tools=tools, tool_choice="auto")
    assistant_message = chat_completion.choices[0].message
    suggested_method_result, tool_message = check_suggested_method(assistant_message)
    messages.append(assistant_message)
    if tool_message:
        messages.append(tool_message)

    additional_input = confirm_additional_input()

    # Get additional input
    while additional_input == True:
        user_input = input_additional_input()
        messages.append({"role": "user", "content": user_input})

        chat_completion = chat_completion_request(messages, tools=tools, tool_choice=None)
        assistant_message = chat_completion.choices[0].message
        suggested_method_result, tool_message = check_suggested_method(assistant_message)
        messages.append(assistant_message)
        if tool_message:
            messages.append(tool_message)
        additional_input = confirm_additional_input()

    method = get_final_method(suggested_method_result)

    # Get dataset structure and upload dataset
    system_content = """
    You are a statistics expert that will guide the user on what is the expected data structure based on the method that will be implemented.
    """

    messages = []
    messages.append({"role": "system", "content": system_content})
    messages.append({"role": "user", "content": f"The method that will be implement is: {method}"})

    tools =[
        {
            "type": "function",
            "function": {
                "name": "get_dataset_structure",
                "description": "Determine the expected dataset structure based on the method that want to be used.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "method": {
                            "type": "string",
                            "description": "The suggested method.",
                            "enum" : ['t-test', 'mann-whitney', 'paired t-test', 'wilcoxon', 'anova', 'kruskal-wallis', 'chi-square', 'pearson', 'spearman', 'kendall', 'other']
                        }
                        },
                    "required": ["method"]
                }
            }
        },
    ]
    chat_completion = chat_completion_request(messages, tools=tools, tool_choice=None)
    assistant_message = chat_completion.choices[0].message

    show_dataset_structure(assistant_message)
    df = upload_dataset()

    # Test the assumption
    system_content = """
    You are a statistics expert that will help the user to determine what assumption test that need to be conducted based on the statistics method that will be conducted.
    Return only the assumption testing is required. Multiple function is allowed.
    From the sample dataset, you need to determine the input value of the function.
    Summarize the result after the function has been run.
    """

    tools = [
        {
            "type": "function",
            "function": {
                "name": "check_normality",
                "description": "Check the normality of a single group only. If the test required normality for each group, use other function",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "value_col": {
                            "type": "string",
                            "description": "The column that contains value to be tested."
                        },
                        },
                    "required": ["value_col"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "check_normality_of_groups",
                "description": "Check the normality of a multiple group where the test required normality of each groups",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "group_col": {
                            "type": "string",
                            "description": "The column that contains groups to be tested."
                        },
                        "value_col": {
                            "type": "string",
                            "description": "The column that contains value to be tested."
                        },
                        },
                    "required": ["group_col","value_col"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "check_homogeneity_of_variances",
                "description": "Check the homogenity of a variance.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "group_col": {
                            "type": "string",
                            "description": "The column that contains groups to be tested."
                        },
                        "value_col": {
                            "type": "string",
                            "description": "The column that contains value to be tested."
                        },
                        },
                    "required": ["group_col","value_col"]
                }
            }
        },
    ]

    available_tools = {
        'check_normality': check_normality,
        'check_normality_of_groups': check_normality_of_groups,
        'check_homogeneity_of_variances': check_homogeneity_of_variances
    }

    messages = []
    messages.append({"role": "system", "content": system_content})
    messages.append({"role": "user", "content": f"The method that will be implement is: {method}. The sample of the dataset: \n {str(df.sample(5))}"})
    chat_completion = chat_completion_request(messages, tools=tools, tool_choice="auto")
    assistant_message = chat_completion.choices[0].message

    messages.append(assistant_message)

    if assistant_message.tool_calls:
        print("[ASSISTANT] Below is the list of assumption testing: \n")
        for tool_call in assistant_message.tool_calls:
            print(tool_call.function.name)
        proceed_assumption_testing = input("[USER_INPUT] Proceed the assumption testing? ")
    else:
        print("[ASSISTANT] No assumption testing is required")
        proceed_assumption_testing = 'N'

    if proceed_assumption_testing=='y':
        function_responses = []
        for tool_call in assistant_message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            function_response = available_tools[function_name](data=df, **function_args)
            function_responses.append(function_response)
            
            messages.append({
                "role": "tool",
                "content": function_response['text'],
                "tool_call_id": tool_call.id,
            })

        # Call the model again to summarize the results
        chat_completion = chat_completion_request(messages)
        assistant_message = chat_completion.choices[0].message.content

        print(f"[ASSISTANT] {assistant_message}")