import pandas as pd
import json

# Before we get started, let’s assume we have a dataframe consisting of payment transactions. 
# When users ask questions about this dataframe, they can use certain tools to answer questions about this data. 
# This is just an example to emulate an external database that the LLM cannot directly access.


# Assuming we have the following data
data = {
    'transaction_id': ['T1001', 'T1002', 'T1003', 'T1004', 'T1005'],
    'customer_id': ['C001', 'C002', 'C003', 'C002', 'C001'],
    'payment_amount': [125.50, 89.99, 120.00, 54.30, 210.20],
    'payment_date': ['2021-10-05', '2021-10-06', '2021-10-07', '2021-10-05', '2021-10-08'],
    'payment_status': ['Paid', 'Unpaid', 'Paid', 'Paid', 'Pending']
}

# Create DataFrame
df = pd.DataFrame(data)

# In many cases, we might have multiple tools at our disposal. 
# For example, let’s consider we have two functions as our two tools: retrieve_payment_status and retrieve_payment_date 
# to retrieve payment status and payment date given transaction ID.

def retrieve_payment_status(df: data, transaction_id: str) -> str:
    if transaction_id in df.transaction_id.values: 
        return json.dumps({'status': df[df.transaction_id == transaction_id].payment_status.item()})
    return json.dumps({'error': 'transaction id not found.'})

def retrieve_payment_date(df: data, transaction_id: str) -> str:
    if transaction_id in df.transaction_id.values: 
        return json.dumps({'date': df[df.transaction_id == transaction_id].payment_date.item()})
    return json.dumps({'error': 'transaction id not found.'})

# In order for Mistral models to understand the functions, we need to outline the function specifications with a JSON schema. 
# Specifically, we need to describe the type, function name, function description, function parameters, and the required parameter for the function. 
# Since we have two functions here, let’s list two function specifications in a list.

tools = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_payment_status",
            "description": "Get payment status of a transaction",
            "parameters": {
                "type": "object",
                "properties": {
                    "transaction_id": {
                        "type": "string",
                        "description": "The transaction id.",
                    }
                },
                "required": ["transaction_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_payment_date",
            "description": "Get payment date of a transaction",
            "parameters": {
                "type": "object",
                "properties": {
                    "transaction_id": {
                        "type": "string",
                        "description": "The transaction id.",
                    }
                },
                "required": ["transaction_id"],
            },
        },
    }
]

import functools

# Then we organize the two functions into a dictionary where keys represent the function name, and values are the function with the df defined. 
# This allows us to call each function based on its function name.

names_to_functions = {
    'retrieve_payment_status': functools.partial(retrieve_payment_status, df=df),
    'retrieve_payment_date': functools.partial(retrieve_payment_date, df=df)
}

# Suppose a user asks the following question: “What’s the status of my transaction?” 
# A standalone LLM would not be able to answer this question, as it needs to query the business logic backend to access the necessary data. 
# But what if we have an exact tool we can use to answer this question? We could potentially provide an answer!

messages = [{"role": "user", "content": "What's the status of my transaction T1001?"}]

# Users can use tool_choice to speficy how tools are used:
# "auto": default mode. Model decides if it uses the tool or not.
# "any": forces tool use.
# "none": prevents tool use.

import os
from mistralai import Mistral

api_key = os.environ["MISTRAL_API_KEY"]
model = "mistral-large-latest"

client = Mistral(api_key=api_key)
response = client.chat.complete(
    model = model,
    messages = messages,
    tools = tools,
    tool_choice = "any",
)

print(f'Response from LLM:\n\n{response}\n')

# Let’s add the response message to the messages list.
messages.append(response.choices[0].message)

# How do we execute the function? Currently, it is the user’s responsibility to execute these functions and the function execution lies on the user side. 
# In the future, we may introduce some helpful functions that can be executed server-side.
# Let’s extract some useful function information from model response including function_name and function_params. 
# It’s clear here that our Mistral model has chosen to use the function retrieve_payment_status with the parameter transaction_id set to T1001.


import json

tool_call = response.choices[0].message.tool_calls[0]
function_name = tool_call.function.name
function_params = json.loads(tool_call.function.arguments)
print("\nfunction_name: ", function_name, "\nfunction_params: ", function_params)

# Now we can execute the function and we get the function output '{"status": "Paid"}'.

function_result = names_to_functions[function_name](**function_params)

print(f'\n\nFunction Result: \n\n{function_result}')

# We can now provide the output from the tools to Mistral models, and in return, 
# the Mistral model can produce a customised final response for the specific user.

messages.append({"role":"tool", "name":function_name, "content":function_result, "tool_call_id":tool_call.id})

response = client.chat.complete(
    model = model, 
    messages = messages
)

print(f'\n\nFinal Response:\n\n{response.choices[0].message.content}\n\n')

