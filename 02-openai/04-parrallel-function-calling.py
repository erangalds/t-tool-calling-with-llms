# tools is an optional parameter in the Chat Completion API which can be used to provide function specifications. 
# The purpose of this is to enable models to generate function arguments which adhere to the provided specifications. 
# Note that the API will not actually execute any function calls. It is up to developers to execute function calls using model outputs.

# Within the tools parameter, if the functions parameter is provided then by default the model will decide when it is appropriate to use one of the functions. 
# The API can be forced to use a specific function by setting the tool_choice parameter to {"type": "function", "function": {"name": "my_function"}}. 
# The API can also be forced to not use any function by setting the tool_choice parameter to "none". 
# If a function is used, the output will contain "finish_reason": "tool_calls" in the response, 
# as well as a tool_calls object that has the name of the function and the generated function arguments.

import json
from openai import OpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt
from termcolor import colored  

GPT_MODEL = "gpt-4o-mini"
client = OpenAI()

# Utilities
# First let's define a few utilities for making calls to the Chat Completions API and for maintaining and keeping track of the conversation state.

@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(messages, tools=None, tool_choice=None, model=GPT_MODEL):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e
def convert_message_output_to_dict_format(llm_response):
    return { 
        'role': llm_response.role,
        'content': llm_response.content
    }

def pretty_print_conversation(messages):
    role_to_color = {
        "system": "red",
        "user": "green",
        "assistant": "blue",
        "function": "magenta",
    }
     
    for message in messages:
        #print(f'\nMessage: \n{message}')
        #print(f'\nType of object: {type(message)}\n')
        
        if str(type(message)) == "<class 'dict'>":
            if message["role"] == "system":
                print(colored(f"system: {message['content']}\n", role_to_color[message["role"]]))
            elif message["role"] == "user":
                print(colored(f"user: {message['content']}\n", role_to_color[message["role"]]))
        else:
            #print('not a dict')
            if message.role == "system":
                print(colored(f"system: {message.content}\n", role_to_color[message.role]))
            elif message.role == "user":
                print(colored(f"user: {message.content}\n", role_to_color[message.role]))
            elif message.role == "assistant" and message.function_call:
                print(colored(f"assistant: {message.function_call}\n", role_to_color[message.role]))
            elif message.role == "assistant" and not message.function_call:
                print(colored(f"assistant: {message.content}\n", role_to_color[message.role]))
            elif message.role == "function":
                print(colored(f"function ({message.name}): {message.content}\n", role_to_color[message.role]))

        
def get_current_weather(location, format):
    import random
    if format == 'celsius':
        # Define the temperature range in Celsius
        min_temp = -89.2
        max_temp = 56.7

        # Generate a random floating-point number within the range
        random_temp = random.uniform(min_temp, max_temp)
    elif format == 'Fahrenheit':
                # Define the temperature range in Celsius
        min_temp = -89.2
        max_temp = 56.7

        # Generate a random floating-point number within the range
        random_temp = random.uniform(min_temp, max_temp)

    return f'The current weather in {location}: {random_temp} in {format}'

# Let's create some function specifications to interface with a hypothetical weather API. 
# We'll pass these function specification to the Chat Completions API in order to generate function arguments that adhere to the specification.

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use. Infer this from the users location.",
                    },
                },
                "required": ["location", "format"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_n_day_weather_forecast",
            "description": "Get an N-day weather forecast",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use. Infer this from the users location.",
                    },
                    "num_days": {
                        "type": "integer",
                        "description": "The number of days to forecast",
                    }
                },
                "required": ["location", "format", "num_days"]
            },
        }
    },
]

# Parallel Function Calling
# Newer models such as gpt-4o or gpt-3.5-turbo can call multiple functions in one turn
print('\n\nPARALLEL FUNCTION CALLING\n\n')
messages = []
messages.append(
    {
        "role": "system", 
        "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."
    }
)
messages.append(
    {
        "role": "user", 
        "content": "what is the weather going to be like in San Francisco and Glasgow over the next 4 days"
    }
)
chat_response = chat_completion_request(
    messages, tools=tools, model=GPT_MODEL
)

assistant_message = chat_response.choices[0].message.tool_calls
print(f'LLS Response:\n\n{assistant_message}\n\n')
pretty_print_conversation(messages=messages)
