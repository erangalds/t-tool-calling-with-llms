import ollama
import requests

available_functions = {
  'request': requests.request,
}

client = ollama.Client(host='http://host.docker.internal:11434')

response = client.chat(
  'llama3.2',
  messages=[{
    'role': 'user',
    'content': 'get the ollama.com webpage?',
  }],
  tools=[requests.request], 
)

print(f'Response from LLM:\n\n{response}\n\n')

for tool in response.message.tool_calls or []:
  function_to_call = available_functions.get(tool.function.name)
  
  print(f'\nFunction to call: {function_to_call}\n')

  if function_to_call == requests.request:
    # Make an HTTP request to the URL specified in the tool call
    resp = function_to_call(
      method=tool.function.arguments.get('method'),
      url=tool.function.arguments.get('url'),
    )
    print(resp.text)
  else:
    print('Function not found:', tool.function.name)