from openai import OpenAI
import os
c = OpenAI(base_url='http://localhost:8000/v1', api_key=os.environ['KEY'])
m = 'Qwen/Qwen3-0.6B'
def rs(msg): return getattr(msg, 'reasoning_content', None) or getattr(msg, 'reasoning', None)
r = c.chat.completions.create(model=m, messages=[{'role': 'user', 'content': 'Say hi in 5 words.'}], max_tokens=400)
print('PLAIN content:', r.choices[0].message.content); print('PLAIN reasoning:', rs(r.choices[0].message)); print(r.usage)
r = c.chat.completions.create(model=m, messages=[{'role': 'user', 'content': 'Say hi in 5 words.'}], max_tokens=100, extra_body={'chat_template_kwargs': {'enable_thinking': False}})
print('NOTHINK content:', r.choices[0].message.content); print(r.usage)
tools = [{'type': 'function', 'function': {'name': 'calculator', 'description': 'Evaluate math', 'parameters': {'type': 'object', 'properties': {'expr': {'type': 'string'}}, 'required': ['expr']}}}]
r = c.chat.completions.create(model=m, messages=[{'role': 'user', 'content': 'What is 123*45? Use the tool.'}], tools=tools, max_tokens=800)
print('TOOLS:', r.choices[0].message.tool_calls, r.choices[0].finish_reason)
