from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, PlainTextResponse
import asyncio, json
app = FastAPI()
@app.get('/health')
async def health(): return PlainTextResponse('ok')
@app.get('/metrics')
async def metrics(): return PlainTextResponse('vllm:num_requests_running 1' + chr(10))
@app.post('/v1/chat/completions')
async def chat(req: Request):
    body = await req.json()
    async def gen():
        for i in range(20):
            await asyncio.sleep(0.01)
            yield 'data: ' + json.dumps({'id': 'x', 'object': 'chat.completion.chunk', 'model': 'm', 'choices': [{'index': 0, 'delta': {'content': 'tok '}, 'finish_reason': None}]}) + chr(10) * 2
        yield 'data: ' + json.dumps({'id': 'x', 'object': 'chat.completion.chunk', 'model': 'm', 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]}) + chr(10) * 2
        yield 'data: ' + json.dumps({'id': 'x', 'object': 'chat.completion.chunk', 'model': 'm', 'choices': [], 'usage': {'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 120}}) + chr(10) * 2
        yield 'data: [DONE]' + chr(10) * 2
    return StreamingResponse(gen(), media_type='text/event-stream')
