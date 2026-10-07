import json
import unittest

import httpx

from resident.domain import ImageAttachment, ToolResult, ToolSpec
from resident.provider import OpenAIResponsesProvider, ResponseRejected
from resident.outputs import output_schema


def final(response_id='response-1'):
    return {'id': response_id, 'status': 'completed', 'output': [
        {'type': 'message', 'content': [{'type': 'output_text', 'text': '{"outputs":[]}'}]}],
        'usage': {'input_tokens': 5000, 'output_tokens': 5, 'input_tokens_details': {'cached_tokens': 4800}}}


class ResponsesTests(unittest.IsolatedAsyncioTestCase):
    def provider(self, replies, **settings):
        requests = []
        def handle(request):
            requests.append((request.method, request.url.path, json.loads(request.content)))
            value = replies.pop(0)
            return httpx.Response(200, json=value)
        provider = OpenAIResponsesProvider('test-key', 'gpt-5.6-luna', **settings)
        provider._client.close()
        provider._client = httpx.Client(base_url='https://openai.test/v1/', transport=httpx.MockTransport(handle))
        self.addCleanup(provider.close)
        return provider, requests

    async def test_empty_conversation_and_response_wire_contract(self):
        provider, requests = self.provider([{'id': 'conv-1'}, final()], reasoning_effort='none',
                                          service_tier='default', compact_threshold=120000)
        conversation_id = await provider.create_conversation()
        turn = await provider.respond('wake', [ToolSpec('read', 'Read current state', {'type': 'object'})], [],
                                      conversation_id=conversation_id, instructions='current role',
                                      output_schema=output_schema([]), request_id='request-1')
        self.assertEqual(('POST', '/v1/conversations', {}), requests[0])
        body = requests[1][2]
        self.assertEqual('conv-1', body['conversation'])
        self.assertNotIn('previous_response_id', body)
        self.assertEqual([{'role': 'user', 'content': 'wake'}], body['input'])
        self.assertEqual('current role', body['instructions'])
        self.assertEqual({'effort': 'none'}, body['reasoning'])
        self.assertTrue(body['text']['format']['strict'])
        self.assertEqual('resident_disposition', body['text']['format']['name'])
        self.assertEqual([{'type': 'compaction', 'compact_threshold': 120000}], body['context_management'])
        self.assertEqual(4800, turn.cached_input_tokens)

    async def test_function_results_only_and_multimodal_camera_output(self):
        provider, requests = self.provider([final()])
        await provider.respond('original wake must not repeat', [], [
            ToolResult('call-1', {'ok': True}, (ImageAttachment(b'jpeg'),)),
            ToolResult('call-2', {'ok': False})], conversation_id='conv-1', instructions='current',
            output_schema=output_schema([]), request_id='request-2')
        items = requests[0][2]['input']
        self.assertEqual(['call-1', 'call-2'], [item['call_id'] for item in items])
        self.assertEqual('input_image', items[0]['output'][1]['type'])
        self.assertTrue(items[0]['output'][1]['image_url'].startswith('data:image/jpeg;base64,'))
        self.assertNotIn('original wake', json.dumps(items))
        self.assertNotIn('context_management', requests[0][2])

    def test_multiple_function_calls_and_reasoning_items(self):
        raw = {'id': 'r', 'status': 'completed', 'output': [{'type': 'reasoning'},
            {'type': 'function_call', 'call_id': 'a', 'name': 'read', 'arguments': '{}'},
            {'type': 'function_call', 'call_id': 'b', 'name': 'read', 'arguments': 'invalid'}]}
        turn = OpenAIResponsesProvider.parse_response(raw)
        self.assertEqual(['a', 'b'], [call.id for call in turn.tool_calls])
        self.assertEqual({'_invalid_json': True}, turn.tool_calls[1].arguments)

    def test_refusal_incomplete_and_missing_disposition_fail(self):
        for raw in [dict(final(), status='incomplete'), dict(final(), output=[]),
                    dict(final(), output=[{'type': 'message', 'content': [{'type': 'refusal'}]}])]:
            with self.subTest(raw=raw), self.assertRaises(RuntimeError):
                OpenAIResponsesProvider.parse_response(raw)

    async def test_http_rejection_and_ambiguous_failure_are_not_retried(self):
        for code in [400, 429, 500]:
            provider = OpenAIResponsesProvider('test', 'model')
            provider._client.close()
            seen = []
            def handle(request):
                seen.append(request)
                return httpx.Response(code)
            provider._client = httpx.Client(base_url='https://openai.test/', transport=httpx.MockTransport(handle))
            try:
                error = ResponseRejected if code < 500 else httpx.HTTPStatusError
                with self.assertRaises(error):
                    await provider.respond('wake', [], [], conversation_id='conv', instructions='i',
                                           output_schema=output_schema([]), request_id='request')
                self.assertEqual(1, len(seen))
            finally:
                provider.close()
