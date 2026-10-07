import json
from resident.domain import ModelTurn


class RecordingProvider:
    def __init__(self, messages=()):
        self.messages = list(messages)
        self.requests, self.contexts, self.tool_sets = [], [], []
        self.created = 0
        self.round = 0

    async def create_conversation(self):
        self.created += 1
        return 'conversation-test'

    async def respond(self, context, tools, results, **request):
        self.round += 1
        self.contexts.append(json.loads(context))
        self.tool_sets.append([tool.name for tool in tools])
        self.requests.append({'context': context, 'tools': tools, 'results': tuple(results), **request})
        value = self.messages.pop(0) if self.messages else '{"outputs":[]}'
        if isinstance(value, Exception):
            raise value
        if isinstance(value, ModelTurn):
            return value
        return ModelTurn(f'response-{self.round}', value)
