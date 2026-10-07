import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from resident.capabilities import Capability
from resident.config import Config
from resident.domain import ModelTurn, ToolCall, WakeEvent
from resident.outputs import output_schema
from resident.provider import ResponseInvalid, ResponseRejected
from resident.runtime import ResidentRuntime
from resident.store import Store, utc_now
from resident.tools import ToolRegistry
from runtime_support import RecordingProvider


class ResponsesRuntimeTests(unittest.IsolatedAsyncioTestCase):
    def runtime(self, directory, provider, **settings):
        return ResidentRuntime(Config(Path(directory), **settings), provider,
                               owner_output=lambda _: None, diagnostic_output=lambda _: None)

    async def test_fresh_conversation_persisted_before_inference_and_reused_on_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            first = RecordingProvider()
            runtime = self.runtime(directory, first)
            await runtime.initialize()
            self.assertEqual('conversation-test', runtime.store.conversation_id())
            await runtime.process(WakeEvent('wake', 'sensor', 'change', utc_now()))
            self.assertEqual(1, first.created)
            self.assertEqual('conversation-test', first.requests[0]['conversation_id'])
            self.assertEqual([], runtime.store.unfinished_response_steps())
            identity = runtime.resident.id
            runtime.close()
            second = RecordingProvider()
            runtime = self.runtime(directory, second)
            await runtime.initialize()
            self.assertEqual(0, second.created)
            self.assertEqual(identity, runtime.resident.id)
            runtime.close()

    async def test_multiple_calls_and_rounds_update_current_guidance(self):
        with tempfile.TemporaryDirectory() as directory:
            provider = RecordingProvider([
                ModelTurn('r1', tool_calls=(ToolCall('a', 'create_intention', {'content': 'remember task'}),
                                           ToolCall('b', 'set_owner_guidance', {'content': 'Be brief'}))),
                ModelTurn('r2', tool_calls=(ToolCall('c', 'diagnostics_current_time', {}),)),
                '{"outputs":[]}'])
            runtime = self.runtime(directory, provider, personality='Curious', role='Watch the home')
            event = runtime.owner_message_event('Always be brief')
            await runtime.process(event)
            self.assertEqual(3, provider.round)
            self.assertEqual(['a', 'b'], [r.call_id for r in provider.requests[1]['results']])
            self.assertEqual(['c'], [r.call_id for r in provider.requests[2]['results']])
            self.assertIn('Be brief', provider.requests[1]['instructions'])
            self.assertIn('Curious', provider.requests[0]['instructions'])
            self.assertIn('Watch the home', provider.requests[0]['instructions'])
            self.assertNotIn('current_time', provider.requests[0]['instructions'])
            self.assertEqual(1, len(runtime.store.pending_intentions()))
            self.assertTrue(all(r.output['ok'] for r in provider.requests[1]['results']))
            self.assertEqual([], runtime.store.unfinished_response_steps())
            runtime.close()

    async def test_tool_validation_and_owner_authority(self):
        with tempfile.TemporaryDirectory() as directory:
            provider = RecordingProvider([ModelTurn('r', tool_calls=(
                ToolCall('a', 'set_owner_guidance', {'content': 'untrusted'}),
                ToolCall('b', 'schedule_wakeup', {'delay_seconds': -1, 'reason': 'bad'}),
                ToolCall('c', 'unavailable', {}))), '{"outputs":[]}'])
            runtime = self.runtime(directory, provider)
            await runtime.process(WakeEvent('wake', 'sensor', 'change', utc_now()))
            self.assertTrue(all(not result.output['ok'] for result in provider.requests[1]['results']))
            self.assertEqual([], runtime.store.active_owner_guidance())
            self.assertNotIn('send_owner_message', provider.tool_sets[0])
            self.assertNotIn('search_long_term_memory', provider.tool_sets[0])
            runtime.close()

    async def test_persisted_mutating_result_replays_without_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            provider = RecordingProvider()
            runtime = self.runtime(directory, provider)
            await runtime.initialize()
            registry = ToolRegistry(runtime.store, [], lambda *_: None)
            call = ToolCall('call', 'create_intention', {'content': 'once'})
            first = await runtime._execute_tool(call, 'r1', registry)
            runtime.close()
            runtime = self.runtime(directory, RecordingProvider())
            registry = ToolRegistry(runtime.store, [], lambda *_: None)
            replay = await runtime._execute_tool(call, 'r1', registry)
            self.assertEqual(first.output, replay.output)
            self.assertEqual(1, len(runtime.store.pending_intentions()))
            with self.assertRaisesRegex(RuntimeError, 'different arguments'):
                await runtime._execute_tool(ToolCall('call', 'create_intention', {'content': 'changed'}), 'r1', registry)
            runtime.close()

    async def test_interrupted_mutation_is_not_repeated(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.runtime(directory, RecordingProvider())
            await runtime.initialize()
            runtime.store.begin_tool_execution(runtime.conversation_id, 'r', 'call', 'create_intention', {'content': 'unknown'})
            result = await runtime._execute_tool(ToolCall('call', 'create_intention', {'content': 'unknown'}),
                                                 'r', ToolRegistry(runtime.store, [], lambda *_: None))
            self.assertEqual('unknown_outcome', result.output['error_code'])
            self.assertEqual([], runtime.store.pending_intentions())
            runtime.close()

    async def test_new_requests_use_updated_role_guidance_and_tools(self):
        with tempfile.TemporaryDirectory() as directory:
            provider = RecordingProvider()
            runtime = self.runtime(directory, provider, personality='First', role='Original role')
            await runtime.process(WakeEvent('one', 'sensor', 'change', utc_now()))
            runtime.store.set_owner_guidance('Current instruction')
            runtime.context_builder.role = 'New role'
            runtime.replace_capabilities([])
            await runtime.process(WakeEvent('two', 'sensor', 'change', utc_now()))
            self.assertIn('New role', provider.requests[1]['instructions'])
            self.assertIn('Current instruction', provider.requests[1]['instructions'])
            self.assertNotIn('diagnostics_current_time', provider.tool_sets[1])
            runtime.close()

    async def test_duplicate_event_does_not_infer_again(self):
        with tempfile.TemporaryDirectory() as directory:
            provider = RecordingProvider()
            runtime = self.runtime(directory, provider)
            event = WakeEvent('wake', 'sensor', 'change', utc_now())
            first = await runtime.process(event)
            self.assertEqual(first, await runtime.process(event))
            self.assertEqual(1, provider.round)
            runtime.close()

    async def test_final_disposition_jobs_recover_from_local_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.runtime(directory, RecordingProvider())
            await runtime.initialize()
            event = runtime.owner_message_event('Hello')
            run_id = runtime.store.start_run(event)
            step = runtime.store.begin_response_step(run_id, runtime.conversation_id, 0, event, runtime._output_schema)
            runtime.store.record_response(step, ModelTurn('response', '{"outputs":[{"type":"notify_owner","content":"Hello"}]}'))
            runtime.close()
            delivered = []
            provider = RecordingProvider()
            runtime = ResidentRuntime(Config(Path(directory)), provider, owner_output=delivered.append,
                                       diagnostic_output=lambda _: None)
            await runtime.initialize()
            self.assertEqual(0, provider.round)
            self.assertEqual([], runtime.store.pending_owner_messages())
            self.assertEqual([], delivered)
            self.assertTrue(await runtime.dispatch_outputs_once())
            self.assertEqual(['Hello'], delivered)
            self.assertFalse(await runtime.dispatch_outputs_once())
            self.assertEqual('completed', runtime.store.connection.execute('SELECT status FROM wake_runs WHERE id=?', (run_id,)).fetchone()[0])
            runtime.close()

    async def test_ambiguous_request_blocks_instance_without_resubmission(self):
        with tempfile.TemporaryDirectory() as directory:
            provider = RecordingProvider([RuntimeError('connection lost')])
            runtime = self.runtime(directory, provider)
            with self.assertRaisesRegex(RuntimeError, 'connection lost'):
                await runtime.process(WakeEvent('one', 'sensor', 'change', utc_now()))
            with self.assertRaisesRegex(RuntimeError, 'Unfinished'):
                await runtime.process(WakeEvent('two', 'sensor', 'change', utc_now()))
            self.assertEqual(1, provider.round)
            runtime.close()
            runtime = self.runtime(directory, RecordingProvider())
            with self.assertRaisesRegex(RuntimeError, 'Unfinished'):
                await runtime.initialize()
            runtime.close()

    async def test_rejection_fails_wake_but_permits_later_request(self):
        with tempfile.TemporaryDirectory() as directory:
            provider = RecordingProvider([ResponseRejected('bad request'), '{"outputs":[]}'])
            runtime = self.runtime(directory, provider)
            with self.assertRaises(ResponseRejected):
                await runtime.process(WakeEvent('one', 'sensor', 'change', utc_now()))
            await runtime.process(WakeEvent('two', 'sensor', 'change', utc_now()))
            self.assertEqual(2, provider.round)
            runtime.close()

    async def test_missing_or_malformed_final_disposition_fails_wake(self):
        for message in [None, 'not json', '{}']:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                runtime = self.runtime(directory, RecordingProvider([ModelTurn('r', message)]))
                with self.assertRaises(RuntimeError):
                    await runtime.process(WakeEvent('wake', 'sensor', 'change', utc_now()))
                self.assertEqual('failed', runtime.store.connection.execute('SELECT status FROM wake_runs').fetchone()[0])
                self.assertEqual(0, runtime.store.connection.execute('SELECT count(*) FROM output_requests').fetchone()[0])
                runtime.close()

    async def test_tool_round_limit_stops_loop(self):
        with tempfile.TemporaryDirectory() as directory:
            provider = RecordingProvider([ModelTurn('r', tool_calls=(ToolCall('a', 'create_intention', {'content': 'task'}),))])
            runtime = self.runtime(directory, provider, max_tool_rounds=0)
            with self.assertRaisesRegex(RuntimeError, 'tool-round limit'):
                await runtime.process(WakeEvent('wake', 'sensor', 'change', utc_now()))
            self.assertEqual([], runtime.store.pending_intentions())
            runtime.close()

    async def test_revocation_while_inference_is_running_prevents_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            executed = []
            class RevokeProvider(RecordingProvider):
                async def respond(self, *args, **request):
                    if self.round == 0:
                        runtime.replace_capabilities([])
                    return await super().respond(*args, **request)
            provider = RevokeProvider([ModelTurn('r', tool_calls=(ToolCall('call', 'mutate', {}),)), '{"outputs":[]}'])
            capability = Capability('test', 'Test', 'mutate', 'Mutate',
                                    {'type': 'object', 'properties': {}, 'additionalProperties': False},
                                    lambda _: executed.append(True) or {})
            runtime = ResidentRuntime(Config(Path(directory)), provider, capabilities=[capability],
                                       owner_output=lambda _: None, diagnostic_output=lambda _: None)
            await runtime.process(WakeEvent('wake', 'sensor', 'change', utc_now()))
            self.assertEqual([], executed)
            self.assertFalse(provider.requests[1]['results'][0].output['ok'])
            self.assertNotIn('mutate', provider.tool_sets[1])
            runtime.close()

    async def test_ephemeral_image_result_is_not_replayed_after_restart(self):
        from resident.domain import ImageAttachment, ToolOutput
        with tempfile.TemporaryDirectory() as directory:
            captures = []
            capability = Capability('camera', 'Camera', 'capture', 'Capture',
                                    {'type': 'object', 'properties': {}, 'additionalProperties': False},
                                    lambda _: captures.append(True) or ToolOutput({'ok': True}, (ImageAttachment(b'jpeg'),)))
            runtime = self.runtime(directory, RecordingProvider())
            await runtime.initialize()
            call = ToolCall('image-call', 'capture', {})
            first = await runtime._execute_tool(call, 'r', ToolRegistry(runtime.store, [capability], lambda *_: None))
            self.assertEqual(b'jpeg', first.attachments[0].data)
            runtime.close()
            runtime = self.runtime(directory, RecordingProvider())
            replay = await runtime._execute_tool(call, 'r', ToolRegistry(runtime.store, [capability], lambda *_: None))
            self.assertEqual('attachment_unavailable', replay.output['error_code'])
            self.assertEqual([], list(replay.attachments))
            self.assertEqual([True], captures)
            self.assertNotIn('jpeg', runtime.store.connection.execute('SELECT output_json FROM tool_executions').fetchone()[0])
            runtime.close()

    async def test_tool_call_budget_rejects_oversized_batch_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            provider = RecordingProvider([ModelTurn('r', tool_calls=(
                ToolCall('a', 'create_intention', {'content': 'one'}),
                ToolCall('b', 'create_intention', {'content': 'two'})))])
            runtime = self.runtime(directory, provider, max_tool_calls=1)
            with self.assertRaisesRegex(RuntimeError, 'tool-call limit'):
                await runtime.process(WakeEvent('wake', 'sensor', 'change', utc_now()))
            self.assertEqual([], runtime.store.pending_intentions())
            self.assertEqual([], runtime.store.connection.execute('SELECT * FROM tool_executions').fetchall())
            runtime.close()

    async def test_failed_output_ingestion_keeps_final_response_recoverable(self):
        import sqlite3
        with tempfile.TemporaryDirectory() as directory:
            provider = RecordingProvider(['{"outputs":[{"type":"notify_owner","content":"Once"}]}'])
            runtime = self.runtime(directory, provider)
            event = runtime.owner_message_event('Hello')
            runtime.store.connection.execute("""
                CREATE TRIGGER fail_output_ingestion BEFORE INSERT ON output_requests
                BEGIN SELECT RAISE(ABORT, 'simulated disk failure'); END
            """)
            with self.assertRaisesRegex(sqlite3.IntegrityError, 'simulated disk failure'):
                await runtime.process(event)
            self.assertEqual(0, runtime.store.connection.execute('SELECT count(*) FROM final_dispositions').fetchone()[0])
            self.assertEqual('returned', runtime.store.unfinished_response_steps()[0]['status'])
            runtime.store.connection.execute('DROP TRIGGER fail_output_ingestion')
            runtime.store.connection.commit()
            runtime.close()
            provider = RecordingProvider()
            runtime = self.runtime(directory, provider)
            await runtime.initialize()
            self.assertEqual(0, provider.round)
            self.assertEqual(1, runtime.store.connection.execute('SELECT count(*) FROM output_requests').fetchone()[0])
            self.assertEqual([], runtime.store.pending_owner_messages())
            runtime.close()

    async def test_invalid_remote_response_retains_known_id_and_fails_wake(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.runtime(directory, RecordingProvider([ResponseInvalid('incomplete', 'response-known')]))
            with self.assertRaises(ResponseInvalid):
                await runtime.process(WakeEvent('wake', 'sensor', 'change', utc_now()))
            self.assertEqual('response-known', runtime.store.unfinished_response_steps()[0]['response_id'])
            self.assertEqual('failed', runtime.store.connection.execute('SELECT status FROM wake_runs').fetchone()[0])
            self.assertEqual(0, runtime.store.connection.execute('SELECT count(*) FROM output_requests').fetchone()[0])
            runtime.close()

    def test_fresh_schema_has_no_agents_or_memory_tables(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory)/'resident.sqlite3')
            names = {row[0] for row in store.connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            self.assertTrue({'conversation_binding', 'response_steps', 'tool_executions', 'owner_guidance', 'output_requests'} <= names)
            self.assertFalse(any('session' in n or 'curator' in n or 'memory' in n or 'agent_' in n for n in names))
            store.close()
