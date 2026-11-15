import copy
import json
from pathlib import Path
import tempfile
import unittest

from op_show.model import ValidationError, validate_model

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'tests' / 'fixtures' / 'qkv_trace.json'
KERNEL = 'synthetic_qkv_trace_kernel'


def _build_fixture(directory):
    """Create a test-only source whose lines contain the model's evidence fragments.

    The v2 validator only checks literal line evidence; the synthetic source is not
    an operator implementation and is never served by the demo server.
    """
    document = json.loads(FIXTURE.read_text())
    refs = document['source']['refs']
    lines = [
        '# Synthetic test-only source for the v2 validator.',
        '# No upstream operator source is bundled with this repository.',
        f'def {KERNEL}():  # synthetic fixture',
    ]
    for key in sorted(refs):
        lines.append(refs[key]['contains'])
        refs[key]['line'] = len(lines)
    source = Path(directory) / 'synthetic_source.py'
    source.write_text('\n'.join(lines) + '\n')
    document['source']['path'] = str(source)
    model = Path(directory) / 'qkv_model.json'
    model.write_text(json.dumps(document))
    return model, source


class TraceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.model_path, cls.source_path = _build_fixture(cls.directory.name)
        cls.model, cls.source = validate_model(cls.model_path)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_fixture_trace_and_tail(self):
        self.assertIn(KERNEL, self.source)
        self.assertEqual(self.model['schema_version'], 2)
        aligned, padded = self.model['scenarios']
        for scenario in (aligned, padded):
            self.assertEqual(scenario['wave_count'], 3)
            self.assertEqual(len(scenario['tasks']), 960)
            self.assertEqual(len(scenario['events']), 4992)
            self.assertEqual({t['id'] for t in scenario['tasks']},
                             {f's{s}-t{t}-b{b}-g{g}' for s in range(4)
                              for t in range(4) for b in range(12) for g in range(5)})
        self.assertTrue(all(t['valid_tokens'] == 112 for t in aligned['tasks']))
        self.assertEqual({t['valid_tokens'] for t in padded['tasks'] if t['block_id'] == 11}, {68})

    def test_reference_claim_and_consumer_destination(self):
        scenario = self.model['scenarios'][0]
        task = next(t for t in scenario['tasks'] if t['id'] == 's0-t2-b1-g3')
        self.assertEqual((task['flat_idx'], task['producer_logical'], task['producer_pid']), (57, 9, 18))
        self.assertEqual((task['consumer_flat_idx'], task['consumer_logical'], task['consumer_pid']), (49, 1, 3))
        self.assertEqual((task['output_row'], task['head_start'], task['head_end']), (112, 12, 16))
        self.assertEqual(task['signal_slot'], ((1 * 4 + 0) * 5 + 3))
        self.assertEqual(task['peer_offsets']['k'] - task['peer_offsets']['q'], 20 * 112 * 128)
        self.assertEqual(task['peer_offsets']['v'] - task['peer_offsets']['k'], 20 * 112 * 128)

    def test_fence_notify_wait_order_and_no_local_wait(self):
        scenario = self.model['scenarios'][0]
        events = {e['id']: e for e in scenario['events']}
        for task in scenario['tasks']:
            if task['route'] == 'local':
                self.assertEqual([e['kind'] for e in scenario['events'] if e['task_id'] == task['id']],
                                 ['load', 'direct'])
                continue
            store = events[f"{task['id']}:peer_store"]
            fence = events[task['fence_id']]
            notify = events[f"{task['id']}:notify"]
            wait = events[f"{task['id']}:wait"]
            consume = events[f"{task['id']}:consume"]
            output = events[f"{task['id']}:output_store"]
            self.assertLess(store['level'], fence['level'])
            self.assertLess(fence['level'], notify['level'])
            self.assertIn(notify['id'], wait['depends_on'])
            self.assertIn(wait['id'], consume['depends_on'])
            self.assertIn(consume['id'], output['depends_on'])
        first = events['s0-t0-b0-g0:load']
        other = events['s0-t0-b1-g0:load']
        self.assertNotIn(other['id'], first['depends_on'])

    def _invalid(self, mutate, message):
        document = json.loads(self.model_path.read_text())
        mutate(document)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.json'
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValidationError, message):
                validate_model(path)

    def test_rejects_missing_dims_bad_formula_overlap_and_refs(self):
        self._invalid(lambda d: d['scenarios'][0]['params'].pop('sequence_length'), 'params must contain')
        self._invalid(lambda d: d['scenarios'][0]['params'].update(stride_a_s=1), 'strides are inconsistent')
        self._invalid(lambda d: d['trace']['formulas'].update(block_id='wave_start'), 'block_id: got')
        self._invalid(lambda d: d['trace']['formulas'].update(head_group='0'), 'head_group: got')
        self._invalid(lambda d: d['trace']['formulas'].update(peer_q='peer_q.__class__'), 'unsupported')
        self._invalid(lambda d: d['source']['refs']['notify'].update(contains='not here'), 'does not match')

    def test_rejects_degenerate_params_and_metadata(self):
        def rank_one(document):
            params = document['scenarios'][0]['params']
            params['rank_size'] = 1
            params['stride_a_s'] = params['heads_per_rank'] * params['block_d']
            params['stride_out_s'] = params['heads_per_rank'] * params['block_d']
        self._invalid(rank_one, 'core or rank count')
        self._invalid(lambda document: document.update(backend=[]), 'v2 backend')
        self._invalid(lambda document: document['scenarios'][0].update(id={}), 'aligned and padded')
        self._invalid(lambda document: document['trace'].update(param_refs={'block_s': 'missing'}),
                      'param_refs')

    def test_param_refs_are_optional_evidence(self):
        document = json.loads(self.model_path.read_text())
        document['trace']['param_refs'] = {'block_s': 'block_s', 's_wave': 's_wave'}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.json'
            path.write_text(json.dumps(document))
            model, _ = validate_model(path)
            self.assertEqual(model['param_refs'], {'block_s': 'block_s', 's_wave': 's_wave'})

    def test_event_levels_and_no_global_wave_barrier(self):
        expected_kinds = {'load': 960, 'direct': 240, 'peer_store': 720,
                          'fence': 192, 'notify': 720, 'wait': 720,
                          'consume': 720, 'output_store': 720}
        for scenario in self.model['scenarios']:
            events = {event['id']: event for event in scenario['events']}
            self.assertEqual(scenario['total_ticks'], 59)
            self.assertEqual(sum(1 for event in scenario['events'] if event['level'] == 0), 64)
            kinds = {}
            for event in scenario['events']:
                kinds[event['kind']] = kinds.get(event['kind'], 0) + 1
                self.assertEqual(
                    event['level'],
                    max((events[dep]['level'] for dep in event['depends_on']), default=-1) + 1)
                for dep in event['depends_on']:
                    if events[dep]['pid'] != event['pid']:
                        self.assertEqual((event['kind'], events[dep]['kind']),
                                         ('wait', 'notify'))
            self.assertEqual(kinds, expected_kinds)
            self.assertEqual(events['s0-t0-b0-g0:load']['level'], 0)
            self.assertEqual(events['s0-t0-b0-g0:direct']['level'], 1)
            self.assertEqual(events['s0-t1-b0-g0:notify']['level'], 11)
            self.assertEqual(events['s0-t1-b0-g0:wait']['level'], 12)
            self.assertEqual(events['s0-t1-b0-g0:consume']['level'], 13)
            self.assertEqual(events['s0-t1-b0-g0:output_store']['level'], 14)

    def test_unknown_mapping_cannot_play(self):
        document = json.loads(self.model_path.read_text())
        document['mapping'] = {'status': 'unknown', 'note': 'Unresolved pointer mapping'}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.json'
            path.write_text(json.dumps(document))
            model, _ = validate_model(path)
            self.assertEqual(model['scenarios'], [])


if __name__ == '__main__':
    unittest.main()
