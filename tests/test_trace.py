import copy
import json
from pathlib import Path
import tempfile
import unittest

from op_show.model import ValidationError, validate_model

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / 'examples/ascend_qkv_alltoall.json'
SOURCE = ROOT / 'examples/sources/ascend_qkv_all2all.py'


class TraceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.source = validate_model(EXAMPLE)

    def test_source_backed_example_and_tail(self):
        self.assertIn('kernel_qkv_fuse_a2a_head_grouped', self.source)
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
        # Same-wave PIDs are independent: no global wave barrier is invented.
        first = events['s0-t0-b0-g0:load']
        other = events['s0-t0-b1-g0:load']
        self.assertNotIn(other['id'], first['depends_on'])

    def _invalid(self, mutate, message):
        document = copy.deepcopy(json.loads(EXAMPLE.read_text()))
        document['source']['path'] = str(SOURCE)
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

    def test_unknown_mapping_cannot_play(self):
        document = json.loads(EXAMPLE.read_text())
        document['source']['path'] = str(SOURCE)
        document['mapping'] = {'status': 'unknown', 'note': 'Unresolved pointer mapping'}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.json'
            path.write_text(json.dumps(document))
            model, _ = validate_model(path)
            self.assertEqual(model['scenarios'], [])


if __name__ == '__main__':
    unittest.main()
