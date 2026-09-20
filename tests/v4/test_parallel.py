import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from jevbench_v4.data import prepare
from jevbench_v4.parallel import assign_gpus, discover_gpus, gpu_worker_context, run_local, run_notebook_command
from jevbench_v4.storage import read_json

INVENTORY = [{"index": i, "selector": str(3 + 4 * i), "name": "Test T4", "total_memory_bytes": 16 * 1024**3}
             for i in range(2)]

# Real child interpreters exercise scheduling and inherited environment without CUDA/SDKs.
PROBE = r'''
import json, os, sys, time
from pathlib import Path
directory, mode = Path(sys.argv[1]), sys.argv[2]
directory.mkdir(parents=True, exist_ok=True)
assignment = json.loads(os.environ['JEVBENCH_V4_GPU_WORKER'])
name = assignment['provider']
started = time.monotonic()
(directory / (name + '.ready')).write_text('ready')
if mode in ('parallel', 'failure'):
    deadline = time.monotonic() + 15
    while len(list(directory.glob('*.ready'))) != 2:
        if time.monotonic() > deadline:
            raise RuntimeError('Sibling was not launched concurrently')
        time.sleep(.01)
if mode == 'failure':
    if name == 'von':
        sys.exit(7)
    time.sleep(20)
else:
    time.sleep(.05)
result = {'pid': os.getpid(), 'started': started, 'finished': time.monotonic(),
          'mask': os.environ['CUDA_VISIBLE_DEVICES'], 'threads': os.environ['OMP_NUM_THREADS']}
(directory / (name + '.json')).write_text(json.dumps(result))
print('Completed', name)
'''


class ParallelTests(unittest.TestCase):
    def setUp(self):
        directory = Path(__file__).resolve().parents[2] / "generated"
        directory.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=directory)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "study"
        prepare(self.root)

    def launch_probe(self, mode, inventory=INVENTORY, **options):
        original = subprocess.Popen
        children, commands = [], []

        def launch(command, **kwargs):
            commands.append(command)
            child = original([sys.executable, "-c", PROBE, str(self.root / "probe"), mode], **kwargs)
            children.append(child)
            return child

        self.children = children
        with patch("jevbench_v4.parallel.discover_gpus", return_value=inventory), \
                patch("jevbench_v4.parallel.subprocess.Popen", side_effect=launch):
            result = run_local(self.root, **options)
        return result, children, commands

    def test_two_real_processes_overlap_and_have_disjoint_gpu_visibility(self):
        result, children, commands = self.launch_probe("parallel", min_gpus=2)
        von = read_json(self.root / "probe/von.json")
        laya = read_json(self.root / "probe/laya.json")
        self.assertNotEqual(von["pid"], laya["pid"])
        self.assertLess(max(von["started"], laya["started"]), min(von["finished"], laya["finished"]))
        self.assertEqual((von["mask"], laya["mask"]), ("3", "7"))
        self.assertEqual((von["threads"], laya["threads"]), ("2", "2"))
        self.assertEqual([r["exit_code"] for r in result], [0, 0])
        for child, command in zip(children, commands):
            self.assertEqual(child.poll(), 0)
            self.assertIn("--isolated-gpu", command)
            self.assertEqual(command[command.index("--device") + 1], "cuda:0")
            self.assertEqual(command[command.index("--min-interval") + 1], "0")
        self.assertFalse((self.root / "execution/local/.lock").exists())
        self.assertEqual(read_json(self.root / "execution/local/pilot/status.json")["state"], "complete")

    def test_single_gpu_runs_serially_without_overlapping_model_processes(self):
        self.launch_probe("serial", inventory=INVENTORY[:1])
        von = read_json(self.root / "probe/von.json")
        laya = read_json(self.root / "probe/laya.json")
        self.assertLessEqual(von["finished"], laya["started"])
        self.assertEqual(von["mask"], laya["mask"])

    def test_worker_failure_stops_sibling_and_releases_lock(self):
        with self.assertRaisesRegex(RuntimeError, "exit 7"):
            self.launch_probe("failure", min_gpus=2)
        self.assertTrue(all(child.poll() is not None for child in self.children))
        self.assertFalse((self.root / "execution/local/.lock").exists())
        result = read_json(self.root / "execution/local/pilot/status.json")
        self.assertEqual(len(result["workers"]), 2)
        self.assertEqual(result["state"], "incomplete")
        self.assertIn("interrupted", [worker.get("status") for worker in result["workers"]])

    def test_gpu_requirements_and_duplicate_assignments_are_rejected(self):
        for inventory, indices, minimum in (([], None, 1), (INVENTORY[:1], None, 2),
                                             (INVENTORY, [0, 0], 1), (INVENTORY, [2], 1)):
            with self.assertRaises(ValueError):
                assign_gpus(inventory, indices, minimum)
        assignments = assign_gpus(INVENTORY, [1, 0], 2)
        self.assertEqual([job["gpu"]["selector"] for job in assignments], ["7", "3"])

    def test_existing_visibility_mask_is_preserved_when_selecting_child_devices(self):
        cuda = SimpleNamespace(device_count=lambda: 2,
                               get_device_properties=lambda i: SimpleNamespace(name="T4", total_memory=16))
        with patch.dict(sys.modules, {"torch": SimpleNamespace(cuda=cuda)}), \
                patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-first,GPU-second"}):
            devices = discover_gpus()
        self.assertEqual([gpu["selector"] for gpu in devices], ["GPU-first", "GPU-second"])

    def test_child_rejects_multiple_visible_gpus_before_allocating(self):
        assignment = assign_gpus(INVENTORY)[0]
        cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 2)
        with patch.dict(sys.modules, {"torch": SimpleNamespace(cuda=cuda)}), \
                patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "3", "JEVBENCH_V4_GPU_WORKER": json.dumps(assignment)}):
            with self.assertRaisesRegex(RuntimeError, "exactly one"):
                gpu_worker_context("von")

    def test_duplicate_launcher_cannot_share_provider_ledgers(self):
        lock = self.root / "execution/local/.lock"
        lock.parent.mkdir(parents=True)
        lock.write_text("another-launcher", encoding="utf-8")
        with patch("jevbench_v4.parallel.discover_gpus", return_value=INVENTORY), \
                patch("jevbench_v4.parallel.subprocess.Popen") as spawn:
            with self.assertRaisesRegex(RuntimeError, "owns this run"):
                run_local(self.root)
            spawn.assert_not_called()
        self.assertEqual(lock.read_text(), "another-launcher")

    def test_changing_gpu_assignment_requires_a_new_run(self):
        self.launch_probe("parallel")
        with patch("jevbench_v4.parallel.discover_gpus", return_value=INVENTORY), \
                patch("jevbench_v4.parallel.subprocess.Popen") as spawn:
            with self.assertRaisesRegex(ValueError, "Frozen manifest changed"):
                run_local(self.root, indices=[1, 0])
            spawn.assert_not_called()
        self.assertFalse((self.root / "execution/local/.lock").exists())

    def test_notebook_interrupt_reaches_launcher_before_forced_kill(self):
        process = Mock()
        process.wait.side_effect = [KeyboardInterrupt(), 0]
        with patch("jevbench_v4.parallel.subprocess.Popen", return_value=process):
            with self.assertRaises(KeyboardInterrupt):
                run_notebook_command(["worker"], cwd=self.root)
        process.send_signal.assert_called_once_with(signal.SIGINT)
        process.kill.assert_not_called()
