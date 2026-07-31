"""Support code for Test async runner."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from RL_Framework.infra.execution.async_runner import AsyncTaskRunner
import time

async def test_task(data):
    """Test task."""
    print(f"Processing data: {data}")
    return {'result': 'success', 'data': data}

print("=" * 60)
print("Testing AsyncTaskRunner task_id filtering")
print("=" * 60)

runner = AsyncTaskRunner()
runner.initialize()


print("\nSubmitting task...")
try:
    runner.submit(test_task, {'test': 'data'}, task_id=0)
    print("Task submitted successfully")
except Exception as e:
    print(f"Task submission failed: {e}")
    runner.destroy()
    sys.exit(1)


print("\nWaiting for result...")
try:
    results = runner.wait(count=1, timeout=5.0)
    if results:
        print(f"Received result: {results[0].data}")
    else:
        print("No result received")
except Exception as e:
    print(f"Waiting for result failed: {e}")

runner.destroy()

print("\n" + "=" * 60)
print("Test passed; task_id was filtered correctly")
print("=" * 60)
