"""Collect gRPC requests until max batch size or timeout, then run them together."""

from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import Future


class RequestBatcher:
    def __init__(self, handler, max_batch=4, timeout=0.05, metrics=None):
        self.handler = handler
        self.max_batch = max(1, int(max_batch))
        self.timeout = float(timeout)
        self.metrics = metrics
        self._queue = queue.Queue()
        self._worker = threading.Thread(target=self._run, name="batcher", daemon=True)
        self._worker.start()

    def submit(self, item):
        future = Future()
        self._queue.put((item, future, time.perf_counter()))
        if self.metrics:
            self.metrics.set_queue(self._queue.qsize())
        return future.result()

    def _run(self):
        while True:
            item, future, _ = self._queue.get()
            batch = [(item, future)]
            deadline = time.perf_counter() + self.timeout
            while len(batch) < self.max_batch:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    batch.append(self._queue.get(timeout=remaining))
                except queue.Empty:
                    break
            if self.metrics:
                self.metrics.set_queue(self._queue.qsize())
                self.metrics.set_batch(len(batch))
            items = [entry[0] for entry in batch]
            futures = [entry[1] for entry in batch]
            try:
                results = self.handler(items)
                if results is None or len(results) != len(futures):
                    raise RuntimeError("batch handler must return one result per item")
                for future, result in zip(futures, results):
                    future.set_result(result)
            except Exception as error:
                for future in futures:
                    future.set_exception(error)
