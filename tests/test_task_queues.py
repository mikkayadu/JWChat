import pytest
from kombu.utils.scheduling import cycle_by_name

from ingestion.tasks import worker


def _queue_for(task_name):
    return worker.app.amqp.router.route({}, task_name)["queue"].name


@pytest.mark.parametrize("task", ["ingest_pdf", "ingest_url", "ingest_video"])
def test_ingest_tasks_use_the_ingest_queue(task):
    assert _queue_for(f"ingestion.tasks.worker.{task}") == worker.INGEST_QUEUE


def test_crawl_tasks_stay_on_the_default_queue():
    assert _queue_for("ingestion.tasks.worker.crawl_and_enqueue") == "celery"


def test_worker_polls_the_ingest_queue_first():
    # Kombu hands the strategy a set of queue names, so set order mustn't matter.
    strategy = worker.app.conf.broker_transport_options["queue_order_strategy"]
    names = [q.name for q in worker.app.conf.task_queues]
    for order in (names, names[::-1]):
        cycle = cycle_by_name(strategy)()
        cycle.update(list(order))
        assert cycle.consume(len(order))[0] == worker.INGEST_QUEUE
