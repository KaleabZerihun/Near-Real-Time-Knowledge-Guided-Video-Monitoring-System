from __future__ import annotations
import time
from src.db import repository

def _cutoff_seconds(days: float) -> float:
    return time.time() - (days * 24 * 60 * 60)

def archive_old_events(*, keep_days: float = 30) -> dict[str, int]:
    # Archives events older than keep_days into events_archive, then deletes them from events.
    # Returns how many were archived and deleted.
   
    cutoff = _cutoff_seconds(keep_days)
    now = time.time()

    with repository._connect() as conn:
        # Copy old rows into archive, ignore if already archived
        cur_ins = conn.execute(
            """
            INSERT OR IGNORE INTO events_archive
              (clip_id, stream_id, ts_start, ts_end, label, confidence, frames_json, vad_json, created_at, archived_at)
            SELECT
              clip_id, stream_id, ts_start, ts_end, label, confidence, frames_json, vad_json, created_at, ?
            FROM events
            WHERE created_at < ?
            """,
            (now, cutoff),
        )

        # Delete from main table after archiving
        cur_del = conn.execute(
            "DELETE FROM events WHERE created_at < ?",
            (cutoff,),
        )

        conn.commit()
    return {
        "events_archived": cur_ins.rowcount,
        "events_deleted_from_main": cur_del.rowcount,
    }
    
# Deletes frame_batches and vad_predictions that never formed an event (or archived event)
# and are older than keep_days.   
def purge_orphaned_inputs(*, keep_days: float = 7) -> dict[str, int]:
    cutoff = _cutoff_seconds(keep_days)
    
    # frames with no matching event (live or archived)
    with repository._connect() as conn:
        cur1 = conn.execute(
            """
            DELETE FROM frame_batches
            WHERE created_at < ?
              AND clip_id NOT IN (SELECT clip_id FROM events)
              AND clip_id NOT IN (SELECT clip_id FROM events_archive)
            """,
            (cutoff,),
        )
        # vad preds with no matching event (live or archived)
        cur2 = conn.execute(
            """
            DELETE FROM vad_predictions
            WHERE created_at < ?
              AND clip_id NOT IN (SELECT clip_id FROM events)
              AND clip_id NOT IN (SELECT clip_id FROM events_archive)
            """,
            (cutoff,),
        )
        conn.commit()
    return {
        "frame_batches_deleted": cur1.rowcount,
        "vad_predictions_deleted": cur2.rowcount,
    }

def vacuum(*, run: bool = False) -> None:
    if not run:
        return

    with repository._connect() as conn:
        conn.execute("VACUUM;")
        conn.commit()

def run_retention_once(
    *,
    keep_events_days: float = 30,
    keep_orphans_days: float = 7,
    do_vacuum: bool = False,
) -> dict[str, object]:
    archive_result = archive_old_events(keep_days=keep_events_days)
    orphans = purge_orphaned_inputs(keep_days=keep_orphans_days)
    vacuum(run=do_vacuum)

    return {
        **archive_result,
        **orphans,
        "vacuum": do_vacuum,
    }