from deep_sort_realtime.deepsort_tracker import DeepSort
import logging

# Flask integration variables
DEFAULT_MAX_AGE = 20
DEFAULT_N_INIT = 2
DEFAULT_NMS_MAX_OVERLAP = 0.3
DEFAULT_MAX_COSINE_DISTANCE = 0.8


class Tracker:
  def __init__(self):
    try:
      self.object_tracker = DeepSort(
          max_age=20,
          n_init=2,
          nms_max_overlap=0.3,
          max_cosine_distance=0.8,
          nn_budget=None,
          override_track_class=None,
          embedder="mobilenet",
          half=True,
          bgr=True,
          embedder_model_name=None,
          embedder_wts=None,
          polygon=False,
          today=None
      )
      logging.info("DeepSort tracker initialized successfully")
    except Exception as e:
      logging.error(f"Failed to initialize tracker: {e}")
      raise e

  def track(self, detections, frame):
    try:
      tracks = self.object_tracker.update_tracks(detections, frame=frame)

      tracking_ids = []
      boxes = []
      for track in tracks:
        if not track.is_confirmed():
          continue
        tracking_ids.append(track.track_id)
        ltrb = track.to_ltrb()
        boxes.append(ltrb)

      return tracking_ids, boxes
    except Exception as e:
      logging.error(f"Tracking error: {e}")
      return [], []

  def update(self, detections, frame):
    """Alias for track method to match the interface expected by main script"""
    return self.track(detections, frame)