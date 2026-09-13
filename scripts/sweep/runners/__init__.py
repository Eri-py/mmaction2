"""Registry mapping a model entry's `type` to its runner function."""
from .recognizer import run_recognizer
from .skeleton import run_skeleton_topdown

RUNNERS = {
    'recognizer': run_recognizer,
    'skeleton_topdown': run_skeleton_topdown,
}
