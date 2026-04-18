"""API Polygraph — detect when cloud providers swap your model.

Two-layer detection:
  Layer 1: Timing profile (cheap, catches gross swaps like 3B→11B)
  Layer 2: Output distribution fingerprint (catches subtle swaps between same-size models)

Optional sidecar weighting: if the model was compressed with HXQ,
sidecar confidence tells you which tokens are most diagnostic.

The detection requires LOCAL GROUND TRUTH — you must run the open-weights
model yourself to know what the "correct" output distribution looks like.
The API response is compared against your local model's output.
"""

__version__ = "0.1.0"
