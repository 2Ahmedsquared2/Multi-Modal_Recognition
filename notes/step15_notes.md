# Step 15: Testing & Debugging

**Estimated Time:** 30 minutes
**Status:** ✅ Completed

---

## Completed
- ✅ Created `tests/` directory with full pytest-based test suite
- ✅ Shared fixtures in `conftest.py` (synthetic data, no dataset dependency)
- ✅ 7 test modules covering all core components
- ✅ 78 tests total — **all passing in 0.74s**
- ✅ Added `pytest` to `requirements.txt`
- ✅ End-to-end integration test (train → evaluate → verify above random chance)
- ✅ Model save/load roundtrip test
- ✅ Numerical gradient check (finite differences vs analytical)

## Test Suite Structure

```
tests/
├── __init__.py
├── conftest.py              # Shared fixtures (synthetic data, network instances)
├── test_activations.py      # 10 tests — ReLU forward/backward, Softmax stability
├── test_dense.py            # 10 tests — forward/backward, He init, gradient check
├── test_loss.py             #  9 tests — CE loss, gradient direction, stability
├── test_network.py          # 13 tests — architecture, forward/backward, predict
├── test_data_prep.py        # 11 tests — flatten, normalize, one-hot, splits, batches
├── test_trainer.py          # 10 tests — training loop, LR decay, early stopping
├── test_evaluator.py        # 12 tests — confusion matrix, P/R/F1, reporting
└── test_integration.py      #  3 tests — full pipeline, save/load, multi-architecture
```

**Total: 78 tests**

## Test Coverage by Module

| Module | Tests | What's Verified |
|--------|------:|-----------------|
| `activations.py` | 10 | ReLU zeros negatives, gradient mask, Softmax sums to 1, numerical stability |
| `dense.py` | 10 | Forward = X@W+b, He init scale, gradient shapes, numerical gradient check |
| `loss.py` | 9 | Perfect→low loss, wrong→high loss, gradient = (pred-true)/N, stability |
| `network.py` | 13 | Architecture building, param count, single step decreases loss, NaN-free |
| `data_prep.py` | 11 | Flatten shape, one-hot encoding, normalization, zero-variance handling, splits |
| `trainer.py` | 10 | Loss decreases, accuracy increases, LR decays, early stopping triggers |
| `evaluator.py` | 12 | CM shape/sum, P/R/F1 ranges, macro F1, print functions |
| `integration.py` | 3 | Full pipeline, save/load roundtrip, multiple architectures |

## How to Run

```bash
# Run all tests
source venv/bin/activate
python -m pytest tests/ -v

# Run a specific module
python -m pytest tests/test_activations.py -v

# Run a specific test
python -m pytest tests/test_network.py::TestNeuralNetwork::test_single_training_step_decreases_loss -v
```

## Test Results

```
tests/test_activations.py    10 passed
tests/test_data_prep.py      11 passed
tests/test_dense.py          10 passed
tests/test_evaluator.py      12 passed
tests/test_integration.py     3 passed
tests/test_loss.py            9 passed
tests/test_network.py        13 passed
tests/test_trainer.py        10 passed
────────────────────────────────────────
78 passed in 0.74s
```

## Design Decisions

1. **pytest over unittest** — cleaner syntax, auto-discovery, fixtures, better output; industry standard
2. **Synthetic data only** — tests don't need the real dataset; fixtures generate data with injected learnable signal
3. **Shared fixtures** in `conftest.py` — DRY; small_network, default_network, synthetic_dataset reused across modules
4. **Numerical gradient check** — verifies backprop correctness with finite differences (gold standard)
5. **Integration test** — single test that exercises the full pipeline (build → train → evaluate → verify)
6. **Save/load roundtrip** — proves model serialization preserves predictions exactly

## Key Tests Worth Highlighting (Portfolio)

| Test | Why It Matters |
|------|---------------|
| `test_numerical_gradient_check` | Proves backprop math is correct to machine precision (1e-10 error) |
| `test_single_training_step_decreases_loss` | Proves gradients point in the right direction |
| `test_normalize_zero_variance_feature` | Proves the zero-variance bug fix works |
| `test_full_pipeline` | End-to-end: train, evaluate, verify above random chance |
| `test_save_and_load_model` | Proves model serialization preserves predictions exactly |
| `test_numerical_stability_extreme_values` | Proves Softmax handles ±1000 logits without NaN |

## Performance Review

**PERFORMANCE REVIEW:**

**What Changed:**
- Added `tests/` directory with 7 test modules + conftest + integration
- Added `pytest` dependency to `requirements.txt`
- ~550 lines of test code total

**Performance Impact:**

1. **Runtime**: ⚡ **0.74 seconds for all 78 tests** — near-instant
2. **No production code changes** — zero impact on training/inference performance
3. **No dataset dependency** — tests use synthetic data, run anywhere
4. **CI-ready** — `python -m pytest tests/ -v` is a one-liner for any CI pipeline

**Portfolio Impact**: 📈 **HIGH**
- Shows you test your code (rare in ML projects)
- Numerical gradient check proves mathematical rigor
- Integration test proves the system works end-to-end
- Professional test organization (fixtures, parametrization, clear naming)

**Bottom Line**: 0.74s to verify the entire codebase is correct. Zero cost to production. Strong portfolio signal.

## Next Steps
1. ✅ ~~Testing & Debugging~~ (DONE!)
2. **Next:** Step 16: Portfolio Materials
