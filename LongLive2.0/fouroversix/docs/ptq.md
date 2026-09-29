# Running PTQ Experiments

## Install Dependencies

Before doing anything, make sure you've installed fouroversix with our test dependencies:

```bash
pip install -e .[evals] --no-build-isolation
```

Also, make sure you've cloned all of our submodules:

```bash
git submodule update --init
```

Then, depending on which PTQ method you would like to test, you may need to run some additional commands.

### AWQ

```bash
pip install --no-deps third_party/llm-awq
```

### GPTQ

1. Install Fast Hadamard Transform

```bash
pip install --no-build-isolation third_party/fast-hadamard-transform
```

2. Install QuTLASS

```bash
pip install --no-build-isolation third_party/qutlass
```

3. Install FP-Quant

```bash
pip install third_party/fp-quant/inference_lib
```

### High Precision

No installation necessary!

### Round-to-Nearest (RTN)

No installation necessary!

### SmoothQuant

No installation necessary!

### SpinQuant

1. Install Fast Hadamard Transform

```bash
pip install --no-build-isolation third_party/fast-hadamard-transform
```

2. Downgrade Transformers if your installation is up-to-date

```bash
pip install "transformers<5.0"
```