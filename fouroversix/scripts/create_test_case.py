from .resources import Dependency, app, get_image

img = get_image(dependencies=[Dependency.transformer_engine, Dependency.fouroversix])

with img.imports():
    import torch
    from fouroversix import AdaptiveBlockScalingRule, QuantizeBackend, quantize_to_fp4
    from fouroversix.quantize import from_blocked


@app.function(image=img, gpu="B200")
def create_test_case(
    backend_a: str = "cuda",
    backend_b: str = "transformer_engine",
    scale_rule: str = "mse",
) -> None:
    M, N = 1024, 1024  # noqa: N806

    torch.set_printoptions(precision=10)

    backend_a = QuantizeBackend(backend_a)
    backend_b = QuantizeBackend(backend_b)
    scale_rule = AdaptiveBlockScalingRule(scale_rule)

    for random_seed in range(10):
        torch.manual_seed(random_seed)

        x = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
        out_a = quantize_to_fp4(
            x,
            backend=backend_a,
            scale_rule=scale_rule,
        )
        out_b = quantize_to_fp4(
            x,
            backend=backend_b,
            scale_rule=scale_rule,
        )
        x_sf_a = from_blocked(out_a.scale_factors.bfloat16(), (M, N // 16))
        x_sf_b = from_blocked(out_b.scale_factors.bfloat16(), (M, N // 16))

        print(f"x absmax: {x.abs().max()}")

        if not torch.allclose(out_a.amax, out_b.amax):
            print("Backends A and B have different amax values!")
            print(f"{backend_a}: {out_a.amax}")
            print(f"{backend_b}: {out_b.amax}")
            return

        if not torch.allclose(x_sf_a.bfloat16(), x_sf_b.bfloat16()):
            mismatch_prop = (x_sf_a != x_sf_b).sum() / x_sf_a.numel()
            print(
                "Backends A and B have different scale factors! "
                f"{mismatch_prop:.2%} mismatch",
            )

            [i, *_], [j, *_] = torch.where(x_sf_a != x_sf_b)
            print(backend_a)
            print("sf", x_sf_a[i, j])
            print("e2m1", out_a.e2m1_values[i, 8 * j : 8 * (j + 1)])
            print(backend_b)
            print("sf", x_sf_b[i, j])
            print("e2m1", out_b.e2m1_values[i, 8 * j : 8 * (j + 1)])
            print("original")
            print("x", x[i, 16 * j : 16 * (j + 1)])
            return

        if not torch.allclose(out_a.e2m1_values, out_b.e2m1_values):
            mismatch_prop = (
                out_a.e2m1_values != out_b.e2m1_values
            ).sum() / out_a.e2m1_values.numel()
            print(
                "Backends A and B have different e2m1 values! "
                f"{mismatch_prop:.2%} mismatch",
            )

            [i, *_], [j, *_] = torch.where(out_a.e2m1_values != out_b.e2m1_values)
            print(i, j)
            print("normconst", out_a.amax)
            print("sf", x_sf_a[i, j // 8])
            print(backend_a)
            print("e2m1", out_a.e2m1_values[i, 8 * (j // 8) : 8 * (j // 8 + 1)])
            print(backend_b)
            print("e2m1", out_b.e2m1_values[i, 8 * (j // 8) : 8 * (j // 8 + 1)])
            print("original")
            print("x", x[i, 16 * (j // 8) : 16 * (j // 8 + 1)])
            return
