import argparse

from sgemm import sgemm_interface

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Entrypoint script for sgemm kernels",
    )
    parser.add_argument(
        "M",
        type=int,
        help="M dimension of the GEMM",
    )
    parser.add_argument(
        "N",
        type=int,
        help="N dimension of the GEMM",
    )
    parser.add_argument(
        "K",
        type=int,
        help="K dimension of the GEMM",
    )
    parser.add_argument(
        "alpha",
        type=int,
        default=1,
        help="value of alpha scaler",
    )
    parser.add_argument(
        "beta",
        type=int,
        default=0,
        help="value of beta scaler",
    )
    parser.add_argument(
        "version",
        type=int,
        default=0,
        choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        help="version of sgemm kernel to use",
    )

    args = parser.parse_args()
    sgemm_interface.sgemm_rand(
        M=args.M,
        N=args.N,
        K=args.K,
        alpha=args.alpha,
        beta=args.beta,
        version=args.version,
    )
