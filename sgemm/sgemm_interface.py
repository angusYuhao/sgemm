import torch


def sgemm_cpu(A, B, C, alpha, beta):
    return alpha * (A @ B) + beta * C


def sgemm_rand(M, N, K, alpha, beta, version):
    assert torch.cuda.is_available(), "cuda must be available"
    # create input matrices A, B, C
    A = torch.rand(M, K)
    B = torch.rand(K, N)
    C = torch.rand(M, N)

    print("sgemm inputs:")
    print("\nA:")
    print(A)
    print("\nB:")
    print(B)
    print("\nC:")
    print(C, flush=True)

    # call cpu impl
    result_cpu = sgemm_cpu(A, B, C, alpha, beta)

    # move inputs to device
    A_d = A.cuda()
    B_d = B.cuda()
    C_d = C.cuda()

    # call cuda impl
    torch.ops.sgemm.sgemm(A_d, B_d, C_d, alpha, beta, version)
    result_cuda = C_d.cpu()

    print("\nsgemm results:")
    print("\n CPU results:")
    print(result_cpu, flush=True)
    print("\n CUDA results:")
    print(result_cuda, flush=True)
