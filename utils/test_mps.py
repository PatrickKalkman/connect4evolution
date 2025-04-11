import time

import torch


def test_mps_availability():
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")
    return torch.backends.mps.is_available()


def compare_performance():
    if not test_mps_availability():
        return

    size = 4000  # Larger matrix size
    iterations = 5  # Multiple iterations

    # Warm-up run for MPS
    print("\n--- Warm-up run ---")
    mps_device = torch.device("mps")
    a_warmup = torch.randn(1000, 1000, device=mps_device)
    b_warmup = torch.randn(1000, 1000, device=mps_device)
    for _ in range(3):  # Multiple warm-up iterations
        c_warmup = torch.matmul(a_warmup, b_warmup)
        torch.mps.synchronize()
    print("Warm-up complete")

    # CPU performance test
    print("\n--- CPU Performance Test ---")
    cpu_device = torch.device("cpu")
    cpu_times = []

    for i in range(iterations):
        a_cpu = torch.randn(size, size, device=cpu_device)
        b_cpu = torch.randn(size, size, device=cpu_device)

        start_time = time.time()
        c_cpu = torch.matmul(a_cpu, b_cpu)
        cpu_time = time.time() - start_time
        cpu_times.append(cpu_time)

        print(f"  Iteration {i + 1}: {cpu_time:.4f} seconds")

    avg_cpu_time = sum(cpu_times) / len(cpu_times)
    print(f"Average CPU time: {avg_cpu_time:.4f} seconds")

    # MPS performance test
    print("\n--- MPS Performance Test ---")
    mps_times = []

    for i in range(iterations):
        a_mps = torch.randn(size, size, device=mps_device)
        b_mps = torch.randn(size, size, device=mps_device)

        start_time = time.time()
        c_mps = torch.matmul(a_mps, b_mps)
        torch.mps.synchronize()  # Wait for MPS operations to complete
        mps_time = time.time() - start_time
        mps_times.append(mps_time)

        print(f"  Iteration {i + 1}: {mps_time:.4f} seconds")

    avg_mps_time = sum(mps_times) / len(mps_times)
    print(f"Average MPS time: {avg_mps_time:.4f} seconds")

    # Compare results
    print(f"\nSpeedup: {avg_cpu_time / avg_mps_time:.2f}x")

    # Test on typical neural network operations (convolution)
    print("\n--- Neural Network Operations Test ---")

    # Create input for conv2d (batch_size, channels, height, width)
    input_size = (32, 3, 224, 224)
    kernel_size = (64, 3, 3, 3)

    # CPU convolution
    input_cpu = torch.randn(*input_size, device=cpu_device)
    kernel_cpu = torch.randn(*kernel_size, device=cpu_device)

    start_time = time.time()
    output_cpu = torch.nn.functional.conv2d(input_cpu, kernel_cpu, padding=1)
    cpu_conv_time = time.time() - start_time
    print(f"CPU convolution time: {cpu_conv_time:.4f} seconds")

    # MPS convolution
    input_mps = torch.randn(*input_size, device=mps_device)
    kernel_mps = torch.randn(*kernel_size, device=mps_device)

    start_time = time.time()
    output_mps = torch.nn.functional.conv2d(input_mps, kernel_mps, padding=1)
    torch.mps.synchronize()
    mps_conv_time = time.time() - start_time
    print(f"MPS convolution time: {mps_conv_time:.4f} seconds")

    print(f"Convolution speedup: {cpu_conv_time / mps_conv_time:.2f}x")


if __name__ == "__main__":
    print("Testing MPS (Metal Performance Shaders) for PyTorch on Mac\n")
    compare_performance()
