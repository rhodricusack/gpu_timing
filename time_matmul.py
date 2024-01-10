import torch
import time



def main():
    # Timing different ways of calculating a random matrix and multiplying
    # Only Method A accelerates well on GPU as others all have CPU bottleneck
    nits = 100
    start = time.time()
    for it in range(nits):
        x = torch.randn(1000, 1000, device='cuda')
        y = torch.randn(1000, 1000, device='cuda')
        z = torch.matmul(x, y)
    print ("Method A: Time per iteration: ", (time.time() - start) / nits)

    for it in range(nits):
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
    print ("Method B: Time per iteration: ", (time.time() - start) / nits)

    x = torch.randn(nits,1000, 1000).cuda()
    y = torch.randn(nits,1000, 1000).cuda()
    z = torch.matmul(x, y)
    print ("Method C: Time per iteration: ", (time.time() - start) / nits)

    x = [torch.randn(1000, 1000).cuda() for it in range(nits)]
    y = [torch.randn(1000, 1000).cuda() for it in range(nits)]
    for it in range(nits):
        z = torch.matmul(x[it], y[it])
    print ("Method D: Time per iteration: ", (time.time() - start) / nits)

if __name__ == "__main__":
    main()