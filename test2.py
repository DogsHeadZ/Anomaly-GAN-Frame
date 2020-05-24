import torch

a = torch.tensor([1, 2, 3.], requires_grad=True)
b = torch.tensor([4, 5, 6.], requires_grad=True)
c = torch.tensor([7, 8, 9.], requires_grad=True)

outa = a.sigmoid()
outb = b.sigmoid()
outc = c.sigmoid()

# loss1 = (outa.detach() + outb).sum()
# loss1.backward(retain_graph=True)
# print(a.grad)
#
#
# loss2 = (2*outa + outc).sum()
#
# loss2.backward()
# print(a.grad)

loss3 = (outa.detach()*outb).sum()
loss3.backward()
print(b.grad)